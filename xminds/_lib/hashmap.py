import numpy
import warnings

from .compat import structured_cast
# from .logger import logger
from ..lib.arraybase import set_or_add_to_structured, to_structured

# Pure numpy implementation of hashmaps
# This file implements low level classes, but as a user you should try to use:
#    unique, unique_count, search, get_dense_labels_map, factorize or Hashmap

UINT64 = numpy.uint64
INV_PHI = UINT64(11400714819323198485)  # 1<<64/phi
STEP_MULT = UINT64(11223344556677889999)  # arbitrary, could be optimized
_DEFAULT = object()


class _BaseHashmap(object):
    """
    template for HashMap keys => values

    * hashmaps gives efficient O(n) search in un-sorted set of keys or map keys => values
    * hashmaps without values are called hashtables
    * only relies on numpy array, so can be moved to shared memory
    * this implementation relies on Fibonacci hashing
    * we support any dtype for the keys and values, by implementing three sub-classes:
    * - vanilla hash for uint64 keys
    * - python's tuple-hash for struct of uint64 fields
    * - python's object hash for any object
    * for the first two, the keys are casted into uint64 or uint64-tuple
    * when possible for struct of a few tiny field, we view them as a single uint64
    """

    @classmethod
    def new(cls, keys, values=None, cast_dtype=None, view_dtype=None, empty=_DEFAULT):
        """
        :param array keys: (n,) key-dtype array
        :param array? values: (n,) val-dtype array
        :param dtype? cast_dtype: dtype to cast ``keys``
        :param dtype? view_dtype: dtype to view ``keys``
        :param int? empty: empty value (default: 0)
        """
        original_dtype = keys.dtype
        cast_dtype = cast_dtype or keys.dtype
        view_dtype = view_dtype or keys.dtype
        _keys = cls._cast(keys, cast_dtype, view_dtype)
        # keys = structured_cast(keys, dtype)
        empty = cls._choose_empty_value(_keys, view_dtype, empty)
        n, = _keys.shape
        log_size = (cls.init_space_ratio(n) * n - 1).bit_length()
        size = 1 << log_size
        table = numpy.full(size, empty, dtype=view_dtype)
        values_table = numpy.zeros(
            size, dtype=values.dtype) if values is not None else None
        hashmap = cls(table, values_table, empty,
                      log_size, original_dtype, cast_dtype)
        hashmap._set_initial(_keys, values)
        return hashmap

    def __init__(self, table, values, empty, log_size, original_dtype, cast_dtype, can_resize=True):
        """ low-level constructor, use .new instead """
        self._table = table
        self.values = values
        self._empty = empty
        self.log_size = log_size
        self.can_resize = can_resize
        self.shift = UINT64(64 - self.log_size)
        self.n_used = (self._table != self._empty).sum()
        self.original_dtype = original_dtype
        self.cast_dtype = cast_dtype

    def get_sharable_arrays(self):
        return {
            'table': self._table,
            'values': self.values,
        }

    def copy_with_shared_arrays(self, table, values):
        cls = type(self)
        return cls(table, values, self._empty, self.log_size, self.original_dtype, self.cast_dtype,
                   can_resize=False)

    @property
    def size(self):
        return self._table.size

    @property
    def nbytes(self):
        summed = self._table.nbytes
        if self.values is not None:
            summed += self.values.nbytes
        return summed

    def set_many(self, keys, values=None):
        """
        :param array keys: (n,) key-dtype array
        :param array? values: (n,) val-dtype array
        """
        _keys = self._cast(keys, self.cast_dtype, self._table.dtype)
        if values is not None and self.values.dtype.names:
            # align fields
            values = values[[k for k in self.values.dtype.names]]
        if _keys.size > 0 and (_keys == self._empty).any():
            self._change_empty(_keys)
        n, = _keys.shape
        if self.min_space_ratio(n) * (self.n_used + n) > self.size:
            self._resize(self.n_used + n)
        # step=0
        step = UINT64(0)
        indexes = self._shifted_hash(_keys, step)
        done = False
        max_steps = self.max_steps(n)
        for _ in range(max_steps):
            available = self._table[indexes] == self._empty
            available_indexes = indexes[available]
            self._table[available_indexes] = _keys[available]
            collisions = self._table[indexes] != _keys
            if values is not None:
                self.values[indexes[~collisions]] = values[~collisions]
            if not collisions.any():
                done = True
                break
            # next step: work only in `collisions`
            step += UINT64(1)
            _keys = _keys[collisions]
            if values is not None:
                values = values[collisions]
            indexes = self._shifted_hash(_keys, step)
        if not done:
            raise RuntimeError(
                'could not set_many within {} steps'.format(max_steps))
        self.n_used = (self._table != self._empty).sum()

    def search(self, keys):
        """
        :param array keys: (n,) key-dtype array
        :returns: tuple(
            indexes: (n,) uint64 array,
            found: (n,) bool array,
        )
        """
        _keys = self._cast(keys, self.cast_dtype, self._table.dtype)
        if _keys.size > 0 and (_keys == self._empty).any():
            self._change_empty(_keys)
        n, = _keys.shape
        # working idx in all (lazy build at first collisions)
        idx_in_all = None
        # step=0
        step = UINT64(0)
        all_indexes = self._shifted_hash(_keys, step)
        indexes = all_indexes
        table_values = self._table[indexes]
        all_found = table_values != self._empty
        found = all_found
        done = False
        max_steps = self.max_steps(n)
        for _ in range(max_steps):
            collisions = found & (table_values != _keys)
            if not collisions.any():
                done = True
                break
            # next step: work only in `collisions`
            step += UINT64(1)
            _keys = _keys[collisions]
            if idx_in_all is None:
                idx_in_all = numpy.where(collisions)[0]
            else:
                idx_in_all = idx_in_all[collisions]
            indexes = self._shifted_hash(_keys, step)
            all_indexes[idx_in_all] = indexes
            table_values = self._table[indexes]
            found = table_values != self._empty
            all_found[idx_in_all] = found
        if not done:
            raise RuntimeError(
                'could not search within {} steps'.format(max_steps))
        return all_indexes, all_found

    def contains(self, keys):
        """
        :param array keys: (n,) key-dtype array
        :returns: (n,) bool array
        """
        _, found = self.search(keys)
        return found

    def get_many(self, keys):
        """
        :param array keys: (n,) key-dtype array
        :returns: tuple(
            values: (n,) val-dtype array,
            found: (n,) bool array,
        )
        """
        if self.values is None:
            raise ValueError(
                '`get_many` is only available when values is not None, use `search`')
        indexes, found = self.search(keys)
        values = self.values[indexes]
        return values, found

    def unique_keys(self, return_table_mask=False, return_values=False):
        """ :returns: (
            (n,) key-dtype array,
            [if return_table_mask] (m,) bool array with n "1s"
            [if return_values] (n,) val-dtype array,
        )
         """
        has_key = self._table != self._empty
        _keys = self._table[has_key]
        keys = self._cast_back(_keys)
        if not return_table_mask and not return_values:
            return keys
        out = (keys,)
        if return_table_mask:
            out = out + (has_key,)
        if return_values:
            out = out + (self.values[has_key],)
        return out

    def keys_hash(self):
        """ returns order-invarient hash of keys (not __hash__ because we don't look at values) """
        # combine raw hash (pre-shift) by global sum
        has_key = self._table != self._empty
        _keys = self._table[has_key]
        # compute raw uint64 hash
        _keys_hsh = self._hash(_keys, UINT64(0))
        # aggregate
        hsh_uint64 = numpy.bitwise_xor.reduce(_keys_hsh)
        return int(hsh_uint64.view(numpy.int64))  # return as int

    @classmethod
    def init_space_ratio(cls, n):
        """ multiplier to set table size """
        return 4 if n < (1<<26) else 2

    @classmethod
    def min_space_ratio(cls, n):
        """ when to trigger resize """
        return 3 if n < (1<<26) else 1.5

    @classmethod
    def max_steps(cls, n):
        """ prevent infinite loop by a cap on the nb of steps (heuristic but very large) """
        return max(64, n // (32 * cls.init_space_ratio(n)))

    def _resize(self, n):
        log_size = int(self.init_space_ratio(n) * n - 1).bit_length()
        has_value = self._table != self._empty
        _keys = self._table[has_value]
        if self.values is not None:
            values = self.values[has_value]
        else:
            values = None
        if values is not None:
            self.values = numpy.zeros(self.size, dtype=values.dtype)
        # re-allocate tables
        self.log_size = log_size
        self.shift = UINT64(64 - self.log_size)
        new_size = 1 << log_size
        self._table = numpy.full(
            new_size, self._empty, dtype=self._table.dtype)
        if values is not None:
            self.values = numpy.zeros(new_size, dtype=values.dtype)
        self._set_initial(_keys, values)

    def _set_initial(self, _keys, values):
        n, = _keys.shape
        # step=0
        step = UINT64(0)
        indexes = self._shifted_hash(_keys, step)
        self._table[indexes] = _keys
        if values is not None:
            self.values[indexes] = values
        done = False
        max_steps = self.max_steps(n)
        for _ in range(max_steps):
            collisions = self._table[indexes] != _keys
            if not collisions.any():
                done = True
                break
            # next step: work only in `collisions`
            step += UINT64(1)
            _keys = _keys[collisions]
            if values is not None:
                values = values[collisions]
            # TOOPTIMIZE re-use computed hashes
            indexes = self._shifted_hash(_keys, step)
            available = self._table[indexes] == self._empty
            available_indexes = indexes[available]
            self._table[available_indexes] = _keys[available]
            if values is not None:
                self.values[available_indexes] = values[available]
        if not done:
            raise RuntimeError(
                'could not _set_initial within {} steps'.format(max_steps))
        self.n_used = (self._table != self._empty).sum()

    def _change_empty(self, new_keys):
        # edge case: the empty value we set clashes with a new key
        _uniq_keys = self._table[self._table != self._empty]
        all_keys = numpy.r_[_uniq_keys, new_keys]
        new_empty = self._choose_empty_value(all_keys, self._table.dtype)
        self._table[self._table == self._empty] = new_empty
        self._empty = new_empty

    @classmethod
    def _hash(cls, _keys, step):
        raise NotImplementedError()

    def _shifted_hash(self, _keys, step):
        _hash = self._hash(_keys, step)
        _hash >>= self.shift
        return _hash

    @classmethod
    def _fibonacci_hash_uint64(cls, _keys, step, copy=True):
        if copy:
            _keys = _keys.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', r'overflow encountered in ulong(long)?_scalars')
            _keys += STEP_MULT * step
            _keys *= INV_PHI
            return _keys

    @classmethod
    def _choose_empty_value(cls, _keys, dtype, empty=_DEFAULT):
        raise NotImplementedError()

    @classmethod
    def _cast(cls, keys, cast_dtype, view_dtype):
        if keys.dtype != cast_dtype:
            keys = structured_cast(keys, cast_dtype)
        if keys.dtype != view_dtype:
            keys = keys.view(view_dtype)
        return keys

    def _cast_back(self, keys):
        if keys.dtype != self.cast_dtype:
            keys = keys.view(self.cast_dtype)
        if keys.dtype != self.original_dtype:
            keys = structured_cast(keys, self.original_dtype)
        return keys


class UInt64Hashmap(_BaseHashmap):
    """
    a mapping from uint64 to arbitrary values in a numpy array
    consider using the higher-level ``Hashmap`` instead
    """

    @classmethod
    def new(cls, keys, values=None, cast_dtype=None, view_dtype=None, empty=_DEFAULT):
        """
        :param array keys: (n,) uint64 array
        :param array? values: (n,) val-dtype array
        :param dtype? cast_dtype: dtype to cast ``keys`` (default: uint64)
        :param dtype? view_dtype: dtype to view ``keys`` (default: uint64)
        :param object? empty: empty value (default: row of 0)
        """
        cast_dtype = cast_dtype or UINT64
        view_dtype = view_dtype or UINT64
        return super(UInt64Hashmap, cls).new(keys, values, cast_dtype, view_dtype, empty)

    @classmethod
    def _hash(cls, _keys, step):
        # (_keys << 21) ^ (_keys >> 33)
        _keys_cpy = _keys << 21
        _keys_cpy ^= _keys >> 33
        return cls._fibonacci_hash_uint64(_keys_cpy, step, copy=False)

    @classmethod
    def _choose_empty_value(cls, _keys, dtype, empty=_DEFAULT):
        # empty defined by user
        if empty is not _DEFAULT:
            return empty
        # use zero if keys are strictly positive
        zero = UINT64(0)
        if zero not in _keys:
            return zero
        # otherwise pick a random number
        while True:
            empty = numpy.random.randint(
                low=1<<62, high=1<<63, dtype='uint64')
            if empty not in _keys:
                return empty


class UInt64StructHashmap(_BaseHashmap):
    """
    a mapping from uint64-struct to arbitrary values in a numpy array
    consider using the higher-level `Hashmap` instead
    """
    # almost INV_PHI but different one (used in xxhash.c)
    _PRIME_1 = UINT64(11400714785074694791)
    _PRIME_2 = UINT64(14029467366897019727)
    _PRIME_5 = UINT64(2870177450012600261)

    @classmethod
    def new(cls, keys, values=None, cast_dtype=None, view_dtype=None, empty=_DEFAULT):
        """
        :param array keys: (n,) uint64-struct array
        :param array? values: (n,) val-dtype array
        :param dtype? cast_dtype: dtype to cast ``keys`` (default: uint64 for each field)
        :param dtype? view_dtype: dtype to view ``keys`` (default: uint64 for each field)
        :param object? empty: empty value (default: row of 0)
        """
        cast_dtype = cast_dtype or [
            (name, 'uint64') for name in keys.dtype.names
        ]
        view_dtype = view_dtype or [
            (name, 'uint64') for name in keys.dtype.names
        ]
        return super(UInt64StructHashmap, cls).new(keys, values, cast_dtype, view_dtype, empty)

    @classmethod
    def _hash(cls, _keys, step):
        """ use Python's algorithm for tuple to get consistent values """
        n, = _keys.shape
        n_cols = len(_keys.dtype)
        acc = numpy.full(n, cls._PRIME_5)
        buf = numpy.empty_like(acc)
        for col in sorted(_keys.dtype.names):
            # acc += _keys[col] * cls._PRIME_2
            buf[:] = _keys[col]
            buf *= cls._PRIME_2
            acc += buf
            # acc = (acc << 31) | (acc >> 33)
            buf[:] = acc
            buf >>= 33
            acc <<= 31
            acc |= buf
            #
            acc *= cls._PRIME_1
        acc += UINT64(n_cols) ^ (cls._PRIME_5 ^ UINT64(3527539))
        return cls._fibonacci_hash_uint64(acc, step, copy=False)

    @classmethod
    def _choose_empty_value(cls, _keys, dtype, empty=_DEFAULT):
        # empty defined by user
        if empty is not _DEFAULT:
            return empty
        # use zeros if keys are strictly positive
        wrapper = numpy.zeros(1, dtype=dtype)
        empty = wrapper[0]
        if empty not in _keys:
            return empty
        # otherwise pick random numbers
        d = len(dtype) or None
        while True:
            rdm = numpy.random.randint(low=1<<62, high=1<<63, size=d, dtype='uint64')
            if d:
                rdm = tuple(rdm)
            wrapper[0] = rdm
            empty = wrapper[0]
            if empty not in _keys:
                return empty


class ObjectHashmap(_BaseHashmap):
    """
    a mapping from arbitrary keys to arbitrary values in a numpy array
    consider using the higher-level ``Hashmap`` instead
    """

    @classmethod
    def new(cls, keys, values=None, cast_dtype=None, view_dtype=None, empty=_DEFAULT):
        """
        :param array keys: (n,) object array
        :param array? values: (n,) val-dtype array
        :param dtype? cast_dtype: dtype to cast ``keys`` (default: keys.type)
        :param dtype? view_dtype: dtype to view ``keys`` (default: keys.type)
        :param object? empty: empty value (default: row of 0)
        """
        cast_dtype = cast_dtype or keys.dtype
        view_dtype = view_dtype or cast_dtype
        return super(ObjectHashmap, cls).new(keys, values, cast_dtype, view_dtype, empty)

    @classmethod
    def _hash(cls, _keys, step):
        n = _keys.shape[0]
        hashes = numpy.fromiter((cls._hash_single_obj(obj) for obj in _keys),
                                count=n, dtype=UINT64)
        return cls._fibonacci_hash_uint64(hashes, step, copy=False)

    @classmethod
    def _hash_single_obj(cls, obj):
        try:
            return hash(obj)
        except TypeError:
            # cast single numpy array to bytes
            if isinstance(obj, numpy.ndarray):
                return hash(obj.tobytes())
            # cast all numpy arrays in tuple/void to bytes
            if isinstance(obj, (tuple, numpy.void)):
                obj_ = tuple((a.tobytes() if isinstance(a, numpy.ndarray) else a)
                             for a in tuple(obj))
                return hash(obj_)
            raise

    @classmethod
    def _choose_empty_value(cls, _keys, dtype, empty=_DEFAULT):
        return UInt64StructHashmap._choose_empty_value(_keys, dtype, empty)


def Hashmap(keys, values=None):
    """
    fake class to select between uint64/struct/object from dtype of arguments
    :param array keys: (n,) key-dtype array
    :param array? values: (n,) val-dtype array
    """
    # switch type from keys
    cls, cast_dtype, view_dtype = _get_optimal_cast(keys.dtype)
    # build hashmap
    return cls.new(keys, values, cast_dtype, view_dtype)


def _get_optimal_cast(dtype):
    """
    select best hashmap type to fit ``dtype``
    :returns: cls, cast_dtype, view_dtype
    """
    kind = dtype.kind
    names = dtype.names
    # scalar input (or strings of less than 8 bytes) we can view as uint64
    if kind in 'buifcSU' and dtype.itemsize <= 8:
        if kind == 'b':
            kind = 'u'
        return UInt64Hashmap, '{}8'.format(kind), UINT64
    # cast string of more than 8 bytes to tuple of uint64
    elif kind in 'SU':
        # number of uint64 (8 bytes) we need to view original dtype, e.g. 5 for 'U9'
        n_uint64 = int(numpy.ceil(float(dtype.itemsize) / 8))
        # how many 'S1' or 'U1' we need for get 8 bytes, e.g. 2 for 'U'
        inner_dtype_len = 8 / numpy.dtype('{}1'.format(kind)).itemsize
        # first cast to bigger string to fit exactly a multiple of 8 bytes, e.g. 'U10'
        cast_dtype = '{}{}'.format(kind, int(n_uint64 * inner_dtype_len))
        # then view as a tuple of uint64, e.g. 'u8,u8,u8,u8,u8'
        view_dtype = [('f{}'.format(i), 'u8') for i in range(n_uint64)]
        return UInt64StructHashmap, cast_dtype, view_dtype
    # struct input
    if names and all(dtype[n].kind in 'buifcSU' for n in names):
        # if all fields fit inside 8 bytes, use uint64 hashmap
        dtypes = [(n, dtype[n]) for n in names]
        sizes = [(n, dt.itemsize) for n, dt in dtypes]
        nbytes = sum(dt for _, dt in sizes)
        if nbytes <= 8:
            npad = 8 - nbytes
            if npad == 0:
                cast_dtype = dtypes  # simply remove offsets
            else:
                # add 'S{npad}' padding field
                cast_dtype = dtypes + [('__pad__', 'S{}'.format(npad))]
            return UInt64Hashmap, cast_dtype, UINT64
        # otherwise if all fields have less than 8B, use the uint64 struct implementation
        if all(sz <= 8 for _, sz in sizes):
            # TO-OPTIMIZE: pack many fields into uint64 to reduce the number of fields
            cast_dtype = [(n, '{}8'.format(dt.kind if dt.kind != 'b' else 'u'))
                          for n, dt in dtypes]
            view_dtype = [(n, 'u8') for n in names]
            return UInt64StructHashmap, cast_dtype, view_dtype
    # otherwise, use object
    return ObjectHashmap, dtype, dtype


def unique(values, return_inverse=False, return_index=False):
    """
    :param array values: (n,) dtype array
    :param bool? return_inverse:
    :param bool? return_index:
    :returns: (
        uniques: (n2,) dtype array with n2<=n,
        [if return_inverse=1] idx_in_uniques: (n,) uint32 array of indexes <= n2,
        [if return_index=1] unique_idx: (n2,) uint32 array of indexes <= n,
    )
    """
    _vals = numpy.arange(
        values.size, dtype='uint32') if return_index or return_inverse else None
    hashmap = Hashmap(keys=values, values=_vals)
    unique_values, table_mask = hashmap.unique_keys(return_table_mask=True)
    if not return_inverse and not return_index:
        return unique_values
    out = (unique_values,)
    if return_index:
        unique_idx = hashmap.values[table_mask]
    if return_inverse:
        # for return_inverse, we change hashmap values to be indexes in `unique_values`
        uniq_idx_in_uniq = numpy.arange(unique_values.size, dtype='uint32')
        hashmap.values[table_mask] = uniq_idx_in_uniq
        idx_in_uniques, found = hashmap.get_many(values)
        assert found.all()
        out = out + (idx_in_uniques,)
    if return_index:
        out = out + (unique_idx,)
    return out


def get_dense_labels_map(values, idx_dtype='uint32'):
    """
    convert unique values into dense int labels [0..n_uniques]
    :param array values: (n,) dtype array
    :param dtype? idx_dtype: (default: 'uint32')
    :returns: tuple(
        labels2values: (n_uniques,) dtype array,
        values2labels: HashMap(dtype->int),
    )
    """
    # get unique values
    unique_values = unique(values)
    # build labels from 0 to n_uniques
    labels = numpy.arange(unique_values.shape[0], dtype=idx_dtype)
    # build small hashmap with just the unique items
    values2labels = Hashmap(unique_values, labels)
    return unique_values, values2labels


def factorize(values):
    """
    Build dense int labels maps and return labels
    :param array values: (n,) dtype array
    :returns: tuple(
        labels: (n,) int array,
        labels2values: (n_uniques,) dtype array,
        values2labels: HashMap(dtype->int),
    )
    """
    labels2values, values2labels = get_dense_labels_map(values)
    values_labels, found = values2labels.get_many(values)
    assert found.all()
    return values_labels, labels2values, values2labels


def update_dense_labels_map(hashmap, values):
    """
    update hashmap values -> dense labels, and return mapped values
    :param HashMap hashmap: HashMap(dtype->int)
    :param array values: (n,) dtype array
    :returns: (n,) int array
    :changes: update ``hashmap`` in-place
    """
    # get current values
    labels, found = hashmap.get_many(values)
    new_values = values[~found]
    if not new_values.size:
        return labels
    # check unique new values
    unique_new_values, new_values_idx_in_uniques = unique(
        new_values, return_inverse=True)
    # build new labels
    idx_dtype = hashmap.values.dtype
    _, current_labels = hashmap.unique_keys(return_values=True)
    if current_labels.size == 0:
        start_at = 0
    else:
        start_at = current_labels.max() + 1
    new_labels = numpy.arange(
        start_at, start_at + unique_new_values.shape[0], dtype=idx_dtype)
    hashmap.set_many(unique_new_values, new_labels)
    # return all labels
    labels, found = hashmap.get_many(values)
    assert found.all()
    return labels


def empty_hashmap(key_dtype, val_dtype='uint32'):
    """
    Build empty Hashmap
    :param dtype key_dtype:
    :param dtype? val_dtype: (default: uint32)
    :returns: Hashmap
    """
    key_dtype = numpy.dtype(key_dtype)
    val_dtype = numpy.dtype(val_dtype)
    return Hashmap(numpy.empty(0, dtype=key_dtype), numpy.empty(0, dtype=val_dtype))


def reverse_values2labels(values2labels, dtype=None):
    """
    Reverse a hashmap values2labels.
    Normally you should not call this function but use the values provided in ``factorize``
    :param Hashmap values2labels: labels hashmap with n items
    :param dtype? dtype:
    :returns: (n,) dtype array
    """
    values = values2labels.unique_keys()
    labels, found = values2labels.get_many(values)
    assert found.all()
    out = numpy.empty_like(values, dtype=dtype)
    out[labels] = values
    return out


def array_hash(array, order_matters):
    """
    Return consistent hash of array that may or may not be order invarient
    :param array array: array with or without structure
    :param bool order_matters:
    :returns: int
    """
    if order_matters:
        order_num = STEP_MULT * numpy.arange(array.size, dtype=UINT64)
        if array.dtype.names is None:
            array = to_structured([('f0', array), ('__index__', order_num)])
        else:
            array = set_or_add_to_structured(array, [('__index__', order_num)])
    hashtable = Hashmap(array)
    return hashtable.keys_hash()


def values_hash(array, step=0):
    """
    Return consistent hash of array values
    :param array array: (n,) array with or without structure
    :param uint64 step: optional step number to modify hash values
    :returns: (n,) uint64 array
    """
    cls, cast_dtype, view_dtype = _get_optimal_cast(array.dtype)
    array = cls._cast(array, cast_dtype, view_dtype)
    return cls._hash(array, UINT64(step))
