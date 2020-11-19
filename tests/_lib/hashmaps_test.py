from __future__ import print_function

import random
import time
from six import binary_type, text_type
import unittest

import numpy

from xminds.lib.arrays import to_structured
from xminds._lib.hashmaps import (UInt64Hashmap, UInt64StructHashmap, ObjectHashmap,
                                  array_hash, empty_hashmap, get_dense_labels_map, reverse_values2labels,
                                  in1d, search, unique, unique_count, update_dense_labels_map, values_hash)


def _cat(ar1, ar2):
    """ concatenation with field alignment """
    if not getattr(ar1.dtype, 'names', None):
        return numpy.r_[ar1, ar2]
    return numpy.r_[ar1, ar2[list(ar1.dtype.names)]]


class BaseHashtableTestCase():
    """ TestCase template to test hashmap without values (a.k.a hashtable) """
    cls = None

    @classmethod
    def get_keys(cls, n=None):
        """ must return keys such that calling twice return an entirely different set """
        raise NotImplementedError()

    def test_search(self):
        """ should succeed at indexing a set of unique keys """
        n = 1<<10
        keys = self.get_keys(n)
        ht = self.cls.new(keys)
        # assert all keys are found
        indexes, found = ht.search(keys)
        assert found.all()
        # assert all indexes are different
        assert numpy.unique(indexes).shape == indexes.shape
        # assert none of other keys are found (very high probability)
        keys2 = self.get_keys(n)
        _, found = ht.search(keys2)
        assert not found.any()
        # combine both
        _, found = ht.search(_cat(keys, keys2))
        assert found[:n].all()
        assert not found[n:].any()

    def test_search_w_dups(self):
        """ should succeed at indexing a set of keys with duplicates """
        keys = self.get_keys()
        ht = self.cls.new(_cat(keys, keys))  # duplicate keys
        # assert all keys are found
        indexes, found = ht.search(keys)
        assert found.all()
        # assert uniques are correct
        uniques = ht.unique_keys()
        assert (numpy.sort(uniques) == numpy.sort(keys)).all()

    def test_set_many(self):
        """ should succeed at re-indexing a set of growing size """
        keys = self.get_keys()
        ht = self.cls.new(keys)
        for i in range(10):
            new_keys = self.get_keys()
            ht.set_many(new_keys)
            _, found = ht.search(new_keys)
            assert found.all()
        # test first keys are still here
        _, found = ht.search(keys)
        assert found.all()

    def test_keys_hash(self):
        """ should succeed at returning consistent keys hash """
        keys = self.get_keys()
        n = keys.size
        # init with half and then add remaining
        ht = self.cls.new(keys[:n//2])
        hsh_before = ht.keys_hash()
        ht.set_many(keys[n//2:])
        hsh = ht.keys_hash()
        assert hsh != hsh_before
        # compare with both at the same time
        ht2 = self.cls.new(keys)
        assert ht2.keys_hash() == hsh
        # shuffle
        numpy.random.shuffle(keys)
        ht2 = self.cls.new(keys)
        assert ht2.keys_hash() == hsh


class UInt64HashtableTestCase(BaseHashtableTestCase, unittest.TestCase):
    cls = UInt64Hashmap

    @classmethod
    def get_keys(cls, n=None):
        n = n or 1<<10
        return numpy.random.randint(1, 1<<63, size=n).astype('uint64')

    # ...all test inherited from BaseHashtableTestCase...

    def test_sequential(self):
        """ test that we can map sequential data """
        n = 1<<10
        keys = numpy.arange(n).astype('uint64')  # start at 0 to also test edge case
        ht = self.cls.new(keys)
        indexes, found = ht.search(keys)
        assert found.all()

    def test_prime_numbers(self):
        """ test that it works with lots of prime numbers """
        n = 1<<10
        # Generate prime-looking numbers
        keys = 1 + 2 * numpy.random.randint(low=1, high=1<<20, size=4 * n).astype('uint64')
        first_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
        looks_prime = (keys % 3) != 0
        for prime in first_primes:
            looks_prime &= (keys % prime) != 0
        keys = keys[looks_prime][:n]
        # Test hashtable
        ht = self.cls.new(keys)
        indexes, found = ht.search(keys)
        assert found.all()

    def test_piecewise_sequential(self):
        """ test that it works with piecewise sequential numbers """
        # Generate staircase numbers [x x+1 x+2 ... x+n y y+1 ... y+n ...]
        n_steps = 1<<7
        n_stairs = 1<<6
        base = numpy.random.randint(low=1, high=1<<63, size=n_stairs).astype('uint64')
        steps = numpy.arange(n_steps * n_stairs, dtype='uint64')
        keys = numpy.repeat(base, n_steps) + steps
        # Test hashtable
        ht = self.cls.new(keys)
        indexes, found = ht.search(keys)
        assert found.all()

    def test_benchmark_against_pandas(self):
        """ assert the common sequence of operations are not 10% slower than pandas """
        try:
            from pandas._libs import hashtable as pd_hashtable
        except ImportError:
            print('pandas is not installed. skip benchmark')
            return
        n = 1<<20
        keys = self.get_keys(n)
        keys2 = self.get_keys(n)
        # run xmlib
        time_start = time.perf_counter()
        ht = self.cls.new(keys)  # init
        ht.search(keys)  # search found
        ht.search(keys2)  # search not found
        ht.set_many(keys2)  # add new
        time_end = time.perf_counter()
        time_xmlib = time_end - time_start
        # run pandas
        time_start = time.perf_counter()
        ht = pd_hashtable.UInt64HashTable()
        ht.map_locations(keys)  # init
        ht.lookup(keys)  # search found
        ht.lookup(keys2)  # search not found
        ht.map_locations(keys2)  # add new
        time_end = time.perf_counter()
        time_pandas = time_end - time_start
        # test xmlib is competitive
        print('xmlib: {:.0f}ms, pandas: {:.0f}ms'.format(1000 * time_xmlib, 1000 * time_pandas))
        assert time_xmlib < 1.2 * time_pandas


class UInt64StructHashtableTestCase(BaseHashtableTestCase, unittest.TestCase):
    cls = UInt64StructHashmap
    dtype = [('a', 'uint64'), ('b', 'uint64')]

    @classmethod
    def get_keys(cls, n=None, dtype=None):
        n = n or 1<<10
        dtype = list(dtype or cls.dtype)
        random.shuffle(dtype)  # shuffle order of fields in dtype to assert implementation re-orders
        keys = numpy.empty(n, dtype=dtype)
        keys[:] = [tuple(t) for t in numpy.random.randint(1, 1<<63, size=(n, len(dtype)))]
        return keys

    # ...all test inherited from BaseHashtableTestCase...

    def test_sanity_check_2d(self):
        """ check the tests are correctly building struct-array """
        ht = self.cls.new(self.get_keys(10))
        uniques = ht.unique_keys()
        assert all(uniques.dtype.fields[f][0] == dt for f, dt in self.dtype)

    def test_zeros(self):
        """ check hashtable is supporting rows of zeros """
        n = 1<<10
        # test by starting without zero and adding it after
        keys = self.get_keys(n)
        ht = self.cls.new(keys)
        keys2 = self.get_keys(n)
        keys2[0] = 0
        ht.set_many(keys2)
        indexes, found = ht.search(keys2)
        assert found.all()
        # test by starting from zero
        ht = self.cls.new(keys2)
        indexes, found = ht.search(keys2)
        assert found.all()


class BytesStringHashtableTestCase(BaseHashtableTestCase, unittest.TestCase):
    cls = ObjectHashmap
    KIND = 'S'

    @classmethod
    def get_keys(cls, n=None, str_len=20):
        n = n or 1<<10
        keys = numpy.empty(n, dtype='{}{}'.format(cls.KIND, str_len))
        keys[:] = [str(i).encode('latin1') for i in numpy.random.randint(1, 1<<63, size=n)]
        return keys

    # ...all test inherited from BaseHashtableTestCase...

    def test_sanity_check_string(self):
        """ check the tests are correctly building str-array """
        keys = self.get_keys(10)
        ht = self.cls.new(keys)
        uniques = ht.unique_keys()
        assert isinstance(uniques[0], binary_type)


class UnicodeStringHashtableTestCase(BytesStringHashtableTestCase):
    KIND = 'U'

    # ...all test inherited from BytesStringHashtableTestCase...

    def test_sanity_check_string(self):
        """ check the tests are correctly building str-array """
        keys = self.get_keys(10)
        ht = self.cls.new(keys)
        uniques = ht.unique_keys()
        assert isinstance(uniques[0], text_type)


class StringTupleHashtableTestCase(BaseHashtableTestCase, unittest.TestCase):
    cls = ObjectHashmap

    @classmethod
    def get_keys(cls, n=None):
        n = n or 1<<10
        dtype = [('s1', 'S20'), ('s2', 'S20')]
        random.shuffle(dtype)  # shuffle order of fields in dtype to assert implementation re-orders
        keys = numpy.empty(n, dtype=dtype)
        keys[:] = [(str(i), str(i + 1)) for i in numpy.random.randint(1, 1<<63, size=n)]
        return keys

    # ...all test inherited from BaseHashtableTestCase...


class UniqueTestCase(unittest.TestCase):
    """ test that `unique` automatically adapts to its arguments """

    def test_unique_on_uint64(self):
        vals = UInt64HashtableTestCase.get_keys()
        self._test_unique(vals)

    def test_unique_on_uint32(self):
        vals = UInt64HashtableTestCase.get_keys().astype('uint32')
        self._test_unique(vals)

    def test_unique_on_int64(self):
        vals = UInt64HashtableTestCase.get_keys().astype('int64')
        self._test_unique(vals)

    def test_unique_on_int32(self):
        vals = UInt64HashtableTestCase.get_keys().astype('int32')
        self._test_unique(vals)

    def test_unique_on_uint64_struct(self):
        vals = UInt64StructHashtableTestCase.get_keys()
        self._test_unique(vals)

    def test_unique_on_mix_int_struct(self):
        dtype = [('a', 'uint8'), ('b', 'int64'), ('c', 'uint64'), ('d', 'int8')]
        vals = UInt64StructHashtableTestCase.get_keys(dtype=dtype)
        self._test_unique(vals)

    def test_unique_on_short_bytes_string(self):
        vals = BytesStringHashtableTestCase.get_keys(str_len=7)
        self._test_unique(vals, vals_are_unique=False)

    def test_unique_on_long_bytes_string(self):
        vals = BytesStringHashtableTestCase.get_keys(str_len=21)
        self._test_unique(vals)

    def test_unique_on_short_unicode_string(self):
        vals = UnicodeStringHashtableTestCase.get_keys(str_len=7)
        self._test_unique(vals, vals_are_unique=False)

    def test_unique_on_long_unicode_string(self):
        vals = UnicodeStringHashtableTestCase.get_keys(str_len=21)
        self._test_unique(vals)

    def test_unique_on_object(self):
        n = 1<<10
        vals = to_structured([
            ('a', UInt64HashtableTestCase.get_keys(n)),
            ('b', UnicodeStringHashtableTestCase.get_keys(n).astype('object')),
        ])
        self._test_unique(vals)

    def _test_unique(self, vals, vals_are_unique=True):
        if vals_are_unique:
            uniq_vals = vals
        else:
            uniq_vals = numpy.unique(vals)
        n_uniq = uniq_vals.size
        # unique on itself
        uniq = unique(vals)
        assert uniq.size == n_uniq
        assert uniq.dtype == vals.dtype
        assert (numpy.sort(uniq) == numpy.sort(uniq_vals)).all()
        uniq, idx_in_uniq, uniq_idx = unique(vals, return_inverse=True, return_index=True)
        assert (uniq[idx_in_uniq] == vals).all()
        assert (uniq == vals[uniq_idx]).all()
        # unique count on itself
        assert unique_count(vals) == n_uniq
        # duplicate values
        dups = _cat(vals, vals)
        numpy.random.shuffle(dups)
        uniq = unique(dups)
        assert uniq.size == n_uniq
        assert (numpy.sort(uniq) == numpy.sort(uniq_vals)).all()
        uniq, idx_in_uniq, uniq_idx = unique(dups, return_inverse=True, return_index=True)
        assert (uniq[idx_in_uniq] == dups).all()
        assert (uniq == dups[uniq_idx]).all()
        # unique count on dups
        assert unique_count(dups) == n_uniq


class BaseHashmapTestCaseMixin(object):
    """ TestCase template to test hashmap with values """
    cls = None
    val_dtype = [('tinyint', 'uint8'), ('bigint', 'uint64'), ('vector', 'float32', 10)]

    @classmethod
    def get_keys(cls, n=None):
        raise NotImplementedError()

    @classmethod
    def get_values(cls, n=None):
        n = n or 1<<10
        keys = numpy.empty(n, dtype=cls.val_dtype)
        keys[:] = [tuple(t) for t in numpy.random.randint(1, 1<<63, size=(n, len(cls.val_dtype)))]
        return keys

    def test_get_many(self):
        n = 1<<10
        keys = self.get_keys(n)
        values = self.get_values(n)
        hm = self.cls.new(keys, values)
        # test found
        found_vals, found = hm.get_many(keys)
        assert found.all()
        assert (found_vals == values).all()
        # test not found
        keys2 = self.get_keys(n)
        _, found = hm.get_many(keys2)
        assert not found.any()
        # test mixed
        found_vals, found = hm.get_many(_cat(keys, keys2))
        assert found[:n].all()
        assert not found[n:].any()
        assert (found_vals[:n] == values).all()

    def test_set_many_no_update(self):
        keys = self.get_keys()
        values = self.get_values()
        hm = self.cls.new(keys, values)
        for _ in range(10):
            new_keys = self.get_keys()
            new_values = self.get_values()
            hm.set_many(new_keys, new_values)
        # test we still have the first keys
        found_vals, found = hm.get_many(keys)
        assert found.all()
        assert (found_vals == values).all()

    def test_set_many_updates(self):
        keys = self.get_keys()
        values = self.get_values()
        hm = self.cls.new(keys, values)
        new_values = self.get_values()
        hm.set_many(keys, new_values)
        found_vals, found = hm.get_many(keys)
        assert found.all()
        assert (found_vals == new_values).all()


class UInt64HashmapTestCase(BaseHashmapTestCaseMixin, unittest.TestCase):
    """ test uint64 -> struct-dtype mapping """
    cls = UInt64Hashmap

    @classmethod
    def get_keys(cls, n=None):
        return UInt64HashtableTestCase.get_keys(n)


class UInt64StructHashmapTestCase(BaseHashmapTestCaseMixin, unittest.TestCase):
    """ test uint64-struct -> struct-dtype mapping """
    cls = UInt64StructHashmap
    keys_dtype = [('a', 'uint64'), ('b', 'uint64')]

    @classmethod
    def get_keys(cls, n=None, dtype=None):
        return UInt64StructHashtableTestCase.get_keys(n, dtype=dtype)


class ObjectHashmapTestCase(BaseHashmapTestCaseMixin, unittest.TestCase):
    """ test str -> struct-dtype mapping """
    cls = ObjectHashmap

    @classmethod
    def get_keys(cls, n=None):
        return UnicodeStringHashtableTestCase.get_keys(n)


class SearchTestCase(unittest.TestCase):
    """ test that `search` automatically adapts to and cast its arguments """

    def test_search_on_uint64(self):
        vals = UInt64HashmapTestCase.get_keys()
        vals2 = UInt64HashmapTestCase.get_keys()
        self._test_search(vals, vals2)

    def test_search_on_uint32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('uint32')
        vals2 = UInt64HashmapTestCase.get_keys().astype('uint32')
        self._test_search(vals, vals2)

    def test_search_on_int64(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int64')
        vals2 = UInt64HashmapTestCase.get_keys().astype('int64')
        self._test_search(vals, vals2)

    def test_search_on_int32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int32')
        vals2 = UInt64HashmapTestCase.get_keys().astype('int32')
        self._test_search(vals, vals2)

    def test_search_on_uint64_struct(self):
        vals = UInt64StructHashmapTestCase.get_keys()
        vals2 = UInt64StructHashmapTestCase.get_keys()
        self._test_search(vals, vals2)

    def test_search_on_mix_int_struct(self):
        dtype = [('a', 'uint8'), ('b', 'int64'), ('c', 'uint64'), ('d', 'int8')]
        vals = UInt64StructHashmapTestCase.get_keys(dtype=dtype)
        vals2 = UInt64StructHashmapTestCase.get_keys(dtype=dtype)
        self._test_search(vals, vals2)

    def test_search_on_mix_tiny_types_struct(self):
        n = 64  # only a few items since duplicates are more likely
        dtype = [('a', 'uint8'), ('b', 'int8'), ('c', 'int8')]
        vals = UInt64StructHashmapTestCase.get_keys(dtype=dtype, n=n)
        vals2 = UInt64StructHashmapTestCase.get_keys(dtype=dtype, n=n)
        self._test_search(vals, vals2)

    def test_search_on_object(self):
        vals = ObjectHashmapTestCase.get_keys()
        vals2 = ObjectHashmapTestCase.get_keys()
        self._test_search(vals, vals2)

    @classmethod
    def _test_search(cls, vals, vals2):
        n = vals.size
        # search itself
        idx, found = search(vals, vals)
        assert found.all()
        assert (idx == numpy.arange(n)).all()
        assert in1d(vals, vals).all()
        # search half of itself within itself
        idx, found = search(vals[:n // 2], vals)
        assert found.all()
        assert (idx == numpy.arange(n // 2)).all()
        assert in1d(vals[:n // 2], vals).all()
        # search duplicate of itself within itself
        dups = _cat(vals, vals)
        idx, found = search(dups, vals)
        assert found.all()
        assert (idx[:n] == numpy.arange(n)).all()
        assert (idx[n:] == numpy.arange(n)).all()
        assert in1d(dups, vals).all()
        # only not found
        idx, found = search(vals2, vals)
        assert not found.any()
        assert not in1d(vals2, vals).any()
        # mix found / not found
        idx, found = search(_cat(vals, vals2), vals)
        assert found[:n].all()
        assert not found[n:].any()
        assert (idx[:n] == numpy.arange(n)).all()
        found = in1d(_cat(vals, vals2), vals)
        assert found[:n].all()
        assert not found[n:].any()
        assert (numpy.invert(in1d(_cat(vals, vals2), vals))
                == in1d(_cat(vals, vals2), vals, invert=True)).all()


class DenseLabelsMapTestCase(unittest.TestCase):
    """ test that `get_dense_labels_map` automatically adapts to and cast its arguments """

    def test_labels_on_uint64(self):
        vals = UInt64HashmapTestCase.get_keys()
        self._test_get_dense_labels_map(vals)

    def test_labels_on_uint32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('uint32')
        self._test_get_dense_labels_map(vals)

    def test_labels_on_int64(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int64')
        self._test_get_dense_labels_map(vals)

    def test_labels_on_int32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int32')
        self._test_get_dense_labels_map(vals)

    def test_labels_on_uint64_struct(self):
        vals = UInt64StructHashmapTestCase.get_keys()
        self._test_get_dense_labels_map(vals)

    def test_labels_on_mix_int_struct(self):
        dtype = [('a', 'uint8'), ('b', 'int64'), ('c', 'uint64'), ('d', 'int8')]
        vals = UInt64StructHashmapTestCase.get_keys(dtype=dtype)
        self._test_get_dense_labels_map(vals)

    def test_labels_on_object(self):
        vals = ObjectHashmapTestCase.get_keys()
        self._test_get_dense_labels_map(vals)

    def test_update_dense_labels_map(self):
        vals = UInt64HashmapTestCase.get_keys().astype('uint32')
        hm = empty_hashmap('uint32')
        # start with empty
        labels = update_dense_labels_map(hm, vals)
        assert labels.size == vals.size
        # update
        vals2 = UInt64HashmapTestCase.get_keys().astype('uint32')
        labels2 = update_dense_labels_map(hm, vals2)
        assert labels2.size == vals2.size
        # test labels are disjoint
        assert not numpy.in1d(labels2, labels).any()
        # get `vals` again
        labels_ = update_dense_labels_map(hm, vals)
        assert (labels == labels_).all()
        # get `vals2` again
        labels2_ = update_dense_labels_map(hm, vals2)
        assert (labels2 == labels2_).all()
        # check max label
        assert labels2.max() == vals.size + vals2.size - 1
        # add a mix of old and new
        vals3 = UInt64HashmapTestCase.get_keys().astype('uint32')
        vals23 = _cat(vals2, vals3)
        labels23 = update_dense_labels_map(hm, vals23)
        assert (labels2 == labels2_).all()
        labels2_ = labels23[:vals2.size]
        labels3 = labels23[vals2.size:]
        assert not numpy.in1d(labels3, _cat(labels, labels2)).any()
        assert (labels2_ == labels2).all()
        # check max label
        assert labels3.max() == vals.size + vals2.size + vals3.size - 1

    def test_reverse_values2labels(self):
        vals = UInt64HashmapTestCase.get_keys()
        labels2values, values2labels = get_dense_labels_map(vals)
        assert (reverse_values2labels(values2labels) == labels2values).all()

    @classmethod
    def _test_get_dense_labels_map(cls, vals):
        n = vals.size
        # test on unique values
        labels2values, values2labels = get_dense_labels_map(vals)
        assert labels2values.size == n  # assert correct amount of labels
        labels, found = values2labels.get_many(vals)  # assert map to labels works
        assert found.all()
        l2l, found = values2labels.get_many(labels2values)
        assert found.all()
        assert (l2l == numpy.arange(n)).all()  # assert respective inverse
        # test with duplicates
        dups = _cat(vals, vals)
        numpy.random.shuffle(dups)
        labels2values, values2labels = get_dense_labels_map(dups)
        assert labels2values.size == n  # assert correct amount of labels
        labels, found = values2labels.get_many(vals)  # assert map to labels works
        assert found.all()
        l2l, found = values2labels.get_many(labels2values)
        assert (l2l == numpy.arange(n)).all()  # assert respective inverse


class ArrayHashTestCase(unittest.TestCase):
    def test_array_hash_on_uint64(self):
        vals = UInt64HashmapTestCase.get_keys()
        self._test_array_hash(vals)

    def test_array_hash_on_uint32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('uint32')
        self._test_array_hash(vals)

    def test_array_hash_on_int64(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int64')
        self._test_array_hash(vals)

    def test_array_hash_on_int32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int32')
        self._test_array_hash(vals)

    def test_array_hash_on_bool(self):
        vals = UInt64HashmapTestCase.get_keys()
        vals = vals > numpy.median(vals)
        self._test_array_hash(vals)

    def test_array_hash_on_uint64_struct(self):
        vals = UInt64StructHashmapTestCase.get_keys()
        self._test_array_hash(vals)

    def test_array_hash_on_mix_int_struct(self):
        dtype = [('a', 'uint8'), ('b', 'int64'), ('c', 'uint64'), ('d', 'int8'), ('e', '?')]
        vals = UInt64StructHashmapTestCase.get_keys(dtype=dtype)
        self._test_array_hash(vals)

    def test_array_hash_on_object(self):
        vals = ObjectHashmapTestCase.get_keys()
        self._test_array_hash(vals)

    def _test_array_hash(self, vals):
        hsh_w_order = array_hash(vals, order_matters=True)
        hsh_wo_order = array_hash(vals, order_matters=False)
        # permute values and check that order_matters=0 is the same
        for _ in range(10):
            numpy.random.shuffle(vals)
            hsh_w_order_ = array_hash(vals, order_matters=True)
            hsh_wo_order_ = array_hash(vals, order_matters=False)
            assert hsh_w_order != hsh_w_order_
            assert hsh_wo_order == hsh_wo_order_
        # edit some value
        vals[0] = next(v for v in vals if v != vals[0])
        hsh_w_order_ = array_hash(vals, order_matters=True)
        hsh_wo_order_ = array_hash(vals, order_matters=False)
        assert hsh_w_order != hsh_w_order_
        # 1st part of assert may fail if we are testing with only few values (e.g. bool)
        assert (hsh_wo_order != hsh_wo_order_) or (len(set(vals)) < 100)


class ValuesHashTestCase(unittest.TestCase):
    def test_values_hash_on_uint64(self):
        vals = UInt64HashmapTestCase.get_keys()
        self._test_values_hash(vals)

    def test_values_hash_on_uint32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('uint32')
        self._test_values_hash(vals)

    def test_values_hash_on_int64(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int64')
        self._test_values_hash(vals)

    def test_values_hash_on_int32(self):
        vals = UInt64HashmapTestCase.get_keys().astype('int32')
        self._test_values_hash(vals)

    def test_values_hash_on_bool(self):
        vals = UInt64HashmapTestCase.get_keys()
        vals = vals > numpy.median(vals)
        self._test_values_hash(vals, assume_unique=False)

    def test_values_hash_on_uint64_struct(self):
        vals = UInt64StructHashmapTestCase.get_keys()
        self._test_values_hash(vals)

    def test_values_hash_on_mix_int_struct(self):
        dtype = [('a', 'uint8'), ('b', 'int64'), ('c', 'uint64'), ('d', 'int8'), ('e', '?')]
        vals = UInt64StructHashmapTestCase.get_keys(dtype=dtype)
        self._test_values_hash(vals)

    def test_values_hash_on_object(self):
        vals = ObjectHashmapTestCase.get_keys()
        self._test_values_hash(vals)

    def _test_values_hash(self, vals, assume_unique=True):
        n = len(vals)
        hsh = values_hash(vals)
        if assume_unique:
            assert len(numpy.unique(hsh)) == n
        # duplicate
        hsh = values_hash(numpy.r_[vals, vals])
        assert (hsh[:n] == hsh[n:]).all()
