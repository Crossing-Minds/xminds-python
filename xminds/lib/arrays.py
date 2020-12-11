"""
Array tools
===========
"""
import sys

import numpy

from xminds._lib.hashmap import Hashmap, factorize
from .arraybase import set_or_add_to_structured, to_structured


def structured_arrays_mean(arrays, keep_missing=False):
    """
    Computes average of each field given a list of struct-arrays.

    By default, will only average and return fields that are present in every struct-array in input.

    The average will be over concatenated fields as if they were in a single structured array,
    as opposed to averaging the results of several ``arr['field'].mean()``

    :param list-of-array arrays: list(struct-array)
    :param boolean? keep_missing: if True, use all fields from all arrays,
        including ones missing from some arrays;
        if False, only keep the intersection of fields defined in all arrays ``(default: False)``
    :returns: struct-array of shape (1,)

    Examples
    ________

    Obtaining the average of every field of a structured array:

    >>> arr1 = to_structured([
    >>>     ('a', numpy.arange(5)),
    >>>     ('b', 2 * numpy.arange(5))
    >>> ])
    >>> structured_arrays_mean([arr1])
    array([(2., 4.)],
      dtype=[('a', '<f8'), ('b', '<f8')])

    Getting the average of the same field scattered accross several structured arrays:

    >>> arr1 = to_structured([
    >>>     ('a', numpy.arange(5)),
    >>>     ('b', 2 * numpy.arange(5))
    >>> ])
    >>> arr2 = to_structured([
    >>>     ('a', numpy.ones(3)),
    >>>     ('b', 2 * numpy.ones(3)),
    >>>     ('c', 3 * numpy.ones(3))
    >>> ])
    >>> structured_arrays_mean([arr1, arr2])
    array([(1.625, 3.25)],
      dtype=[('a', '<f8'), ('b', '<f8')])

    If you want to include all fields, including fields that are only available in a few arrays,
    use option ``keep_missing=True``:

    >>> structured_arrays_mean([arr1, arr2], keep_missing=True)
    array([(1.625, 3.25, 3.)],
      dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
    """
    fields0 = set(arrays[0].dtype.names)
    if not keep_missing:
        fields_set = fields0.intersection(
            *(set(array.dtype.names) for array in arrays[1:]))
    else:
        fields_set = fields0.union(*(set(array.dtype.names)
                                     for array in arrays[1:]))
    return to_structured([
        (
            field,
            numpy.atleast_1d(numpy.concatenate(
                [array[field] for array in arrays if field in array.dtype.names]
            ).mean(axis=0, keepdims=True)))
        for field in fields_set
    ])


def first(a, predicate=None, batch_size=None, offset=0):
    """
    Efficiently find next index satisfying predicate in a 1d array.

    Can also be used on a array of booleans without a predicate.

    Will be added in numpy2.0: https://github.com/numpy/numpy/issues/2269

    :param array a: ``(n,) dtype array``
    :param callable? predicate: function(dtype-array -> bool-array) (or ``None`` if dtype=``bool``)
    :param int? batch_size: ``(default: 4k)``
    :param int? offset: ``(default: 0)``
    :returns: int, index of first value satisfying predicate
    :raises StopIteration: if there is no index satisfying predicate after ``offset``

    Examples
    ________

    With predicate:

    >>> a = numpy.array([0, 1, 1, 2, 3, 2, 4, 2])
    >>> idx = first(a, lambda x: x == 2)
    >>> idx
    3

    >>> idx = first(a, lambda x: x == 2, offset=idx + 1)
    >>> idx
    5

    >>> idx = first(a, lambda x: x == 2, offset=idx + 1)
    >>> idx
    7

    >>> idx = first(a, lambda x: x == 2, offset=idx + 1)
    StopIteration:

    Without predicate on array of booleans:

    >>> mask = numpy.array([0, 0, 0, 1, 0, 1, 0, 0], '?')
    >>> idx = first(mask)
    3
    """
    batch_size = batch_size or (1 << 12)
    n, = a.shape
    assert predicate is not None or a.dtype == '?', (
        'first without a predicate can only be used on an array of booleans')
    n_batches = int(numpy.ceil(float(n) / batch_size))
    for b in range(n_batches):
        batch_start = offset + b * batch_size
        batch_end = batch_start + batch_size
        batch = a[batch_start:batch_end]
        tests = batch if predicate is None else predicate(batch)
        inds = tests.nonzero()[0]
        try:
            return batch_start + inds[0]
        except IndexError:
            pass
    raise StopIteration()


def set_or_reallocate(array, values, offset, growing_factor=2., fill=None):
    """
    Assigns ``values`` (of length ``n2``) in ``array`` (of length ``n1``),
    starting at array's ``offset`` index.

    Returns the same array if there is enough space in array for assigning
    ``array[offset:offset+n2, :]`` = values and otherwise returns a new array,
    expanded by ``growing_factor`` when ``n2 > n1 - offset``

    :param array array: ``(n1, *extra-dims)`` array
    :param array values: ``(n2, *extra-dims)`` array
    :param int offset: index of ``array`` to start assigning ``values``.
              ``array[offset:offset+n2, :] = values``
    :param float? growing_factor: growing factor > 1 (``default:2``)
    :param float? fill: fill value or ``None`` to leave empty
    :returns: same array if ``n1 >= offset + n2``, or new array with copied data otherwise
              assigns ``array[offset:offset+n2, :] = values``

    Examples
    ________

    If ``n1 >= offset + n2``, the same array is returned with
    ``array[offset:offset+n2, :] = values``.
    In this case, the assignment is inplace and the input array will be affected.

    >>> array  = numpy.arange(10)
    >>> values = - 2 * numpy.arange(5)
    >>> set_or_reallocate(array, values, offset=5)
    array([ 0,  1,  2,  3,  4,  0, -2, -4, -6, -8])
    >>> array
    array([ 0,  1,  2,  3,  4,  0, -2, -4, -6, -8])

    Otherwise, a new expanded array with copied data is created and with
    ``array[offset:offset+n2, :] = values``.
    Without specifying a ``fill`` value, the newly expanded data will be arbitrary,
    starting at index ``offset + n2``.
    Since a new array is created, the input array will not be affected.

    >>> array  = numpy.arange(10)
    >>> set_or_reallocate(array, values, offset=10)
    array([                  0,                   1,                   2,
                             3,                   4,                   5,
                             6,                   7,                   8,
                             9,                   0,                  -2,
                            -4,                  -6,                  -8,
           5572452860762084442,     140512663005448,     670512663005448,
                             0,    2814751914590207])
    >>> array
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> set_or_reallocate(array, values, offset=10, fill=0)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0, -2, -4, -6, -8,  0,  0,
        0,  0,  0])
    >>> array
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    values = numpy.asarray(values)
    end = offset + values.shape[0]
    n1 = array.shape[0]
    extra_dims = array.shape[1:]
    assert values.shape[1:] == extra_dims
    # Nothing to do if it fits
    if end <= n1:
        array[offset:end] = values
        return array
    # Grow if it does not fit
    if growing_factor <= 1:
        raise ValueError('growing_factor must be larger than 1')
    new_length = max(end, int(growing_factor * n1))
    dtype = array.dtype
    if fill is not None:
        new_array = numpy.full((new_length,) + extra_dims, fill, dtype=dtype)
    else:
        new_array = numpy.empty((new_length,) + extra_dims, dtype=dtype)
    # Fill previous values
    new_array[:n1] = array
    # Sets new values
    new_array[offset:end] = values
    return new_array


def lexsort_uint32_pair(a, b):
    """
    faster alternative to numpy.lexsort for two uint32

    :param array a: (n,) uint32-array
    :param array b: (n,) uint32-array
    :returns: permutation array of n to sort by a first and b second
    """
    assert a.min() >= 0 and a.max() < (1<<32), (a.min(), a.max())
    assert b.min() >= 0 and b.max() < (1<<32), (b.min(), b.max())
    assert a.size == b.size
    dtype = ([('b', 'u4'), ('a', 'u4')] if sys.byteorder == 'little'
             else [('a', 'u4'), ('b', 'u4')])
    combined = numpy.empty(a.size, dtype=dtype)
    combined['a'] = a
    combined['b'] = b
    return numpy.argsort(combined.view('u8'))


def kargmax(a, k, axis=0, do_sort=False):
    """
    Returns the indices of the k maximum values along a specified axis.
    Set ``do_sort`` at ``True`` to get these indices ordered to get ``a[output]`` sorted.
    User-friendly wrapper around numpy.argpartition

    :param arr a: (n,) input array
    :param int k: sets the k in 'top k maximum values'
    :param int? axis: axis along which to look for argmax
        Default: 0
    :param bool? do_sort: if False, then the order of the output will be arbitrary
        if True, then ``a[output]`` will be sorted, starting with the a.max().
        Default: False
    :return arr: (k,) indexes of the k maximum values along the specified axis

    Examples
    ________
    >>> a = numpy.array([3, 1, 9, 6, 4, 4, 0, 6, 4, 8, 1, 3])
    >>> kargmax(a, 2)
    array([9, 2])

    The output array's order is arbitrary - to have the output ordered so that ``a[output]``
    is sorted, then use ``do_sort=True``:

    >>> kargmax(a, 2, do_sort=True)
    array([2, 9])

    On ndarrays:

    >>> a = numpy.array([3, 1, 9, 5, 6, 4, 0, 6, 4, 8, 1, 3]).reshape((3, 4))
    array([[3, 1, 9, 5],
           [6, 4, 0, 6],
           [4, 8, 1, 3]])

    >>> kargmax(a, 2, axis=0)
    array([[2, 1, 2, 0],
           [1, 2, 0, 1]])

    >>> kargmax(a, 2, axis=1)
    array([[3, 2],
           [0, 3],
           [0, 1]])
    """
    a = numpy.asarray(a)
    ndim = len(a.shape)
    n = a.shape[axis]
    if n <= k:
        topk = numpy.indices(a.shape)[axis]
    else:
        argp = numpy.argpartition(a, -k, axis=axis)
        if axis == 0:
            topk = argp[-k:]
        elif axis == 1:
            topk = argp[:, -k:]
        else:
            raise NotImplementedError()
    if not do_sort:
        return topk
    if ndim == 1:
        args = numpy.argsort(-a[topk], axis=axis)
        return topk[args]
    elif ndim == 2:
        if axis == 0:
            cols = numpy.indices(topk.shape)[1]
            args = numpy.argsort(-a[topk, cols], axis=0)
            return topk[args, cols]
        else:
            assert axis == 1
            rows = numpy.indices(topk.shape)[0]
            args = numpy.argsort(-a[rows, topk], axis=1)
            return topk[rows, args]
    raise NotImplementedError(
        '`kargmax` with `do_sort` only supports 1d or 2d')


def kargmin(a, k, axis=0, do_sort=False):
    """
    Returns the indices of the k minimum values along a specified axis.
    Set ``do_sort`` at ``True`` to get these indices ordered to get a[output] sorted.
    User-friendly wrapper around numpy.argpartition.

    :param arr a: (n,) input array
    :param int k: sets the k in 'top k minimum values'
    :param int? axis: axis along which to look for argmin
        Default: 0
    :param bool? do_sort: if False, then the order of the output will be arbitrary
        if True, then ``a[output]`` will be sorted, starting with the a.min().
        Default: False
    :return arr: (k,) indexes of the k minimum values along the specified axis

    Examples
    ________
    >>> a = numpy.array([3, 1, 9, 6, 4, 4, 0, 6, 4, 8, 1, 3])
    >>> kargmin(a, 2)
    array([1, 6])

    The output array's order is arbitrary - to have the output ordered so that ``a[output]``
    is sorted, then use ``do_sort=True``:

    >>> kargmin(a, 2)
    array([6, 1])

    On ndarrays:

    >>> a = numpy.array([3, 1, 9, 5, 6, 4, 0, 6, 4, 8, 1, 3]).reshape((3, 4))
    array([[3, 1, 9, 5],
           [6, 4, 0, 6],
           [4, 8, 1, 3]])

    >>> kargmin(a, 2, axis=0)
    array([[0, 0, 1, 2],
           [2, 1, 2, 0]])

    >>> kargmin(a, 2, axis=1)
    array([[1, 0],
           [2, 1],
           [2, 3]])


    """
    a = numpy.asarray(a)
    ndim = len(a.shape)
    n = a.shape[axis]
    if n <= k:
        topk = numpy.indices(a.shape)[axis]
    else:
        argp = numpy.argpartition(a, k, axis=axis)
        if axis == 0:
            topk = argp[:k]
        elif axis == 1:
            topk = argp[:, :k]
        else:
            raise NotImplementedError()
    if not do_sort:
        return topk
    if ndim == 1:
        args = numpy.argsort(a[topk], axis=axis)
        return topk[args]
    elif ndim == 2:
        if axis == 0:
            cols = numpy.indices(topk.shape)[1]
            args = numpy.argsort(a[topk, cols], axis=0)
            return topk[args, cols]
        else:
            assert axis == 1
            rows = numpy.indices(topk.shape)[0]
            args = numpy.argsort(a[rows, topk], axis=1)
            return topk[rows, args]
    raise NotImplementedError(
        '`kargmin` with `do_sort` only supports 1d or 2d')


def in1d(needles, haystack, invert=False):
    """
    Test whether each element of a 1d array is also present in a second 1d array.

    :param array needles: (n1,) dtype array
    :param array haystack: (n2,) dtype array
    :param bool? invert: if True, the values in the returned array are inverted,
        ``in1d(a, b, invert=True)`` is equivalent to ``~in1d(a, b)``. (``default: False``)
    :returns: (n1,) bool array

    Example
    _______
    >>> needles = numpy.array([5,10,20])
    >>> haystack = numpy.arange(15)
    >>> in1d(needles, haystack)
    array([True, True, False])
    >>> in1d(needles, haystack, invert=True)
    array([False, False, True])
    """
    if invert:
        return ~Hashmap(haystack).contains(needles)
    return Hashmap(haystack).contains(needles)


def search(needles, haystack, idx_dtype='uint32'):
    """
    Return whether each element of a 1d array is present in a second 1d array
    and the corresponding indexes.

    If an element of ``needles`` is not found in ``haystack``, the corresponding value
    returned in ``indexes`` is 0.

    :param array needles: (n1,) dtype array
    :param array haystack: (n2,) dtype array
    :param dtype? idx_dtype: (``default: uint32``)
    :returns: tuple(
        indexes: (n1,) idx_dtype array <n2 of indexes in haystack,
        found: (n1,) bool array,)

    Example
    _______
    >>> needles = numpy.array([1000, 2000, 3000])
    >>> haystack = numpy.arange(50)
    >>> haystack[10] = needles[0]
    >>> haystack[20] = needles[1]
    >>> search(needles, haystack)
    (array([10, 20,  0], dtype=uint32), array([ True,  True, False]))
    """
    hashmap = Hashmap(haystack, numpy.arange(haystack.size, dtype=idx_dtype))
    idx, found = hashmap.get_many(needles)
    return idx, found


def cumcount_by_value(values, assume_sorted=False):
    """
    Compute the rank of appearance of entries grouped by input value.

    If a value appears for the 5th time in the ``values`` array at index ``i``,
    ``output[i]`` will be a 5.

    :param array values: uint32 array
    :param bool? assume_sorted: if True, the input values are assumed sorted (``default: False``)
    :returns: array of int

    Example
    _______
    >>> array = numpy.array([0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
    >>> cumcount_by_value(array)
    array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4])
    >>> array = numpy.array(['beta', 'alpha', 'gamma', 'alpha', 'beta', 'alpha', 'delta'])
    >>> cumcount_by_value(array)
    array([0, 0, 0, 1, 1, 2, 0])
    """
    if len(values) == 0:
        return []
    convert = True
    if values.dtype.kind in 'ui':
        if min(values) >= 0 and max(values) < (1 << 6) * len(values):
            convert = False
    if not assume_sorted:
        argsort = numpy.argsort(values, kind='stable')
        values = values[argsort]
    if convert:
        values, _, _ = factorize(values)
    # values        0 0 0 0 1 1 1 3 3 3 3 3
    # n_per_idx     4 3 0 5
    n_per_idx = numpy.bincount(values)
    # rank          0 1 2 3 0 1 2 0 1 2 3 4
    rank = _arange_sequence(n_per_idx)
    if assume_sorted:
        return rank
    # reverse argsort
    out = numpy.empty_like(rank)
    out[argsort] = rank
    return out


def _arange_sequence(lengths):
    """
    :param array lengths: int array like [4 3 0 5]
    :returns: concat(arange(i) for i length) like [0 1 2 3 0 1 2 0 1 2 3 4]
    """
    # lengths     4 3 0 5
    # cum_counts  0 4 7 7 12
    cum_counts = numpy.zeros(lengths.size + 1, dtype=lengths.dtype)
    numpy.cumsum(lengths, out=cum_counts[1:])
    # offset      0 0 0 0 4 4 4 7 7 7 7 7
    offset = numpy.repeat(cum_counts[:-1], lengths)
    # out         0 1 2 3 0 1 2 0 1 2 3 4
    return numpy.arange(offset.size) - offset


def unique_count(values):
    """
    Count the number of unique elements in ``values``.

    :param array values: (n,) dtype array
    :returns: int number of unique values
    """
    return Hashmap(keys=values, values=None).n_used
