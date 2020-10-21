"""
Array tools
===========
"""
import numpy
from numpy.lib.recfunctions import unstructured_to_structured

from .iterables import split


def to_structured(arrays):
    """
    Casts list of array and names (and optional dtypes) to numpy structured array.

    See `Numpy's documentation <https://numpy.org/doc/stable/user/basics.rec.html>`_
    for how to use numpy's structured arrays.

    A pandas DataFrame can be converted into a numpy record array,
    using DataFrame.to_records() (`to_records documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_records.html>`_,
    `documentation on record arrays
    <https://numpy.org/doc/stable/user/basics.rec.html#record-arrays>`_).
    A record array can then be converted into a structured array by using:

    >>> recordarr.view(recordarr.dtype.fields, numpy.ndarray)

    :param list-of-tuple arrays: ``list((name, array-like, *dtype-infos)))``
        or ``{name: array-like}``.
        A lone value will be broadcasted as an array full of this value
        and of the size of the other arrays
    :returns: struct-array

    Examples
    ________
    >>> to_structured([
    >>>     ('a', numpy.arange(5), 'uint32'),
    >>>     ('b', 2 * numpy.arange(5), 'float32')
    >>> ])
    array([(0, 0.), (1, 2.), (2, 4.), (3, 6.), (4, 8.)],
      dtype=[('a', '<u4'), ('b', '<f4')])

    A single value can also be used, and will be broadcasted as an array full of this value,
    and of the size of the other arrays, for instance:

    >>> to_structured([
    >>>     ('a', numpy.arange(5), 'uint32'),
    >>>     ('b', 2, 'float32')
    >>> ])
    array([(0, 2.), (1, 2.), (2, 2.), (3, 2.), (4, 2.)],
      dtype=[('a', '<u4'), ('b', '<f4')])

    Not specifying the dtype in the tuples will cause the function to use the array's dtype,
    or to infer it in case of sequences of Python objects.

    >>> to_structured([
    >>>    ('a', numpy.arange(5, dtype='uint32')),
    >>>    ('b', [2 * i for i in range(5)])
    >>> ])
    array([(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)],
      dtype=[('a', '<u4'), ('b', '<i8')])

    Using dictionaries:

    >>> to_structured({
    >>>     'a': numpy.arange(5, dtype='uint32'),
    >>>     'b': [2 * i for i in range(5)]
    >>> })
    array([(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)],
      dtype=[('a', '<u4'), ('b', '<i8')])

    (n, m) 2D numpy arrays are viewed as (n,) arrays of (m,) 1D arrays:

    >>> to_structured([
    >>> ('a', numpy.arange(5)),
    >>> ('b', numpy.arange(15).reshape(5, 3))
    >>> ])
    array([(0, [ 0,  1,  2]), (1, [ 3,  4,  5]), (2, [ 6,  7,  8]),
       (3, [ 9, 10, 11]), (4, [12, 13, 14])],
      dtype=[('a', '<i8'), ('b', '<i8', (3,))])
    """
    if isinstance(arrays, dict):
        arrays = arrays.items()
    else:
        assert isinstance(arrays, list), (
            'arrays must be a dict or a list, not {}'.format(type(arrays)))
    all_n_rows = {len(a) for _, a, *_ in arrays if hasattr(a, '__len__')}
    if len(all_n_rows) > 1:
        raise ValueError('all arrays must have the same shape[0], got: {}'.format(all_n_rows))
    n_rows, = all_n_rows
    # set default dtype
    dtypes = []
    for t in arrays:
        name, array = t[:2]
        if len(t) >= 3:  # given dtype infos
            dtype_infos = t[2:]
        else:  # infer dtype and shape
            array = numpy.asarray(array)
            if len(array.shape) == 1:
                dtype_infos = (array.dtype,)
            else:
                # do not use e.g. ('int', 3) but always '(3,)int' because the former flattens '1'
                dtype_infos = (f'{array.shape[1:]}{array.dtype}',)
        dtypes.append((name,) + dtype_infos)
    # combine to structured-array
    out = numpy.empty(n_rows, dtype=dtypes)
    for t in arrays:
        out[t[0]] = t[1]
    return out


def set_or_add_to_structured(array, data, copy=True):
    """
    Updates existing structured array, either by replacing the data for an existing field,
    or by adding a new field to the array

    :param struct-array array: array to update
    :param list-of-tuple data: list((name, array))
    :param bool? copy: set to False to avoid copy when possible ``(default: True)``
    :returns: struct-array

    Examples
    ________
    Adding field to existing structured array:

    >>> array = to_structured([
    >>>     ('a', numpy.arange(5, dtype='uint8')),
    >>>     ('b', 2 * numpy.arange(5, dtype='uint16')),
    >>> ])
    >>> new_data = 3 * numpy.arange(5, dtype='float32')
    >>> updated_array = set_or_add_to_structured(array, [
    >>>     ('c', new_data),
    >>> ])
    >>> updated_array
    array([(0, 0,  0.), (1, 2,  3.), (2, 4,  6.), (3, 6,  9.), (4, 8, 12.)],
      dtype=[('a', 'u1'), ('b', '<u2'), ('c', '<f4')])

    Replacing data from a structured array:

    >>> updated_array = set_or_add_to_structured(array, [
    >>>     ('b', new_data)
    >>> ])
    array([(0,  0), (1,  3), (2,  6), (3,  9), (4, 12)],
      dtype=[('a', 'u1'), ('b', '<u2')])

    Or doing both:

    >>> updated_array = set_or_add_to_structured(array, [
    >>>     ('b', new_data),
    >>>     ('c', 2 * new_data)
    >>> ])
    array([(0,  0,  0.), (1,  3,  6.), (2,  6, 12.), (3,  9, 18.),
       (4, 12, 24.)], dtype=[('a', 'u1'), ('b', '<u2'), ('c', '<f4')])
    """
    existing, new = split(data, lambda t: t[0] in array.dtype.names)
    if existing:
        if copy:
            array = array.copy()
        for k, v in existing:
            array[k] = v
    if new:
        arrays = [array]
        for name, data in new:
            if len(data.shape) == 1:
                data = data.astype([(name, data.dtype)])
            else:
                # pack 2d arrays into structure
                dtype = numpy.dtype([(name, data.dtype, data.shape[1:])])
                data = unstructured_to_structured(data, dtype)
            arrays.append(data)
        all_dtypes = [
            (name, a.dtype.fields[name][0])
            for a in arrays
            for name in a.dtype.names
        ]
        # ``merge_arrays`` is really slow, so we copy manually
        array = numpy.empty(array.size, dtype=all_dtypes)
        for a in arrays:
            for name in a.dtype.names:
                array[name] = a[name]
    return array


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
        fields_set = fields0.intersection(*(set(array.dtype.names) for array in arrays[1:]))
    else:
        fields_set = fields0.union(*(set(array.dtype.names) for array in arrays[1:]))
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
    batch_size = batch_size or (1<<12)
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
    raise NotImplementedError('`kargmax` with `do_sort` only supports 1d or 2d')


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
    raise NotImplementedError('`kargmin` with `do_sort` only supports 1d or 2d')
