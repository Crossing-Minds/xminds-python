"""
Arrays base tools
=================
"""
import numpy
from numpy.lib.recfunctions import unstructured_to_structured

from .iterable import split


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
        raise ValueError(
            'all arrays must have the same shape[0], got: {}'.format(all_n_rows))
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
    Updates existing structured array, either by replacing the data for existing fields,
    or by adding new fields to the array.

    Fast `alternative <https://github.com/numpy/numpy/issues/7811>`_ to
    ``numpy.lib.recfunctions.append_fields``.

    :param struct-array array: array to update
    :param list-of-tuple data: list((name, array-or-scalar))  (scalars are broadcasted)
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

    Or doing both, while adding broadcasted constants:

    >>> updated_array = set_or_add_to_structured(array, [
    >>>     ('b', new_data),
    >>>     ('c', 2 * new_data),
    >>>     ('d', 1),
    >>>     ('e', b'1')
    >>> ])
    array([
        (0,  0,  0., 1, b'1'),
        (1,  3,  6., 1, b'1'),
        (2,  6, 12., 1, b'1'),
        (3,  9, 18., 1, b'1'),
        (4, 12, 24., 1, b'1')],
        dtype=[('a', 'u1'), ('b', '<u2'), ('c', '<f4'), ('d', '<i8'), ('e', 'S1')])
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
            data = numpy.atleast_1d(data)
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
