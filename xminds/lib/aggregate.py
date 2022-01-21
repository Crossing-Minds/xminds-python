"""
Aggregate tools
===============
"""
import sys

import numpy

from .._lib.hashmap import factorize
from ..compat import tqdm
from ..ds.scaling import linearscaling
from .arrays import first, lexsort_uint32_pair, to_structured


def igroupby(ids, values, n=None, logging_prefix=None, assume_sorted=False, stable=False,
             find_next_hint=512):
    """
    Efficiently converts two arrays representing a relation
    (the ``ids`` and the associated ``values``) to an iterable ``(id, values_associated)``.

    The ``values`` are grouped by ``ids`` and a sequence of tuples is generated.

    The ``i`` th tuple generated is ``(id_i, values[ids == id_i])``,
    ``id_i`` being the ``i`` th element of the ``ids`` array, once sorted in ascending order.

    :param array ids: ``(>=n,) dtype array``
    :param array values: ``(>=n, *shape) uint32 array``
    :param int? n: length of array to consider,
     applying igroupby to ``(ids[:n], values[:n])``. Uses full array when not set.
    :param string? logging_prefix: prefix to include while logging progress.
        ``(default:`` Does not log``)``.
    :param bool? assume_sorted: whether ids is sorted. ``(default: False)``
    :param bool? stable: when True, ensures that the relative order of values is preserved
        (will slightly impact performance)
        When False, nothing guarantees that the values in ``(id_i, values[ids == id_i])``
        will be aligned on the input values (see example below)
        The stable option is only used when `assume_sorted` is False (general case)
    :param int? find_next_hint: hint for find_next_lookup. ``(default: 512)``
    :generates: tuple(id:int, values_associated:``(m, *shape) array slice``)

    Example
    _______
    >>> ids     = numpy.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3])
    >>> values  = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> gen = igroupby(ids, values)
    >>> next(gen)
    (0, array([0, 2, 4, 6]))

    >>> next(gen)
    (1, array([0, 1, 2, 3, 4]))

    >>> next(gen)
    (3, array([10,  9,  8,  7,  0,  5,  6,  4, 11,  6, 12]))
    # note how the values are not in the same order as they were in the input

    with option `stable`:
    >>> gen = igroupby(ids, values, stable=True)
    >>> next(gen)
    >>> next(gen)
    >>> next(gen)
    (3, array([0, 4, 6, 5, 6, 7, 8, 9, 10, 11, 12]))
    # this time, the values are in the same order as they were in the input

    Example with strings as ids:

    >>> ids = numpy.array(["alpha", "alpha", "beta", "omega", "alpha", "gamma", "beta"])
    >>> values = numpy.array([1, 2, 10, 100, 3, 1000, 20])
    >>> gen = igroupby(ids, values)
    >>> next(gen)
    ('alpha', array([1, 2, 3]))
    >>> next(gen)
    ('beta', array([10, 20]))
    >>> next(gen)
    ('gamma', array([1000]))
    >>> next(gen)
    ('omega', array([100]))
    """
    # convert to numpy arrays
    ids = numpy.asarray(ids)
    values = numpy.asarray(values)
    # check input shape
    assert len(ids.shape) == 1
    if n is None:
        n = ids.shape[0]
    assert ids.shape[0] >= n and values.shape[0] >= n, values.shape
    # sort if needed
    if not assume_sorted:
        ids = ids[:n]
        values = values[:n]
        if stable:
            # slower, but ensures relative order of values will be preserved
            asort = numpy.argsort(ids, kind='stable')
        else:
            asort = numpy.argsort(ids)
        ids = ids[asort]
        values = values[asort]
    # init
    start_block = 0
    find_next_lookup = find_next_hint
    # search next change block by block
    disable = logging_prefix is None
    with tqdm(total=n, desc=logging_prefix, disable=disable, file=sys.stdout) as pbar:
        while start_block < n:
            # find all items having id by block boundaries
            current_id = ids[start_block]
            try:
                end_block = first(ids,
                                  lambda x: x != current_id,
                                  offset=start_block,
                                  batch_size=find_next_lookup)
                find_next_lookup = max(
                    find_next_hint, 2 * (end_block - start_block))
            except StopIteration:
                end_block = n
            current_id_values = values[start_block:end_block]
            assert (ids[start_block:end_block] == current_id).all()
            pbar.update(end_block - start_block)
            start_block = end_block
            yield current_id, current_id_values


def ufunc_group_by_idx(idx, values, ufunc, init, minlength=None):
    """
    Abstract wrapper to compute ufunc grouped by values in array ``idx``.

    Return an array containing the results of ``ufunc`` applied to ``values``
    grouped by the indexes in array ``idx``.
    (See available ufuncs `here <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_).

    Warning: the ``init`` parameter is not a filling value for missing indexes.
    If index ``i`` is missing, then ``out[i] = init``
    but this value also serves as the initialization of ``ufunc`` on all the groups of ``values``.

    For example, if ``ufunc`` is ``numpy.add`` and ``init = -1`` then for each index,
    the sum of the corresponding values will be decreased by one.

    :param array idx: ``(n,) int array``
    :param array values: ``(n,) dtype array``
    :param numpy.ufunc ufunc: universal function applied to the groups of ``values``
    :param dtype init: initialization value
    :param int? minlength: ``(default: idx.max() + 1)``
    :returns: (min-length,) dtype array, such that ``out[i] = ufunc(values[idx==i])``

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> ufunc_group_by_idx(idx, values, numpy.maximum, -1)
    array([ 4, 6,  -1,  6])
    >>> ufunc_group_by_idx(idx, values, numpy.add, -1)
    array([ 9, 11, -1,  9])
    >>> ufunc_group_by_idx(idx, values, numpy.add, 0)
    array([ 10, 12, -0,  10])
    """
    length = max(idx.max() + 1 if idx.size else 0, minlength or 0)
    out = numpy.full(length, init)
    ufunc.at(out, idx, values)
    return out


def min_by_idx(idx, values, minlength=None, fill=None):
    """
    Given array of indexes ``idx`` and array ``values``,
    outputs the max value by idx, aligned on ``arange(idx.max() + 1)``.
    See also ``argmin_by_idx`` and ``value_at_argmin_by_idx``.

    :param array idx: (n,) int array
    :param array values: (n,) float array
    :param int? minlength: (default: idx.max() + 1)
    :param float? fill: filling value for missing idx (default: +inf)
    :returns: (min-length,) float array, such that out[i] = min(values[idx==i])

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([1, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> min_by_idx(idx, values, fill=100)
    array([  1,   0, 100,   0])
    >>> min_by_idx(idx, values)
    array([1, 0, 9223372036854775807, 0])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where (idx >= 0).all()')
    if fill is None:
        fill = numpy.inf if values.dtype.kind == 'f' else numpy.iinfo(values.dtype).max
    else:
        assert fill >= values.max()
    return ufunc_group_by_idx(idx, values, numpy.minimum, fill, minlength=minlength)


def max_by_idx(idx, values, minlength=None, fill=None):
    """
    Given array of indexes ``idx`` and array ``values``,
    outputs the max value by idx, aligned on ``arange(idx.max() + 1)``.
    See also ``argmax_by_idx`` and ``value_at_argmax_by_idx``.

    :param array idx: (n,) int array
    :param array values: (n,) float array
    :param int? minlength: (default: idx.max() + 1)
    :param float? fill: filling value for missing idx (default: -inf)
    :returns: (min-length,) float array, such that out[i] = max(values[idx==i])

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> max_by_idx(idx, values, fill=-1)
    array([ 4, 6,  -1,  6])
    >>> max_by_idx(idx, values, minlength=10, fill=-1)
    array([ 4,  6, -1,  6, -1, -1, -1, -1, -1, -1])
    >>> max_by_idx(idx, values)
    array([ 4, 6, -9223372036854775808, 6])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    if fill is None:
        fill = - numpy.inf if values.dtype.kind == 'f' else numpy.iinfo(values.dtype).min
    else:
        assert fill <= values.min()
    return ufunc_group_by_idx(idx, values, numpy.maximum, fill, minlength=minlength)


def argmin_by_idx(idx, values, minlength=None, fill=None):
    """
    Given array of indexes ``idx`` and array ``values``,
    outputs the argmin of the values by idx,
    aligned on ``arange(idx.max() + 1)``.
    See also ``min_by_idx`` and ``value_at_argmin_by_idx``.

    :param array idx: (n,) int array
    :param array values: (n,) float array
    :param int? minlength: (default: idx.max() + 1)
    :param float? fill: filling value for missing idx (default: -1)
    :returns: (min-length,) int32 array, such that
              out[i] = argmin_{idx}(values[idx] : idx[idx] == i)

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> argmin_by_idx(idx, values, fill=-1)
    array([ 0,  5, -1,  9])
    >>> argmin_by_idx(idx, values, minlength=10, fill=-1)
    array([ 0,  5, -1,  9, -1, -1, -1, -1, -1, -1])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    if fill is None:
        fill = -1
    min_values_by_idx = min_by_idx(idx, values, minlength)  # (n-idx,)
    is_min = values == min_values_by_idx[idx]
    out = numpy.full(min_values_by_idx.size, fill)
    out[idx[is_min]] = numpy.where(is_min)[0]
    return out


# TODO: improve test
def value_at_argmin_by_idx(idx, sorting_values, fill, output_values=None, minlength=None):
    """
    Wrapper around argmin_by_idx and get_value_by_idx.
    Allows to use a different value for the output and for detecting the minimum
    Allows to set a specific fill value that is not compared with the sorting_values

    :param array idx: (n,) uint array with values < max_idx
    :param array values: (n,) array
    :param fill: filling value for output[i] if there is no idx == i
    :param array? output_values: (n,) dtype array
        Useful if you want to select the min based on one array,
        and get the value on another array
    :param int? minlength: minimum shape for the output array.
    :returns array: (max_idx+1,), dtype array such that
        out[i] = min(values[idx==i])

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> value_at_argmin_by_idx(idx, values, fill=-1)
    array([ 0,  0, -1,  0])
    >>> value_at_argmin_by_idx(idx, values, minlength=10, fill=-1)
    array([ 0,  0, -1,  0, -1, -1, -1, -1, -1, -1])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    length = max(idx.max() + 1, minlength or 0)
    if output_values is None:
        output_values = sorting_values
    out = numpy.full(length, fill, dtype=output_values.dtype)
    argmin = argmin_by_idx(idx,
                           sorting_values,
                           minlength=minlength)
    mask = (argmin != -1)
    out[:mask.size][mask] = output_values[argmin[mask]]
    return out


def argmax_by_idx(idx, values, minlength=None, fill=None):
    """
    Given array of indexes ``idx`` and array ``values``,
    outputs the argmax of the values by idx,
    aligned on ``arange(idx.max() + 1)``.
    See also ``max_by_idx`` and ``value_at_argmax_by_idx``.

    :param array idx: (n,) int array
    :param array values: (n,) float array
    :param int? minlength: (default: idx.max() + 1)
    :param float? fill: filling value for missing idx (default: -1)
    :returns: (min-length,) int32 array, such that
              out[i] = argmax_{idx}(values[idx] : idx[idx] == i)

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> argmax_by_idx(idx, values, fill=-1)
    array([ 4,  8, -1, 11])
    >>> argmax_by_idx(idx, values, minlength=10, fill=-1)
    array([ 4,  8, -1, 11, -1, -1, -1, -1, -1, -1])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    if fill is None:
        fill = -1
    max_values_by_idx = max_by_idx(idx, values, minlength)  # (n-idx,)
    is_max = values == max_values_by_idx[idx]
    out = numpy.full(max_values_by_idx.size, fill)
    out[idx[is_max]] = numpy.where(is_max)[0]
    return out


# TODO: improve test
def value_at_argmax_by_idx(idx, sorting_values, fill, output_values=None, minlength=None):
    """
    Wrapper around ``argmax_by_idx`` and ``get_value_by_id``.
    Allows to use a different value for the output and for detecting the minimum
    Allows to set a specific fill value that is not compared with the sorting_values

    :param array idx: (n,) uint array with values < max_idx
    :param array values: (n,) array
    :param fill: filling value for output[i] if there is no idx == i
    :param array? output_values: (n,) dtype array
        Useful if you want to select the min based on one array,
        and get the value on another array
    :param int? minlength: minimum shape for the output array.
    :returns array: (max_idx+1,), dtype array such that
        out[i] = max(values[idx==i])

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> value_at_argmax_by_idx(idx, values, fill=-1)
    array([ 4,  6, -1,  6])
    >>> value_at_argmax_by_idx(idx, values, minlength=10, fill=-1)
    array([ 4,  6, -1,  6, -1, -1, -1, -1, -1, -1])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    length = max(idx.max() + 1, minlength or 0)
    if output_values is None:
        output_values = sorting_values
    out = numpy.full(length, fill, dtype=output_values.dtype)
    argmax = argmax_by_idx(idx,
                           sorting_values,
                           minlength=minlength)
    mask = (argmax != -1)
    out[:mask.size][mask] = output_values[argmax[mask]]
    return out


def connect_adjacents_in_groups(group_ids, values, max_gap):
    """
    For each group_id in ``group_ids``, connect values that are closer than ``max_gap`` together.

    Return an array mapping the values to the indexes of
    the newly formed connected components they belong to.

    Two values that don't have the same input group_id can's be connected in the same
    connected component.

    ``connect_adjacents_in_groups`` is faster when an array of indexes is provided as ``group_ids``,
    but also accepts other types of ids.

    :param array group_ids: ``(n,) dtype array``
    :param array values: ``(n,) float array``
    :param float max_gap: maximum distance between a value and the nearest value in the same group.
    :returns: ``(n,) uint array``,
        such that ``out[s[i]]==out[s[i+1]]`` :math:`\iff`
        ``group_ids[s[i]]==group_ids[s[i+1]]`` & ``|values[s[i]]-values[s[i+1]]| <= max_gap``
        where ``s[i]`` is the ``i`` -th index when sorting by id and value

    Example
    _______
    >>> group_ids = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3])
    >>> values = numpy.array([ 0, 35, 20, 25, 30,  0,  5, 10, 20,  0,  5, 10, 15])
    >>> connect_adjacents_in_groups(group_ids, values, max_gap = 5)
    array([0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4], dtype=uint32)

    Example with string ``group_ids``:

    >>> group_ids = numpy.array(['alpha', 'alpha', 'alpha', 'alpha', 'alpha', 'beta', 'beta', 'beta', 'beta', 'gamma', 'gamma', 'gamma', 'gamma'])
    >>> values = numpy.array([ 0, 35, 20, 25, 30,  0,  5, 10, 20,  0,  5, 10, 15])
    >>> connected_components_ids = connect_adjacents_in_groups(group_ids, values, max_gap = 5)

    The function does not require the ``group_ids`` or the ``values`` to be sorted:

    >>> shuffler = numpy.random.permutation(len(group_ids))
    >>> group_ids_shuffled = group_ids[shuffler]
    >>> values_shuffled = values[shuffler]
    >>> connect_adjacents_in_groups(group_ids_shuffled, values_shuffled, max_gap = 5)
    array([2, 1, 0, 2, 4, 1, 1, 4, 1, 4, 3, 2, 4], dtype=uint32)
    >>> connected_components_ids[shuffler]
    array([2, 1, 0, 2, 4, 1, 1, 4, 1, 4, 3, 2, 4], dtype=uint32)
    """
    as_idx = False
    if group_ids.dtype.kind in 'ui':
        if min(group_ids) >= 0 and max(group_ids) < (1 << 6) * len(group_ids):
            as_idx = True
    # FIXME: add old max and old min for it to work with pandas DataFrames
    if as_idx:
        values_for_uint32 = linearscaling(
            values, 1, (1 << 32) - float(1 << 8) - 1)
        args = lexsort_uint32_pair(group_ids, values_for_uint32)
    else:
        args = numpy.lexsort((values, group_ids))
    group_ids = group_ids[args]  # e.g.  1 1 1 1 1 1 1 1 1 2 2 2 2
    values = values[args]  # e.g.        1 1 1 2 2 3 3 9 9 1 2 2 9
    # to_split  e.g.                     0 0 0 0 0 0 1 0 1 0 0 1
    to_split = ((group_ids[1:] != group_ids[:-1])
                | ((values[1:] - values[:-1]) > max_gap))
    # group_idx  e.g.                    0 0 0 0 0 0 0 1 1 2 2 2 3
    group_idx = numpy.empty(group_ids.size, dtype='uint32')
    group_idx[0] = 0
    numpy.cumsum(to_split, out=group_idx[1:])
    # reverse argsort
    aligned_group_idx = numpy.empty_like(group_idx)
    aligned_group_idx[args] = group_idx
    return aligned_group_idx


# TODO: improve test
def get_value_by_idx(idx, values, default, check_unique=True, minlength=None):
    """
    Given array of indexes ``idx`` and array ``values`` (unordered, not necesarilly full),
    output array such that ``out[i] = values[idx==i]``.

    If all indexes in ``idx`` are unique, it is equivalent to sorting the ``values``
    by their ``idx`` and filling  with ``default`` for missing ``idx``.

    If ``idx`` elements are not unique and you still want to proceed,
    you can set ``check_unique`` to ``False``. The output values for the non-unique indexes
    will be chosen arbitrarily among the multiple values corresponding.

    :param array idx: ``(n,) uint array`` with values < max_idx
    :param array values: ``(n,) dtype array``
    :param dtype default: filling value for ``output[i]`` if there is no ``idx == i``
    :param bool check_unique: if ``True``, will check that ``idx`` are unique
        If ``False``, if the ``idx`` are not unique, then an arbitrary value
        will be chosen.
    :param int? minlength: minimum shape for the output array (``default: idx.max() + 1``).
    :returns array: (max_idx+1,), dtype array such that
        ``out[i] = values[idx==i]``.

    Example
    _______
    >>> idx = numpy.array([8,2,4,7])
    >>> values = numpy.array([100, 200, 300, 400])
    >>> get_value_by_idx(idx, values, -1, check_unique=False, minlength=None)
    array([ -1,  -1, 200,  -1, 300,  -1,  -1, 400, 100])

    Example with non-unique elements in ``idx``:

    >>> idx = numpy.array([2,2,4,7])
    >>> values = numpy.array([100, 200, 300, 400])
    >>> get_value_by_idx(idx, values, -1, check_unique=False, minlength=None)
    array([ -1,  -1, 200,  -1, 300,  -1,  -1, 400])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer indexes in `idx`, where (idx >= 0).all()')
    if check_unique:
        assert numpy.unique(idx).shape == idx.shape, "indexes in `idx` should be unique"
    length = max(idx.max() + 1, minlength or 0)
    out = numpy.full(length, default, dtype=values.dtype)
    out[idx] = values
    return out


# TODO: improve test and add example in doc
def get_most_common_by_idx(idx, values, fill, minlength=None):
    """
    Given array of indexes ``idx`` and array ``values``,
    outputs the most common value by idx.

    :param array idx: (n,) uint array with values < max_idx
    :param array values: (n,) non-float, dtype array
    :param fill: filling value for output[i] if there is no idx == i
    :param minlength: minimum shape for the output array.
    :returns: (max_idx+1,), dtype array such that
        out[i] = the most common value such that (values[idx==i])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where all idx >= 0')
    assert values.dtype.kind != 'f', ('values dtype is float - Please convert to other dtype'
                                      'or digitize values before using get_most_common_by_idx')
    length = max(idx.max() + 1, minlength or 0)

    sessions_uv = to_structured([
        ('idx', idx),
        ('value', values),
    ])
    sessions_uv_idx, uv_idx2id, _ = factorize(sessions_uv)
    uv_idx2id = numpy.asarray(
        uv_idx2id, [('idx', 'uint32'), ('value', values.dtype)])  # cast to struct
    count_by_uv_idx = numpy.bincount(sessions_uv_idx)
    top_uv_by_id = argmax_by_idx(
        uv_idx2id['idx'], count_by_uv_idx, minlength=length)
    out = uv_idx2id[top_uv_by_id]['value']
    out[top_uv_by_id == -1] = fill
    return out


def average_by_idx(idx, values, weights=None, minlength=None, fill=0, dtype='float64'):
    """
    Compute average-by-idx given array of indexes ``idx``, ``values``, and optional ``weights``

    :param array idx: (n,) int array
    :param array values: (n,) float array
    :param array? weights: (n,) float array
    :param int? minlength: (default: idx.max() + 1)
    :param float? fill: filling value for missing idx (default: 0)
    :param str? dtype: (default: 'float32')
    :returns: (min-length,) float array, such that out[i] = mean(values[idx==i])

    Example
    _______
    >>> idx  = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> average_by_idx(idx, values, fill=0)
    array([ 2.        ,  3.        , 0.        ,  3.33333333])
    >>> weights = numpy.array([0, 1, 0, 0, 0, 1, 2, 3, 4, 1, 1, 0])
    >>> average_by_idx(idx, values, weights=weights, fill=0)
    array([ 1.,  4., 0.,  2.])
    """
    assert idx.dtype.kind == 'u' or (idx.dtype.kind == 'i' and (idx >= 0).all()), (
        'Can only use get_xx_by_idx with integer idx, where (idx >= 0).all()')
    # FIXME: define dtype whitelist instead
    assert values.dtype.kind not in 'USOb', ('values dtype not supported')
    norm_by_idx = numpy.bincount(
        idx, weights, minlength=minlength).astype(dtype)
    if weights is not None:
        values = values * weights
    sum_by_idx = numpy.bincount(idx, values, minlength=minlength).astype(dtype)
    with numpy.warnings.catch_warnings():
        numpy.warnings.filterwarnings('ignore', r'.*divide.*')
        return numpy.where(norm_by_idx > 0, sum_by_idx / norm_by_idx, fill)
