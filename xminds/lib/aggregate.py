"""
Aggregage tools
===============
"""
import sys

import numpy

from .arrays import first
from ..compat import tqdm


def igroupby(ids, values, n=None, logging_prefix=None, assume_sorted=False,
             find_next_hint=512):
    """
    Efficiently converts two arrays representing a relation to an iterable (id, values_associated)

    :param array ids: ``(>=n,) uint32 array``
    :param array values: ``(>=n, *shape) uint32 array``
    :param int? n: length of array to consider. Uses full array when not set
    :param string? logging_prefix: prefix to include while logging progress. Default: Does not log
    :param bool? assume_sorted: whether ids is sorted. Default: False
    :param int? find_next_hint: hint for find_next_lookup. Default: 512
    :generates: tuple(id:int, values_associated:`(m, *shape) array slice`)

    Example
    _______
    >>> ids      = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    >>> values   = numpy.array([0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6])
    >>> gen = igrouby(ids, values)
    >>> next(gen)
    (0, array([0, 1, 2, 3, 4]))

    >>> next(gen)
    (1, array([0, 2, 4, 6]))

    >>> next(gen)
    (2, array([0, 4, 6]))
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
                find_next_lookup = max(find_next_hint, 2 * (end_block - start_block))
            except StopIteration:
                end_block = n
            current_id_values = values[start_block:end_block]
            assert (ids[start_block:end_block] == current_id).all()
            pbar.update(end_block - start_block)
            start_block = end_block
            yield current_id, current_id_values
