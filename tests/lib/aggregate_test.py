import unittest

import numpy

from xminds.lib.arrays import to_structured
from xminds.lib.aggregate import (
    igroupby, min_by_idx, max_by_idx, argmin_by_idx, argmax_by_idx,
    average_by_idx, connect_adjacents_in_groups,
    value_at_argmax_by_idx, value_at_argmin_by_idx,
    get_value_by_idx, get_most_common_by_idx
)


def test_igroupby():
    tests = [
        # basic
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
         [0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6],
         {'assume_sorted': True, 'find_next_hint': 2},
         [(0, [0, 1, 2, 3, 4]), (1, [0, 2, 4, 6]), (2, [0, 4, 6])]),
        # 2d to_array
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
         [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [0, 10],
          [2, 12], [4, 14], [6, 16], [0, 10], [4, 14], [6, 16]],
         {'assume_sorted': True, 'find_next_hint': 2},
         [(0, [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]]),
          (1, [[0, 10], [2, 12], [4, 14], [6, 16]]),
          (2, [[0, 10], [4, 14], [6, 16]])]),
        # sort
        ([0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 2, 2],
         [0, 0, 0, 3, 4, 1, 2, 4, 6, 2, 4, 6],
         {'assume_sorted': False, 'find_next_hint': 2},
         [(0, [0, 1, 2, 3, 4]), (1, [0, 2, 4, 6]), (2, [0, 4, 6])]),
        # big `find_next_hint`
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
         [0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6],
         {'assume_sorted': True, 'find_next_hint': 64},
         [(0, [0, 1, 2, 3, 4]), (1, [0, 2, 4, 6]), (2, [0, 4, 6])]),
        # small `n`
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9],
         [0, 1, 2, 3, 4, 0, 2, 4, 6, 0, 4, 6, 9, 9, 9, 9, 9, 9, 9],
         {'assume_sorted': True, 'find_next_hint': 2, 'n': 11},
         [(0, [0, 1, 2, 3, 4]), (1, [0, 2, 4, 6]), (2, [0, 4, 6])]),
    ]
    for fr, to, kwargs, expected in tests:
        generator = igroupby(numpy.asarray(fr), numpy.asarray(to), **kwargs)
        results = [(fr_id, sorted(to_ids.tolist()))
                   for fr_id, to_ids in generator]
        assert results == expected


class UtilsTestCase(unittest.TestCase):
    def test_group_by_idx(self):  # TODO: add all functions
        n = 1 << 10
        n_idx = n // 10
        idx = numpy.random.randint(0, n_idx, n)
        vals = numpy.random.randn(n)
        self.min_max_by_idx_test_util(idx, vals)
        self.average_by_idx_test_util(idx, vals)
        # following util simply runs the functions, should add additional tests
        self.run_util_by_idx(idx, vals)

        # TODO: generate random datetime64, strings...
        # test everything except average_by_id
        # TODO: test with a pandas DataFrame

    def test_connect_adjacents_in_groups(self):
        group_ids = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        val = numpy.array([1, 1, 1, 2, 2, 3, 3, 9, 9, 1, 2, 2, 9])
        groups = connect_adjacents_in_groups(group_ids, val, 3)
        grp = numpy.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3])
        assert (groups == grp).all()

    @classmethod
    def min_max_by_idx_test_util(cls, idx, vals):
        n_idx = idx.max() + 1
        fill_min = vals.max() + 1
        fill_max = vals.min() - 1
        min_ = min_by_idx(idx, vals, fill=fill_min)
        max_ = max_by_idx(idx, vals, fill=fill_max)
        amin_ = argmin_by_idx(idx, vals, fill=-1)
        # TODO: add value at argmax, value at argmin
        amax_ = argmax_by_idx(idx, vals, fill=-1)
        for i in range(n_idx):
            i_vals = vals[idx == i]
            if not i_vals.size:
                assert numpy.isclose(min_[i], fill_min)
                assert numpy.isclose(max_[i], fill_max)
                assert amin_ == -1
                assert amax_ == -1
            else:
                assert numpy.isclose(min_[i], i_vals.min())
                assert numpy.isclose(max_[i], i_vals.max())
                assert numpy.isclose(vals[amin_[i]], min_[i])
                assert numpy.isclose(vals[amax_[i]], max_[i])

    @classmethod
    def run_util_by_idx(cls, idx, vals):
        # TODO - add tests - currently simply running the methods
        value_at_argmax_by_idx(idx, vals, fill=-1, output_values=vals)
        value_at_argmin_by_idx(idx, vals, fill=-1, output_values=vals)
        get_value_by_idx(idx, vals, default=-1, check_unique=False)
        get_most_common_by_idx(idx, (10 * vals).astype('int32'), fill=-1)

    @classmethod
    def average_by_idx_test_util(cls, idx, vals):
        n_idx = idx.max() + 1
        fill_avg = 0
        avg_ = average_by_idx(idx, vals, fill=fill_avg)
        for i in range(n_idx):
            i_vals = vals[idx == i]
            if not i_vals.size:
                assert numpy.isclose(avg_[i], fill_avg)
            else:
                assert numpy.isclose(avg_[i], i_vals.mean())
