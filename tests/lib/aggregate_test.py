import numpy

from xminds.lib.aggregate import igroupby


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
