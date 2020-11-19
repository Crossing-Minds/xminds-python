import unittest

import numpy

from xminds.ds.scaling import linearscaling


class LinearToolsTestCase(unittest.TestCase):
    def test_linearscaling_scalar(self):
        assert numpy.isclose(linearscaling(0, 0, 10, 0, 1), 0)
        assert numpy.isclose(linearscaling(1, 0, 10, 0, 1), 10)
        assert numpy.isclose(linearscaling(.5, 0, 10, 0, 1), 5.)
        assert numpy.isclose(linearscaling(-1, 0, 10, -1, 1), 0)
        assert numpy.isclose(linearscaling(1, 0, 10, -1, 1), 10)
        assert numpy.isclose(linearscaling(0, 0, 10, -1, 1), 5.)
        assert numpy.isclose(linearscaling(-1, 10, -10, -1, 1), 10)
        assert numpy.isclose(linearscaling(1, 10, -10, -1, 1), -10)
        assert numpy.isclose(linearscaling(0, 10, -10, -1, 1), 0)

    def test_linearscaling_1d(self):
        x = numpy.asarray([0., 0.5, 1.])
        assert numpy.allclose(linearscaling(x, 0, 10), [0, 5, 10])
        x = numpy.asarray([-1, 0, 1])
        assert numpy.allclose(linearscaling(x, 0, 10), [0, 5, 10])
        x = numpy.asarray([-1, 0, 1])
        assert numpy.allclose(linearscaling(x, 10, -10), [10, 0, -10])

    def test_linearscaling_2d(self):
        x = numpy.asarray([[-1, -2],
                           [0, 0],
                           [1, 2]], dtype='float32')
        new_min = numpy.asarray([0, 0])
        new_max = numpy.asarray([1, 2])
        expected = numpy.asarray([[0, 0],
                                  [0.5, 1],
                                  [1, 2]])
        assert numpy.allclose(linearscaling(
            x, new_min, new_max, axis=0), expected)
