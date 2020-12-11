"""
Scaling tools
===================
"""
import numpy


def linearscaling(x, new_min, new_max, old_min=None, old_max=None, axis=None):
    """
    Linearly rescale input from its original range to a new range.

    :param scalar-or-array x:  scalar or arrays of scalars in ``[old_min, old_max]`` of shape (n, *shape)
    :param scalar-or-array new_min: scalar or array of shape (*shape,)
    :param scalar-or-array new_max: scalar or array of shape (*shape,)
    :param scalar-or-array? old_min: (``default=x.min()``)
    :param scalar-or-array? old_max: (``default=x.max()``)
    :param int? axis: (``default=None``)
    :return: scalar or array of scalars in ``[new_min, new_max]`` of shape (n, *shape)

    Example
    _______
    >>> linearscaling(0, -10, 10, 0, 1)
    -10.0

    When the original range is not passed on, it is considered to be the interval in which the input values are.

    >>> x = numpy.arange(0, 1, 0.1)
    >>> linearscaling(x, 0, 10)
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> linearscaling(x, 0, 10, 0, 2)
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

    Linear scaling can be performed on dfferent series of data having different ranges provided that the new minima/maxima are specfied for each serie.

    >>> new_min = numpy.array([0, 0])
    >>> new_max = numpy.array([1, 100])
    >>> old_min = numpy.array([0, 0])
    >>> old_max = numpy.array([10, 10])
    >>> x = numpy.arange(0, 1, 0.1).reshape(5, 2)
    >>> x
    array([[0. , 0.1],
       [0.2, 0.3],
       [0.4, 0.5],
       [0.6, 0.7],
       [0.8, 0.9]])
    >>> linearscaling(x, new_min, new_max, old_min, old_max)
    array([[ 0.1, 10. ],
       [ 0.2, 20. ],
       [ 0.3, 30. ],
       [ 0.4, 40. ],
       [ 0.5, 50. ]])
    """
    if not isinstance(x, numpy.ndarray):
        assert old_min is not None
        assert old_max is not None
        assert old_min <= x <= old_max
    else:
        # set default old to current
        if old_min is None:
            old_min = x.min(axis=axis)
        if old_max is None:
            old_max = x.max(axis=axis)
        if axis is not None:
            old_min = numpy.expand_dims(old_min, axis)
        if axis is not None:
            old_max = numpy.expand_dims(old_max, axis)
        assert ((old_min <= x) & (x <= old_max)).all(), (x.min(), x.max())
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min + 1e-32)
