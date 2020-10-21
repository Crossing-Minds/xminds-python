"""
Iterable tools
==============
"""

def split(iterable, function):
    """
    Split an iterable into two lists according to test function

    Example
    _______

    >>> split([1,2,3,4,5,6], lambda x: x<3)
    ([1, 2], [3, 4, 5, 6])
    """
    match = []
    unmatch = []
    for value in iterable:
        if function(value):
            match.append(value)
        else:
            unmatch.append(value)
    return match, unmatch
