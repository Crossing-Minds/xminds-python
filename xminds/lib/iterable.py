"""
Iterable tools
==============
"""
from itertools import chain, islice


def split(iterable, function):
    """
    Split an iterable into two lists according to test function

    :param iterable iterable: iterable of values to be split
    :param function function: decision function ``value => bool``
    :returns: tuple(
        list with values for which function is `True`,
        list with values for which function is `False`,)

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


def unzip(zipped):
    """
    Unzip a zipped list

    :param list-of-tuple zipped: list of tuples to be disaggregated
    :returns: list of tuples

    Example
    _______

    >>> unzip([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
    """
    return list(zip(*zipped))


def is_empty_iterable(iterable):
    """
    Check if a sequence is empty.

    Most useful on generator types.

    :param iterable iterable: input iterable
    :returns: tuple(iterable, is_empty). If a generator is passed,
        a new generator will be returned preserving the original values

    Example
    _______
    >>> a = []
    >>> b = (str(i) for i in range(0))
    >>> c = (str(i) for i in range(5))

    >>> a, is_empty = is_empty_iterable(a)
    >>> a, is_empty
    ([], True)
    >>> b, is_empty = is_empty_iterable(b)
    >>> is_empty
    True

    When the generator ``c`` is given, a new generator is returned by ``is_empty_iterable``
    to preserve original values of ``c``:

    >>> c, is_empty = is_empty_iterable(c)
    >>> next(c), is_empty
    ('0', False)
    """
    if hasattr(iterable, '__len__'):
        is_empty = len(iterable) == 0
        return iterable, is_empty
    # iterable is likely a generator-like, we need to consume one item and re-build a new generator
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        return iterable, True
    return chain([first], iterable), False


def get_first_of_iterable(iterable):
    """
    Return the first element of the given sequence.

    Most useful on generator types.

    :param iterable iterable: input iterable
    :returns: tuple(iterable, first_element). If a generator is passed,
        a new generator will be returned preserving the original values.

    :raises: IndexError

    Example
    _______
    >>> a = [1,2,3]
    >>> b = (str(i) for i in range(3))

    >>> a, first_element = get_first_of_iterable(a)
    >>> a, first_element
    ([1, 2, 3], 1)

    When the generator ``b`` is given, a new generator is returned by ``is_empty_iterable``
    to preserve original values of ``b``:

    >>> b, first_element = get_first_of_iterable(b)
    >>> next(b), first_element
    ('0', '0')
    """
    if hasattr(iterable, '__getitem__'):
        return iterable, iterable[0]
    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        raise IndexError('`iterable` is empty')
    return chain([first], iterable), first


def ichunk_iterable(iterable, chunk_length=2**12):
    """
    Split a sequence into consecutive sub-sequences of given length by returning
    a generator of generators generating the sub sequences.

    Most useful on generator types.

    :param iterable iterable: input iterable
    :param int? chunk_length: length of the chunks. (``default: 4096``)
    :generates: generators

    Example
    _______
    >>> iterable = range(15)
    >>> chunks = ichunk_iterable(iterable, 4)
    >>> for chunk in chunks:
    >>>    print(list(chunk))
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14]
    """
    iterable = iter(iterable)
    for head in iterable:
        def chunk():
            yield head
            for val in islice(iterable, chunk_length - 1):
                yield val
        yield chunk()
