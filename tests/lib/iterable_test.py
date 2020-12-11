import numpy

from xminds.lib.iterable import (is_empty_iterable,
                                 get_first_of_iterable, ichunk_iterable)


def _empty_generator_function():
    if False:
        yield 1


def test_is_empty_iterable():
    tests = [
        ([], True),
        (range(0), True),
        (_empty_generator_function(), True),
        ((x for x in _empty_generator_function()), True),
        ([0], False),
        ([1, 2], False),
        (range(1), False),
        ((str(i) for i in range(3)), False),
    ]
    for iterable, expected_empty in tests:
        _, is_empty = is_empty_iterable(iterable)
        assert is_empty == expected_empty


def test_is_empty_iterable_preserves_values():
    tests = [
        ([], []),
        (range(0), []),
        (_empty_generator_function(), []),
        ((x for x in _empty_generator_function()), []),
        ([0], [0]),
        ([1, 2], [1, 2]),
        (range(1), [0]),
        ((str(i) for i in range(3)), ['0', '1', '2']),
    ]
    for iterable, expected_values in tests:
        new_iterable, _ = is_empty_iterable(iterable)
        assert list(new_iterable) == expected_values


def test_get_first_of_iterable():
    tests = [
        ([0], [0]),
        ([1, 2], [1, 2]),
        (range(1), [0]),
        ((str(i) for i in range(3)), ['0', '1', '2']),
    ]
    for iterable, values_list in tests:
        new_iterable, first = get_first_of_iterable(iterable)
        assert list(new_iterable) == values_list
        assert first == values_list[0]


def test_ichunk_iterable():
    tests = [
        range(2**10),
        range(1 + 2**10),
        range(0),
        _empty_generator_function(),
        list(range(2**10)),
    ]
    for iterable in tests:
        chunks = ichunk_iterable(iterable, 64)
        concat_out = []
        for chunk in chunks:
            values = list(chunk)
            assert len(values) <= 64
            concat_out.extend(values)
        assert list(concat_out) == list(iterable)
