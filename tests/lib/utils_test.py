import numpy

import scipy.sparse

import unittest

import pytest

from xminds.lib.utils import deep_hash, classproperty, retry


def test_deep_hash():
    val = {
        'arrays': {
            'int': numpy.arange(1000),
            'struct': numpy.array([(11, 12), (21, 22)], [('a', int), ('b', int)]),
            'struct-sliced': numpy.array([(11, 12), (21, 22)], [('a', int), ('b', int)])[['a']],
            'transposed': numpy.arange(3*5).reshape((3, 5)).T,
        },
        'sparse': {
            'csr': scipy.sparse.random(5, 4, 0.1, 'csr'),
            'csc': scipy.sparse.random(5, 4, 0.1, 'csc'),
            'coo': scipy.sparse.random(5, 4, 0.1, 'coo'),
        },
        'scalars': [1, 2.5, 'str', b'bytes', numpy.float32(1.5), numpy.bytes_(b'npbytes')]
    }
    h1 = deep_hash(val)
    # test prefix works
    assert h1 != deep_hash(val, prefix=b'my-prefix')
    # test modify something inside the array
    val['arrays']['int'][50] = 999
    h2 = deep_hash(val)
    assert h2 != h1
    # test `fmt`
    h_long = deep_hash(val, fmt='long')
    assert isinstance(h_long, int) and h_long.bit_length(
    ) <= 160 and h_long.bit_length() > 64
    h_int = deep_hash(val, fmt='int')
    assert isinstance(h_int, int) and h_int.bit_length(
    ) <= 64 and h_int.bit_length() > 32
    h_hex = deep_hash(val, fmt='hex40')
    assert isinstance(h_hex, str) and len(h_hex) == 40
    h_bytes = deep_hash(val, fmt='bytes20')
    assert isinstance(h_bytes, bytes) and len(h_bytes) == 20


class ClassPropertyTestCase(unittest.TestCase):
    def test_access(self):
        class MyTest(object):
            @classproperty
            def name(cls):
                return cls.__name__

        assert MyTest.name == 'MyTest'
        instance = MyTest()
        assert instance.name == 'MyTest'

    def test_setter(self):
        class MyTest(object):
            _val = 42

            @classproperty
            def val(cls):
                return cls._val

            @val.setter
            def val(cls, value):
                cls._val = value

        assert MyTest.val == 42
        instance = MyTest()
        assert instance.val == 42
        MyTest.val = 43
        assert MyTest.val == 43
        assert instance.val == 43


def test_retry():
    lst = []

    @retry(base=0.0001, max_retry=3)
    def dummy(arg):
        lst.append(arg)
        raise ValueError('failed')

    try:
        dummy(2)
    except ValueError:
        pass

    assert lst == [2, 2, 2, 2]


def test_not_retry():
    lst = []

    @retry(base=0.0001, max_retry=3)
    def dummy(arg):
        lst.append(arg)
        raise ValueError('failed')

    try:
        dummy(2, __retry__=False)
    except ValueError:
        pass

    assert lst == [2]
