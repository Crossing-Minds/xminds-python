"""
Utils
=====
"""

from base64 import b64decode
from functools import wraps
import hashlib
import os
import time

import numpy

HAS_SCIPY = True
try:
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
except ImportError:
    HAS_SCIPY = False

from .arraybase import clean_dtype


def deep_hash(value, prefix=None, fmt=None):
    """
    Hash values in a nested structure such as `dict` or `list`.
    Most useful to check the integrity of nested values

    :param nested-object value: nested structure to hash, supporting numpy arrays and scipy sparse
    :param bytes? prefix: optional prefix to make hash unique
    :param str? fmt: output format. Can be ``'long'`` (default)
        or ``'int'`` or ``'bytes20'`` or ``'hex40'``
    :returns: ``long`` or ``int`` or ``bytes`` or ``str``, depending on ``fmt``

    Example
    _______
    >>> val = {
    >>>     'arrays': {
    >>>         'int': numpy.arange(1000),
    >>>         'struct': numpy.array([(11, 12), (21, 22)], [('a', int), ('b', int)]),
    >>>         'struct-sliced': numpy.array([(11, 12), (21, 22)], [('a', int), ('b', int)])[['a']],
    >>>         'transposed': numpy.arange(3*5).reshape((3, 5)).T,
    >>>     },
    >>>     'sparse': {
    >>>         'csr': scipy.sparse.random(5, 4, 0.1, 'csr'),
    >>>         'csc': scipy.sparse.random(5, 4, 0.1, 'csc'),
    >>>         'coo': scipy.sparse.random(5, 4, 0.1, 'coo'),
    >>>     },
    >>>     'scalars': [1, 2.5, 'str', b'bytes', numpy.float32(1.5), numpy.bytes_(b'npbytes')]
    >>> }

    The hash can returned with different formats:

    >>> h1 = deep_hash(val, fmt='long')
    >>> h2 = deep_hash(val, fmt='hex40')
    >>> h3 = deep_hash(val, fmt='bytes20')
    >>> h1, h2, h3
    (699019679910377672527134164600537195154359546715,
    '7a7120641ee1b531deb2f14d04c986be5d89735b',
    b'zq d\\x1e\\xe1\\xb51\\xde\\xb2\\xf1M\\x04\\xc9\\x86\\xbe]\\x89s[')

    A prefix can be added to make the hash unique:

    >>> h1 = deep_hash(val)
    >>> h2 = deep_hash(val, prefix=b'my-prefix')
    >>> h1 == h2
    False

    Two inputs that are not identical lead to different hash values:

    >>> val['arrays']['int'][50] += 1
    >>> h1 == deep_hash(val)
    False
    >>> val['arrays']['int'][50] -= 1
    >>> h1 == deep_hash(val)
    True
    """
    fmt = fmt or 'long'
    h = hashlib.sha1()
    if prefix:
        h.update(prefix)
    _deep_hash_update(h, value)
    if fmt == 'long':
        return int.from_bytes(h.digest(), byteorder='big')
    elif fmt == 'int':
        return hash(int.from_bytes(h.digest(), byteorder='big'))
    elif fmt == 'bytes20':
        return h.digest()
    elif fmt == 'hex40':
        return h.hexdigest()
    raise NotImplementedError(fmt)


def _deep_hash_update(h, value):
    if value is None:
        h.update(b'None')
    elif isinstance(value, bytes):
        h.update(value)
    elif isinstance(value, str):
        h.update(value.encode('utf8'))
    elif isinstance(value, (int, float, numpy.number, numpy.dtype)):
        h.update(str(value).encode('ascii'))
    elif isinstance(value, dict):
        for k, v in value.items():
            h.update(k.encode('ascii'))
            _deep_hash_update(h, v)
    elif isinstance(value, numpy.ndarray):
        if value.dtype.names:
            h.update(value.astype(clean_dtype(value.dtype, sort=True)))
        else:
            h.update(numpy.ascontiguousarray(value))
    elif HAS_SCIPY and isinstance(value, (csr_matrix, csc_matrix)):
        h.update(value.data)
        h.update(value.indices)
        h.update(value.indptr)
    elif HAS_SCIPY and isinstance(value, coo_matrix):
        h.update(value.data)
        h.update(value.row)
        h.update(value.col)
    elif hasattr(value, '__iter__'):
        for v in value:
            _deep_hash_update(h, v)
    else:
        raise NotImplementedError(type(value))


class ClassPropertyDescriptor(object):
    """
    By `Mahmoud Abdelkader <https://stackoverflow.com/a/5191224/710358>`_.

    Example
    _______
    >>> class MyTest(object):
    >>>    _val = 42

    >>>    @classproperty
    >>>    def val(cls):
    >>>        return cls._val

    >>>    @val.setter
    >>>    def val(cls, value):
    >>>        cls._val = value
    """

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        return self.fget.__get__(obj, objtype)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        objtype = type(obj)
        return self.fset.__get__(obj, objtype)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """ Equivalent of @property for class """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def getenv(name, default=None, cast=None):
    """
    Return the value of the environment variable ``name`` if it exists,
    otherwise return the default value.

    Specifying a value for the cast argument enables to return the value of the environement
     variable with the specified format (instead of the default ``'str'`` format).

    :param str name: the name of the environement variable
    :param str? default: string denoting the default value
        in case name does not exists. (``default: None``)
    :param function? cast: either function to cast the environment variable,
        or ``list`` to split comma separated values,
        or ``bool`` to handle strings: ``'1'``, ``'true'``, ``'yes'`` (case insensitive),
        or ``'b64'`` to decode base64-encoded bytes.
    :returns: environment variable
    """
    val = os.getenv(name, default)
    if cast is not None and isinstance(val, str):
        if cast == list:
            return val.split(',')
        elif cast == bool:
            return is_true(val)
        elif cast == 'b64':
            return b64decode(val)
        else:
            return cast(val)
    return val


def is_true(value):
    """
    Return ``True`` if the input value is ``'1'``, ``'true'`` or ``'yes'`` (case insensitive)

    :param str value: value to be evaluated
    :returns: bool

    Example
    _______
    >>> is_true('1')
    True
    """
    return str(value).lower() in ['true', '1', 'yes']


def is_false(value):
    """
    Return ``True`` if the input value is ``'0'``, ``'false'`` or ``'no'`` (case insensitive)

    :param str value: value to be evaluated
    :returns: bool

    Example
    _______
    >>> is_false('0')
    True
    >>> is_false('1')
    False
    """
    return str(value).lower() in ['false', '0', 'no']


def retry(base=1, multiplier=8, max_retry=2, exception=Exception,
          reraise=True):
    """
    Decorator retrying funtcions with exponential backoff.

    The retry mechanism can be aborted by the client by using myfunction(..., __retry__=False).

    The maximum time of execution is :math:`\sum_{k=1}^{max\_retry}base \\times multiplier^k`.

    :param int base: base time
    :param int multiplier: multiplier
    :param int max_retry: maximum number of attempts
    :param exception: exception
    """
    def _decorator(func):
        @wraps(func)
        def _wrap(*args, **kwargs):
            message = '{name} failed with {exc_ty}: {exc}'
            # if the caller sets __retry__=False, don't do anything
            if not kwargs.pop('__retry__', True):
                return func(*args, **kwargs)
            # retry loop
            wait = base
            for _ in range(max_retry):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    print(message.format(name=func.__name__,
                                         exc_ty=type(e).__name__, exc=str(e)))
                    print('Retrying in {}s'.format(wait))
                    time.sleep(wait)
                    wait *= multiplier
            # rerun without try when reraise
            if reraise:
                return func(*args, **kwargs)
            # rerun with try when not reraise
            try:
                return func(*args, **kwargs)
            except exception as e:
                print(message.format(name=func.__name__,
                                     exc_ty=type(e).__name__, exc=str(e)))
        return _wrap
    return _decorator
