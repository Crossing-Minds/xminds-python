"""
xminds.api.exceptions
~~~~~~~~~~~~~~~~~~~~~

This module defines all custom API exceptions.
All exceptions inherit from `XMindsError`.
"""

class XMindsError(Exception):
    """ Base class for all Crossing Minds Exceptions """
    retry_after = None

    def __init__(self, data=None):
        self.message = self.MSG
        self.data = data
        if self.message and data:
            try:
                self.message = self.message.format(**data)
            except KeyError as e:
                print(f'Missing key {e} in `error_extra_data`')

    def __str__(self):
        msg = self.message
        if self.data:
            msg = msg + ' ' + str(self.data)
        return msg


# === Server Errors ===


class ServerError(XMindsError):
    MSG = 'Unknown error from server'
    CODE = 0
    HTTP_STATUS = 500


class ServerUnavailable(XMindsError):
    MSG = 'The server is currently unavailable, please try again later'
    CODE = 1
    HTTP_STATUS = 503
    retry_after = 1


class TooManyRequests(XMindsError):
    MSG = 'The amount of requests exceeds the limit of your subscription'
    CODE = 2
    HTTP_STATUS = 429
    retry_after = 1  # should be passed in __init__ instead

    def __init__(self, retry_after=None):
        if retry_after:
            self.retry_after = retry_after
        super().__init__()


# === Authentication Errors ====


class AuthError(XMindsError):
    HTTP_STATUS = 401
    MSG = 'Cannot perform authentication: {error}'
    CODE = 21


class JwtTokenExpired(AuthError):
    MSG = 'The JWT token has expired'
    CODE = 22


class RefreshTokenExpired(AuthError):
    MSG = 'The refresh token has expired'
    CODE = 28


# === Request Errors ===


class RequestError(XMindsError):
    HTTP_STATUS = 400


class WrongData(RequestError):
    MSG = 'There is an error in the submitted data'
    CODE = 40


class DuplicatedError(RequestError):
    MSG = 'The {type} {key} is duplicated'
    CODE = 42


class ForbiddenError(XMindsError):
    HTTP_STATUS = 403
    MSG = 'Do not have enough permissions to access this resource: {error}'
    CODE = 50



# === Resource Errors ===


class NotFoundError(XMindsError):
    HTTP_STATUS = 404
    MSG = 'The {type} {key} does not exist'
    CODE = 60


class MethodNotAllowed(XMindsError):
    HTTP_STATUS = 405
    MSG = 'Method "{method}" not allowed'
    CODE = 70


# === Utils to build exception from code ===


def _get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        {s for c in cls.__subclasses__() for s in _get_all_subclasses(c)})


_ERROR_CLASSES = _get_all_subclasses(XMindsError)


@classmethod
def from_code(cls, code, data=None):
    if data is None:
        data = {}
    try:
        c = next(c for c in _ERROR_CLASSES if getattr(c, 'CODE', -1) == code)
    except StopIteration:
        print(f'unknown error code {code}')
        c = ServerError
    exc = c.__new__(c)
    XMindsError.__init__(exc, data)
    return exc


XMindsError.from_code = from_code
