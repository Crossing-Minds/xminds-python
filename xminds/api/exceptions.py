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


class BaseAuthError(XMindsError):
    HTTP_STATUS = 401


class MalformedAuthHeader(BaseAuthError):
    MSG = 'Authorization header must contain the Bearer type'
    CODE = 20


class WrongAuthToken(BaseAuthError):
    MSG = 'The token is corrupted'
    CODE = 21


class AuthTokenExpired(BaseAuthError):
    MSG = 'The token has expired'
    CODE = 22


class AuthTokenMissing(BaseAuthError):
    MSG = 'Authorization token is required'
    CODE = 23


class WrongActivationCode(BaseAuthError):
    MSG = ('The given code is invalid. Maybe you have generated a new '
           'code, please check your email. You can also create a new one '
           'and we will email it to you')
    CODE = 24


class AccountNotVerified(BaseAuthError):
    MSG = 'Your account has not been verified yet. Please check your email'
    CODE = 25


class InvalidPassword(BaseAuthError):
    MSG = 'Invalid password'
    CODE = 26


class WrongAuthRefreshToken(BaseAuthError):
    MSG = 'The refresh token is invalid'
    CODE = 27


class AuthRefreshTokenExpired(BaseAuthError):
    MSG = 'The refresh token has expired'
    CODE = 28


class WrongAccountType(BaseAuthError):
    MSG = 'Does not match the account type'
    CODE = 29


# === Request Errors ===


class BaseRequestError(XMindsError):
    HTTP_STATUS = 400


class WrongData(BaseRequestError):
    MSG = 'There is an error in the submitted data'
    CODE = 40


class MalformedData(BaseRequestError):
    MSG = 'Request data cannot be parsed: {error}'
    CODE = 41


class DuplicatedName(BaseRequestError):
    MSG = 'The name {name} already exists'
    CODE = 42


class DuplicatedOrganization(BaseRequestError):
    MSG = 'The organization {name} already exists'
    CODE = 43


class AccountVerified(BaseRequestError):
    MSG = 'The account {name} has already been verified'
    CODE = 44


class NoDatabaseSelected(BaseRequestError):
    MSG = 'There is no database assigned to the token. Check the db_id at login.'
    CODE = 45


class DuplicatedProperty(BaseRequestError):
    MSG = 'Property {name} already exists'
    CODE = 46



# === Forbidden Errors ===

class ForbiddenError(XMindsError):
    HTTP_STATUS = 403


class ResourcesForbidden(ForbiddenError):
    MSG = 'Do not have enough permissions to access this resource'
    CODE = 50


class FrontendUserResourcesForbidden(ForbiddenError):
    MSG = 'The user registered during the login cannot access the user resources in the request'
    CODE = 51


# === Resource Errors ===


class BaseNotFoundError(XMindsError):
    HTTP_STATUS = 404


class ItemNotFound(BaseNotFoundError):
    MSG = 'The item {item_id} does not exist'
    CODE = 60


class UserNotFound(BaseNotFoundError):
    MSG = 'The user {user_id} does not exist'
    CODE = 61


class RatingNotFound(BaseNotFoundError):
    MSG = 'The rating for user {user_id} and item {item_id} does not exist'
    CODE = 62


class OrganizationNotFound(BaseNotFoundError):
    MSG = 'The organization {org_id} does not exist'
    CODE = 63


class AccountNotFound(BaseNotFoundError):
    MSG = 'The account {email} does not exist'
    CODE = 64


class DatabaseNotFound(BaseNotFoundError):
    MSG = 'The database {db_id} does not exist'
    CODE = 65


class MethodNotAllowed(XMindsError):
    HTTP_STATUS = 405
    MSG = 'Method "{method}" not allowed'
    CODE = 70

class DatabaseNotReady(XMindsError):
    HTTP_STATUS = 503
    MSG = 'The database {db_id} is not ready. Wait using /status/'
    CODE = 80


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
