"""
xminds.api.apirequest
~~~~~~~~~~~~~~~~~~~~~

This module implements the low level request logic of Crossing Minds API:
* headers and JTW token autentication
* serialization
* translation from HTTP status code to XMindsError
"""

import json
import pickle
import requests

from ..compat import PYV
from .exceptions import ServerError, XMindsError


CLIENT_VERSION = 'v1.0'


class _BaseCrossingMindsApiRequest:
    HOST = 'https://api.crossingminds.com'
    HEADERS = {}
    API_VERSION = 'v1'
    DEFAULT_TIMEOUT = 6
    _REQUEST_KWARGS = {}

    def __init__(self, host=None, api_version=None, headers=None):
        self.host = host or self.HOST
        headers = dict(self.HEADERS, **headers or {})
        self.api_version = api_version or self.API_VERSION
        self.session = requests.Session()
        self.session.headers.update(headers)
        self._jwt_token = None

    def get(self, path, params=None, **kwargs):
        return self._request('GET', path, params=params, **kwargs)

    def put(self, path, data, **kwargs):
        return self._request('PUT', path, data=data, **kwargs)

    def post(self, path, data, **kwargs):
        return self._request('POST', path, data=data, **kwargs)

    def patch(self, path, data, **kwargs):
        return self._request('PATCH', path, data=data, **kwargs)

    def delete(self, path, data=None, **kwargs):
        return self._request('DELETE', path, data=data, **kwargs)

    @property
    def jwt_token(self):
        return self._jwt_token

    def set_jwt_token(self, jwt_token):
        self._jwt_token = jwt_token
        self.session.headers.update({'Authorization': f'Bearer {jwt_token}'})

    def clear_jwt_token(self):
        self.session.headers.pop('Authorization', None)

    def _request(self, method, path, params=None, data=None, timeout=None):
        url = f'{self.host}/{self.api_version}/{path}'
        if path and not path.endswith('/'):
            url += '/'

        request_kwargs = {
            'timeout': timeout or self.DEFAULT_TIMEOUT,
            **self._REQUEST_KWARGS
        }
        if data:
            request_kwargs['data'] = self._serialize_data(data)
        if params:
            request_kwargs['params'] = params
        resp = self.session.request(method, url, **request_kwargs)

        if resp.status_code >= 500:
            raise ServerError()
        elif resp.status_code >= 400:
            data = self._parse_response(resp, fallback=True)
            try:
                exc = XMindsError.from_code(data.get('error_code', 0), data.get('error_data'))
            except (KeyError, AttributeError, TypeError, ValueError):
                exc = ServerError({'response': data})
            raise exc

        data = self._parse_response(resp)

        return data

    def _serialize_data(self, data):
        raise NotImplementedError()

    def _parse_response(self, response, fallback=False):
        raise NotImplementedError()

    @staticmethod
    def _parse_token(headers):
        if 'Authorization' not in headers:
            return None

        authorization_header = headers['Authorization']
        token_start_index = authorization_header.index('Bearer') + len('Bearer ')
        jwt = authorization_header[token_start_index:]
        return jwt


class CrossingMindsApiJsonRequest(_BaseCrossingMindsApiRequest):
    HEADERS = {
        'User-Agent': f'CrossingMinds/{CLIENT_VERSION} (Python/{PYV}; JSON)',
        'Content-type': 'application/json',
        'Accept': 'application/json',
    }

    def _parse_response(self, response, fallback=False):
        if not response.text:
            return None
        if not fallback:
            return response.json()
        try:
            return response.json()
        except ValueError:
            return response.text

    def _serialize_data(self, data):
        return json.dumps(data)


class CrossingMindsApiPythonRequest(_BaseCrossingMindsApiRequest):
    HEADERS = {
        'User-Agent': f'CrossingMinds/{CLIENT_VERSION} (Python/{PYV}; pickle)',
        'Content-type': 'application/xminds-pkl',
        'Accept': 'application/xminds-pkl',
    }
    _REQUEST_KWARGS = {
        'stream': True,
    }

    def _serialize_data(self, data):
        return pickle.dumps(data)

    def _parse_response(self, response, fallback=False):
        if response.status_code == 204:
            return None
        # decode gzip in case it is used
        response.raw.decode_content = True
        # un-pickle from iterable
        if not fallback:
            return pickle.load(response.raw)
        try:
            return pickle.load(response.raw)
        except pickle.UnpicklingError:
            return response.raw.read()
