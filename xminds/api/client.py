"""
xminds.api.client
~~~~~~~~~~~~~~~~~

This module implements the requests for all API endpoints.
The client handles the logic to automatically get a new JWT token using the refresh token
"""

import base64
from functools import wraps
import logging
import sys
import time
from urllib.parse import quote
from binascii import Error as BinasciiError

import numpy

from ..compat import tqdm
from .apirequest import CrossingMindsApiJsonRequest, CrossingMindsApiPythonRequest
from .exceptions import DuplicatedError, JwtTokenExpired, ServerError


def require_login(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except JwtTokenExpired:
            if not self._refresh_token or not self.auto_refresh_token:
                raise
            self.login_refresh_token()
            return method(self, *args, **kwargs)
    return wrapped


class CrossingMindsApiClient:

    def __init__(self, serializer='pkl', **api_kwargs):
        self._database = None
        self._refresh_token = None
        self.auto_refresh_token = True
        if serializer.lower() in ['pkl', 'pickle']:
            cls = CrossingMindsApiPythonRequest
            self.b64_encode_bytes = False
        elif serializer.lower() == 'json':
            cls = CrossingMindsApiJsonRequest
            self.b64_encode_bytes = True
        else:
            raise NotImplementedError(f'unknown serializer {serializer}')
        self.api = cls(**api_kwargs)

    # === Account ===

    @require_login
    def create_individual_account(self, first_name, last_name, email, password, role='backend'):
        """
        Create a new individual account

        :param str first_name:
        :param str last_name:
        :param str email:
        :param str password:
        :returns: {'id': str}
        """
        path = f'accounts/individual/'
        data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
            'role': role,
        }
        return self.api.post(path=path, data=data)

    @require_login
    def create_service_account(self, name, password, role='frontend'):
        """
        Create a new service account

        :param str name:
        :param str password:
        :param str? role:
        :returns: {'id': str}
        """
        path = f'accounts/service/'
        data = {
            'name': name,
            'password': password,
            'role': role,
        }
        return self.api.post(path=path, data=data)

    def resend_verification_code(self, email):
        """
        Resend the verification code to the account email

        :param str email:
        """
        path = f'accounts/resend-verification-code/'
        return self.api.put(path=path, data={'email': email})

    def verify_account(self, code, email):
        """
        Verify the email of an account by entering the verification code

        :param str code:
        :param str email:
        """
        path = f'accounts/verify/'
        data = {
            'code': code,
            'email': email,
        }
        return self.api.get(path=path, params=data)

    @require_login
    def list_accounts(self):
        """
        Get all accounts on the current organization

        :returns: {
            'individual_accounts': [
                {
                    'first_name': str,
                    'last_name': str,
                    'email': str,
                    'role': str,
                    'verified': bool,
                },
            ],
            'service_accounts': [
                {
                    'name': str,
                    'role': str,
                },
            ],
        }
        """
        path = f'organizations/accounts/'
        return self.api.get(path=path)

    # === Login ===

    def login_individual(self, email, password, db_id, frontend_user_id=None):
        """
        Login on a database with an account

        :param str email:
        :param str password:
        :param str db_id:
        :param ID? frontend_user_id: user ID
        :returns: {
            'token': str,
            'database': {
                'id': str,
                'name': str,
                'description': str,
                'item_id_type': str,
                'user_id_type': str,
            },
        }
        """
        path = f'login/individual/'
        data = {
            'email': email,
            'password': password,
            'db_id': db_id,
        }
        return self._login(path, data, frontend_user_id)

    def login_service(self, name, password, db_id, frontend_user_id=None):
        """
        Login on a database with a service account

        :param str name:
        :param str password:
        :param str db_id:
        :param ID? frontend_user_id: user ID
        :returns: {
            'token': str,
            'database': {
                'id': str,
                'name': str,
                'description': str,
                'item_id_type': str,
                'user_id_type': str,
            },
        }
        """
        path = f'login/service/'
        data = {
            'name': name,
            'password': password,
            'db_id': db_id,
        }
        return self._login(path, data, frontend_user_id)

    def login_root(self, email, password):
        """
        Login with the root account without selecting a database

        :param str email:
        :param str password:
        :returns: {
            'token': str,
        }
        """
        path = f'login/root/'
        data = {
            'email': email,
            'password': password
        }
        resp = self.api.post(path=path, data=data)
        jwt_token = resp['token']
        self.set_jwt_token(jwt_token)
        self._database = None
        self._refresh_token = None
        return resp

    def login_refresh_token(self, refresh_token=None):
        """
        Login again using a refresh token

        :param str? refresh_token: (default: self._refresh_token)
        :returns: {
            'token': str,
            'database': {
                'id': str,
                'name': str,
                'description': str,
                'item_id_type': str,
                'user_id_type': str,
            },
        }
        """
        refresh_token = refresh_token or self._refresh_token
        path = f'login/refresh-token/'
        data = {
            'refresh_token': refresh_token
        }
        return self._login(path, data, None)

    def _login(self, path, data, frontend_user_id):
        if frontend_user_id:
            data['frontend_user_id'] = self._userid2body(frontend_user_id)
        resp = self.api.post(path=path, data=data)
        jwt_token = resp['token']
        self.set_jwt_token(jwt_token)
        self._database = resp['database']
        self._refresh_token = resp['refresh_token']
        return resp

    # === Org metadata ===

    @require_login
    def get_organization(self):
        """
        get organization meta-data
        """
        path = f'organizations/current/'
        return self.api.get(path=path)

    @require_login
    def create_or_partial_update_organization(self, metadata, preserve=None):
        """
        create, or apply deep partial update of meta-data
        :param dict metadata: meta-data to store structured as unvalidated JSON-compatible dict
        :param bool? preserve: set to `True` to append values instead of replace as in RFC7396
        """
        path = f'organizations/current/'
        data = {'metadata': metadata}
        if preserve is not None:
            data['preserve'] = preserve
        return self.api.patch(path=path, data=data)

    @require_login
    def partial_update_organization_preview(self, metadata, preserve=None):
        """
        preview deep partial update of extra data, without changing any state
        :param dict metadata: extra meta-data to store structured as unvalidated JSON-compatible dict
        :param bool? preserve: set to `True` to append values instead of replace as in RFC7396
        :returns: {
            'metadata_old': {
                'description': str,
                'extra': {**any-key: any-val},
            },
            'metadata_new': {
                'description': str,
                'extra': {**any-key: any-val},
            },
        }
        """
        path = f'organizations/current/preview/'
        data = {'metadata': metadata}
        if preserve is not None:
            data['preserve'] = preserve
        return self.api.patch(path=path, data=data)

    # === Database ===

    @require_login
    def create_database(self, name, description, item_id_type, user_id_type):
        """
        Create a new database

        :param str name: Database name, must be unique
        :param str description:
        :param str item_id_type: Item ID type
        :param str user_id_type: User ID type
        """
        path = f'databases/'
        data = {
            'name': name,
            'description': description,
            'item_id_type': item_id_type,
            'user_id_type': user_id_type,
        }
        return self.api.post(path=path, data=data)

    @require_login
    def get_all_databases(self, amt=None, page=None):
        """
        Get all databases on the current organization

        :param int? amt: amount of databases by page (default: API default)
        :param int? page: page number (default: 1)
        :returns: {
            'has_next': bool,
            'next_page': int,
            'databases': [
                {
                    'id': int,
                    'name': str,
                    'description': str,
                    'item_id_type': str,
                    'user_id_type': str
                },
            ]
        }
        """
        path = f'databases/'
        params = {}
        if amt:
            params['amt'] = amt
        if page:
            params['page'] = page
        return self.api.get(path=path, params=params)

    @require_login
    def get_database(self):
        """
        Get details on current database

        :returns: {
            'id': str,
            'name': str,
            'description': str,
            'item_id_type': str,
            'user_id_type': str,
            'counters': {
                'rating': int,
                'user': int,
                'item': int,
            },
            'metadata': {**any-key: any-val},
        }
        """
        path = f'databases/current/'
        return self.api.get(path=path)

    @require_login
    def partial_update_database(self, description=None, metadata=None, preserve=None):
        """
        update description, and apply deep partial update of extra meta-data
        :param str? description: description of DB
        :param dict? metadata: extra data to store structured as unvalidated JSON-compatible dict
        :param bool? preserve: set to `True` to append values instead of replace as in RFC7396
        """
        path = f'databases/current/'
        data = {}
        assert description is not None or metadata is not None
        if description is not None:
            data['description'] = description
        if metadata is not None:
            data['metadata'] = metadata
        if preserve is not None:
            data['preserve'] = preserve
        return self.api.patch(path=path, data=data)

    @require_login
    def partial_update_database_preview(self, description=None, metadata=None, preserve=None):
        """
        preview deep partial update of extra data, without changing any state
        :param str? description: description of DB
        :param dict? metadata: extra data to store structured as unvalidated JSON-compatible dict
        :param bool? preserve: set to `True` to append values instead of replace as in RFC7396
        :returns: {
            'metadata_old': {
                'description': str,
                'metadata': {**any-key: any-val},
            },
            'metadata_new': {
                'description': str,
                'metadata': {**any-key: any-val},
            },
        }
        """
        path = f'databases/current/preview/'
        data = {}
        assert description is not None or metadata is not None
        if description is not None:
            data['description'] = description
        if metadata is not None:
            data['metadata'] = metadata
        if preserve is not None:
            data['preserve'] = preserve
        return self.api.patch(path=path, data=data)

    @require_login
    def delete_database(self):
        """
        Delete current database.
        """
        path = f'databases/current/'
        return self.api.delete(path=path, timeout=29)

    @require_login
    def status(self):
        """
        Get readiness status of current database.
        """
        path = f'databases/current/status/'
        return self.api.get(path=path)

    # === User Property ===

    @require_login
    def get_user_property(self, property_name):
        """
        Get one user-property.

        :param str property_name: property name
        :returns: {
            'property_name': str,
            'value_type': str,
            'repeated': bool,
        }
        """
        path = f'users-properties/{self.escape_url(property_name)}/'
        return self.api.get(path=path)

    @require_login
    def list_user_properties(self):
        """
        Get all user-properties for the current database.

        :returns: {
            'properties': [{
                'property_name': str,
                'value_type': str,
                'repeated': bool,
            }],
        }
        """
        path = f'users-properties/'
        return self.api.get(path=path)

    @require_login
    def create_user_property(self, property_name, value_type, repeated=False):
        """
        Create a new user-property.

        :param str property_name: property name
        :param str value_type: property type
        :param bool? repeated: whether the property has many values (default: False)
        """
        path = f'users-properties/'
        data = {
            'property_name': property_name,
            'value_type': value_type,
            'repeated': repeated,
        }
        return self.api.post(path=path, data=data)

    @require_login
    def delete_user_property(self, property_name):
        """
        Delete an user-property given by its name

        :param str property_name: property name
        """
        path = f'users-properties/{self.escape_url(property_name)}/'
        return self.api.delete(path=path)

    # === User ===

    @require_login
    def get_user(self, user_id):
        """
        Get one user given its ID.

        :param ID user_id: user ID
        :returns: {
            'item': {
                'id': ID,
                *<property_name: property_value>,
            }
        }
        """
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/'
        resp = self.api.get(path=path)
        resp['user']['user_id'] = self._body2userid(resp['user']['user_id'])
        return resp

    @require_login
    def list_users(self, users_id):
        """
        Get multiple users given their IDs.
        The users in the response are not aligned with the input.
        In particular this endpoint does not raise NotFoundError if any user in missing.
        Instead, the missing users are simply not present in the response.

        :param ID-array users_id: users IDs
        :returns: {
            'users': array with fields ['id': ID, *<property_name: value_type>]
                contains only the non-repeated values,
            'users_m2m': dict of arrays for repeated values:
                {
                    *<repeated_property_name: {
                        'name': str,
                        'array': array with fields ['user_index': uint32, 'value_id': value_type],
                    }>
                }
        }
        """
        users_id = self._userid2body(users_id)
        path = f'users-bulk/list/'
        data = {'users_id': users_id}
        resp = self.api.post(path=path, data=data)
        resp['users'] = self._body2itemid(resp['users'])
        return resp

    @require_login
    def list_users_paginated(self, amt=None, cursor=None):
        """
        Get multiple users by page.
        The response is paginated, you can control the response amount and offset
        using the query parameters ``amt`` and ``cursor``.

        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :returns: {
            'users': array with fields ['id': ID, *<property_name: value_type>]
                contains only the non-repeated values,
            'users_m2m': dict of arrays for repeated values:
                {
                    *<repeated_property_name: {
                        'name': str,
                        'array': array with fields ['user_index': uint32, 'value_id': value_type],
                    }>
                },
            'has_next': bool,
            'next_cursor': str, pagination cursor to use in next request to get more users,
        }
        """
        path = f'users-bulk/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        resp = self.api.get(path=path, params=params)
        resp['users'] = self._body2userid(resp['users'])
        return resp

    @require_login
    def create_or_update_user(self, user):
        """
        Create a new user, or update it if the ID already exists.

        :param dict user: user ID and properties {'user_id': ID, *<property_name: property_value>}
        """
        user = dict(user)
        user_id = self._userid2url(user.pop('user_id'))
        path = f'users/{user_id}/'
        data = {
            'user': user,
        }
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_users_bulk(self, users, users_m2m=None, wait_for_completion=None,
                                    chunk_size=(1<<10)):
        """
        Create many users in bulk, or update the ones for which the id already exist.

        :param array users: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param dict? users_m2m: dict of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['user_index': uint32, 'value_id': value_type],
                }>
            }
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        :param bool? wait_for_completion: (default: True)
        """
        path = f'users-bulk/'
        for users_chunk, users_m2m_chunk in self._chunk_users(users, users_m2m, chunk_size):
            data = {
                'users': users_chunk,
                'users_m2m': users_m2m_chunk
            }
            if wait_for_completion is not None:
                data['wait_for_completion'] = wait_for_completion
            self.api.put(path=path, data=data, timeout=60)

    @require_login
    def partial_update_user(self, user, create_if_missing=None):
        """
        Partially update some properties of an user

        :param dict user: user ID and properties {'user_id': ID, *<property_name: property_value>}
        :param bool? create_if_missing: control whether an error should be returned or a new user
        should be created when the ``user_id`` does not already exist. (default: False)
        """
        user = dict(user)
        user_id = self._userid2url(user.pop('user_id'))
        path = f'users/{user_id}/'
        data = {
            'user': user,
        }
        if create_if_missing is not None:
            data['create_if_missing'] = create_if_missing
        return self.api.patch(path=path, data=data)

    @require_login
    def partial_update_users_bulk(self, users, users_m2m=None, create_if_missing=None,
                                  chunk_size=(1 << 10)):
        """
        Partially update some properties of many users.

        :param array users: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param dict? users_m2m: dict of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['user_index': uint32, 'value_id': value_type],
                }>
            }
        :param bool? create_if_missing: to control whether an error should be returned or new users
        should be created when the ``user_id`` does not already exist. (default: False)
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        """
        path = f'users-bulk/'
        data = {}
        if create_if_missing is not None:
            data['create_if_missing'] = create_if_missing
        for users_chunk, users_m2m_chunk in self._chunk_users(users, users_m2m, chunk_size):
            data['users'] = users_chunk
            data['users_m2m'] = users_m2m_chunk
            self.api.patch(path=path, data=data, timeout=60)

    @require_login
    def delete_user(self, user_id):
        """
        Delete a single user;  doesn't wait for task completion

        :param bytes user_id:
        """
        user_id_url = self._userid2url(user_id)
        self.api.delete(f'users/{user_id_url}')

    @require_login
    def delete_users(self, users_id):
        """
        Delete users; doesn't wait for task completion

        :param ID-array users_id: user IDs
        """
        data = {'users_id': self._userid2body(users_id)}
        self.api.delete(path='users-bulk/', data=data)

    def _chunk_users(self, users, users_m2m, chunk_size):
        users_m2m = users_m2m or []
        # cast dict to list of dict
        if isinstance(users_m2m, dict):
            users_m2m = [{'name': name, 'array': array}
                         for name, array in users_m2m.items()]
        n_chunks = int(numpy.ceil(len(users) / chunk_size))
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            users_chunk = users[start_idx:end_idx]
            # split M2M array-optimized if any
            users_m2m_chunk = []
            for m2m in users_m2m:
                array = m2m['array']
                if isinstance(array, numpy.ndarray):
                    mask = (array['user_index'] >= start_idx) & (array['user_index'] < end_idx)
                    array_chunk = array[mask]  # does copy
                    array_chunk['user_index'] -= start_idx
                else:
                    logging.warning('array-optimized many-to-many format is not efficient '
                                    'with JSON. Use numpy arrays and pkl serializer instead')
                    array_chunk = [
                        {'user_index': row['user_index'] - start_idx, 'value_id': row['value_id']}
                        for row in array
                        if start_idx <= row['user_index'] < end_idx
                    ]
                users_m2m_chunk.append({'name': m2m['name'], 'array': array_chunk})
            yield self._userid2body(users_chunk), users_m2m_chunk

    # === Item Property ===

    @require_login
    def get_item_property(self, property_name):
        """
        Get one item-property.

        :param str property_name: property name
        :returns: {
            'property_name': str,
            'value_type': str,
            'repeated': bool,
        }
        """
        path = f'items-properties/{self.escape_url(property_name)}/'
        return self.api.get(path=path)

    @require_login
    def list_item_properties(self):
        """
        Get all item-properties for the current database.

        :returns: {
            'properties': [{
                'property_name': str,
                'value_type': str,
                'repeated': bool,
            }],
        }
        """
        path = f'items-properties/'
        return self.api.get(path=path)

    @require_login
    def create_item_property(self, property_name, value_type, repeated=False):
        """
        Create a new item-property.

        :param str property_name: property name
        :param str value_type: property type
        :param bool? repeated: whether the property has many values (default: False)
        """
        path = f'items-properties/'
        data = {
            'property_name': property_name,
            'value_type': value_type,
            'repeated': repeated,
        }
        return self.api.post(path=path, data=data)

    @require_login
    def delete_item_property(self, property_name):
        """
        Delete an item-property given by its name

        :param str property_name: property name
        """
        path = f'items-properties/{self.escape_url(property_name)}/'
        return self.api.delete(path=path)

    # === Item ===

    @require_login
    def get_item(self, item_id):
        """
        Get one item given its ID.

        :param ID item_id: item ID
        :returns: {
            'item': {
                'id': ID,
                *<property_name: property_value>,
            }
        }
        """
        item_id = self._itemid2url(item_id)
        path = f'items/{item_id}/'
        resp = self.api.get(path=path)
        resp['item']['item_id'] = self._body2itemid(resp['item']['item_id'])
        return resp

    @require_login
    def list_items(self, items_id):
        """
        Get multiple items given their IDs.
        The items in the response are not aligned with the input.
        In particular this endpoint does not raise NotFoundError if any item in missing.
        Instead, the missing items are simply not present in the response.

        :param ID-array items_id: items IDs
        :returns: {
            'items': array with fields ['id': ID, *<property_name: value_type>]
                contains only the non-repeated values,
            'items_m2m': dict of arrays for repeated values:
                {
                    *<repeated_property_name: {
                        'name': str,
                        'array': array with fields ['item_index': uint32, 'value_id': value_type],
                    }>
                }
        }
        """
        items_id = self._itemid2body(items_id)
        path = f'items-bulk/list/'
        data = {'items_id': items_id}
        resp = self.api.post(path=path, data=data)
        resp['items'] = self._body2itemid(resp['items'])
        return resp

    @require_login
    def list_items_paginated(self, amt=None, cursor=None):
        """
        Get multiple items by page.
        The response is paginated, you can control the response amount and offset
        using the query parameters ``amt`` and ``cursor``.

        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :returns: {
            'items': array with fields ['id': ID, *<property_name: value_type>]
                contains only the non-repeated values,
            'items_m2m': dict of arrays for repeated values:
                {
                    *<repeated_property_name: {
                        'name': str,
                        'array': array with fields ['item_index': uint32, 'value_id': value_type],
                    }>
                },
            'has_next': bool,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        path = f'items-bulk/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        resp = self.api.get(path=path, params=params)
        resp['items'] = self._body2itemid(resp['items'])
        return resp

    @require_login
    def create_or_update_item(self, item):
        """
        Create a new item, or update it if the ID already exists.

        :param dict item: item ID and properties {'item_id': ID, *<property_name: property_value>}
        """
        item = dict(item)
        item_id = self._itemid2url(item.pop('item_id'))
        path = f'items/{item_id}/'
        data = {
            'item': item,
        }
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_items_bulk(self, items, items_m2m=None, wait_for_completion=None,
                                    chunk_size=(1<<10)):
        """
        Create many items in bulk, or update the ones for which the id already exist.

        :param array items: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param array? items_m2m: dict of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['item_index': uint32, 'value_id': value_type],
                }>
            }
        :param bool? wait_for_completion: (default: true)
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        """
        path = f'items-bulk/'
        for items_chunk, items_m2m_chunk in self._chunk_items(items, items_m2m, chunk_size):
            data = {
                'items': items_chunk,
                'items_m2m': items_m2m_chunk,
            }
            if wait_for_completion is not None:
                data['wait_for_completion'] = wait_for_completion
            self.api.put(path=path, data=data, timeout=60)

    @require_login
    def partial_update_item(self, item, create_if_missing=None):
        """
        Partially update some properties of an item.

        :param dict item: item ID and properties {'item_id': ID, *<property_name: property_value>}
        :param bool? create_if_missing: control whether an error should be returned or a new item
        should be created when the ``item_id`` does not already exist. (default: false)
        """
        item = dict(item)
        item_id = self._itemid2url(item.pop('item_id'))
        path = f'items/{item_id}/'
        data = {
            'item': item,
        }
        if create_if_missing is not None:
            data['create_if_missing'] = create_if_missing
        return self.api.patch(path=path, data=data)

    @require_login
    def partial_update_items_bulk(self, items, items_m2m=None, create_if_missing=None,
                                  chunk_size=(1 << 10)):
        """
        Partially update some properties of many items.

        :param array items: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param array? items_m2m: dict of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['item_index': uint32, 'value_id': value_type],
                }>
            }
        :param bool? create_if_missing: control whether an error should be returned or a new item
        should be created when the ``item_id`` does not already exist. (default: false)
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        """
        path = f'items-bulk/'
        data = {}
        if create_if_missing is not None:
            data['create_if_missing'] = create_if_missing
        for items_chunk, items_m2m_chunk in self._chunk_items(items, items_m2m, chunk_size):
            data['items'] = items_chunk
            data['items_m2m'] = items_m2m_chunk
            self.api.patch(path=path, data=data, timeout=60)

    @require_login
    def delete_item(self, item_id):
        """
        Delete a single item;  doesn't wait for task completion

        :param bytes item_id:
        """
        item_id_url = self._itemid2url(item_id)
        self.api.delete(path=f'items/{item_id_url}/')

    @require_login
    def delete_items(self, items_id):
        """
        Delete items; doesn't wait for task completion

        :param ID-array items_id: items IDs
        """
        data = {'items_id': self._itemid2body(items_id)}
        self.api.delete(path='items-bulk/', data=data)

    def _chunk_items(self, items, items_m2m, chunk_size):
        items_m2m = items_m2m or []
        # cast dict to list of dict
        if isinstance(items_m2m, dict):
            items_m2m = [{'name': name, 'array': array}
                         for name, array in items_m2m.items()]
        n_chunks = int(numpy.ceil(len(items) / chunk_size))
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            items_chunk = items[start_idx:end_idx]
            # split M2M array-optimized if any
            items_m2m_chunk = []
            for m2m in items_m2m:
                array = m2m['array']
                if isinstance(array, numpy.ndarray):
                    mask = (array['item_index'] >= start_idx) & (array['item_index'] < end_idx)
                    array_chunk = array[mask]  # does copy
                    array_chunk['item_index'] -= start_idx
                else:
                    logging.warning('array-optimized many-to-many format is not efficient '
                                    'with JSON. Use numpy arrays and pkl serializer instead')
                    array_chunk = [
                        {'item_index': row['item_index'] - start_idx, 'value_id': row['value_id']}
                        for row in array
                        if start_idx <= row['item_index'] < end_idx
                    ]
                items_m2m_chunk.append({'name': m2m['name'], 'array': array_chunk})
            yield self._itemid2body(items_chunk), items_m2m_chunk

    # === Reco: Item-to-item ===

    @require_login
    def get_reco_item_to_items(self, item_id, amt=None, cursor=None,
                               scenario=None, filters=None, reranking=None,
                               skip_default_scenario=None):
        """
        Get similar items.

        :param ID item_id: item ID
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param str? scenario: name of scenario
        :param list-str? filters: Item-property filters. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :param list-str? reranking: Item-property reranking. Format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_WEIGHT>:<OPTIONS>']
        :param bool? skip_default_scenario: True to skip default scenario if any
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        item_id = self._itemid2url(item_id)
        path = f'recommendation/items/{item_id}/items/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        if filters:
            params['filters'] = filters
        if reranking:
            params['reranking'] = reranking
        if scenario:
            params['scenario'] = scenario
        if skip_default_scenario is not None:
            params['skip_default_scenario'] = skip_default_scenario
        resp = self.api.get(path=path, params=params)
        resp['items_id'] = self._body2itemid(resp['items_id'])
        return resp

    # === Reco: Session-to-item ===

    @require_login
    def get_reco_session_to_items(self, ratings=None, interactions=None, user_properties=None,
                                  amt=None, cursor=None, scenario=None, filters=None,
                                  reranking=None, exclude_rated_items=None,
                                  skip_default_scenario=None):
        """
        Get items recommendations given the ratings or interactions of an anonymous session.
        Ratings and interactions are mutually exclusive.

        :param array? ratings: ratings array with fields ['item_id': ID, 'rating': float]
        :param array? interactions: interactions to calculate ratings from.
            Timestamp is optional and defaults to now.
            ['item_id': ID, 'interaction_type': 'O' 'timestamp?': float]
        :param dict? user_properties: user properties {**property_name: property_value(s)}
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param str? scenario: scenario name
        :param list-str? filters: Item-property filters. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :param list-str? reranking: Item-property reranking. Format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_WEIGHT>:<OPTIONS>']
        :param bool? exclude_rated_items: exclude rated items from response
        :param bool? skip_default_scenario: True to skip default scenario if any
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        path = f'recommendation/sessions/items/'
        data = {}
        if ratings is not None:
            data['ratings'] = self._itemid2body(ratings)
        if user_properties:
            data['user_properties'] = user_properties
        if amt:
            data['amt'] = amt
        if cursor:
            data['cursor'] = cursor
        if filters:
            data['filters'] = filters
        if reranking:
            data['reranking'] = reranking
        if exclude_rated_items is not None:
            data['exclude_rated_items'] = exclude_rated_items
        if scenario:
            data['scenario'] = scenario
        if skip_default_scenario is not None:
            data['skip_default_scenario'] = skip_default_scenario
        if interactions is not None:
            data['interactions'] = interactions
        resp = self.api.post(path=path, data=data)
        resp['items_id'] = self._body2itemid(resp['items_id'])
        return resp

    # === Reco: User-to-item ===

    @require_login
    def get_reco_user_to_items(self, user_id, amt=None, cursor=None, scenario=None,
                               filters=None, reranking=None,
                               exclude_rated_items=None, skip_default_scenario=None):
        """
        Get items recommendations given a user ID.

        :param ID user_id: user ID
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param str? scenario: scenario's name
        :param list-str? filters: Item-property filters. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :param list-str? reranking: Item-property reranking. Format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_WEIGHT>:<OPTIONS>']
        :param bool? exclude_rated_items: exclude rated items from response
        :param bool? skip_default_scenario: True to skip default scenario if any
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        user_id = self._userid2url(user_id)
        path = f'recommendation/users/{user_id}/items/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        if filters:
            params['filters'] = filters
        if reranking:
            params['reranking'] = reranking
        if exclude_rated_items is not None:
            params['exclude_rated_items'] = exclude_rated_items
        if scenario:
            params['scenario'] = scenario
        if skip_default_scenario is not None:
            params['skip_default_scenario'] = skip_default_scenario
        resp = self.api.get(path=path, params=params)
        resp['items_id'] = self._body2itemid(resp['items_id'])
        return resp

    # === Reco: User-to-item-property ===

    @require_login
    def get_reco_user_to_item_properties(self, user_id, property_name: str, amt=None):
        """
        Recommends item-property values given a user ID
        :param bytes user_id:
        :param str property_name:
        :param int? amt: (default 16)  maximal number of property values to return for each property
        :raises: NotFoundError when data not found
        :raises: RequestError if property missing
        :return: {'properties': [n,] np.array, n<=amt}
        """
        user_id = self._userid2url(user_id)
        path = f'recommendation/users/{user_id}/items-properties/{property_name}/'
        params = {}
        if amt:
            params['amt'] = amt
        resp = self.api.get(path=path, params=params)
        return resp

    # === Reco: Session-to-item with Context Items ===

    @require_login
    def get_reco_session_to_items_w_ctx_items(
            self, context_items, ratings=None, user_properties=None, amt=None, cursor=None,
            scenario=None, filters=None, reranking=None, exclude_rated_items=None,
            skip_default_scenario=None):
        """
        Get items recommendations given the ratings of an anonymous session and context items ID.

        :param array context_items: context items ID array with fields ['item_id': ID]
        :param array? ratings: ratings array with fields ['item_id': ID, 'rating': float]
        :param dict? user_properties: user properties {**property_name: property_value(s)}
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param str? scenario: scenario name
        :param list-str? filters: Item-property filters. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :param list-str? reranking: Item-property reranking. Format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_WEIGHT>:<OPTIONS>']
        :param bool? exclude_rated_items: exclude rated items from response
        :param bool? skip_default_scenario: True to skip default scenario if any
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        path = f'recommendation/context-items/sessions/items/'
        data = {
            'context_items': context_items,
        }
        if ratings is not None:
            data['ratings'] = self._itemid2body(ratings)
        if user_properties:
            data['user_properties'] = user_properties
        if amt:
            data['amt'] = amt
        if cursor:
            data['cursor'] = cursor
        if filters:
            data['filters'] = filters
        if reranking:
            data['reranking'] = reranking
        if exclude_rated_items is not None:
            data['exclude_rated_items'] = exclude_rated_items
        if scenario:
            data['scenario'] = scenario
        if skip_default_scenario is not None:
            data['skip_default_scenario'] = skip_default_scenario
        resp = self.api.post(path=path, data=data)
        resp['items_id'] = self._body2itemid(resp['items_id'])
        return resp

    # === Reco: User-to-item with Context Items ===

    @require_login
    def get_reco_user_to_items_w_ctx_items(
            self, context_items, user_id, amt=None, cursor=None, scenario=None, filters=None,
            reranking=None, exclude_rated_items=None, skip_default_scenario=None):
        """
        Get items recommendations given a user ID and context items ID.

        :param array context_items: context items ID array with fields ['item_id': ID]
        :param ID user_id: user ID
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param str? scenario: scenario's name
        :param list-str? filters: Item-property filters. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :param list-str? reranking: Item-property reranking. Format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_WEIGHT>:<OPTIONS>']
        :param bool? exclude_rated_items: exclude rated items from response
        :param bool? skip_default_scenario: True to skip default scenario if any
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        user_id = self._userid2url(user_id)
        path = f'recommendation/context-items/users/{user_id}/items/'
        data = {
            'context_items': context_items,
        }
        if amt:
            data['amt'] = amt
        if cursor:
            data['cursor'] = cursor
        if filters:
            data['filters'] = filters
        if reranking:
            data['reranking'] = reranking
        if exclude_rated_items is not None:
            data['exclude_rated_items'] = exclude_rated_items
        if scenario:
            data['scenario'] = scenario
        if skip_default_scenario is not None:
            data['skip_default_scenario'] = skip_default_scenario
        resp = self.api.post(path=path, data=data)
        resp['items_id'] = self._body2itemid(resp['items_id'])
        return resp

    # === User Ratings ===

    @require_login
    def create_or_update_rating(self, user_id, item_id, rating, timestamp=None):
        """
        Create or update a rating for a user and an item.
        If the rating exists for the tuple (user_id, item_id) then it is updated,
        otherwise it is created.

        :param ID user_id: user ID
        :param ID item_id: item ID
        :param float rating: rating value
        :param float? timestamp: rating timestamp (default: now)
        """
        user_id = self._userid2url(user_id)
        item_id = self._itemid2url(item_id)
        path = f'users/{user_id}/ratings/{item_id}/'
        data = {
            'rating': rating,
        }
        if timestamp is not None:
            data['timestamp'] = timestamp
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_user_ratings_bulk(self, user_id, ratings):
        """
        Create or update bulks of ratings for a single user and many items.

        :param ID user_id: user ID
        :param array ratings: ratings array with fields:
            ['item_id': ID, 'rating': float, 'timestamp': float]
        """
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/ratings/'
        data = {
            'ratings': self._itemid2body(ratings),
        }
        return self.api.put(path=path, data=data, timeout=10)

    @require_login
    def create_or_update_ratings_bulk(self, ratings, chunk_size=(1<<14)):
        """
        Create or update large bulks of ratings for many users and many items.

        :param array ratings: ratings array with fields:
            ['user_id': ID, 'item_id': ID, 'rating': float, 'timestamp': float]
        :param int? chunk_size: split the requests in chunks of this size (default: 16K)
        """
        path = f'ratings-bulk/'
        n_chunks = int(numpy.ceil(len(ratings) / chunk_size))
        sleep = chunk_size / 500
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            ratings_chunk = ratings[i*chunk_size:(i+1)*chunk_size]
            ratings_chunk = self._userid2body(self._itemid2body(ratings_chunk))
            data = {
                'ratings': ratings_chunk,
            }
            self.api.put(path=path, data=data, timeout=60)
            if n_chunks > 1:
                time.sleep(sleep)
        return

    @require_login
    def list_user_ratings(self, user_id, amt=None, page=None):
        """
        List the ratings of one user (paginated)

        :param ID user_id: user ID
        :param int? amt: amount of ratings by page (default: API default)
        :param int? page: page number (default: 1)
        :returns: {
            'has_next': bool,
            'next_page': int,
            'user_ratings': ratings array with fields
                ['item_id': ID, 'rating': float, 'timestamp': float]
        }
        """
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/ratings/'
        params = {}
        if amt:
            params['amt'] = amt
        if page:
            params['page'] = page
        resp = self.api.get(path=path, params=params)
        resp['user_ratings'] = self._body2itemid(resp['user_ratings'])
        return resp

    @require_login
    def list_ratings(self, amt=None, cursor=None):
        """
        List the ratings of one database

        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :returns: {
            'has_next': bool,
            'next_cursor': str,
            'ratings': array with fields
                ['item_id': ID, 'user_id': ID, 'rating': float, 'timestamp': float]
        }
        """
        path = f'ratings-bulk/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        resp = self.api.get(path=path, params=params)
        resp['ratings'] = self._body2userid(self._body2itemid(resp['ratings']))
        return resp

    @require_login
    def delete_rating(self, user_id, item_id):
        """
        Delete a single rating for a given user.

        :param ID user_id: user ID
        :param ID item_id: item ID
        """
        user_id = self._userid2url(user_id)
        item_id = self._itemid2url(item_id)
        path = f'users/{user_id}/ratings/{item_id}'
        return self.api.delete(path=path)

    @require_login
    def delete_user_ratings(self, user_id):
        """
        Delete all ratings of a given user.

        :param ID user_id: user ID
        """
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/ratings/'
        return self.api.delete(path=path)

    # === User Interactions ===

    @require_login
    def create_interaction(self, user_id, item_id, interaction_type, timestamp=None):
        """
        This endpoint allows you to create a new interaction for a user and an item.
        An inferred rating will be created or updated for the tuple (user_id, item_id).
        The taste profile of the user will then be updated in real-time by the online machine learning algorithm.

        :param ID user_id: user ID
        :param ID item_id: item ID
        :param str interaction_type: Interaction type
        :param float? timestamp: rating timestamp (default: now)
        """
        user_id = self._userid2url(user_id)
        item_id = self._itemid2url(item_id)
        path = f'users/{user_id}/interactions/{item_id}/'
        data = {
            'interaction_type': interaction_type,
        }
        if timestamp is not None:
            data['timestamp'] = timestamp
        return self.api.post(path=path, data=data)

    @require_login
    def create_interactions_bulk(self, interactions, chunk_size=(1<<14)):
        """
        Create or update large bulks of interactions for many users and many items.
        Inferred ratings will be created or updated for all tuples (user_id, item_id).

        :param array interactions: interactions array with fields:
            ['user_id': ID, 'item_id': ID, 'interaction_type': str, 'timestamp': float]
        :param int? chunk_size: split the requests in chunks of this size (default: 16K)
        """
        path = f'interactions-bulk/'
        n_chunks = int(numpy.ceil(len(interactions) / chunk_size))
        sleep = chunk_size / 500
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            interactions_chunk = interactions[i*chunk_size:(i+1)*chunk_size]
            interactions_chunk = self._userid2body(self._itemid2body(interactions_chunk))
            data = {
                'interactions': interactions_chunk,
            }
            self.api.post(path=path, data=data, timeout=10)
            if n_chunks > 1:
                time.sleep(sleep)

        return

    @require_login
    def list_interactions(self, amt=None, cursor=None):
        """
        List the interactions of one database

        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :returns: {
            'has_next': bool,
            'next_cursor': str,
            'interactions': array with fields
                ['item_id': ID, 'user_id': ID, 'intraction_type': str, 'timestamp': float]
        }
        """
        path = f'interactions-bulk/'
        params = {}
        if amt:
            params['amt'] = amt
        if cursor:
            params['cursor'] = cursor
        resp = self.api.get(path=path, params=params)
        resp['interactions'] = self._body2userid(self._body2itemid(resp['interactions']))
        return resp

    # === Data Dump Storage ===

    @require_login
    def get_data_dump_signed_urls(self, name, content_type, resource):
        """
        Get signed url to upload a file. (url_upload and url_report)

        :param str? name: filename
        :param str? content_type:
        :param str? resource: values allowed are `items`, `users`, `ratings` and `ratings_implicit`.
        :returns: {
            'url_upload': str,
            'url_report': str,
        }
        """
        path = f'data-dump-storage/signed-url/'
        params = {'name': name,
                  'content_type': content_type,
                  'resource': resource}
        return self.api.get(path=path, params=params)

    # === Scheduled Background Tasks ===

    @require_login
    def trigger_background_task(self, task_name, payload=None):
        """
        Trigger background task such as retraining of ML models.
        You should not have to call this endpoint yourself, as this is done automatically.

        :param str task_name: for instance ``'ml_model_retrain'``
        :param dict? payload: optional task payload
        :returns: {
            'task_id': str,
        }
        :raises: DuplicatedError with error name 'TASK_ALREADY_RUNNING'
            if this task is already running
        """
        path = f'tasks/{self.escape_url(task_name)}/'
        data = {}
        if payload:
            data['payload'] = payload
        return self.api.post(path=path, data=data)

    @require_login
    def get_background_tasks(self, task_name, page=None, amt=None):
        """
        List currently running background tasks such as ML models training.

        :param str task_name: names allowed ``ml_model_retrain``,
        ``item_popularity_score_recalibrate``, ``ml_new_users``, ``ml_new_items``
        :param int? page: page number (default: 1)
        :param int? amt: amount of tasks by page (default: 50)
        :returns: {
            'tasks': [{
                'task_id': str,
                'name': str,
                'start_time': int,
                'status': str,
                'progress?': str,
            }],
        }
        """
        params = {}
        if page is not None:
            params['page'] = page
        if amt is not None:
            params['amt'] = amt
        path = f'tasks/{self.escape_url(task_name)}/recents/'
        return self.api.get(path=path, params=params)

    def wait_until_ready(self, timeout=600, sleep=1):
        """
        Wait until the current database status is ready, meaning at least one model has been trained

        :param int? timeout: maximum time to wait, raise RuntimeError if exceeded (default: 10min)
        :param int? sleep: time to wait between polling (default: 1s)
        """
        assert sleep > 0.1
        resp = None
        time_start = time.time()
        while time.time() - time_start < timeout:
            time.sleep(sleep)
            resp = self.status()
            if resp['status'] == 'ready':
                return
        raise RuntimeError(f'API not ready before {timeout}s. Last response: {resp}')

    @require_login
    def trigger_and_wait_background_task(self, task_name, timeout=600, lock_wait_timeout=None,
                                         sleep=1, verbose=None):
        """
        Trigger background task such as retraining of ML models.
        You don't necessarily have to call this endpoint yourself,
        model training is also triggered automatically.
        By default this waits for an already running task before triggering the new one

        :param str task_name: for instance ``'ml_model_retrain'``
        :param int? timeout: maximum time to wait after the new task is triggered (default: 10min)
        :param int? lock_wait_timeout: if another task is already running, maximum time to wait
            for it to finish before triggering the new task (default: ``timeout``)
        :param int? sleep: time to wait between polling (default: 1s)
        :returns: {
            'task_id': str,
        }
        :raises: RuntimeError if either ``timeout`` or ``lock_wait_timeout`` is reached
        """
        assert sleep > 0.1
        if lock_wait_timeout is None:
            lock_wait_timeout = timeout
        if verbose is None:
            verbose = sys.stdout.isatty()
        # wait for already running task (if any)
        if lock_wait_timeout > 0:
            msg = 'waiting for already running...' if verbose else None
            self.wait_for_background_task(
                task_name, lock_wait_timeout, sleep, msg=msg, wait_if_no_task=False,
                filtr=lambda t: t['status'] == 'RUNNING')
        # trigger
        try:
            task_id = self.trigger_background_task(task_name)['task_id']
        except DuplicatedError as exc:
            if getattr(exc, 'data', {})['name'] != 'TASK_ALREADY_RUNNING':
                raise
            # edge case: something else triggered the same task at the same time
            tasks = self.get_background_tasks(task_name)['tasks']
            task_id = next(t['task_id'] for t in tasks if t['status'] != 'COMPLETED')
        # wait for new task
        msg = 'waiting...' if verbose else None
        self.wait_for_background_task(
            task_name, timeout, sleep, msg=msg, filtr=lambda t: t['task_id'] == task_id)

    def wait_for_background_task(self, task_name, timeout=600, sleep=1, msg=None, filtr=None,
                                 wait_if_no_task=True, func_list_tasks=None):
        """
        Wait for a certain background task. Optionally specified with ``filtr`` function

        :param str task_name: for instance ``'ml_model_retrain'``
        :param int? timeout: maximum time to wait after the new task is triggered (default: 10min)
        :param int? sleep: time to wait between polling (default: 1s)
        :param str? msg: either ``None`` to disable print, or message prefix (default: None)
        :param func? filtr: filter function(task: bool)
        :param bool? wait_if_no_task: wait (instead of return) if there is no task satisfying filter
        :param func? func_list_tasks: function to list background tasks (default: self.get_background_tasks)
        :returns: True is a task satisfying filters successfully ran, False otherwise
        :raises: RuntimeError if ``timeout`` is reached or if the task failed
        """
        spinner = '|/-\\'
        task = None
        time_start = time.time()
        time_waited = 0
        i = 0
        func_list_tasks = func_list_tasks or self.get_background_tasks
        while time_waited < timeout:
            time.sleep(sleep)
            time_waited = time.time() - time_start
            print_time = f'{int(time_waited) // 60:d}m{int(time_waited) % 60:02d}s'
            tasks = func_list_tasks(task_name)['tasks']
            try:
                task = max(filter(filtr, tasks), key=lambda t: t['start_time'])
            except ValueError:
                if wait_if_no_task:
                    continue
                else:
                    return (task is not None)
            progress = task.get('progress', '')
            if task['status'] == 'COMPLETED':
                if msg is not None:
                    print(f'\r{msg} {print_time} done   {progress:80s}')
                return True
            elif task['status'] == 'FAILED':
                raise RuntimeError(f'task {task_name} failed with: {progress}')
            if msg is not None:
                print(f'\r{msg} {print_time} {spinner[i%len(spinner)]} {progress:80s}', end='')
                sys.stdout.flush()
            i += 1
        raise RuntimeError(f'task {task_name} not done before {timeout}s. Last response: {task}')

    # === Trained Models ===

    @require_login
    def list_trained_models(self):
        """
        Get all trained models
        :return: trained models
        {
            'trained_models': [
                {
                    'id': str,
                    'trained_timestamp': str
                    'algorithm': str,
                }
            ]
        }
        """
        path = f'trained-models/'
        return self.api.get(path=path)

    # === Scenarios ===

    @require_login
    def get_scenario(self, reco_type, name):
        """
        Get a scenario
        :param str reco_type
        :param str name:
        :raise: NotFoundError if not found
        :return:
        {
            'name': str,
            'reco_type': str
            'scenario': {
                'scenario_type': 'case',
                'case': {
                    'filters?': [dict],
                    'reranking?': [dict],
                    'exclude_rated_items?': bool,
                    'algorithms?': str,
                    'candidates_preselection?': dict,
                }
            }
        }
        or
        {
            'name': str,
            'reco_type': str
            'scenario': {
                'scenario_type': 'ab_test',
                'ab_test': {
                    'id': str,
                    'scenario_a': str,
                    'scenario_b': str
                }
            }
        }
        or
        {
            'name': str,
            'reco_type': str
            'scenario': {
                'scenario_type': 'condition',
                'condition': {
                    'condition_type': str,
                    'if': dict,
                    'then': str,
                    'else': str
                }
            }
        }
        """
        path = f'scenarios/{reco_type}/{name}/'
        return self.api.get(path=path)

    @require_login
    def list_scenarios(self):
        """
        Get all scenarios
        :return: scenarios
        {
            'scenarios': [
                {
                    'name': str,
                    'reco_type': str
                    'scenario': {
                        'scenario_type': str,
                        'case?ab_test?condition?': dict
                    }
                }
            ]
        }
        """
        path = f'scenarios/'
        return self.api.get(path=path)

    @require_login
    def create_scenario(self, reco_type, name, scenario):
        """
        Create a new scenario.
        A scenario should take the following form, with only one of the keys
        among case|condition|ab_test, corresponding to the scenario type:

        :param str reco_type: accepted values
            "item_to_items", "profile_to_items", "session_to_items"
        :param str name: name of the scenario
        :param dict scenario: a ``?`` in a key means the key is optional
            {
                'scenario_type': str,   # 'case', 'ab_test' or 'condition'
                'case?: {
                    'filters?': [dict],
                    'reranking?': [dict],
                    'exclude_rated_items?': bool,
                    'algorithms?': str,
                    'candidates_preselection?': dict,
                },
                'ab_test?': {
                    'id': str,
                    'scenario_a': str,
                    'scenario_b': str
                },
                'condition': {
                    'condition_type': str,
                    'if': dict,  # FIXME
                    'then': str,   # scenario name
                    'else': str    # scenario name
                }
            }
        :raise: RequestError if some business rule is invalid
        :raise: NotFoundError if an AB-test or a then/else scenario is not found

        Example
        _______
        api.create_scenario(
            reco_type='profile_to_items',
            name='my_case_scenario',
            scenario = {
                'scenario_type': 'case',
                'case': {
                    'filters': [
                            {'property_name': 'tags', 'op': 'EQ', 'value': 'pi'},
                            {'property_name': 'price', 'op': 'GEQ', 'value': 3.14}
                    ],
                    'reranking': [
                            {'property_name"': 'director', 'op': 'diversity', 'weight': 0.8}
                    ],
                    'algorithms': 'algo1|algo2',
                    'candidates_preselection': {
                        'method': 'rated_items_nearest_neighbor',
                        'params': {'nearest_neighbor_algorithms': ['algo3']}
                    },
                    # only for reco types "profile_to_items", "session_to_items":
                    'exclude_rated_items': True,
                }
            }
        )

        Example
        _______
        api.create_scenario(
            reco_type='profile_to_items',
            name='my_ab_test_scenario',
            scenario = {
                'scenario_type': 'ab_test',
                'ab_test': {
                    'id': 'my_ab_test_id',
                    'scenario_a': 'scenar1',
                    'scenario_b': 'scenar2'
                }
            }
        )

        Example
        _______
        api.create_scenario(
            reco_type='profile_to_items',
            name='my_condition_scenario',
            scenario = {
                'scenario_type': 'condition',
                'condition': {
                    'condition_type': 'user_function',
                    'if': {
                        'function_name': 'n_ratings',
                        'op': 'lt',
                        'value': 10
                    },
                    'then': 'scenar1',
                    'else': 'scenar2'
                }
            }
        )
        """
        path = f'scenarios/{reco_type}/{name}/'
        data = scenario
        self.api.put(path=path, data=data)

    @require_login
    def delete_scenario(self, reco_type, name):
        """
        Delete scenario with provided name.
        :param str reco_type:
        :param str name:
        :raise: NotFoundError if not found
        """
        path = f'scenarios/{reco_type}/{name}/'
        self.api.delete(path=path)

    # === Default Scenario ===

    @require_login
    def get_default_scenario(self, reco_type):
        """
        Get default scenario
        :param str reco_type:
        :raise: NotFoundError if no default scenario
        :return:
        {
            'name': str,
        }
        """
        path = f'scenarios-default/{reco_type}/'
        return self.api.get(path=path)

    @require_login
    def set_default_scenario(self, reco_type, name):
        """
        Set scenario as default for a reco type
        """
        path = f'scenarios-default/{reco_type}/'
        data = {'name': name}
        self.api.patch(path=path, data=data)

    @require_login
    def unset_default_scenario(self, reco_type):
        """
        Unset default scenario for a reco type
        """
        path = f'scenarios-default/{reco_type}/'
        self.api.delete(path=path)

    # === Utils ===

    def clear_jwt_token(self):
        return self.api.clear_jwt_token()

    @property
    def jwt_token(self):
        return self.api.jwt_token

    def set_jwt_token(self, jwt_token):
        self.api.set_jwt_token(jwt_token)

    def _userid2url(self, user_id):
        """ base64 encode if needed """
        return self._id2url(user_id, 'user')

    def _itemid2url(self, item_id):
        """ base64 encode if needed """
        return self._id2url(item_id, 'item')

    def _id2url(self, data, field):
        """ base64 encode if needed """
        assert self._database is not None, f'You need to login to a database first'
        if self._database[f'{field}_id_type'].startswith('bytes'):
            return self._b64_encode(data)
        if isinstance(data, bytes):
            return data.decode('ascii')
        return data

    def _userid2body(self, data):
        return self._base_field_id(data, 'user', self._id2body)

    def _itemid2body(self, data):
        return self._base_field_id(data, 'item', self._id2body)

    def _body2userid(self, data):
        return self._base_field_id(data, 'user', self._body2id)

    def _body2itemid(self, data):
        return self._base_field_id(data, 'item', self._body2id)

    def _base_field_id(self, data, field, cast_func):
        if not self.b64_encode_bytes:
            return data
        d_type = self._database[f'{field}_id_type']
        if not d_type.startswith(('bytes', 'uuid', 'hex', 'urlsafe')):
            pass
        elif not isinstance(data, list):
            data = cast_func(data, d_type)
        elif all(isinstance(d, dict) for d in data):
            data = [{**row, f'{field}_id': cast_func(row[f'{field}_id'], d_type)} for row in data]
        else:
            data = [cast_func(row, d_type) for row in data]
        return data

    def _id2body(self, data, d_type):
        if d_type.startswith(('uuid', 'hex', 'urlsafe')):
            return data.decode('ascii')
        else:  # Bytes
            return self._b64_encode(data)

    def _body2id(self, data, d_type):
        if d_type.startswith(('uuid', 'hex', 'urlsafe')):
            return data.encode('ascii')
        else:  # Bytes
            return self._b64_decode(data.encode('ascii'))

    def _b64_encode(self, data):
        return base64.urlsafe_b64encode(data).replace(b'=', b'').decode('ascii')

    def _b64_decode(self, data):
        n_pad = (4 - len(data) % 4) % 4
        if n_pad <= 2:
            data = data + b'=' * n_pad
        elif n_pad == 3:
            raise TypeError()  # TODO
        try:
            return base64.b64decode(data, b'-_')
        except BinasciiError:
            raise TypeError()

    def escape_url(self, param):
        return quote(param, safe='')

    def _get_latest_task_progress_message(self, task_name,
                                          default=None, default_running=None, default_failed=None):
        tasks = self.get_background_tasks(task_name)['tasks']
        if not tasks:
            return default
        latest_task = max(tasks, key=lambda t: t['start_time'])
        if latest_task['status'] == 'RUNNING':
            return latest_task.get('progress', default_running)
        if latest_task['status'] == 'FAILED':
            raise ServerError({'error': latest_task.get('error', default_failed)})
        return latest_task['status']
