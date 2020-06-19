"""
xminds.api.client
~~~~~~~~~~~~~~~~~

This module implements the requests for all API endpoints.
The client handles the logic to automatically get a new JWT token using the refresh token
"""

import base64
from functools import wraps
import logging
import numpy
import sys
import time

from ..compat import tqdm
from .apirequest import CrossingMindsApiJsonRequest, CrossingMindsApiPythonRequest
from .exceptions import AuthTokenExpired


def require_login(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except AuthTokenExpired:
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
    def create_account(self, first_name, last_name, email, password):
        """
        Create a new account

        :param str first_name:
        :param str last_name:
        :param str email:
        :param str password:
        :returns: {'id': int}
        """
        path = 'accounts/create/'
        data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
        }
        return self.api.post(path=path, data=data)

    def resend_verification_code(self, email):
        """
        Resend the verification code to the account email

        :param str email:
        """
        path = 'accounts/resend-verification-code/'
        return self.api.put(path=path, data={'email': email})

    def verify_account(self, code, email):
        """
        Verify the email of an account by entering the verification code

        :param str code:
        :param str email:
        """
        path = 'accounts/verify/'
        data = {
            'code': code,
            'email': email,
        }
        return self.api.post(path=path, data=data)

    # === Login ===

    def login(self, email, password, db_id):
        """
        Login on a database with an account

        :param str email:
        :param str password:
        :param int db_id:
        :returns: {
            'token': str,
            'database': {
                'id': int,
                'name': str,
                'description': str,
                'item_id_type': str,
                'user_id_type': str,
            },
        }
        """
        path = 'login/'
        data = {
            'email': email,
            'password': password,
            'db_id': db_id,
        }
        resp = self.api.post(path=path, data=data)
        jwt_token = resp['token']
        self.set_jwt_token(jwt_token)
        self._database = resp['database']
        self._refresh_token = resp['refresh_token']
        return resp

    def login_root(self, email, password):
        """
        Login with the root account without selecting a database

        :param str email:
        :param str password:
        :returns: {
            'token': str,
        }
        """
        path = 'login/root/'
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
                'id': int,
                'name': str,
                'description': str,
                'item_id_type': str,
                'user_id_type': str,
            },
        }
        """
        refresh_token = refresh_token or self._refresh_token
        path = 'login/refresh-token/'
        data = {
            'refresh_token': refresh_token
        }
        resp = self.api.post(path=path, data=data)
        jwt_token = resp['token']
        self.set_jwt_token(jwt_token)
        self._database = resp['database']
        self._refresh_token = resp['refresh_token']
        return resp

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
        path = 'databases/'
        data = {
            'name': name,
            'description': description,
            'item_id_type': item_id_type,
            'user_id_type': user_id_type,
        }
        return self.api.post(path=path, data=data)

    @require_login
    def get_database(self):
        """
        Get details on current database

        :returns: {
            'id': int,
            'name': str,
            'description': str,
            'item_id_type': str,
            'user_id_type': str,
        }
        """
        path = 'databases/current/'
        return self.api.get(path=path)

    @require_login
    def delete_database(self):
        """
        Delete current database.
        """
        path = 'databases/current/'
        return self.api.delete(path=path, timeout=29)

    @require_login
    def status(self):
        """
        Get readiness status of current database.
        """
        path = 'databases/current/status/'
        return self.api.get(path=path)

    def wait_until_ready(self, timeout=600, sleep=1, verbose=None):
        """
        Wait until the current database status is ready.

        :param int? timeout: maximum time to wait, raise RuntimeError if exceeded (default: 10min)
        :param int? sleep: time to wait between polling (default: 1s)
        :param bool? verbose: whether to print ascii spinner (default: only on TTY)
        """
        assert sleep > 0.1
        if verbose is None:
            verbose = sys.stdout.isatty()
        spinner = '|/-\\'
        for i in range(int(numpy.ceil(float(timeout) / sleep))):
            time.sleep(sleep)
            print_time = f'{int(i * sleep) // 60:d}m{int(i * sleep) % 60:02d}s'
            resp = self.status()
            if resp['status'] == 'ready':
                if verbose:
                    print(f'\rready in {print_time}    ')
                return
            if verbose:
                print(f'\rwaiting... {print_time} {spinner[i%len(spinner)]} ', end='')
                sys.stdout.flush()
        if verbose:
            print('')
        raise RuntimeError(f'API not ready before {timeout}s. Last response: {resp}')

    # === User Property ===

    @require_login
    def get_user_property(self, property_name):
        path = f'users-properties/{property_name}/'
        return self.api.get(path=path)

    @require_login
    def list_user_properties(self):
        path = 'users-properties/'
        return self.api.get(path=path)

    @require_login
    def create_or_update_user_property(self, property_name, value_type, repeated=False):
        path = 'users-properties/'
        data = {
            'property_name': property_name,
            'value_type': value_type,
            'repeated': repeated,
        }
        return self.api.put(path=path, data=data)

    # === User ===

    @require_login
    def get_user(self, user_id):
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/'
        return self.api.get(path=path)

    @require_login
    def create_or_update_user(self, user):
        path = 'users/'
        if self.b64_encode_bytes:
            user = dict(user, user_id=self._userid2url(user['user_id']))
        data = {
            'user': user,
        }
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_users_bulk(self, users, users_m2m):
        if self.b64_encode_bytes:
            raise NotImplementedError('bulk is not implement for json serializer, '
                                      'use pkl serializer instead')
        path = 'users-bulk/'
        data = {
            'users': users,
            'users_m2m': users_m2m,
        }
        return self.api.put(path=path, data=data, timeout=60)

    # === Item Property ===

    @require_login
    def get_item_property(self, property_name):
        """
        Get one item property.

        :param str property_name: property name
        :returns: {
            'property_name': str,
            'value_type': str,
            'repeated': bool,
        }
        """
        path = f'items-properties/{property_name}/'
        return self.api.get(path=path)

    @require_login
    def list_item_properties(self):
        """
        Get all items properties for the current database.

        :returns: {
            'properties': [{
                'property_name': str,
                'value_type': str,
                'repeated': bool,
            }],
        }
        """
        path = 'items-properties/'
        return self.api.get(path=path)

    @require_login
    def create_item_property(self, property_name, value_type, repeated=False):
        """
        Create a new item property.

        :param str property_name: property name
        :param str value_type: property type
        :param bool? repeated: whether the property has many values (default: False)
        """
        path = 'items-properties/'
        data = {
            'property_name': property_name,
            'value_type': value_type,
            'repeated': repeated,
        }
        return self.api.post(path=path, data=data)

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
        return self.api.get(path=path)

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
            'items_m2m': object of arrays for repeated values:
                {
                    *<repeated_property_name: {
                        'name': str,
                        'array': array with fields ['item_index': uint32, 'value_id': value_type],
                    }>
                }
        }
        """
        path = f'items-bulk/list/'
        data = {'items_id': items_id}
        return self.api.post(path=path, data=data)

    @require_login
    def create_or_update_item(self, item):
        """
        Create a new item, or update it if the ID already exists.

        :param object item: item ID and properties {'id': ID, *<property_name: property_value>}
        """
        path = 'items/'
        if self.b64_encode_bytes:
            item = dict(item, item_id=self._itemid2url(item['item_id']))
        data = {
            'item': item,
        }
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_items_bulk(self, items, items_m2m=None, chunk_size=(1<<10)):
        """
        Create many items in bulk, or update the ones for which the id already exist.

        :param array items: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param array? items_m2m: object of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['item_index': uint32, 'value_id': value_type],
                }>
            }
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        """
        items_m2m = items_m2m or []
        # cast dict to list of dict
        if isinstance(items_m2m, dict):
            items_m2m = [{'name': name, 'array': array}
                         for name, array in items_m2m.items()]
        path = 'items-bulk/'
        n_chunks = int(numpy.ceil(len(items) / chunk_size))
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            start_idx = i * chunk_size
            end_idx = (i+1) * chunk_size
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
            data = {
                'items': items_chunk,
                'items_m2m': items_m2m_chunk,
            }
            self.api.put(path=path, data=data, timeout=60)

    # === Reco: Item-to-item ===

    @require_login
    def get_reco_item_to_items(self, item_id, amt=None):
        """
        Get similar items.

        :param ID item_id: item ID
        :param int? amt: amount to return (default: use the API default)
        :returns: {
            'items_id': array of items IDs
        }
        """
        item_id = self._itemid2url(item_id)
        path = f'recommendation/items/{item_id}/items/'
        params = {}
        if amt:
            params['amt'] = amt
        return self.api.get(path=path, params=params)

    # === Reco: Session-to-item ===

    @require_login
    def get_reco_session_to_items(self, ratings, amt=None):
        """
        Get items recommendations given the ratings of an anonymous session.

        :param array ratings: ratings array with fields ['item_id': ID, 'rating': float]
        :param int? amt: amount to return (default: use the API default)
        :returns: {
            'items_id': array of items IDs
        }
        """
        path = f'recommendation/sessions/items/'
        data = {
            'ratings': ratings,
        }
        if amt:
            data['amt'] = amt
        return self.api.post(path=path, data=data)

    # === Reco: User-to-item ===

    @require_login
    def get_reco_user_to_items(self, user_id, amt=None):
        """
        Get items recommendations given a user ID.

        :param ID user_id: user ID
        :param int? amt: amount to return (default: use the API default)
        :returns: {
            'items_id': array of items IDs
        }
        """
        user_id = self._userid2url(user_id)
        path = f'recommendation/users/{user_id}/items/'
        params = {}
        if amt:
            params['amt'] = amt
        return self.api.get(path=path, params=params)

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
        path = f'ratings-bulk/'
        data = {
            'user_id': user_id,
            'ratings': ratings,
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
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            data = {
                'ratings': ratings[i*chunk_size:(i+1)*chunk_size],
            }
            self.api.put(path=path, data=data, timeout=60)
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
        return self.api.get(path=path, params=params)

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
        if self._database['user_id_type'].startswith('S'):
            return self._b64(user_id)
        return user_id

    def _itemid2url(self, item_id):
        """ base64 encode if needed """
        if self._database['item_id_type'].startswith('S'):
            return self._b64(item_id)
        return item_id

    def _b64(self, data):
        return base64.urlsafe_b64encode(data).replace(b'=', b'').decode('ascii')