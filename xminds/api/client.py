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
from .exceptions import JwtTokenExpired, ServerError


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
            data['frontend_user_id'] = frontend_user_id
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
                    'user_id_type': str,
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
        }
        """
        path = f'databases/current/'
        return self.api.get(path=path)

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
        path = f'users-properties/{property_name}/'
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
        path = f'users-properties/{property_name}/'
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
        return self.api.get(path=path)

    @require_login
    def create_or_update_user(self, user):
        """
        Create a new user, or update it if the ID already exists.

        :param object user: user ID and properties {'user_id': ID, *<property_name: property_value>}
        """
        user = dict(user)
        user_id = self._userid2url(user.pop('user_id'))
        path = f'users/{user_id}/'
        data = {
            'user': user,
        }
        return self.api.put(path=path, data=data)

    @require_login
    def create_or_update_users_bulk(self, users, users_m2m=None, chunk_size=(1<<10)):
        """
        Create many users in bulk, or update the ones for which the id already exist.

        :param array users: array with fields ['id': ID, *<property_name: value_type>]
            contains only the non-repeated values,
        :param array? users_m2m: object of arrays for repeated values:
            {
                *<repeated_property_name: {
                    'name': str,
                    'array': array with fields ['user_index': uint32, 'value_id': value_type],
                }>
            }
        :param int? chunk_size: split the requests in chunks of this size (default: 1K)
        """
        users_m2m = users_m2m or []
        # cast dict to list of dict
        if isinstance(users_m2m, dict):
            users_m2m = [{'name': name, 'array': array}
                         for name, array in users_m2m.items()]
        path = f'users-bulk/'
        n_chunks = int(numpy.ceil(len(users) / chunk_size))
        for i in tqdm(range(n_chunks), disable=(True if n_chunks < 4 else None)):
            start_idx = i * chunk_size
            end_idx = (i+1) * chunk_size
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
            data = {
                'users': users_chunk,
                'users_m2m': users_m2m_chunk,
            }
            self.api.put(path=path, data=data, timeout=60)

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
        path = f'items-properties/{property_name}/'
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
        path = f'items-properties/{property_name}/'
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

        :param object item: item ID and properties {'item_id': ID, *<property_name: property_value>}
        """
        item = dict(item)
        item_id = self._itemid2url(item.pop('item_id'))
        path = f'items/{item_id}/'
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
        path = f'items-bulk/'
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
            'items_id': array of items IDs,
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
    def get_reco_session_to_items(self, ratings, amt=None, cursor=None, filters=None):
        """
        Get items recommendations given the ratings of an anonymous session.

        :param array ratings: ratings array with fields ['item_id': ID, 'rating': float]
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param list-str? filters: Filter by properties. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
        :returns: {
            'items_id': array of items IDs,
            'next_cursor': str, pagination cursor to use in next request to get more items,
        }
        """
        path = f'recommendation/sessions/items/'
        data = {
            'ratings': ratings,
        }
        if amt:
            data['amt'] = amt
        if cursor:
            data['cursor'] = cursor
        if filters:
            data['filters'] = filters
        return self.api.post(path=path, data=data)

    # === Reco: User-to-item ===

    @require_login
    def get_reco_user_to_items(self, user_id, amt=None, cursor=None, filters=None):
        """
        Get items recommendations given a user ID.

        :param ID user_id: user ID
        :param int? amt: amount to return (default: use the API default)
        :param str? cursor: Pagination cursor
        :param list-str? filters: Filter by properties. Filter format: ['<PROP_NAME>:<OPERATOR>:<OPTIONAL_VALUE>',...]
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
        user_id = self._userid2url(user_id)
        path = f'users/{user_id}/ratings/'
        data = {
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

    # === Scheduled Background Tasks ===

    @require_login
    def trigger_background_task(self, task_name):
        """
        Trigger background task such as retraining of ML models.
        You should not have to call this endpoint yourself, as this is done automatically.

        :param str task_name: for instance `'ml_model_retrain'`
        :returns: {
            'task_id': str,
        }
        :raises: DuplicatedError with error name 'TASK_ALREADY_RUNNING'
            if this task is already running
        """
        path = f'tasks/{task_name}/'
        return self.api.post(path=path, data={})

    @require_login
    def get_background_tasks(self, task_name):
        """
        List currently running background tasks such as ML models training.

        :param str task_name: for instance `'ml_model_retrain'`
        :returns: {
            'tasks': [{
                'name': string, Task name
                'start_time': int, Start timestamp
                'details': dict, Execution details, like progress message
            }],
        }
        """
        path = f'tasks/{task_name}/recents/'
        return self.api.get(path=path)

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

        :param str task_name: for instance `'ml_model_retrain'`
        :param int? timeout: maximum time to wait after the new task is triggered (default: 10min)
        :param int? lock_wait_timeout: if another task is already running, maximum time to wait
            for it to finish before triggering the new task (default: `timeout`)
        :param int? sleep: time to wait between polling (default: 1s)
        :returns: {
            'task_id': str,
        }
        :raises: RuntimeError if either `timeout` or `lock_wait_timeout` is reached
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
                filtr=lambda t: t['status'] != 'COMPLETED')
        # trigger
        task_id = self.trigger_background_task(task_name)['task_id']
        # wait for new task
        msg = 'waiting...' if verbose else None
        self.wait_for_background_task(
            task_name, timeout, sleep, msg=msg, filtr=lambda t: t['task_id'] == task_id)

    def wait_for_background_task(self, task_name, timeout=600, sleep=1, msg=None, filtr=None,
                                 wait_if_no_task=True):
        """
        Wait for a certain background task. Optionally specified with `filtr` function

        :param str task_name: for instance `'ml_model_retrain'`
        :param int? timeout: maximum time to wait after the new task is triggered (default: 10min)
        :param int? sleep: time to wait between polling (default: 1s)
        :param str? msg: either `None` to disable print, or message prefix (default: None)
        :param func? filtr: filter function(task: bool)
        :param bool? wait_if_no_task: wait (instead of return) if there is no task satisfying filter
        :returns: True is a task satisfying filters successfully ran, False otherwise
        :raises: RuntimeError if `timeout` is reached or if the task failed
        """
        if filtr is None:
            filtr = lambda t: True
        spinner = '|/-\\'
        task = None
        time_start = time.time()
        time_waited = 0
        i = 0
        while time_waited < timeout:
            time.sleep(sleep)
            time_waited = time.time() - time_start
            print_time = f'{int(time_waited) // 60:d}m{int(time_waited) % 60:02d}s'
            tasks = self.get_background_tasks(task_name)['tasks']
            try:
                task = next(task for task in tasks if filtr(task))
            except StopIteration:
                if wait_if_no_task:
                    continue
                else:
                    return False
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
        if self._database['user_id_type'].startswith('bytes'):
            return self._b64(user_id)
        return user_id

    def _itemid2url(self, item_id):
        """ base64 encode if needed """
        if self._database['item_id_type'].startswith('bytes'):
            return self._b64(item_id)
        return item_id

    def _b64(self, data):
        return base64.urlsafe_b64encode(data).replace(b'=', b'').decode('ascii')

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
