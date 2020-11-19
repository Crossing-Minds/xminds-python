import unittest

import numpy

from xminds.lib.arrays import  (kargmin, kargmax, to_structured,
                                set_or_add_to_structured, structured_arrays_mean,
                                set_or_reallocate)


def test_set_or_reallocate():
    # 1D
    array = numpy.arange(3)  # 0 1 2
    array = set_or_reallocate(array, [3], 0)  # 3 1 2
    assert list(array) == [3, 1, 2]
    array = set_or_reallocate(array, [4, 5, 6], 2)  # 3 1 4 5 6
    assert list(array[:5]) == [3, 1, 4, 5, 6]
    array = set_or_reallocate(array, numpy.ones(8), 0, fill=0)  # 1 .. 1 0 .. 0
    assert array[-1] == 0
    # 2D
    array = numpy.arange(3*2).reshape((3, 2))  # [[0 1][2 3][4 5]]
    array = set_or_reallocate(array, [[6, 7], [8, 9]], 3)
    assert (array[:5, :] == numpy.arange(5*2).reshape((5, 2))).all()


class StructuredArrayTestCase(unittest.TestCase):
    def test_to_structured(self):
        n = 20
        items_ty = numpy.random.randint(1, 1<<7, n, dtype='uint8')
        items_id = numpy.random.randint(1, 1<<15, n, dtype='uint16')
        items_uuid = numpy.random.randint(1, 1<<31, n, dtype='uint32')

        def _assert_ty_id(items):
            assert (items['ty'] == items_ty).all()
            assert (items['id'] == items_id).all()

        # default dtype
        items = to_structured([('ty', items_ty), ('id', items_id), ('uuid', items_uuid)])
        _assert_ty_id(items)
        # default dtype with object
        items = to_structured([('ty', items_ty), ('id', items_id),
                               ('uuid', items_uuid.astype('object'))])
        _assert_ty_id(items)
        assert (items['uuid'] == items_uuid.astype('object')).all()
        # give dtype
        items = to_structured([('ty', items_ty, 'uint8'),
                               ('id', items_id, 'uint16'),
                               ('uuid', items_uuid, 'uint32')])
        _assert_ty_id(items)
        assert (items['uuid'] == items_uuid).all()
        # 2d: default dtype
        items_embs = numpy.random.randn(n, 3).astype('float32')
        items = to_structured([('ty', items_ty), ('id', items_id), ('embs', items_embs)])
        _assert_ty_id(items)
        assert numpy.allclose(items['embs'], items_embs)
        # 2d: given dtype
        items = to_structured([('ty', items_ty, 'uint8'),
                               ('id', items_id, 'uint16'),
                               ('embs', items_embs, 'float32', 3)])
        _assert_ty_id(items)
        assert numpy.allclose(items['embs'], items_embs)
        # 2d: given dtype with shape in dtype
        items = to_structured([('ty', items_ty, 'uint8'),
                               ('id', items_id, 'uint16'),
                               ('embs', items_embs, '(3,)float32')])
        _assert_ty_id(items)
        assert numpy.allclose(items['embs'], items_embs)
        # 2d but flat: default dtype
        flat_embs = numpy.random.randn(n, 1).astype('float32')
        items = to_structured([('ty', items_ty),
                               ('id', items_id),
                               ('embs', flat_embs)])
        assert items.shape == (n,)
        _assert_ty_id(items)
        assert items['embs'].shape == (n, 1)
        # 2d but flat: given dtype with shape in dtype
        items = to_structured([('ty', items_ty, 'uint8'),
                               ('id', items_id, 'uint16'),
                               ('embs', flat_embs, '(1,)float32')])
        assert items.shape == (n,)
        _assert_ty_id(items)
        assert items['embs'].shape == (n, 1)
        # dict input
        items = to_structured({'ty': items_ty, 'id': items_id})
        _assert_ty_id(items)
        # broadcast scalar
        items = to_structured({'ty': items_ty, 'id': items_id, 'uuid': 2})
        _assert_ty_id(items)
        assert (items['uuid'] == 2).all()

        # long long shouldn't be cast to float
        items = to_structured([
            ('id', [12345678901234567890, 123], 'uint64'),
            ('v', [1, 2]),
        ])
        assert items['id'][0] == 12345678901234567890

    def test_set_or_add_to_structured(self):
        n = 20
        array = to_structured([
            ('a', numpy.random.randint(1, 1<<7, n, dtype='uint8')),
            ('b', numpy.random.randint(1, 1<<15, (n, 5), dtype='uint16')),
        ])
        # just new
        data_c = numpy.random.randint(1, 1<<31, (n, 3), dtype='uint32')
        array2 = set_or_add_to_structured(array, [
            ('c', data_c),
        ])
        assert (array2['a'] == array['a']).all()
        assert (array2['b'] == array['b']).all()
        assert (array2['c'] == data_c).all()
        # just old
        data_a = numpy.random.randint(1, 1<<7, n, dtype='uint32')
        data_b = numpy.random.randint(1, 1<<15, (n, 5), dtype='uint32')
        array2 = set_or_add_to_structured(array, [
            ('a', data_a),
            ('b', data_b),
        ])
        assert (array2['a'] == data_a).all()
        assert (array2['b'] == data_b).all()
        # old and new
        array2 = set_or_add_to_structured(array, [
            ('a', data_a),
            ('b', data_b),
            ('c', data_c)
        ])
        assert (array2['a'] == data_a).all()
        assert (array2['b'] == data_b).all()
        assert (array2['c'] == data_c).all()
        # old and new with object
        array = to_structured([
            ('a', numpy.random.randint(1, 1<<7, n, dtype='uint8')),
            ('b', numpy.random.randint(1, 1<<15, n).astype('object')),
        ])
        array2 = set_or_add_to_structured(array, [
            ('a', data_a),
            ('b', numpy.random.randint(1, 1<<15, n).astype('object')),
            ('c', numpy.random.randint(1, 1<<31, n).astype('object')),
        ])

    def test_structured_arrays_mean(self):
        n1 = 20
        array1 = to_structured([
            ('a', numpy.random.randint(1, 1<<7, n1, dtype='uint8')),
            ('b', numpy.random.randint(1, 1<<15, (n1, 5), dtype='uint16')),
            ('zz', numpy.random.randint(1, 1<<15, (n1, 5), dtype='uint16')),
        ])
        n2 = 1
        array2 = to_structured([
            ('a', numpy.random.randint(1, 1<<7, n2, dtype='uint8')),
            ('b', numpy.random.randint(1, 1<<15, (n2, 5), dtype='uint16')),
            ('yy', numpy.random.randint(1, 1<<15, n2, dtype='uint16')),
        ])
        # intersection
        mean_intrsct = structured_arrays_mean([array1, array2], keep_missing=False)
        assert mean_intrsct['a'] == numpy.r_[array1['a'], array2['a']].mean()
        assert (mean_intrsct['b'] == numpy.r_[array1['b'], array2['b']].mean(axis=0)).all()
        # union
        mean_union = structured_arrays_mean([array1, array2], keep_missing=True)
        assert mean_intrsct['a'] == numpy.r_[array1['a'], array2['a']].mean()
        assert (mean_intrsct['b'] == numpy.r_[array1['b'], array2['b']].mean(axis=0)).all()
        assert (mean_union['zz'] == array1['zz'].mean(axis=0)).all()
        assert mean_union['yy'] == array2['yy'].mean()


class KArgmaxTestCase(unittest.TestCase):
    def test_kargmax_1d(self):
        arr = numpy.random.rand(10)
        arr[4] = 2.
        arr[7] = 3.
        top = kargmax(arr, 2)
        assert set(top) == {4, 7}
        top_sorted = kargmax(arr, 2, do_sort=True)
        assert list(top_sorted) == [7, 4]

    def test_kargmin_1d(self):
        arr = numpy.random.rand(10)
        arr[4] = -2.
        arr[7] = -3.
        top = kargmin(arr, 2)
        assert set(top) == {4, 7}
        top_sorted = kargmin(arr, 2, do_sort=True)
        assert list(top_sorted) == [7, 4]

    def test_kargmax_big_k(self):
        arr = numpy.random.rand(10)
        top = kargmax(arr, 10)
        assert set(top) == set(range(10))
        top = kargmax(arr, 99)
        assert set(top) == set(range(10))
        top = kargmax(arr, 10, do_sort=True)
        argsort = numpy.argsort(-arr)
        assert (top == argsort).all()

    def test_kargmin_big_k(self):
        arr = numpy.random.rand(10)
        top = kargmin(arr, 10)
        assert set(top) == set(range(10))
        top = kargmin(arr, 99)
        assert set(top) == set(range(10))
        top = kargmin(arr, 10, do_sort=True)
        argsort = numpy.argsort(arr)
        assert (top == argsort).all()

    def test_kargmax_2d(self):
        # axis=0
        arr = numpy.random.rand(10, 20)
        arr[4, :] = 2.
        arr[7, :] = 3.
        top = kargmax(arr, 2, axis=0, do_sort=True)
        assert top.shape == (2, 20)
        assert (top[0, :] == 7).all()
        assert (top[1, :] == 4).all()
        # axis=1
        arr = numpy.random.rand(20, 10)
        arr[:, 4] = 2.
        arr[:, 7] = 3.
        top = kargmax(arr, 2, axis=1, do_sort=True)
        assert top.shape == (20, 2)
        assert (top[:, 0] == 7).all()
        assert (top[:, 1] == 4).all()

    def test_kargmin_2d(self):
        # axis=0
        arr = numpy.random.rand(10, 20)
        arr[4, :] = -2.
        arr[7, :] = -3.
        top = kargmin(arr, 2, axis=0, do_sort=True)
        assert top.shape == (2, 20)
        assert (top[0, :] == 7).all()
        assert (top[1, :] == 4).all()
        # axis=1
        arr = numpy.random.rand(20, 10)
        arr[:, 4] = -2.
        arr[:, 7] = -3.
        top = kargmin(arr, 2, axis=1, do_sort=True)
        assert top.shape == (20, 2)
        assert (top[:, 0] == 7).all()
        assert (top[:, 1] == 4).all()