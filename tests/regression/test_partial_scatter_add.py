"""Partial windows must not dispatch to the row-only TfScatterAdd API."""
import jax
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['empty', 'original', 'duplicate', 'offset', 'oob', 'window2d'])
def test_partial_scatter_add(case):
    base = np.arange(32 * 6, dtype=np.float32).reshape(32, 6)
    indices = np.array([[1, 0]], np.int32)
    updates = np.array([[1, 2, 3]], np.float32)
    window = (1,)
    inserted = (0,)
    if case == 'duplicate':
        indices = np.array([[1, 0], [1, 0], [1, 1]], np.int32)
        updates = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    if case == 'offset':
        indices = np.array([[31, 3]], np.int32)
    if case == 'oob':
        indices = np.array([[-1, 0], [32, 0], [1, 4], [2, 3]], np.int32)
        updates = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    if case == 'empty':
        indices = np.empty((0, 2), np.int32)
        updates = np.empty((0, 3), np.float32)
    if case == 'window2d':
        indices = np.array([[1, 2], [2, 3]], np.int32)
        updates = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
        window, inserted = (1, 2), ()
    dims = jax.lax.ScatterDimensionNumbers(window, inserted, (0, 1))
    def fn(b, i, u):
        return jax.lax.scatter_add(b, i, u, dims, mode='drop')
    results = {}
    for platform in ('cpu', 'npu'):
        args = jax.device_put((base, indices, updates), jax.devices(platform)[0])
        for original, transferred in zip((base, indices, updates), args):
            np.testing.assert_array_equal(original, np.asarray(transferred))
        results[platform] = np.asarray(jax.jit(fn)(*args))
    np.testing.assert_array_equal(results['npu'].view(np.uint32),
                                  results['cpu'].view(np.uint32))
