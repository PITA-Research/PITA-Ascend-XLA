"""Scatter point grids are independent of the number of mapped operand axes."""
import jax
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['unique', 'duplicate', 'oob'])
def test_scatter_index_grid(case):
    base = np.arange(200, dtype=np.float32).reshape(20, 10)
    indices = np.arange(9, dtype=np.int32).reshape(3, 3, 1)
    updates = np.arange(1, 181, dtype=np.float32).reshape(20, 3, 3)
    if case == 'duplicate':
        indices = np.array([1,1,1,4,4,4,9,9,9], np.int32).reshape(3,3,1)
    if case == 'oob':
        indices = np.array([-1,10,2147483647,0,1,2,8,9,9], np.int32).reshape(3,3,1)
        updates[:,0,:] = np.nan  # Dropped updates must not contaminate the base.
    expected = base.copy()
    for i in range(3):
        for k in range(3):
            col = indices[i,k,0]
            if 0 <= col < 10:
                expected[:,col] += updates[:,i,k]
    def fn(b,i,u):
        dims = jax.lax.ScatterDimensionNumbers((0,), (1,), (1,))
        return jax.lax.scatter_add(b,i,u,dims,mode='drop')
    for platform in ('cpu','npu'):
        values = jax.device_put((base,indices,updates),jax.devices(platform)[0])
        for original, value in zip((base,indices,updates),values):
            np.testing.assert_array_equal(original.view(np.uint32), np.asarray(value).view(np.uint32))
        actual = np.asarray(jax.jit(fn)(*values))
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
