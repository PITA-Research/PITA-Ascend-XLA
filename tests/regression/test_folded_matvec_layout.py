"""Folded scatter/transpose literals must retain their logical dot values."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('rows,cols', [(14, 20), (3, 5)])
@pytest.mark.parametrize('kind', ['direct', 'folded', 'rhs', 'matrix',
    'batch'])
def test_folded_constant_dot(rows, cols, kind):
    idx = np.arange(rows, dtype=np.int32)
    matrix = np.eye(rows, cols, dtype=np.float32)

    def fn(x):
        a = jnp.zeros((rows, cols), dtype=jnp.float32).at[idx, idx].set(1.)
        if kind == 'direct':
            return jnp.asarray(matrix.T) @ x
        if kind == 'rhs':
            return x @ a.T
        if kind == 'batch':
            return jax.vmap(lambda v: a.T @ v)(x)
        return a.T @ x

    shape = {'rhs': (2, cols), 'matrix': (rows, 3),
             'batch': (2, rows)}.get(kind, (rows,))
    x = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    expected = (x @ matrix.T if kind == 'rhs' else
                x @ matrix if kind == 'batch' else matrix.T @ x)
    for platform in ('cpu', 'npu'):
        value = jax.device_put(x, jax.devices(platform)[0])
        np.testing.assert_array_equal(np.asarray(value).view(np.uint32), x.view(np.uint32))
        actual = np.asarray(jax.jit(fn)(value))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
