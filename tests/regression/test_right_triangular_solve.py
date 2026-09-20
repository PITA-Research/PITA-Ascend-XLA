"""Right solves and the existing left path, including rectangular batched B."""
import jax
import numpy as np
import pytest


def inputs(batched, lower, unit, left):
    a = np.arange(25, dtype=np.float32).reshape(5, 5) / 128
    a = np.tril(a) if lower else np.triu(a)
    a[np.diag_indices(5)] = 0 if unit else 2
    a += (np.triu(np.ones_like(a), 1) if lower else
          np.tril(np.ones_like(a), -1)) * 19  # Must ignore unused triangle.
    b = np.arange(1, 16, dtype=np.float32).reshape((5, 3) if left else (3, 5)) / 16
    if batched:
        a = np.broadcast_to(a, (2, 2, 5, 5)).copy()
        b = np.stack((b, b * 2, b * 3, b * 4)).reshape((2, 2) + b.shape)
    return a, b


@pytest.mark.parametrize('left', [True, False])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('unit', [False, True])
def test_triangular_solve(left, batched, lower, transpose, unit):
    a, b = inputs(batched, lower, unit, left)
    def fn(a, b):
        return jax.lax.linalg.triangular_solve(a, b, left_side=left, lower=lower,
                                             transpose_a=transpose, unit_diagonal=unit)
    results = {}
    for platform in ('cpu', 'npu'):
        args = jax.device_put((a, b), jax.devices(platform)[0])
        for expected, actual in zip((a, b), args):
            np.testing.assert_array_equal(expected.view(np.uint32), np.asarray(actual).view(np.uint32))
        results[platform] = np.asarray(jax.jit(fn)(*args))
    np.testing.assert_allclose(results['npu'], results['cpu'], rtol=2e-5, atol=2e-6)
    effective = np.tril(a) if lower else np.triu(a)
    if unit:
        effective[..., np.arange(5), np.arange(5)] = 1
    if transpose:
        effective = effective.swapaxes(-1, -2)
    reconstructed = effective @ results['npu'] if left else results['npu'] @ effective
    np.testing.assert_allclose(reconstructed, b, rtol=2e-5, atol=2e-6)
