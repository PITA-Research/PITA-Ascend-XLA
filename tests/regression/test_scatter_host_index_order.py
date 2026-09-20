"""Host-linearized scatter indices must wait for device producers."""
import jax
import jax.numpy as jp
import numpy as np
import pytest


@pytest.mark.parametrize('mode', ['replace', 'mul', 'add'])
def test_scatter_waits_for_device_indices(mode):
    def fn(x, sign, base):
        index = (jp.sum(x @ (x * sign)) > 0).astype(jp.int32)
        indices = jp.stack([jp.stack([index, jp.int32(1)]),
                            jp.stack([index + 1, jp.int32(2)])])
        dims = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1))
        op = {'replace': jax.lax.scatter, 'mul': jax.lax.scatter_mul,
              'add': jax.lax.scatter_add}[mode]
        return indices, op(base, indices, jp.array([31., 41.], jp.float32),
                           dims, mode='promise_in_bounds')

    x = np.ones((4096, 4096), np.float32)
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    reference = {}
    for platform in ['cpu', 'npu']:
        device = jax.devices(platform)[0]
        xx, bb = jax.device_put((x, base), device)
        np.testing.assert_array_equal(np.asarray(xx).view(np.uint32), x.view(np.uint32))
        compiled = jax.jit(fn).lower(xx, jax.device_put(np.float32(1), device), bb).compile()
        for i in range(2 if platform == 'cpu' else 20):
            sign = 1 if i % 2 == 0 else -1
            indices, actual = map(np.asarray, compiled(
                xx, jax.device_put(np.float32(sign), device), bb))
            index = int(sign > 0)
            np.testing.assert_array_equal(indices, [[index, 1], [index + 1, 2]])
            if platform == 'cpu':
                reference[sign] = actual
            else:
                np.testing.assert_array_equal(actual.view(np.uint32), reference[sign].view(np.uint32))
