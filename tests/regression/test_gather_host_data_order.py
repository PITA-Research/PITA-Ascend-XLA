"""Business-shaped gather must wait for data, even with constant indices."""
import jax
import jax.numpy as jp
import numpy as np


def test_constant_index_gather_waits_for_device_data():
    def fn(x, sign, base):
        value = jp.sum(x @ (x * sign)) / jp.float32(2 ** 36)
        operand = base + value
        dims = jax.lax.GatherDimensionNumbers(
            offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=(0, 1))
        result = jax.lax.gather(
            operand, jp.zeros((1, 2), jp.int32), dims, (1, 3), mode='clip')
        return operand, result

    x = np.ones((4096, 4096), np.float32)
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    reference = {}
    for platform in ('cpu', 'npu'):
        device = jax.devices(platform)[0]
        xx, bb = jax.device_put((x, base), device)
        for actual, expected in ((xx, x), (bb, base)):
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint32), expected.view(np.uint32))
        compiled = jax.jit(fn).lower(
            xx, jax.device_put(np.float32(1), device), bb).compile()
        for i in range(2 if platform == 'cpu' else 20):
            sign = np.float32(1 if i % 2 == 0 else -1)
            operand, actual = map(np.asarray, compiled(
                xx, jax.device_put(sign, device), bb))
            expected_operand = base + sign
            expected = expected_operand[:1][None]
            np.testing.assert_array_equal(
                operand.view(np.uint32), expected_operand.view(np.uint32))
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32))
            if platform == 'cpu':
                reference[float(sign)] = actual
            else:
                np.testing.assert_array_equal(
                    actual.view(np.uint32), reference[float(sign)].view(np.uint32))
