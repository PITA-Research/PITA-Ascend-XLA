"""Cover host-copy branches beyond the original P07/P08 index regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _check_alternating(fn, base, expected):
    x = np.ones((4096, 4096), np.float32)
    reference = {}
    for platform in ("cpu", "npu"):
        device = jax.devices(platform)[0]
        xx, bb = jax.device_put((x, base), device)
        np.testing.assert_array_equal(np.asarray(bb), base)
        compiled = jax.jit(fn).lower(
            xx, jax.device_put(np.float32(1), device), bb).compile()
        for iteration in range(2 if platform == "cpu" else 20):
            sign = np.float32(1 if iteration % 2 == 0 else -1)
            actual = np.asarray(compiled(xx, jax.device_put(sign, device), bb))
            np.testing.assert_array_equal(actual, expected(sign))
            if platform == "cpu":
                reference[float(sign)] = actual
            else:
                np.testing.assert_array_equal(actual, reference[float(sign)])


@pytest.mark.parametrize("path", ["flatten", "extract_column"])
def test_gather_host_index_copy_branches(path):
    # Both columns are dynamic; the second is ignored by extract_column because
    # its slice covers the entire operand dimension. The other path linearizes
    # two genuinely gathered dimensions.
    base = np.arange(16, dtype=np.float32).reshape(4, 4)

    def fn(x, sign, data):
        index = (jnp.sum(x @ (x * sign)) > 0).astype(jnp.int32)
        indices = jnp.stack([jnp.stack([index, index + 1]),
                             jnp.stack([index + 1, index])])
        column = path == "extract_column"
        dims = jax.lax.GatherDimensionNumbers(
            offset_dims=(1,) if column else (),
            collapsed_slice_dims=(0,) if column else (0, 1),
            start_index_map=(0, 1))
        result = jax.lax.gather(
            data, indices, dims, (1, 4) if column else (1, 1), mode="clip")
        return result * jnp.float32(2) + jnp.float32(1)

    def expected(sign):
        index = int(sign > 0)
        if path == "extract_column":
            result = base[[index, index + 1]]
        else:
            result = base[[index, index + 1], [index + 1, index]]
        return result * np.float32(2) + np.float32(1)

    _check_alternating(fn, base, expected)


@pytest.mark.parametrize("mode,dtype", [
    ("replace", np.float32), ("mul", np.float32), ("replace", np.int32)])
def test_scatter_device_base_and_updates_with_constant_indices(mode, dtype):
    # Constant indices alone do not exercise ordering of data and updates.
    # Integer replace also uses host D2H updates and per-element H2D writes.
    base = np.arange(1, 17, dtype=dtype).reshape(4, 4)

    def fn(x, sign, data):
        value = (jnp.sum(x @ (x * sign)) / jnp.float32(2 ** 36)).astype(data.dtype)
        operand = data + value
        updates = jnp.asarray([3, 5], dtype=data.dtype) + value
        dims = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1))
        op = jax.lax.scatter if mode == "replace" else jax.lax.scatter_mul
        result = op(operand, jnp.asarray([[1, 1], [2, 2]], jnp.int32),
                    updates, dims, mode="promise_in_bounds")
        return result * jnp.asarray(2, data.dtype) + jnp.asarray(1, data.dtype)

    def expected(sign):
        result = base + dtype(sign)
        updates = np.asarray([3, 5], dtype=dtype) + dtype(sign)
        if mode == "replace":
            result[[1, 2], [1, 2]] = updates
        else:
            result[[1, 2], [1, 2]] *= updates
        return result * dtype(2) + dtype(1)

    _check_alternating(fn, base, expected)
