"""Non-scalar broadcast fusion end-to-end correctness tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_utils import assert_close, gen_data


class TestBroadcastFusion:
    @pytest.mark.parametrize(
        "input_shape,output_shape,broadcast_dimensions",
        [
            ((64,), (64, 64), (1,)),
            ((32, 16), (32, 4, 16), (0, 2)),
            ((32,), (8, 32, 8), (1,)),
        ],
        ids=["leading_axis", "middle_axis", "leading_and_trailing_axes"],
    )
    def test_broadcast_add_matches_numpy_and_is_deterministic(
        self, input_shape, output_shape, broadcast_dimensions
    ):
        source = gen_data(input_shape, np.float32, seed=0)
        bias = gen_data(output_shape, np.float32, seed=1)

        @jax.jit
        def broadcast_add(x, y):
            return jax.lax.broadcast_in_dim(
                x, output_shape, broadcast_dimensions
            ) + y

        source_npu = jnp.asarray(source)
        bias_npu = jnp.asarray(bias)
        results = []
        for _ in range(3):
            result = broadcast_add(source_npu, bias_npu)
            result.block_until_ready()
            results.append(np.asarray(result))

        reshape = [1] * len(output_shape)
        for input_axis, output_axis in enumerate(broadcast_dimensions):
            reshape[output_axis] = input_shape[input_axis]
        expected = np.broadcast_to(source.reshape(reshape), output_shape) + bias
        for result in results:
            assert_close(result, expected, np.float32)
        for previous, current in zip(results, results[1:]):
            np.testing.assert_array_equal(previous, current)
