"""SOP chapter 9: while carry must not contaminate shared zero output leaves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def make_zero_leaf_case(packing):
    def run(limit):
        zero = jnp.zeros((2,), dtype=jnp.int32)

        def cond(state):
            return state[0] < limit

        def body(state):
            return state[0] + jnp.int32(1), state[1] + jnp.int32(1)

        _, carry = jax.lax.while_loop(cond, body, (jnp.int32(0), zero))
        if packing == "single":
            return zero, carry
        if packing == "repeated":
            return (zero,) * 16 + (carry,)
        return {"zeros": (zero, (zero, zero)), "carry": carry,
                "after_while": carry == zero}

    return run


@pytest.mark.parametrize("packing", ["single", "repeated", "nested"])
def test_zero_output_leaves_isolated_from_while_carry(packing):
    function = make_zero_leaf_case(packing)
    references = {}
    for platform in ("cpu", "npu"):
        device = jax.devices(platform)[0]
        compiled = jax.jit(function).lower(
            jax.device_put(np.int32(2), device)).compile()
        # Changing the runtime trip count also checks reuse of the executable.
        for limit in (2, 1, 0, 3, 2):
            output = jax.device_get(compiled(jax.device_put(np.int32(limit), device)))
            if packing == "nested":
                zero_leaves = jax.tree.leaves(output["zeros"])
                carry = output["carry"]
                np.testing.assert_array_equal(output["after_while"],
                                              np.full(2, limit == 0, np.bool_))
            else:
                zero_leaves, carry = output[:-1], output[-1]
            for leaf in zero_leaves:
                np.testing.assert_array_equal(leaf, np.zeros(2, np.int32))
            np.testing.assert_array_equal(carry, np.full(2, limit, np.int32))
            leaves = jax.tree.leaves(output)
            if platform == "cpu":
                references[limit] = leaves
            else:
                assert len(leaves) == len(references[limit])
                for actual, expected in zip(leaves, references[limit]):
                    np.testing.assert_array_equal(actual, expected)
