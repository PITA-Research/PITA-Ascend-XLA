"""Run with the NPU plugin configured and JAX_PLATFORMS=npu,cpu."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def cases():
    return [
        ("vector", np.arange(6, dtype=np.float32), -1),
        ("batch", np.arange(72, dtype=np.float32).reshape(12, 6), -1),
        ("ties_odd", np.array([2, 7, 7, -4, -4], np.float32), 0),
        ("nan_odd", np.array([0, np.nan, 2, np.nan, 4], np.float32), 0),
        ("nan_even", np.array([0, np.nan, 2, np.nan, 4, 5], np.float32), 0),
        ("all_nan", np.full(5, np.nan, np.float32), 0),
        ("positive_inf", np.full(5, np.inf, np.float32), 0),
        ("negative_inf", np.full(5, -np.inf, np.float32), 0),
        ("signed_zero", np.array([0., -0., 0., -0., 0.], np.float32), 0),
        ("singleton", np.array([3.], np.float32), 0),
        ("axis0", np.array([[0, 4, 2], [5, 3, 2], [1, 4, 7]], np.float32), 0),
        ("middle_axis", np.arange(30, dtype=np.float32).reshape(2, 5, 3), 1),
        ("flatten", np.array([[0, 4, 2], [5, 3, 2]], np.float32), None),
        ("int32", np.array([3, -7, 5, 5, -7], np.int32), 0),
    ]


def compare_platforms(fn, host):
    results = []
    for platform in ("cpu", "npu"):
        x = jax.device_put(host, jax.devices(platform)[0])
        np.testing.assert_array_equal(np.asarray(x).view(np.uint8), host.view(np.uint8))
        result = jax.jit(fn)(x)
        jax.block_until_ready(result)
        results.append(jax.tree.map(np.asarray, result))
    for cpu, npu in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
        np.testing.assert_array_equal(npu, cpu)
    return results


@pytest.mark.parametrize("maximum", [False, True], ids=["argmin", "argmax"])
@pytest.mark.parametrize("name,host,axis", cases(), ids=[case[0] for case in cases()])
def test_argreduce(maximum, name, host, axis):
    op = jnp.argmax if maximum else jnp.argmin
    # Exercise both a direct result and a tuple result, including keepdims.
    results = compare_platforms(lambda x: (op(x, axis=axis),
                                          op(x, axis=axis, keepdims=True)), host)
    expected = (np.argmax if maximum else np.argmin)(host, axis=axis)
    np.testing.assert_array_equal(results[1][0], expected)


def test_frozen_collision_scores():
    path = Path(__file__).with_name("data") / "collision_support_12x6.npz"
    with np.load(path) as data:
        host = data["support"]
    results = compare_platforms(
        lambda x: (jnp.argmax(x, axis=1), jnp.argmin(x, axis=1)), host)
    np.testing.assert_array_equal(results[1][0], [4, 4, 4, 5, 3, 3, 3, 4, 5, 5, 0, 5])


def test_coupled_value_and_custom_index():
    # The index operand need not be an iota. Both reducer outputs are live.
    def fn(x):
        indices = jnp.array([9, 3, 7, 2, 4], dtype=jnp.int32)

        def reducer(lhs, rhs):
            lv, li = lhs
            rv, ri = rhs
            pred = (lv > rv) | (lv != lv)
            first = pred | ((lv == rv) & (li < ri))
            return jnp.where(pred, lv, rv), jnp.where(first, li, ri)

        return jax.lax.reduce((x, indices),
                              (jnp.float32(-np.inf), jnp.int32(0)), reducer, (0,))

    compare_platforms(fn, np.array([1, 8, 8, 3, 2], np.float32))


def test_mixed_variadic_reducers():
    def fn(x):
        return jax.lax.reduce(
            (x, x), (jnp.float32(0), jnp.float32(-np.inf)),
            lambda a, b: (a[0] + b[0], jnp.maximum(a[1], b[1])), (0,))

    compare_platforms(fn, np.array([1, 2, 3, 4, 5], np.float32))


@pytest.mark.parametrize("kind", ["xor", "abs_max"])
def test_unsupported_reducer_is_rejected(kind):
    # Names/metadata containing "add" must not turn an unknown reducer into sum.
    def reduce_sum_add(x):
        reducer = (jax.lax.bitwise_xor if kind == "xor" else
                   lambda a, b: jnp.maximum(jnp.abs(a), jnp.abs(b)))
        return jax.lax.reduce(x, jnp.int32(0), reducer, (0,))

    x = jax.device_put(np.arange(6, dtype=np.int32), jax.devices("npu")[0])
    with pytest.raises(jax.errors.JaxRuntimeError, match="Unsupported NPU reducer"):
        jax.jit(reduce_sum_add).lower(x).compile()


@pytest.mark.parametrize("op", [jnp.sum, jnp.max, jnp.min, jnp.all, jnp.any])
def test_existing_scalar_reducers(op):
    compare_platforms(lambda x: op(x, axis=-1),
                      np.array([[0, 1, 2, 3], [0, 0, 0, 0]], np.float32))
