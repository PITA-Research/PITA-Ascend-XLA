"""NPU Cholesky results must not change when returned through tuple aliases.

Run with the NPU plugin environment configured and JAX_PLATFORMS=npu,cpu.
"""

from pathlib import Path

import jax
import jax.scipy.linalg as jl
import numpy as np
import pytest


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("output", ["single", "pair", "nested"])
def test_cholesky_tuple_layout(lower, output):
    with np.load(Path(__file__).with_name("data") / "cholesky_20x20.npz") as data:
        matrix = data["M"]

    def factor(x):
        triangular = jl.cho_factor(x, lower=lower)[0]
        if output == "single":
            return triangular
        if output == "pair":
            return x, triangular
        return x, (triangular, x)

    results = []
    for platform in ("cpu", "npu"):
        x = jax.device_put(matrix, jax.devices(platform)[0])
        np.testing.assert_array_equal(np.asarray(x).view(np.uint32),
                                      matrix.view(np.uint32))
        result = jax.jit(factor)(x)
        jax.block_until_ready(result)
        if output == "single":
            triangular = np.asarray(result)
        else:
            np.testing.assert_array_equal(np.asarray(result[0]), matrix)
            triangular = np.asarray(result[1] if output == "pair" else result[1][0])
            if output == "nested":
                np.testing.assert_array_equal(np.asarray(result[1][1]), matrix)
        unused = np.triu(triangular, 1) if lower else np.tril(triangular, -1)
        np.testing.assert_array_equal(unused, np.zeros_like(unused))
        reconstructed = (triangular @ triangular.T if lower
                         else triangular.T @ triangular)
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-6, rtol=1e-5)
        results.append(triangular)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6, rtol=1e-5)
