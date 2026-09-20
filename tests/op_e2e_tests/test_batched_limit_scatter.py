"""Regression test for nested-vmap joint-limit one-hot scatter.

This mirrors the MJX limit-constraint row construction:

    zeros(nv).at[dofadr].set(sign) * active

The failing case previously zeroed sparse active rows on NPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _inputs(batch=4096, rows=12, nv=18):
    active = np.zeros((batch, rows), dtype=np.bool_)
    sign = np.ones((batch, rows), dtype=np.float32)
    seeds = [
        (2478, 0, -1.0),
        (3180, 3, 1.0),
        (3830, 8, -1.0),
        (10, 1, 1.0),
        (11, 2, -1.0),
    ]
    for env_id, row_id, value in seeds:
        active[env_id, row_id] = True
        sign[env_id, row_id] = value

    dofadr = np.arange(6, 6 + rows, dtype=np.int32)
    expected = np.zeros((batch, rows, nv), dtype=np.float32)
    for env_id, row_id, value in seeds:
        expected[env_id, row_id, dofadr[row_id]] = value
    return active, sign, dofadr, expected


def _scatter_postmul(active, sign, dofadr, nv):
    def one_env(sign_e, active_e):
        def one_row(s, a, dof):
            return (
                jnp.zeros(nv, dtype=jnp.float32).at[dof].set(s)
                * a.astype(jnp.float32)
            )

        return jax.vmap(one_row)(sign_e, active_e, dofadr)

    return jax.vmap(one_env)(sign, active)


def _scatter_premul(active, sign, dofadr, nv):
    def one_env(sign_e, active_e):
        def one_row(s, a, dof):
            return jnp.zeros(nv, dtype=jnp.float32).at[dof].set(
                s * a.astype(jnp.float32)
            )

        return jax.vmap(one_row)(sign_e, active_e, dofadr)

    return jax.vmap(one_env)(sign, active)


def _scatter_where(active, sign, dofadr, nv):
    def one_env(sign_e, active_e):
        def one_row(s, a, dof):
            row = jnp.zeros(nv, dtype=jnp.float32).at[dof].set(s)
            return jnp.where(a, row, jnp.zeros_like(row))

        return jax.vmap(one_row)(sign_e, active_e, dofadr)

    return jax.vmap(one_env)(sign, active)


@pytest.mark.parametrize(
    "fn",
    [_scatter_postmul, _scatter_premul, _scatter_where],
    ids=["postmul", "premul", "where"],
)
def test_nested_vmap_limit_scatter_sparse_active_rows(fn):
    active, sign, dofadr, expected = _inputs()
    compiled = jax.jit(fn, static_argnames=("nv",))
    result = np.asarray(
        compiled(
            jnp.asarray(active),
            jnp.asarray(sign),
            jnp.asarray(dofadr),
            expected.shape[-1],
        )
    )

    np.testing.assert_array_equal(result, expected)
