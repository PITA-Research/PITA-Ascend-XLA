"""Semantic scatter-add coverage independent of application shapes.

The NumPy oracle decodes HLO window/inserted/index-grid dimensions directly.
Integer-valued float32 inputs keep duplicate accumulation exactly representable.
"""
from dataclasses import dataclass
import jax
import numpy as np
import pytest


@dataclass(frozen=True)
class Case:
    name: str
    base: tuple
    grid: tuple
    inserted: tuple
    mapping: tuple
    window: tuple
    window_axes: tuple


CASES = [
    Case('vector_unit', (21,), (14,), (), (0,), (1,), (1,)),
    Case('vector_window', (11,), (4,), (), (0,), (3,), (1,)),
    Case('vector_point', (11,), (4,), (0,), (0,), (), ()),
    Case('scalar_index_window', (11,), (), (), (0,), (3,), (0,)),
    Case('full_rows', (4,7), (3,), (0,), (0,), (7,), (1,)),
    Case('columns', (4,7), (3,), (1,), (1,), (4,), (0,)),
    Case('partial_rows', (4,7), (3,), (0,), (0,1), (3,), (1,)),
    Case('grid_columns', (4,7), (2,3), (1,), (1,), (4,), (0,)),
    Case('grid_interleaved', (4,7), (2,3), (0,), (0,1), (3,), (1,)),
    Case('matrix_windows', (4,7), (3,), (), (1,0), (2,3), (0,2)),
    Case('all_points', (4,7), (2,3), (0,1), (1,0), (), ()),
    Case('rank3_window', (3,4,5), (2,2), (1,), (2,1,0), (2,3), (1,3)),
    Case('rank4_window', (2,3,4,5), (2,), (0,2), (3,0,2), (2,3), (0,2)),
    Case('grid_rank3', (7,), (2,2,2), (), (0,), (1,), (2,)),
    Case('empty_points', (4,7), (0,), (0,), (0,1), (3,), (1,)),
    Case('empty_window', (4,7), (2,), (0,), (0,1), (0,), (1,)),
    Case('scalar_operand', (), (3,), (), (), (), ()),
]


# Deterministic generated geometry, independent of previously reported shapes.
_rng = np.random.default_rng(20260914)
for _k in range(12):
    _rank = int(_rng.integers(1,5))
    _base = tuple(map(int,_rng.integers(2,6,size=_rank)))
    _inserted = tuple(d for d in range(_rank) if _rng.integers(2))
    _window = tuple(int(_rng.integers(1,_base[d]+1)) for d in range(_rank) if d not in _inserted)
    _grid = tuple(map(int,_rng.integers(1,4,size=int(_rng.integers(0,4)))))
    _mapping = tuple(map(int,_rng.permutation(_rank)[:int(_rng.integers(1,_rank+1))]))
    _axes = tuple(sorted(map(int,_rng.choice(len(_window)+len(_grid),len(_window),replace=False))))
    CASES.append(Case(f'generated_{_k}',_base,_grid,_inserted,_mapping,_window,_axes))


def payload(case, mode):
    rank = len(case.base)
    window_operand_axes = [d for d in range(rank) if d not in case.inserted]
    widths = [1] * rank
    for d, width in zip(window_operand_axes, case.window):
        widths[d] = width
    updates_shape = [None] * (len(case.grid) + len(case.window))
    for d, width in zip(case.window_axes, case.window):
        updates_shape[d] = width
    scatter_axes = [d for d in range(len(updates_shape)) if d not in case.window_axes]
    for d, width in zip(scatter_axes, case.grid):
        updates_shape[d] = width
    indices = np.zeros(case.grid + (len(case.mapping),), dtype=np.int32)
    for p, coord in enumerate(np.ndindex(case.grid)):
        for k, d in enumerate(case.mapping):
            maxstart = case.base[d] - widths[d]
            indices[coord + (k,)] = (p * 3 + k) % (maxstart + 1)
        if mode == 'edge' and case.mapping:
            if p % 4 == 0:
                indices[coord + (0,)] = -1
            elif p % 4 == 1:
                indices[coord + (0,)] = np.iinfo(np.int32).max
            else:
                indices[coord] = 0  # Duplicates and overlapping windows.
    base = np.arange(np.prod(case.base, dtype=int), dtype=np.float32).reshape(case.base)
    updates = (np.arange(np.prod(updates_shape, dtype=int), dtype=np.float32) % 17 - 8).reshape(updates_shape)
    return base, indices, updates


def reference(case, base, indices, updates):
    out = base.copy()
    window_operand_axes = [d for d in range(base.ndim) if d not in case.inserted]
    scatter_axes = [d for d in range(updates.ndim) if d not in case.window_axes]
    widths = [1] * base.ndim
    for d, width in zip(window_operand_axes, case.window):
        widths[d] = width
    for point in np.ndindex(case.grid):
        start = [0] * base.ndim
        for k, d in enumerate(case.mapping):
            start[d] = int(indices[point + (k,)])
        if any(s < 0 or s + w > n for s, w, n in zip(start, widths, base.shape)):
            continue
        for offset in np.ndindex(case.window):
            target = start.copy()
            ucoord = [0] * updates.ndim
            for d, value in zip(scatter_axes, point):
                ucoord[d] = value
            for od, ud, value in zip(window_operand_axes, case.window_axes, offset):
                target[od] += value
                ucoord[ud] = value
            out[tuple(target)] += updates[tuple(ucoord)]
    return out


def function(case):
    dims = jax.lax.ScatterDimensionNumbers(case.window_axes, case.inserted, case.mapping)
    return lambda b, i, u: jax.lax.scatter_add(b, i, u, dims, mode='drop')


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.name)
@pytest.mark.parametrize('mode', ['valid', 'edge'])
def test_scatter_add_semantics(case, mode):
    host = payload(case, mode)
    expected = reference(case, *host)
    for platform in ('cpu','npu'):
        # CPU XLA's ScatterSimplifier fails on a rank-0 operand with multiple
        # updates. Track it separately; the independent NumPy oracle above
        # still verifies the original scatter directly on NPU.
        if platform == 'cpu' and case.base == ():
            continue
        values = jax.device_put(host, jax.devices(platform)[0])
        actual = np.asarray(jax.jit(function(case))(*values))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('nested', [False, True])
def test_scatter_add_batch_dimensions(nested):
    case = CASES[6]  # Partial rows with nontrivial batch-dependent indices.
    host = payload(case, 'edge')
    b,i,u = [np.stack([v,v]) for v in host]
    i[1] = 0
    expected = np.stack([reference(case,b[k],i[k],u[k]) for k in range(2)])
    fn = jax.vmap(function(case))
    if nested:
        b,i,u = [np.stack([v,v]) for v in (b,i,u)]
        expected = np.stack([expected,expected])
        fn = jax.vmap(fn)
    for platform in ('cpu','npu'):
        actual = np.asarray(jax.jit(fn)(*jax.device_put((b,i,u),jax.devices(platform)[0])))
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@pytest.mark.xfail(strict=True, raises=jax.errors.JaxRuntimeError,
                   reason='CPU XLA ScatterSimplifier produces malformed scalar add')
def test_cpu_scalar_scatter_reference_bug():
    case = next(c for c in CASES if c.name == 'scalar_operand')
    host = payload(case, 'valid')
    try:
        actual = np.asarray(jax.jit(function(case))(
            *jax.device_put(host, jax.devices('cpu')[0])))
    except jax.errors.JaxRuntimeError as exc:
        assert 'Failed after scatter_simplifier' in str(exc)
        raise
    np.testing.assert_array_equal(actual, reference(case, *host))
