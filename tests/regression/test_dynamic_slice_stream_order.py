"""A host-lowered slice must wait for its device-computed index.

The large producer makes the missing stream dependency observable. The
predicate itself is exact; this is not a floating-point tolerance test.
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest


@pytest.mark.parametrize('op', ['slice', 'update'])
def test_dynamic_slice_waits_for_device_index(op):
    x = np.ones((4096,4096), np.float32)
    data = np.array([3.,7.], np.float32)
    def fn(x, sign, data):
        index = (jp.sum(x @ (x * sign)) > 0).astype(jp.int32)
        if op == 'slice':
            result = jax.lax.dynamic_slice(data, (index,), (1,))
        else:
            # An intermediate base can be reused in-place by buffer assignment.
            result = jax.lax.dynamic_update_slice(data + sign, jp.array([11.],jp.float32), (index,))
        return index, result
    reference = {}
    for platform in ['cpu','npu']:
        device = jax.devices(platform)[0]
        xx, dd = jax.device_put((x,data),device)
        np.testing.assert_array_equal(np.asarray(xx).view(np.uint32), x.view(np.uint32))
        compiled = jax.jit(fn).lower(xx,jax.device_put(np.float32(1),device),dd).compile()
        for i in range(2 if platform == 'cpu' else 20):
            sign = np.float32(1 if i % 2 == 0 else -1)
            index, chosen = map(np.asarray,compiled(xx,jax.device_put(sign,device),dd))
            assert int(index) == int(sign > 0)
            if platform == 'cpu':
                reference[int(sign)] = chosen
            else:
                np.testing.assert_array_equal(chosen.view(np.uint32),
                                              reference[int(sign)].view(np.uint32))
