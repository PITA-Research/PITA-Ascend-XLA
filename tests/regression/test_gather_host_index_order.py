"""Host-expanded gather indices must wait for their device producer."""
import jax
import jax.numpy as jp
import numpy as np

def test_partial_window_gather_waits_for_device_indices():
    def fn(x,sign,data):
        index=(jp.sum(x@(x*sign))>0).astype(jp.int32)
        indices=jp.stack([index,index+1])[:,None]
        dims=jax.lax.GatherDimensionNumbers(offset_dims=(1,),collapsed_slice_dims=(),start_index_map=(0,))
        return indices,jax.lax.gather(data,indices,dims,(2,),mode='clip')
    x=np.ones((4096,4096),np.float32)
    data=np.array([3.,7.,11.,13.],np.float32)
    reference={}
    for platform in ['cpu','npu']:
        device=jax.devices(platform)[0]
        xx,dd=jax.device_put((x,data),device)
        np.testing.assert_array_equal(np.asarray(xx).view(np.uint32),x.view(np.uint32))
        compiled=jax.jit(fn).lower(xx,jax.device_put(np.float32(1),device),dd).compile()
        for i in range(2 if platform=='cpu' else 20):
            sign=1 if i%2==0 else -1
            indices,actual=map(np.asarray,compiled(xx,jax.device_put(np.float32(sign),device),dd))
            np.testing.assert_array_equal(indices,np.array([[int(sign>0)],[int(sign>0)+1]],np.int32))
            if platform=='cpu':
                reference[sign]=actual
            else:
                np.testing.assert_array_equal(actual.view(np.uint32),reference[sign].view(np.uint32))
