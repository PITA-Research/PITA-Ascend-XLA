/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ASC_FWK_TQUE_TD
#define ASC_FWK_TQUE_TD
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// TQueBind operations 
//===----------------------------------------------------------------------===//

def AscendC_TQueBindAllocTensorOp : APIOp<"que_bind.alloc_tensor", "AllocTensor"> {
  let summary = "Allocate tensor on queue wrapped buffer";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue attr-dict `:` qualified(type($queue)) `,` qualified(type($tensor))
  }];
}

def AscendC_TQueBindAllocTensorInPlaceOp : APIOp<"que_bind.alloc_tensor_in_place", "AllocTensor"> {
  let summary = "Allocate tensor on queue wrapped buffer";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue, AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue `,` $tensor attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}

def AscendC_TQueBindDequeTensorOp : APIOp<"que_bind.deque_tensor", "DeQue"> {
  let summary = "Pop tensor out from queue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue attr-dict `:` qualified(type($queue)) `,` qualified(type($tensor))
  }];
}

def AscendC_TQueBindDequeTensorInPlaceOp : APIOp<"que_bind.deque_tensor_in_place", "DeQue"> {
  let summary = "Pop tensor out from queue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue, AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue `,` $tensor attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}

def AscendC_TQueBindDequeTensorPosOp : APIOp<"que_bind.deque_tensor_pos", "DeQue"> {
  let summary = "Pop tensor out from queue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue,
                       AscendC_TPositionAttr:$srcUserPos, AscendC_TPositionAttr:$dstUserPos);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue `,` $srcUserPos `,` $dstUserPos attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}

def AscendC_TQueBindDestroyOp : APIOp<"que_bind.destroy", "Destroy", [AscMemberFunc]> {
  let summary = "Push tensor back to queue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let assemblyFormat = [{
    $queue attr-dict `:` qualified(type($queue))
  }];
}

def AscendC_TQueBindEnqueTensorOp : APIOp<"que_bind.enque_tensor", "EnQue", [AscMemberFunc]> {
  let summary = "Push tensor back to queue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue,
                       AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue `,` $tensor attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}

def AscendC_TQueBindEnqueTensorPosOp : APIOp<"que_bind.enque_tensor_pos", "EnQue"> {
  let summary = "Push tensor back to queue with position";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue, AscendC_LocalTensor:$tensor, 
                       AscendC_TPositionAttr:$srcUserPos, AscendC_TPositionAttr:$dstUserPos);
  let assemblyFormat = [{
    $queue `,` $tensor `,` $srcUserPos `,` $dstUserPos attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}
  
def AscendC_TQueBindFreeTensorOp : APIOp<"que_bind.free_tensor", "FreeTensor", [AscMemberFunc]> {
  let summary = "Release buffer held by tensor";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue,
                       AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $queue `,` $tensor attr-dict `:` qualified(type($queue)) `,`
    qualified(type($tensor))
  }];
}

def AscendC_TQueBindGetTensorCountInQueOp : APIOp<"que_bind.get_tensor_count_in_que", "GetTensorCountInQue", [AscMemberFunc]> {
  let summary = "GetTensorCountInQue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $queue  attr-dict `:` qualified(type($queue)) `,` type($value)
  }];
}

def AscendC_TQueBindHasIdleBufferOp : APIOp<"que_bind.has_idle_buffer", "HasIdleBuffer", [AscMemberFunc]> {
  let summary = "HasIdleBuffer";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $queue  attr-dict `:` qualified(type($queue)) `,` type($value)
  }];
}

def AscendC_TQueBindHasTensorInQueOp : APIOp<"que_bind.has_tensor_in_que", "HasTensorInQue", [AscMemberFunc]> {
  let summary = "TQueBind HasTensorInQue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $queue  attr-dict `:` qualified(type($queue)) `,` type($value)
  }];
}

def AscendC_TQueBindVacantInQueOp : APIOp<"que_bind.vacant_in_que", "VacantInQue", [AscMemberFunc]> {
  let summary = "VacantInQue";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $queue  attr-dict `:` qualified(type($queue)) `,` type($value)
  }];
}

def AscendC_ToQueBindOp : AscendC_Op<"to_que_bind", [Pure]> {
  let summary = "Obtain AscendC::TQueBind from its derivative class instances";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$source);
  let results = (outs AscendC_QueBind:$result);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `,` qualified(type($result))
  }];
}

def AscendC_TQueBindInitBufHandleOp : APIOp<"que_bind.init_buf_handle", "InitBufHandle"> {
  let summary = "TQueBind InitBufHandle";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue, AnyType:$bufPool,
                       UI32:$index, AnyType:$bufhandle, I32:$curPoolAddr, UI32:$len);
  let assemblyFormat = [{
    $queue `,` $bufPool `,` $index `,` $bufhandle `,` $curPoolAddr `,` $len attr-dict `:` 
    qualified(type($queue)) `,` qualified(type($bufPool)) `,` qualified(type($index)) `,`
    qualified(type($bufhandle)) `,` qualified(type($curPoolAddr)) `,` qualified(type($len))
  }];
}

def AscendC_TQueBindInitStartBufHandleOp : APIOp<"que_bind.init_start_buf_handle", "InitStartBufHandle"> {
  let summary = "TQueBind InitStartBufHandle";
  let arguments = (ins AscendC_BaseQueueTypeInterface:$queue,
                       AnyType:$startBufhandle, UI8:$num, UI32:$len);
  let assemblyFormat = [{
    $queue `,` $startBufhandle `,` $num `,` $len attr-dict `:` qualified(type($queue)) `,`
    qualified(type($startBufhandle)) `,` qualified(type($num)) `,` qualified(type($len))
  }];
}


//===----------------------------------------------------------------------===//
// Queue operations (TQue)
//===----------------------------------------------------------------------===//

def AscendC_QueueOp : AscendC_Op<"queue", [AscConstructor]> {
  let summary = "Instantiate queue";
  let results = (outs AscendC_Queue:$queue);
  let assemblyFormat = "attr-dict `:` type($queue)";
}

//===----------------------------------------------------------------------===//
// TQueBind operations (TQueBind)
//===----------------------------------------------------------------------===//

def AscendC_QueBindOp : AscendC_Op<"que_bind", [AscConstructor]> {
  let summary = "Instantiate QueBind";
  let results = (outs AscendC_QueBind:$que_bind);
  let assemblyFormat = "attr-dict `:` type($que_bind)";
}

#endif
