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

#ifndef ASC_FWK_TPIPE_TD
#define ASC_FWK_TPIPE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// Pipe operations (TPipe)
//===----------------------------------------------------------------------===//

def AscendC_PipeOp : AscendC_Op<"pipe", [AscConstructor]> {
  let summary = "Instantiate pipe";
  let results = (outs AscendC_Pipe:$pipe);
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getType<PipeType>());
    }]>,
  ];
}

def AscendC_TPipeInitBufferOp : APIOp<"pipe.init_buffer", "InitBuffer", [AscMemberFunc]> {
  let summary = "Initialize raw buffer";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_TBuf:$buffer,
                       AnyType:$length);
  let assemblyFormat = [{
    $pipe `,` $buffer `,` $length attr-dict `:` qualified(type($buffer)) `,`
    type($length)
  }];
}

def AscendC_TPipeInitQueueOp : APIOp<"pipe.init_queue", "InitBuffer", [AscMemberFunc]> {
  let summary = "Initialize queue";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_BaseQueueTypeInterface:$queue, AnyType:$num,
                       AnyType:$length);
  let assemblyFormat = [{
    $pipe `,` $queue `,` $num `,` $length attr-dict `:` qualified(type($queue))
    `,` type($num) `,` type($length)
  }];
}

def AscendC_TPipeInitOp : APIOp<"pipe.init", "Init", [AscMemberFunc]> {
  let summary = "TPipe Initialize";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeTQueBindDestroyOp : APIOp<"pipe.destroy", "Destroy", [AscMemberFunc]> {
  let summary = "TPipe Destroy";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeResetOp : APIOp<"pipe.reset", "Reset", [AscMemberFunc]> {
  let summary = "TPipe Reset";
  let arguments = (ins AscendC_Pipe:$pipe);
}

def AscendC_TPipeAllocEventIDOp : APIOp<"pipe.alloc_event_id", "AllocEventID", [AscMemberFunc]> {
  let summary = "TPipe AllocEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_HardEventAttr:$event);
  let results = (outs AnySignlessIntegerOrIndex:$event_id);
  let paramTypeLists = [0, 3];
  let assemblyFormat = "$pipe `,` $event  attr-dict `:` type($event_id)";
}

def AscendC_TPipeReleaseEventIDOp : APIOp<"pipe.release_event_id", "ReleaseEventID", [AscMemberFunc]> {
  let summary = "TPipe ReleaseEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AnyType:$id, AscendC_HardEventAttr:$event);
  let paramTypeLists = [0, 0, 3];
}

def AscendC_TPipeFetchEventIDOp : APIOp<"pipe.fetch_event_id", "FetchEventID", [AscMemberFunc]> {
  let summary = "TPipe FetchEventID";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_HardEventAttr:$event);
  let results = (outs AnySignlessIntegerOrIndex:$event_id);
  let paramTypeLists = [0, 3];
  let assemblyFormat = "$pipe `,` $event  attr-dict `:` type($event_id)";
}

def AscendC_TPipeGetBaseAddrOp : APIOp<"pipe.get_base_addr", "GetBaseAddr", [AscMemberFunc]> {
  let summary = "TPipe GetBaseAddr";
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_TPositionAttr:$logicPos);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$pipe `,` $logicPos  attr-dict `:` type($value)";
}

def AscendC_TPipeInitBufPoolOp : APIOp<"pipe.init_buf_pool", "InitBufPool", [AscMemberFunc]> {
  let summary = "TPipe InitBufPool";
  let arguments = (ins AscendC_Pipe:$pipe, AnyType:$bufPool, AnyType:$len, Optional<AnyType>:$shareBuf);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    `(` $pipe `,` $bufPool `,` $len `)` 
    attr-dict (` ` `,` ` ` $shareBuf^)?
    `:` type($pipe) `,` type($bufPool) `,` type($len)
    (` ` `,` ` ` type($shareBuf)^)? `->` type($value)
  }];
}

#endif
