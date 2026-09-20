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

#ifndef ASC_BASIC_OP_BLOCKSYNC_TD
#define ASC_BASIC_OP_BLOCKSYNC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data Sync Barrier Op
//===----------------------------------------------------------------------===//

def AscendC_DataSyncBarrierOp
    : APIOp<"data_sync_barrier", "DataSyncBarrier", [AscFunc]> {
  let description = "Synchronize data accesses by waiting for specified memory operations, "
                    "controlled by the MemDsbT parameter.";
  let arguments = (ins
    AscendC_MemDsbAttr:$memDsbType
  );
  let paramTypeLists = [3];
}

//===----------------------------------------------------------------------===//
// Synchronization operations
//===----------------------------------------------------------------------===//

def PipeBarrierOp : APIOp<"pipe_barrier", "PipeBarrier"> {
  let arguments = (ins AscendC_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
  let hasCanonicalizeMethod = 1;
}

def SetFlagOp : APIOp<"set_flag", "SetFlag", [AscFunc]> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
  let paramTypeLists = [3, 0];
}

def WaitFlagOp : APIOp<"wait_flag", "WaitFlag"> {
  let arguments = (ins AscendC_HardEventAttr:$event, AnyType:$eventId);
  let assemblyFormat = "$event `,` $eventId attr-dict `:` type($eventId)";
}
#endif // ASC_BASIC_OP_BLOCKSYNC_TD
