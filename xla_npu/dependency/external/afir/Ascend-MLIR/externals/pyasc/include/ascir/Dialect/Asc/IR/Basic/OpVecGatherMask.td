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

#ifndef ASC_BASIC_OP_GATHER_MASK_TD
#define ASC_BASIC_OP_GATHER_MASK_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// GatherMask/GetGatherMaskRemainCount operation definition
//===----------------------------------------------------------------------===//

def AscendC_GatherMaskOp : APIOp<"gather_mask", "GatherMask"> {
  let summary = "Call `AscendC::GatherMask` function";
  let description = "Gather elements from source tensor using mask pattern";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                  AnyType:$src1Pattern,
                  AnyType:$reduceMode, AnyType:$mask,
                  AscendC_GatherMaskParams:$params, AnyType:$rsvdCnt,
                  DefaultValuedAttr<AscendC_GatherMaskModeAttr, 
                  "::mlir::ascendc::GatherMaskMode::DEFAULT">:$gatherMaskMode);
}

def AscendC_GetGatherMaskRemainCountOp : APIOp<"get_gather_mask_remain_count", "GetGatherMaskRemainCount",[AscFunc]> {
  let summary = "Call `AscendC::GetGatherMaskRemainCount` function";
  let description = "Get remaining count from last GatherMask operation";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}


#endif //ASC_BASIC_OP_GATHER_MASK_TD
