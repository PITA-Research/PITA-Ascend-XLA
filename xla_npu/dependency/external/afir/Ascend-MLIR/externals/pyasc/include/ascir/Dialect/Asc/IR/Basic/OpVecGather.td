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

#ifndef ASC_BASIC_OP_GATHER_TD
#define ASC_BASIC_OP_GATHER_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Other vector binary operations
//===----------------------------------------------------------------------===//

def AscendC_GatherbL0Op : VectorOp<"gatherb_l0", "Gatherb", [OpWithDstInterface]> {
    let description = "Gatherb elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$offset, AnyType:$repeatTimes,
                         AscendC_GatherRepeatParams:$repeatParams);
}

def AscendC_GatherL0Op : VectorOp<"gather_l0", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$srcOffset, AnyType:$srcBaseAddr,
                         AnyType:$mask, AnyType:$repeatTimes,
                         AnyType:$dstRepStride);
}

def AscendC_GatherL1Op : VectorOp<"gather_l1", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L1 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$dstOffset, AnyType:$dstBase,
                         Variadic<AnyType>:$mask, AnyType:$repeatTimes,
                         AnyType:$srcRepStride);
}

def AscendC_GatherL2Op : VectorOp<"gather_l2", "Gather", [OpWithDstInterface]> {
    let description = "Gather elements from source tensor (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                         AscendC_LocalTensor:$srcOffset, AnyType:$srcBaseAddr,
                         AnyType:$count);
}
#endif //ASC_BASIC_OP_GATHER_TD
