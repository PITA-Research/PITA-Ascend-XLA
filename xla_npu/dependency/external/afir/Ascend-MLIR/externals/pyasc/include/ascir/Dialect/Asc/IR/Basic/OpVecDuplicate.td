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

#ifndef ASC_BASIC_OP_DUPLICATE_TD
#define ASC_BASIC_OP_DUPLICATE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_DuplicateL0Op : VectorOp<"duplicate_l0", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L0 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar, AnyType:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlockStride,
                         AnyType:$dstRepeatStride, UnitAttr:$isSetMask);
}

def AscendC_DuplicateL1Op : VectorOp<"duplicate_l1", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L1 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar, Variadic<UI64>:$mask,
                         AnyType:$repeatTimes, AnyType:$dstBlockStride,
                         AnyType:$dstRepeatStride, UnitAttr:$isSetMask);
}

def AscendC_DuplicateL2Op : VectorOp<"duplicate_l2", "Duplicate", [OpWithDstInterface]> {
    let description = "Duplicate elements from source tensor (L2 API)";
    let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$scalar,
                         AnyType:$calCount);
}
#endif //ASC_BASIC_OP_DUPLICATE_TD
