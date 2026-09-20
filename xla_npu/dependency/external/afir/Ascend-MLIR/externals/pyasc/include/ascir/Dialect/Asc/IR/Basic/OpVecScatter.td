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

#ifndef ASC_BASIC_OP_SCATTER_TD
#define ASC_BASIC_OP_SCATTER_TD

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

def AscendC_ScatterL0Op : VectorOp<"scatter_l0", "Scatter", [AscFunc, OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L0 API). ";
  let arguments = (ins 
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    AnyType:$mask,
    AnyType:$repeatTimes,
    AnyType:$srcRepStride
  );
}

def AscendC_ScatterL1Op : VectorOp<"scatter_l1", "Scatter", [OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L1 API). ";
  let arguments = (ins 
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    Variadic<AnyType>:$mask,
    AnyType:$repeatTimes,
    AnyType:$srcRepStride
  );
}

def AscendC_ScatterL2Op : VectorOp<"scatter_l2", "Scatter", [AscFunc, OpWithDstInterface]> {
  let description = "Scatter elements from source tensor to destination tensor (L2 API). ";
  let arguments = (ins 
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$dstOffset,
    AnyType:$dstBase,
    AnyType:$count
  );
}
#endif //ASC_BASIC_OP_SCATTER_TD