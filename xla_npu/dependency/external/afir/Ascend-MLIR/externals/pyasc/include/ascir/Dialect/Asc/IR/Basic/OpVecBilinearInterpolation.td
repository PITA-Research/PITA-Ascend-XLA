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

#ifndef ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD
#define ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_BilinearInterpolationL0Op : VectorOp<"bilinearInterpolation_l0", "BilinearInterpolation", [OpWithDstInterface]> {
    let description = "AscendC::BilinearInterpolation is a vector binary operation (L0 API).\n";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$src0Offset, AscendC_LocalTensor:$src1,
                         AnyType:$mask, AnyType:$hRepeat, AnyType:$repeatMode,
                         AnyType:$dstBlkStride, AnyType:$vROffset, AnyType:$vRepeat,
                         AscendC_LocalTensor:$sharedTmpBuffer);
}

def AscendC_BilinearInterpolationL1Op : VectorOp<"bilinearInterpolation_l1", "BilinearInterpolation", [OpWithDstInterface]> {
    let description = "AscendC::BilinearInterpolation is a vector binary operation (L1 API).\n";
    let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                         AscendC_LocalTensor:$src0Offset, AscendC_LocalTensor:$src1,
                         Variadic<UI64>:$mask, AnyType:$hRepeat, AnyType:$repeatMode,
                         AnyType:$dstBlkStride, AnyType:$vROffset, AnyType:$vRepeat,
                         AscendC_LocalTensor:$sharedTmpBuffer);
}
#endif //ASC_BASIC_OP_VEC_BILINEARINTERPOLATION_TD
