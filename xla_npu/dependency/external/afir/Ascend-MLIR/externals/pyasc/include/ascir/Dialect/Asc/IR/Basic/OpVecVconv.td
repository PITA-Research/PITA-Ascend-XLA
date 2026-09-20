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

#ifndef ASC_BASIC_OP_VEC_VCONV_TD
#define ASC_BASIC_OP_VEC_VCONV_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_CastL0Op : VectorOp<"cast_l0", "Cast", [OpWithDstInterface]> {
  let description = "Convert the input tensor to the specified data type (L0 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, AnyType:$mask,
                       AnyType:$repeatTimes, AnyType:$dstBlkStride,
                       AnyType:$srcBlkStride, AnyType:$dstRepStride,
                       AnyType:$srcRepStride);
}

def AscendC_CastL1Op : VectorOp<"cast_l1", "Cast", [OpWithDstInterface]> {
  let description = "Convert the input tensor to the specified data type (L1 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, Variadic<UI64>:$mask,
                       AnyType:$repeatTimes, AnyType:$dstBlkStride,
                       AnyType:$srcBlkStride, AnyType:$dstRepStride,
                       AnyType:$srcRepStride);
}

def AscendC_CastL2Op : VectorOp<"cast_l2", "Cast", [OpWithDstInterface]> {
  let description = "Convert the input tensor to the specified data type (L2 API)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_RoundModeAttr:$roundMode, AnyType:$calCount);
}

def AscendC_CastDeqL0Op : VectorOp<"cast_deq_l0", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L0 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask, UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_CastDeqL1Op : VectorOp<"cast_deq_l1", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L1 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, Variadic<UI64>:$mask,
                       AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                       UnitAttr:$isSetMask, UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_CastDeqL2Op : VectorOp<"cast_deq_l2", "CastDeq"> {
  let description = "Quantize and convert the input tensor to the specified data type (L2 API)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$calCount,
                       UnitAttr:$isVecDeq, UnitAttr:$halfBlock);
}

def AscendC_SetDeqScaleOp : APIOp<"set_deq_scale", "SetDeqScale", [AttrSizedOperandSegments, AscFunc]> {
  let arguments = (ins AnyType:$scale, Optional<I16>:$offset, Optional<AnyType>:$signMode);
  let assemblyFormat = [{
    $scale attr-dict (`,` $offset^)? (`,` $signMode^)? `:`
    qualified(type($scale)) (`,` type($offset)^)? (`,` type($signMode)^)?
  }];
}

defm AddReluCast : BinaryCastL012Op<"add_relu_cast", "AddReluCast">;
defm SubReluCast : BinaryCastL012Op<"sub_relu_cast", "SubReluCast">;

#endif // ASC_BASIC_OP_VEC_VCONV_TD
