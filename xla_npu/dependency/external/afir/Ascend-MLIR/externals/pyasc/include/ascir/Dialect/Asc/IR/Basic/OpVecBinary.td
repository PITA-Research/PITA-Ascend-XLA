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

#ifndef ASC_BASIC_OP_VEC_BINARY_TD
#define ASC_BASIC_OP_VEC_BINARY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

defm Add : BinaryTemplateL0123Op<"add", "Add", "operator+">;
defm AddDeqRelu : BinaryCastL012Op<"add_deq_relu", "AddDeqRelu">;
defm AddRelu : BinaryTemplateL012Op<"add_relu", "AddRelu">;
defm And : BinaryTemplateL012Op<"and", "And">;
defm Div : BinaryTemplateL0123Op<"div", "Div", "operator/">;
def FusedAbsSubL2Op : BinaryL2Op<"fused_abs_sub_l2", "FusedAbsSub">;
def FusedExpSubL2Op : BinaryL2Op<"fused_exp_sub_l2", "FusedExpSub">;
defm FusedMulAdd : BinaryTemplateL012Op<"fused_mul_add", "FusedMulAdd">;
defm FusedMulAddRelu : BinaryTemplateL012Op<"fused_mul_add_relu", "FusedMulAddRelu">;
defm Max : BinaryTemplateL012Op<"max", "Max">;
defm Min : BinaryTemplateL012Op<"min", "Min">;
defm Mul : BinaryTemplateL0123Op<"mul", "Mul", "operator*">;
defm MulAddDst : BinaryCastL012Op<"mul_add_dst", "MulAddDst">;
defm Or : BinaryTemplateL012Op<"or", "Or">;
def PreluL2Op : BinaryL2Op<"prelu_l2", "Prelu">;
defm Sub : BinaryTemplateL0123Op<"sub", "Sub", "operator-">;
defm SubRelu : BinaryTemplateL012Op<"sub_relu", "SubRelu">;

#endif //ASC_BASIC_OP_VEC_BINARY_TD
