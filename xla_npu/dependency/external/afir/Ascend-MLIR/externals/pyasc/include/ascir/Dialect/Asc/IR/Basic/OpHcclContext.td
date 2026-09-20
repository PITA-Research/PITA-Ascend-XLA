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

#ifndef ASC_BASIC_OP_HCCL_CONTEXT_TD
#define ASC_BASIC_OP_HCCL_CONTEXT_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_SetHcclContextOp : APIOp<"set_hccl_context", "SetHcclContext", [AscFunc]> {
  let arguments = (ins AnyType:$index, AnyType:$context);
  let paramTypeLists = [4, 0];
  let assemblyFormat = "$index `,` $context attr-dict `:` type($index) `,` type($context)";
}

def AscendC_GetHcclContextOp : APIOp<"get_hccl_context", "GetHcclContext", [AscFunc]> {
  let arguments = (ins AnyType:$index);
  let results = (outs AnyType:$context);
  let paramTypeLists = [4];
  let assemblyFormat = "$index attr-dict `:` type($index) `,` type($context)";
}

#endif // ASC_BASIC_OP_HCCL_CONTEXT_TD
