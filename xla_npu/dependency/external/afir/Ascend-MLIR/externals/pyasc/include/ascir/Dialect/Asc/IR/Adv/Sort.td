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

#ifndef ASC_ADV_SORT_TD
#define ASC_ADV_SORT_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Concat Op
//===----------------------------------------------------------------------===//

def AscendC_ConcatOp
    : APIOp<"concat", "Concat", [AscFunc]> {
  let description = "Preprocess source tensor data and merge it into the target tensor concat before sort operation.";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AscendC_LocalTensor:$tmp,
    AnyType:$repeatTime
  );
}

//===----------------------------------------------------------------------===//
// Extract Op
//===----------------------------------------------------------------------===//

def AscendC_ExtractOp
    : APIOp<"extract", "Extract", [AscFunc]> {
  let description = "Extract sorted values and indices from source tensor after sort operation.";
  let arguments = (ins
    AscendC_LocalTensor:$dstValue,
    AscendC_LocalTensor:$dstIndex,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime
  );
}

#endif // ASC_ADV_SORT_TD
