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

#ifndef ASC_ADV_NORMALIZATION_TD
#define ASC_ADV_NORMALIZATION_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_RmsNormOp : APIOp<"rmsnorm", "RmsNorm"> {
  let arguments = (ins UnitAttr:$basicBlock, AscendC_LocalTensor:$dst,
                   AscendC_LocalTensor:$src, AscendC_LocalTensor:$gamma,
                   AnyType:$epsilon, AscendC_RmsNormTiling:$tiling,
                   Optional<AscendC_LocalTensor>:$sharedTmpBuffer);
}
#endif //ASC_ADV_NORMALIZATION_TD
