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

#ifndef ASC_BASIC_OP_FIXPIPE_TD
#define ASC_BASIC_OP_FIXPIPE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
 
def AscendC_FixpipeOp : APIOp<"fixpipe", "Fixpipe", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst, AscendC_LocalTensor:$src,
                    Optional<AscendC_LocalTensor>:$workspace,
                    AnyType:$intriParams);
}
#endif // ASC_BASIC_OP_FIXPIPE_TD
