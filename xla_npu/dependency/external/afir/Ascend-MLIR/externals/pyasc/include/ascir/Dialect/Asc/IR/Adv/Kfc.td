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

#ifndef ASC_ADV_KFC_TD
#define ASC_ADV_KFC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


def AscendC_KfcInitOp : APIOp<"kfc_server.init", "Init"> {
  let arguments = (ins AscendC_KfcServer:$server, AnyType:$workspace);
}

def AscendC_KfcIsRunOp : APIOp<"kfc_server.is_run", "isRun"> {
  let arguments = (ins AscendC_KfcServer:$server);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$server attr-dict `:` type($value)";
}

def AscendC_KfcInitObjOp : APIOp<"kfc_server.init_obj", "InitObj"> {
  let arguments = (ins AscendC_KfcServer:$server, AscendC_Pipe:$pipe,
                    Variadic<AnyType>:$ops);
}

def AscendC_KfcRunOp : APIOp<"kfc_server.run", "Run"> {
  let arguments = (ins AscendC_KfcServer:$server, AscendC_Matmul:$matmul);
}

def AscendC_KfcQuitOp : APIOp<"kfc_server.quit", "Quit"> {
  let arguments = (ins AscendC_KfcServer:$server);
}
#endif //ASC_ADV_KFC_TD
