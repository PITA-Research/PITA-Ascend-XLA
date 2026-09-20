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

#ifndef ASC_CORE_INTERFACES_TD
#define ASC_CORE_INTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class AscendC_TypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::ascendc";
}

def AscendC_BaseQueueTypeInterface : AscendC_TypeInterface<"BaseQueueType"> {
  let summary = "base queue type";
}

def AscendC_BaseTensorTypeInterface
    : AscendC_TypeInterface<"BaseTensorType", [ShapedTypeInterface]> {
  let summary = "base tensor type";
}

def AscendC_BaseLoadDataParamsTypeInterface
    : AscendC_TypeInterface<"BaseLoadDataParamsType"> {
  let summary = "base load data params type";
}

#endif // ASC_CORE_INTERFACES_TD
