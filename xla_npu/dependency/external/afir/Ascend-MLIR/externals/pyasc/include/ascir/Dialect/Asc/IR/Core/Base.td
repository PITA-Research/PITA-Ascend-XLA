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

#ifndef ASC_CORE_BASE_TD
#define ASC_CORE_BASE_TD

include "Dialect.td"
include "Core/Interfaces.td"

class AscendC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AscendC_Dialect, name, [MemRefElementTypeInterface] # traits> {
  let mnemonic = typeMnemonic;
}

class AscendC_BaseQueueType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseQueueTypeInterface] # traits> {
}

class AscendC_BaseLoadDataParamsType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseLoadDataParamsTypeInterface] # traits> {
}

class AscendC_BaseTensorType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseTensorTypeInterface,
        DeclareTypeInterfaceMethods<ShapedTypeInterface>] # traits> {
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::ascendc::BaseTensorType":$baseType)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif // ASC_CORE_BASE_TD
