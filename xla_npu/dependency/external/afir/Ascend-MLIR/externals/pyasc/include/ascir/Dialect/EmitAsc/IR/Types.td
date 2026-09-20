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

#ifndef EMITASC_TYPES_TD
#define EMITASC_TYPES_TD

include "Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class EmitAsc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EmitAsc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PyStruct : EmitAsc_Type<"PyStruct", "py_struct", [MemRefElementTypeInterface]> {
  let parameters = (ins "StringAttr":$nameAttr, "ArrayAttr":$typesAttr, "ArrayAttr":$namesAttr);
  let assemblyFormat = [{  `<` $nameAttr `,` $typesAttr `,` $namesAttr `>` }];
}

#endif // EMITASC_TYPES_TD
