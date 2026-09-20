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

#ifndef ASC_BASIC_OP_SCALAR_TD
#define ASC_BASIC_OP_SCALAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_ScalarCastOp : APIOp<"scalar_cast", "ScalarCast", []> {
  let description = "Convert the data type of a scalar value.";
  let arguments = (ins
    AnyType:$valueIn,                 
    TypeAttr:$dtype,                  
    AscendC_RoundModeAttr:$roundMode  
  );
  let results = (outs AnyType:$valueOut);
  let assemblyFormat = "$valueIn attr-dict `:` type($valueIn) `->` type($valueOut)";
}

def AscendC_ScalarGetSFFValueOp : APIOp<"scalar_get_sff_value", "ScalarGetSFFValue", [AscFunc]> {
    let description = "Get the position of the first 0 or 1 (from LSB) in a uint64_t value. Returns -1 if not found.";
    let arguments = (ins 
        UI64:$valueIn, 
        AnyInteger:$countValue                  
    );
    let results = (outs AnySignlessIntegerOrIndex:$position); 
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($position)";
    let paramTypeLists = [0, 4]; 
}

#endif //ASC_BASIC_OP_SCALAR_TD
