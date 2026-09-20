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

//===- concat.td - AFIR concat operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the concat operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONCAT_OPS
#define AFIR_CONCAT_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def AFIR_ConcatOp : AFIR_VariadicOp<"concat", [Pure, InferTensorTypeAdaptor], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Concat", "Vector", 1> {
  let arguments = !con(
    (ins 
      Variadic<TensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, F16, F32]>>:$inputs,
      I32Attr:$concat_axis
    ),
    commonArgs
  );
  let extraClassDeclaration = [{
    /// Returns when two result types are compatible for this op; method used by
    /// InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
  let extraClassDefinition = [{
    bool $cppClass::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size() || l.size() != 1) return false;
      return succeeded(verifyCompatibleShape(l[0], r[0]));
    }
  }];
}

#endif // AFIR_CONCAT_OPS