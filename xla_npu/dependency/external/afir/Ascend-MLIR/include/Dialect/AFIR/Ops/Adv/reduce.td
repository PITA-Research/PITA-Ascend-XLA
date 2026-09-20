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

//===- reduce.td - AFIR reduce operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the reduce operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_REDUCE_OPS
#define AFIR_REDUCE_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class AFIR_UnaryReduceOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_UnaryOpBase<mnemonic, traits # [InferTensorTypeAdaptor], types, types, "Reduce", "Vector", 1> {
  let arguments = !con(
    (ins 
      TensorOf<types>:$input,
      I32Attr:$axis
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

def AFIR_MaxOp : AFIR_UnaryReduceOp<"max", [Pure], [F16, F32]>;
def AFIR_MinOp : AFIR_UnaryReduceOp<"min", [Pure], [F16, F32]>;
def AFIR_SumOp : AFIR_UnaryReduceOp<"sum", [Pure], [I32, F32]>;
def AFIR_MeanOp : AFIR_UnaryReduceOp<"mean", [Pure], [F32]>;
def AFIR_ProdOp : AFIR_UnaryReduceOp<"prod", [Pure], [F32]>;
def AFIR_AnyOp : AFIR_UnaryReduceOp<"any", [Pure], [F32]>;
def AFIR_AllOp : AFIR_UnaryReduceOp<"all", [Pure], [F32]>;

#endif // AFIR_REDUCE_OPS