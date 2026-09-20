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

//===- data.td - AFIR data operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the data operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_DATA_OPS
#define AFIR_DATA_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def AFIR_ScalarOp : AFIR_NullaryOpBase<"scalar", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Scalar", 1> {
  let arguments = !con(
    (ins 
      AnyAttrOf<[I8Attr, UI8Attr, I16Attr, UI16Attr, I32Attr, UI32Attr, I64Attr, UI64Attr, F16Attr, F32Attr]>:$value
    ),
    commonArgs
  );
}
def AFIR_IndexExprOp : AFIR_NullaryOpBase<"index_expr", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32], "Invalid", "Vector", 1>;
def AFIR_LoadOp : AFIR_UnaryOp<"load", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Load", "MTE2", 1>;
def AFIR_StoreOp : AFIR_UnaryOp<"store", [Pure], [I8, UI8, I16, UI16, I32, UI32, I64, F16, F32], "Store", "MTE3", 1>;

#endif // AFIR_DATA_OPS