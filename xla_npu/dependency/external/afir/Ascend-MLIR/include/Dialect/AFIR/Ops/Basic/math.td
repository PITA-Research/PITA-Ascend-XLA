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

//===- math.td - AFIR math operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the math operations for the AFIR dialect, organized into
// three categories: unary, binary, and compare operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_MATH_OPS
#define AFIR_MATH_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class AFIR_UnaryMathOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_UnaryOp<mnemonic, traits # [SameOperandsAndResultType], types, "Elementwise", "Vector", 1> {
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

class AFIR_BinaryMathOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_BinaryOp<mnemonic, traits # [InferTensorTypeAdaptor], types, "Elementwise", "Vector", 1> {
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

class AFIR_BinaryHomoMathOp<string mnemonic, list<Trait> traits = [], list<Type> inputTypes, list<Type> resultTypes> :
    AFIR_BinaryHomoOp<mnemonic, traits # [InferTensorTypeAdaptor], inputTypes, resultTypes, "Elementwise", "Vector", 1>;

class AFIR_TernaryMathOp<string mnemonic, list<Trait> traits = [], list<Type> types> :
    AFIR_TernaryOp<mnemonic, traits # [InferTensorTypeAdaptor], types, "Elementwise", "Vector", 1>;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

def AFIR_AbsOp : AFIR_UnaryMathOp<"abs", [Pure], [F16, F32]>;
def AFIR_ExpOp : AFIR_UnaryMathOp<"exp", [Pure], [F16, F32]>;
def AFIR_LnOp : AFIR_UnaryMathOp<"ln", [Pure], [F16, F32]>;
def AFIR_SqrtOp : AFIR_UnaryMathOp<"sqrt", [Pure], [F16, F32]>;
def AFIR_RsqrtOp : AFIR_UnaryMathOp<"rsqrt", [Pure], [F16, F32]>;
def AFIR_ReciprocalOp : AFIR_UnaryMathOp<"reciprocal", [Pure], [F16, F32]>;
def AFIR_ErfOp : AFIR_UnaryMathOp<"erf", [Pure], [F16, F32]>;
def AFIR_TanhOp : AFIR_UnaryMathOp<"tanh", [Pure], [F16, F32]>;
def AFIR_ReluOp : AFIR_UnaryMathOp<"relu", [Pure], [I32, F16, F32]>;
def AFIR_NegOp : AFIR_UnaryMathOp<"neg", [Pure], [I16, I32, F16, F32]>;
def AFIR_SigmoidOp : AFIR_UnaryMathOp<"sigmoid", [Pure], [F16, F32]>;
def AFIR_LogicalNotOp : AFIR_UnaryMathOp<"logical_not", [Pure], [UI8, I16, I32, F16, F32]>;

def AFIR_IsnanOp : AFIR_UnaryOpBase<"isnan", [Pure], [F16, F32], [UI8], "Elementwise", "Vector", 1>;
def AFIR_IsFiniteOp : AFIR_UnaryOpBase<"isfinite", [Pure], [F16, F32], [UI8], "Elementwise", "Vector", 1>;

def AFIR_LeakyReluOp : AFIR_UnaryMathOp<"leaky_relu", [Pure], [F16, F32]> {
  let arguments = !con(
    (ins 
      TensorOf<[F16, F32]>:$input,
      F32Attr:$negative_slop
    ),
    commonArgs
  );
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

def AFIR_AddOp : AFIR_BinaryMathOp<"add", [Pure], [F16, F32, I16, I32]>;
def AFIR_SubOp : AFIR_BinaryMathOp<"sub", [Pure], [F16, F32, I16, I32]>;
def AFIR_MulOp : AFIR_BinaryMathOp<"mul", [Pure], [F16, F32, I16, I32]>;
def AFIR_DivOp : AFIR_BinaryMathOp<"div", [Pure], [F16, F32]>;
def AFIR_MinimumOp : AFIR_BinaryMathOp<"minimum", [Pure], [I16, I32, F16, F32]>;
def AFIR_MaximumOp : AFIR_BinaryMathOp<"maximum", [Pure], [I16, I32, F16, F32]>;
def AFIR_TrueDivOp : AFIR_BinaryMathOp<"truediv", [Pure], [F16, F32]>;
def AFIR_PowOp : AFIR_BinaryMathOp<"pow", [Pure], [I32, F16, F32]>;
def AFIR_BitwiseAndOp : AFIR_BinaryMathOp<"bitwise_and", [Pure], [UI8, I16, UI16, I32]>;
def AFIR_FloorDivOp : AFIR_BinaryMathOp<"floor_div", [Pure], [I16, I32]>;
def AFIR_GeluOp : AFIR_BinaryMathOp<"gelu", [Pure], [F16, F32]>;
def AFIR_SignOp : AFIR_BinaryMathOp<"sign", [Pure], [I32, F16, F32]>;
def AFIR_LogicalOrOp : AFIR_BinaryHomoMathOp<"logical_or", [Pure], [UI8, I16, I32, F16, F32], [UI8]>;
def AFIR_LogicalAndOp : AFIR_BinaryHomoMathOp<"logical_and", [Pure], [UI8, I16, I32, F16, F32], [UI8]>;

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

def AFIR_GeOp : AFIR_BinaryHomoMathOp<"ge", [Pure], [F16, F32, F64], [UI8]>;
def AFIR_EqOp : AFIR_BinaryHomoMathOp<"eq", [Pure], [I32, F16, F32, I64], [UI8]>;
def AFIR_NeOp : AFIR_BinaryHomoMathOp<"ne", [Pure], [F16, F32, F64], [UI8]>;
def AFIR_GtOp : AFIR_BinaryHomoMathOp<"gt", [Pure], [I32, F16, F32, I64], [UI8]>;
def AFIR_LeOp : AFIR_BinaryHomoMathOp<"le", [Pure], [F16, F32, I64], [UI8]>;
def AFIR_LtOp : AFIR_BinaryHomoMathOp<"lt", [Pure], [F16, F32], [UI8]>;

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//
def AFIR_ClipByValue : AFIR_TernaryMathOp<"clip_by_value", [Pure], [F16, F32]> {
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

#endif // AFIR_MATH_OPS
