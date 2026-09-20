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

#ifndef AFIR_BASE
#define AFIR_BASE

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Core/Attrs.td"

//===----------------------------------------------------------------------===//
// Common properties of each operator
//===----------------------------------------------------------------------===//
class AFIR_CommonAttrs {
  dag commonArgs = (ins
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
    DefaultValuedOptionalAttr<I32Attr, "-1">:$loop_axis,
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$ir_attr_def,
    OptionalAttr<TmpBufDescArrayAttr>:$tmp_buffers,
    DefaultValuedAttr<AscTensorGroupsArrayAttr, "::mlir::afir::Position::GM">:$outputs
  );
}

//===----------------------------------------------------------------------===//
// Base AFIR Operation
//===----------------------------------------------------------------------===//
class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// Given the base class of compute info, it serves as the base class for all subsequent operators
//===----------------------------------------------------------------------===//
class AFIR_BaseOp<string mnemonic,
                          list<Trait> traits,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid"> :
    AFIR_Op<mnemonic, traits> {
  code getComputeInfo = [{
    afir::ComputeType getComputeType() {
      return afir::ComputeType::}] # computeType # [{;
    }
    afir::ComputeUnit getComputeUnit() {
      return afir::ComputeUnit::}] # computeUnit # [{;
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// N-ary operation: base class for given input and output operands
//===----------------------------------------------------------------------===//
class AFIR_NaryOp<string mnemonic,
                          list<Trait> traits,
                          int numOperands,
                          int numResults = 1,
                          string computeType = "Invalid",
                          string computeUnit = "Invalid",
                          bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit> {
  code baseMethods = [{
      int getNumberOfOperands() { return }] # numOperands # [{; }
      int getNumberOfResults() { return }] # numResults # [{; }
    }] # getComputeInfo;
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethods, "");
}

//===----------------------------------------------------------------------===//
// Zero element operation: base class with input number 0 and output number 1
//===----------------------------------------------------------------------===//
class AFIR_NullaryOpBase<string mnemonic, 
                       list<Trait> traits,
                       list<Type> types,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 0, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = commonArgs;
  let results = (outs TensorOf<types>:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Unary operation: base class where both the input number and the output number are 1
//===----------------------------------------------------------------------===//
class AFIR_UnaryOpBase<string mnemonic, 
                       list<Trait> traits,
                       list<Type> inputTypes,
                       list<Type> resultTypes,
                       string computeType = "Invalid",
                       string computeUnit = "Invalid",
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 1, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins TensorOf<inputTypes>:$input),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_UnaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_UnaryOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Binary operations: base class with input number 2 and output number 1
//===----------------------------------------------------------------------===//
class AFIR_BinaryOpBase<string mnemonic, 
                        list<Trait> traits,
                        list<Type> lhsTypes,
                        list<Type> rhsTypes,
                        list<Type> resultTypes,
                        string computeType = "Invalid",
                        string computeUnit = "Invalid",
                        bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 2, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins 
      TensorOf<lhsTypes>:$lhs, 
      TensorOf<rhsTypes>:$rhs
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_BinaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, types, types, types, computeType, computeUnit, finalize>;

class AFIR_BinaryHomoOp<string mnemonic, 
                        list<Trait> traits = [], 
                        list<Type> inputTypes, 
                        list<Type> resultTypes, 
                        string computeType, 
                        string computeUnit,
                        bit finalize = 0> :
    AFIR_BinaryOpBase<mnemonic, traits, inputTypes, inputTypes, resultTypes, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Ternary operation: base class with input number 3 and output number 1
//===----------------------------------------------------------------------===//
class AFIR_TernaryOpBase<string mnemonic, 
                         list<Trait> traits,
                         list<Type> input1Types,
                         list<Type> input2Types,
                         list<Type> input3Types,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 3, 1, computeType, computeUnit, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins 
      TensorOf<input1Types>:$input1,
      TensorOf<input2Types>:$input2,
      TensorOf<input3Types>:$input3
    ),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input1 `,` $input2 `,` $input3 attr-dict `:` 
    `(` type($input1) `,` type($input2) `,` type($input3) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

class AFIR_TernaryOp<string mnemonic, 
                    list<Trait> traits = [], 
                    list<Type> types, 
                    string computeType, 
                    string computeUnit,
                    bit finalize = 0> :
    AFIR_TernaryOpBase<mnemonic, traits, types, types, types, types, computeType, computeUnit, finalize>;

//===----------------------------------------------------------------------===//
// Variable input number: base class with unknown input number and output number 1
//===----------------------------------------------------------------------===//
class AFIR_VariadicOpBase<string mnemonic, 
                         list<Trait> traits,
                         list<Type> inputTypes,
                         list<Type> resultTypes,
                         string computeType = "Invalid",
                         string computeUnit = "Invalid",
                         bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits, computeType, computeUnit>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins Variadic<TensorOf<inputTypes>>:$inputs),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($result)
  }];
  code baseMethods = [{
      int getNumberOfOperands() { return getInputs().size(); }
      int getNumberOfResults() { return 1; }
    }] # getComputeInfo;

  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethods, "");
}

class AFIR_VariadicOp<string mnemonic, 
                      list<Trait> traits = [], 
                      list<Type> types, 
                      string computeType, 
                      string computeUnit,
                      bit finalize = 0> :
    AFIR_VariadicOpBase<mnemonic, traits, types, types, computeType, computeUnit, finalize>;

#endif // AFIR_BASE