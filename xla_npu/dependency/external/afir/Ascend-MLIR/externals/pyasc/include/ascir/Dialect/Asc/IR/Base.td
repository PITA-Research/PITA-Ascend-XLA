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

#ifndef ASC_BASE_TD
#define ASC_BASE_TD

include "Dialect.td"
include "Interfaces.td"
include "Core/Attributes.td"
include "Core/Types.td"

def AscConstructor : NativeOpTrait<"AscConstructorTrait">;
def AscMemberFunc : NativeOpTrait<"AscMemberFuncTrait">;
def AscFunc : NativeOpTrait<"AscFuncTrait">;

class AscendC_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendC_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascendc";
  let assemblyFormat = "operands attr-dict `:` qualified(type(operands))";
  // `genEmitter` is a boolean flag indicating whether this operation can auto code generation
  // for it's emitter function. It is set to true if any of the following traits are present
  // - AscConstructor: The operation represents a constructor call
  // - AscMemberFunc: The operation represents a member function call
  // - AscFunc: The operation represents a regular function call
  // The flag defaulting to false if none of these traits are present.
  // When true, the operation's auto emitter function will be generated during  code emission phase
  bit genEmitter = !foldl(
    0, traits, init, trait,
    !or(init,
        !eq(trait, AscConstructor),
        !eq(trait, AscMemberFunc),
        !eq(trait, AscFunc)
    )
  );
  // `paramTypeLists` defines the mapping between operation arguments and parameters.
  // The list corresponds one-to-one with the operation's arguments, where each value specifies
  // how the arguments should be treated for template parameters generation:
  // -3: Pointer type parameter
  // -2: Pointer to int type parameter
  // -1: The argument is an attributes (not a template)
  //  0: Regular parameter (pass directly as a function parameter)
  //  1: Extract the template type from the argument at this position (e.g., <typename T> form function(T arg))
  //  2: Extract the element type from a template type (e.g., LocalTensor<T> -> T)
  //  3：Non-type template parameter (enum value)
  //  4: Non-type template parameter (regular value)
  //  5: Type template paramter (eg., <typename T> function())
  // This allows flexible template argument generation from operation arguments while supporting
  // various patterns like type extraction and non-type template
  list<int> paramTypeLists = [];
}

//===----------------------------------------------------------------------===//
// Base operations
//===----------------------------------------------------------------------===//

class APIOp<string mnemonic, string apiName, list<Trait> traits = []>
    : AscendC_Op<mnemonic, [APIOpInterface] # traits> {
  let summary = "Call `AscendC::" # apiName # "` function";
  string comment = "";
  code extraClassDeclarationBase = [{
    static StringRef getAPIName() { return "}] # apiName # [{"; }
    static StringRef getComment() { return "}] # comment # [{"; }
  }];
  let extraClassDeclaration = extraClassDeclarationBase;
}

class DataCopyOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [DataCopyOpInterface] # traits>;

class VectorOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [VectorOpInterface] # traits>;

//===----------------------------------------------------------------------===//
// Vector unary operations
//===----------------------------------------------------------------------===//

class UnaryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [UnaryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation.\n";
}

class UnaryL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, [UnaryL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$mask,
                   AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class UnaryL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, Variadic<UI64>:$mask,
                   AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class UnaryL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, [UnaryL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$calCount);
}

multiclass UnaryL012Op<string mnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : UnaryL0Op<mnemonic # "_l0", apiName, traits>;
  def L1Op : UnaryL1Op<mnemonic # "_l1", apiName, traits>;
  def L2Op : UnaryL2Op<mnemonic # "_l2", apiName, traits>;
}

//===----------------------------------------------------------------------===//
// Vector binary operations
//===----------------------------------------------------------------------===//

class BinaryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [BinaryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation.\n";
}

class BinaryL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes, 
                  AscendC_BinaryRepeatParams:$repeatParams);
}

class BinaryL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                   AscendC_BinaryRepeatParams:$repeatParams);
}

class BinaryL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount);
}

class BinaryTemplateL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes, 
                  AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryTemplateL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryTemplateL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount, UnitAttr:$isSetMask);
}

class BinaryCastL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes, 
                  AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryCastL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryCastL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount, UnitAttr:$isSetMask);
}

class BinaryL3Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL3OpInterface] # traits> {
  let summary = "Call `LocalTensor::" # apiName # "` method";
  let description = "`LocalTensor::" # apiName # "` performs a vector binary operation (L3 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1);
}

multiclass BinaryL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryTemplateL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryTemplateL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryTemplateL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryCastL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryCastL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryCastL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryL0123Op<string baseMnemonic, string apiName, string l3operator,
                        list<Trait> traits = []> {
  defm "" : BinaryL012Op<baseMnemonic, apiName, traits>;
  def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
}

multiclass BinaryTemplateL0123Op<string baseMnemonic, string apiName, string l3operator,
                        list<Trait> traits = []> {
  defm "" : BinaryTemplateL012Op<baseMnemonic, apiName, traits>;
  def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
}

//===----------------------------------------------------------------------===//
// Vector-scalar operations
//===----------------------------------------------------------------------===//

class VecScalarOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [VecScalarOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation.\n";
}

class VecScalarL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, [VecScalarL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   AnyType:$mask, AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class VecScalarL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class VecScalarL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, [VecScalarL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   AnyType:$calCount, UnitAttr:$isSetMask);
}

multiclass VecScalarL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : VecScalarL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : VecScalarL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : VecScalarL2Op<baseMnemonic # "_l2", apiName, traits>;
}

//===----------------------------------------------------------------------===//
// Math library operations
//===----------------------------------------------------------------------===//

class MathLibraryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [MathLibraryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a math library operation.\n";
}

class UnaryMathOp<string mnemonic, string apiName, list<Trait> traits = []>
    : MathLibraryOp<mnemonic, apiName, [UnaryMathOpInterface, AttrSizedOperandSegments] # traits> {
  let description = "`AscendC::" # apiName # "` is a unary math library operation.\n";
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src, Optional<AnyType>:$sharedTmpBuffer,
                   Optional<AnyType>:$calCount, AnyType:$isReuseSource);
}

class BinaryMathOp<string mnemonic, string apiName, list<Trait> traits = []>
    : MathLibraryOp<mnemonic, apiName, [BinaryMathOpInterface, AttrSizedOperandSegments] # traits> {
  let description = "`AscendC::" # apiName # "` is a binary math library operation.\n";
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src0, AnyType:$src1,
                   Optional<AnyType>:$sharedTmpBuffer, Optional<AnyType>:$calCount, AnyType:$isReuseSource);
}

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

class DataConversionOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [DataConversionOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a data conversion operation.\n";
}

class TransposeOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AscFunc] # traits> {
}

class TransposeSimpleOp<string mnemonic, string apiName, list<Trait> traits = []>
    : TransposeOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` method (simple version)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src);
}

class TransposeExtOp<string mnemonic, string apiName, list<Trait> traits = []>
    : TransposeOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` method (extended version)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer,
                       AscendC_TransposeParamsExt:$params);
}

class TransDataTo5HDTensorListOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AttrSizedOperandSegments] # traits> {
  let summary = "Call `AscendC::" # apiName # "` with LocalTensor lists";
  let arguments = (ins Variadic<AscendC_LocalTensor>:$dstList,
                       Variadic<AscendC_LocalTensor>:$srcList,
                       AscendC_TransDataTo5HDParams:$params);
}

class TransDataTo5HDUintListOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AttrSizedOperandSegments] # traits> {
  let summary = "Call `AscendC::" # apiName # "` with uint64_t address lists";
  let arguments = (ins Variadic<AnyUnsignedInteger>:$dstList,
                       Variadic<AnyUnsignedInteger>:$srcList,
                       AscendC_TransDataTo5HDParams:$params);
}

class TransDataTo5HDSingleOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` with LocalTensor<uint64_t>";
  let arguments = (ins AscendC_LocalTensor:$dst,
                       AscendC_LocalTensor:$src,
                       AscendC_TransDataTo5HDParams:$params);
}

#endif // ASC_BASE_TD
