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

#ifndef ASC_INTERFACES_TD
#define ASC_INTERFACES_TD

include "mlir/IR/OpBase.td"

def AscendC_CalCount : AnyTypeOf<[I8, I16, I32, I64]>;

class GetMethod<string fieldName, string methodName, string returnType = "::mlir::Value">
    : InterfaceMethod<"Obtain `" # fieldName # "` parameter", returnType, methodName>;

class AscendC_InterfaceMethods {
  // Method list should be kept sorted by name.
  InterfaceMethod getAPIName = InterfaceMethod<"Obtain Ascend C library name", "::llvm::StringRef", "getAPIName">;
  InterfaceMethod getCalCount = GetMethod<"calCount", "getCalCount">;
  InterfaceMethod getComment = InterfaceMethod<"Obtain describing comment", "::llvm::StringRef", "getComment">;
  InterfaceMethod getDst = GetMethod<"dst", "getDst">;
  InterfaceMethod getDstBlkStride = GetMethod<"dstBlkStride", "getDstBlkStride">;
  InterfaceMethod getDstRepStride = GetMethod<"dstRepStride", "getDstRepStride">;
  InterfaceMethod getMask = GetMethod<"mask", "getMask">;
  InterfaceMethod getParams = GetMethod<"params", "getParams">;
  InterfaceMethod getRepeatParams = GetMethod<"repeatParams", "getRepeatParams">;
  InterfaceMethod getRepeatTimes = GetMethod<"repeatTimes", "getRepeatTimes">;
  InterfaceMethod getIsSetMask = GetMethod<"isSetMask", "getIsSetMask", "bool">;
  InterfaceMethod getScalar = GetMethod<"scalar", "getScalar">;
  InterfaceMethod getSharedTmpBuffer = GetMethod<"sharedTmpBuffer", "getSharedTmpBuffer">;
  InterfaceMethod getSrc = GetMethod<"src", "getSrc">;
  InterfaceMethod getSrcBlkStride = GetMethod<"srcBlkStride", "getSrcBlkStride">;
  InterfaceMethod getSrcRepStride = GetMethod<"srcRepStride", "getSrcRepStride">;
  InterfaceMethod getSrc0 = GetMethod<"src0", "getSrc0">;
  InterfaceMethod getSrc0BlkStride = GetMethod<"src0BlkStride", "getSrc0BlkStride">;
  InterfaceMethod getSrc0RepStride = GetMethod<"src0RepStride", "getSrc0RepStride">;
  InterfaceMethod getSrc1 = GetMethod<"src1", "getSrc1">;
  InterfaceMethod getSrc1BlkStride = GetMethod<"src1BlkStride", "getSrc1BlkStride">;
  InterfaceMethod getSrc1RepStride = GetMethod<"src1RepStride", "getSrc1RepStride">;
  InterfaceMethod getIsReuseSource = GetMethod<"isReuseSource", "getIsReuseSource">;
}

class AscendC_OpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>, AscendC_InterfaceMethods {
  let cppNamespace = "::mlir::ascendc";
}

//===----------------------------------------------------------------------===//
// Ascend C API interfaces
//===----------------------------------------------------------------------===//

def APIOpInterface : AscendC_OpInterface<"APIOp"> {
  let description = "Base interface for operations representing Ascend C API";
  let methods = [getAPIName, getComment];
}

def OpWithDstInterface : AscendC_OpInterface<"OpWithDst"> {
  let description = "Operation with `dst` operand (typically tensor)";
  let methods = [getDst];
}

def VectorOpInterface : AscendC_OpInterface<"VectorOp", [APIOpInterface]> {
  let description = "Base interface for vector operations";
}

def DataCopyOpInterface :
    AscendC_OpInterface<"DataCopyOp", [APIOpInterface, OpWithDstInterface]> {
  InterfaceMethod getDirection = InterfaceMethod<"Get copy direction",
    "::mlir::ascendc::CopyDirection", "getDirection", (ins), "", [{
    auto dstType = $_op.getDst().getType();
    auto srcType = $_op.getSrc().getType();
    if (isa<GlobalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::gm_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::gm_ubuf;
    } else if (isa<LocalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::ubuf_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::ubuf_ubuf;
    }
    return CopyDirection::Unknown;
  }]>;
  InterfaceMethod setSrc = InterfaceMethod<"Set src operand", "void", "setSrc",
                                           (ins "Value":$src), "", [{
    $_op.getSrcMutable().assign(src);
  }]>;

  let description = "Data copy operation";
  let methods = [getSrc, getDirection, setSrc];
}

//===----------------------------------------------------------------------===//
// Vector unary operations
//===----------------------------------------------------------------------===//

def UnaryOpInterface
    : AscendC_OpInterface<"UnaryOp", [VectorOpInterface, OpWithDstInterface]> {
  let description = "Vector unary operation";
  let methods = [getSrc];
}

def UnaryL0OpInterface : AscendC_OpInterface<"UnaryL0Op", [UnaryOpInterface]> {
  let description = "Vector unary operation (L0 API)";
  let methods = [getMask, getRepeatTimes, getRepeatParams];
}

def UnaryL2OpInterface : AscendC_OpInterface<"UnaryL2Op", [UnaryOpInterface]> {
  let description = "Vector unary operation (L2 API)";
  let methods = [getCalCount];
}

//===----------------------------------------------------------------------===//
// Vector binary operations
//===----------------------------------------------------------------------===//

def BinaryOpInterface
    : AscendC_OpInterface<"BinaryOp", [VectorOpInterface, OpWithDstInterface]> {
  let description = "Vector binary operation";
  let methods = [getSrc0, getSrc1];
}

def BinaryL3OpInterface : AscendC_OpInterface<"BinaryL3Op", [BinaryOpInterface]> {
  let description = "Vector binary operation (L3 API)";
}

//===----------------------------------------------------------------------===//
// Vector-scalar operations
//===----------------------------------------------------------------------===//

def VecScalarOpInterface
    : AscendC_OpInterface<"VecScalarOp", [VectorOpInterface, OpWithDstInterface]> {
  let description = "Vector-scalar operation";
  let methods = [getSrc, getScalar];
}

def VecScalarL0OpInterface : AscendC_OpInterface<"VecScalarL0Op", [VecScalarOpInterface]> {
  let description = "Vector-scalar operation (L0 API)";
  let methods = [getMask, getRepeatTimes, getRepeatParams];
}

def VecScalarL2OpInterface : AscendC_OpInterface<"VecScalarL2Op", [VecScalarOpInterface]> {
  let description = "Vector-scalar operation (L2 API)";
  let methods = [getCalCount];
}

//===----------------------------------------------------------------------===//
// Math library operations
//===----------------------------------------------------------------------===//

def MathLibraryOpInterface
    : AscendC_OpInterface<"MathLibraryOp", [VectorOpInterface, OpWithDstInterface]> {
  let description = "Math library operation";
}

def UnaryMathOpInterface : AscendC_OpInterface<"UnaryMathOp", [MathLibraryOpInterface]> {
  let description = "Unary math library operation";
  let methods = [getSrc, getSharedTmpBuffer, getCalCount, getIsReuseSource];
}

def BinaryMathOpInterface : AscendC_OpInterface<"BinaryMathOp", [MathLibraryOpInterface]> {
  let description = "Binary math library operation";
  let methods = [getSrc0, getSrc1, getSharedTmpBuffer, getCalCount, getIsReuseSource];
}

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

def DataConversionOpInterface
    : AscendC_OpInterface<"DataConversionOp", [APIOpInterface]> {
  let description = "Data conversion operation";
}

#endif // ASC_INTERFACES_TD
