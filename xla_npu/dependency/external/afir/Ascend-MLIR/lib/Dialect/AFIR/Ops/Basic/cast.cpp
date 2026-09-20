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

//===- cast.cpp - AFIR cast operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Type ID enumeration definition
//===----------------------------------------------------------------------===//
enum class CastTypeId : unsigned {
  F32 = 0,
  F16 = 1,
  I64 = 2,
  I32 = 3,
  I16 = 4,
  BF16 = 5,
  I8 = 6,
  UI8 = 7,
  I4 = 8,
  UI32 = 9,
  UI16 = 10,
  UI64 = 11,
  Unknown = 999
};

//===----------------------------------------------------------------------===//
// Type to ID mapping
//===----------------------------------------------------------------------===//
static CastTypeId getTypeId(Type type) {
  // floating point type
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    if (floatTy.isF32()) return CastTypeId::F32;
    if (floatTy.isF16()) return CastTypeId::F16;
    if (floatTy.isBF16()) return CastTypeId::BF16;
    return CastTypeId::Unknown;
  }

  // integer type
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    unsigned width = intTy.getWidth();
    bool isUnsigned = intTy.isUnsigned();
    bool isSignless = intTy.isSignless();

    switch (width) {
      case 64:
        if (isUnsigned) return CastTypeId::UI64;
        if (isSignless || intTy.isSigned()) return CastTypeId::I64;
        break;
      case 32:
        if (isUnsigned) return CastTypeId::UI32;
        if (isSignless || intTy.isSigned()) return CastTypeId::I32;
        break;
      case 16:
        if (isUnsigned) return CastTypeId::UI16;
        if (isSignless || intTy.isSigned()) return CastTypeId::I16;
        break;
      case 8:
        if (isUnsigned) return CastTypeId::UI8;
        if (isSignless || intTy.isSigned()) return CastTypeId::I8;
        break;
      case 4:
        return CastTypeId::I4;
    }
  }

  return CastTypeId::Unknown;
}

//===----------------------------------------------------------------------===//
// Supported type conversion pair lookup table
//===----------------------------------------------------------------------===//
static const llvm::DenseSet<std::pair<CastTypeId, CastTypeId>>&
getSupportedCastPairs() {
  using CT = CastTypeId;

  static const llvm::DenseSet<std::pair<CT, CT>> supportedCasts = {
      // F32 can be converted to: F32, F16, I64, I32, I16, BF16
      {CT::F32, CT::F32},
      {CT::F32, CT::F16},
      {CT::F32, CT::I64},
      {CT::F32, CT::I32},
      {CT::F32, CT::I16},
      {CT::F32, CT::BF16},

      // F16 can be converted to: F32, I32, I16, I8, UI8, I4, I64
      {CT::F16, CT::F32},
      {CT::F16, CT::I32},
      {CT::F16, CT::I16},
      {CT::F16, CT::I8},
      {CT::F16, CT::UI8},
      {CT::F16, CT::I4},
      {CT::F16, CT::I64},

      // I4 can be converted to: F16
      {CT::I4, CT::F16},

      // UI8 can be converted to: F16, F32, I32, I16, I8, I4
      {CT::UI8, CT::F16},
      {CT::UI8, CT::F32},
      {CT::UI8, CT::I32},
      {CT::UI8, CT::I16},
      {CT::UI8, CT::I8},
      {CT::UI8, CT::I4},

      // I8 can be converted to: F16, UI8
      {CT::I8, CT::F16},
      {CT::I8, CT::UI8},

      // I16 can be converted to: F16, F32, UI16
      {CT::I16, CT::F16},
      {CT::I16, CT::F32},
      {CT::I16, CT::UI16},

      // I32 can be converted to: F32, I64, I16, F16, UI32
      {CT::I32, CT::F32},
      {CT::I32, CT::I64},
      {CT::I32, CT::I16},
      {CT::I32, CT::F16},
      {CT::I32, CT::UI32},

      // I64 can be converted to: I32
      {CT::I64, CT::I32},

      // BF16 can be converted to: F32, I32
      {CT::BF16, CT::F32},
      {CT::BF16, CT::I32},

      // UI32 can be converted to: I32
      {CT::UI32, CT::I32},

      // UI16 can be converted to: I16
      {CT::UI16, CT::I16},

      // UI64 can be converted to: I64
      {CT::UI64, CT::I64}};

  return supportedCasts;
}

//===----------------------------------------------------------------------===//
// Check if type conversion is supported
//===----------------------------------------------------------------------===//
static bool isSupportedCast(Type inputType, Type outputType) {
  CastTypeId inputId = getTypeId(inputType);
  CastTypeId outputId = getTypeId(outputType);

  if (inputId == CastTypeId::Unknown || outputId == CastTypeId::Unknown) {
    return false;
  }

  return getSupportedCastPairs().count({inputId, outputId}) > 0;
}

//===----------------------------------------------------------------------===//
// CastOp Verify implementation
//===----------------------------------------------------------------------===//
LogicalResult CastOp::verify() {
  // 1. Check that the input and output are both ranked tensor
  auto inputTensorType = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputTensorType =
      mlir::dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputTensorType || !outputTensorType) {
    return emitOpError("input and output must be ranked tensors");
  }

  // 2. Check that shape must match exactly
  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output shapes must match, got input shape ")
           << inputTensorType.getShape() << " and output shape "
           << outputTensorType.getShape();
  }

  // 3. Check whether element type conversion is supported
  Type inputElemType = inputTensorType.getElementType();
  Type outputElemType = outputTensorType.getElementType();

  if (!isSupportedCast(inputElemType, outputElemType)) {
    return emitOpError("unsupported cast from ")
           << inputElemType << " to " << outputElemType
           << ". See supported cast pairs in Cast operation definition.";
  }

  return success();
}
