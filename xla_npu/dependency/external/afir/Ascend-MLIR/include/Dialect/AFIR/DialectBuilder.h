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

//===- AFIRDialectBuilder.h - Builder for AFIR operations -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// This file is inspired by onnx-mlir's DialectBuilder design.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIRDIALECTBUILDER_H
#define MLIR_DIALECT_AFIR_AFIRDIALECTBUILDER_H

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// AFIRBuilder - Convenient builder for AFIR operations
//===----------------------------------------------------------------------===//

/// AFIRBuilder provides a convenient interface for creating AFIR operations.
/// It wraps an OpBuilder and Location to simplify operation creation.
///
/// Usage:
///   AFIRBuilder afirBuilder(builder, loc);
///   Value result = afirBuilder.add(lhs, rhs);
///
class AFIRBuilder {
 public:
  AFIRBuilder(OpBuilder &builder, Location loc) : builder(builder), loc(loc) {}

  AFIRBuilder(OpBuilder &builder, Operation *op)
      : builder(builder), loc(op->getLoc()) {}

  // Get the underlying builder and location
  OpBuilder &getBuilder() const { return builder; }
  Location getLoc() const { return loc; }

  //===--------------------------------------------------------------------===//
  // Arithmetic Operations
  //===--------------------------------------------------------------------===//

  /// Create an element-wise addition operation.
  Value add(Value lhs, Value rhs) const {
    return builder.create<AddOp>(loc, lhs, rhs);
  }

  /// Create an element-wise subtraction operation.
  Value sub(Value lhs, Value rhs) const {
    return builder.create<SubOp>(loc, lhs, rhs);
  }

  /// Create an element-wise multiplication operation.
  Value mul(Value lhs, Value rhs) const {
    return builder.create<MulOp>(loc, lhs, rhs);
  }

  /// Create an element-wise division operation.
  Value div(Value lhs, Value rhs) const {
    return builder.create<DivOp>(loc, lhs, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Compound Operations
  //===--------------------------------------------------------------------===//

  /// Create a fused multiply-add: result = a * b + c
  Value fma(Value a, Value b, Value c) const {
    Value mulResult = mul(a, b);
    return add(mulResult, c);
  }

  /// Create a squared difference: result = (a - b) * (a - b)
  Value squaredDiff(Value a, Value b) const {
    Value diff = sub(a, b);
    return mul(diff, diff);
  }

 protected:
  OpBuilder &builder;
  Location loc;
};

//===----------------------------------------------------------------------===//
// ScopedAFIRBuilder - Builder with automatic scope management
//===----------------------------------------------------------------------===//

/// ScopedAFIRBuilder extends AFIRBuilder with scope management capabilities.
/// It can be used to insert operations at specific points in the IR.
class ScopedAFIRBuilder : public AFIRBuilder {
 public:
  ScopedAFIRBuilder(OpBuilder &builder, Location loc)
      : AFIRBuilder(builder, loc), insertionGuard(builder) {}

  /// Create a builder that inserts at the beginning of a block.
  static ScopedAFIRBuilder atBlockBegin(OpBuilder &builder, Block *block) {
    builder.setInsertionPointToStart(block);
    return ScopedAFIRBuilder(builder, builder.getUnknownLoc());
  }

  /// Create a builder that inserts at the end of a block.
  static ScopedAFIRBuilder atBlockEnd(OpBuilder &builder, Block *block) {
    builder.setInsertionPointToEnd(block);
    return ScopedAFIRBuilder(builder, builder.getUnknownLoc());
  }

  /// Create a builder that inserts before an operation.
  static ScopedAFIRBuilder before(OpBuilder &builder, Operation *op) {
    builder.setInsertionPoint(op);
    return ScopedAFIRBuilder(builder, op->getLoc());
  }

  /// Create a builder that inserts after an operation.
  static ScopedAFIRBuilder after(OpBuilder &builder, Operation *op) {
    builder.setInsertionPointAfter(op);
    return ScopedAFIRBuilder(builder, op->getLoc());
  }

 private:
  OpBuilder::InsertionGuard insertionGuard;
};

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_DIALECT_AFIR_AFIRDIALECTBUILDER_H
