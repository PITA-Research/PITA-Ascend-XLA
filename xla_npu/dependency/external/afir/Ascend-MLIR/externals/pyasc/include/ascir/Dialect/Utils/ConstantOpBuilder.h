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

#ifndef ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H
#define ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascir {

struct ConstantOpBuilder {
  using ConstantOp = ::mlir::arith::ConstantOp;

  ConstantOpBuilder(OpBuilder &builder) : builder(builder) {}
  ~ConstantOpBuilder() = default;

  ConstantOp create(TypedAttr attr) {
    auto op = ConstantOp::create(builder, builder.getUnknownLoc(), attr);
    return op;
  }

  ConstantOp create(IndexType type, int64_t value) {
    return create(builder.getIndexAttr(value));
  }

  ConstantOp create(IntegerType type, int64_t value) {
    return create(builder.getIntegerAttr(type, value));
  }

  ConstantOp create(FloatType type, double value) {
    return create(builder.getFloatAttr(type, value));
  }

  Value index(int64_t value) { return create(builder.getIndexType(), value); }

  Value i64(int64_t value) { return create(builder.getI64Type(), value); }

  Value i32(int32_t value) { return create(builder.getI32Type(), value); }

  Value i16(int16_t value) { return create(builder.getI16Type(), value); }

  Value i8(int8_t value) { return create(builder.getI8Type(), value); }

  Value f64(double value) { return create(builder.getF64Type(), value); }

  Value f32(float value) { return create(builder.getF32Type(), value); }

  Value f16(float value) { return create(builder.getF16Type(), value); }

  template <typename IntType>
  Value integer(IntType value) {
    return create(builder.getIntegerType(sizeof(IntType) * CHAR_BIT), value);
  }

 private:
  OpBuilder &builder;
};

}  // namespace ascir
}  // namespace mlir

#endif  // ASCIR_DIALECT_UTILS_CONSTANTOPBUILDER_H
