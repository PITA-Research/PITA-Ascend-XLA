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

//===- DialectBuilde.h - ASC-IR dialect builder -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file contains the dialect builder for the ASC-IR dialect.
// Follows the implementation pattern from onnx-mlir.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFIRTOASCIR_DIALECT_BUILDER_H
#define MLIR_CONVERSION_AFIRTOASCIR_DIALECT_BUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

namespace mlir {
namespace afir {

struct DialectBuilder {
  DialectBuilder(mlir::Location loc) : builder(nullptr), location(loc) {}
  DialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  DialectBuilder(const DialectBuilder &db)
      : builder(db.builder), location(db.location) {}
  virtual ~DialectBuilder() {}
  DialectBuilder(DialectBuilder &&) = delete;
  DialectBuilder &operator=(const DialectBuilder &) = delete;
  DialectBuilder &&operator=(const DialectBuilder &&) = delete;

  mlir::OpBuilder &getBuilder() const { return b(); }
  mlir::OpBuilder *getBuilderPtr() const { return builder; }
  mlir::Location getLoc() const { return loc(); }

 protected:
  mlir::OpBuilder &b() const {
    assert(builder);
    return *builder;
  }
  mlir::Location loc() const { return location; }

 private:
  mlir::OpBuilder *builder;
  mlir::Location location;
};

struct AscendCBuilder : DialectBuilder {
  AscendCBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  AscendCBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  AscendCBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~AscendCBuilder() {}

  mlir::Value createTBuf(ascendc::TPosition pos) const;
  mlir::Value getTensorFromTBuf(mlir::Value tbuf, mlir::Type tensorType) const;

  mlir::Value addL3(mlir::Value dst, mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value subL3(mlir::Value dst, mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value mulL3(mlir::Value dst, mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value divL3(mlir::Value dst, mlir::Value lhs, mlir::Value rhs) const;

  mlir::Value createLocalTensor(mlir::Type tensorType, mlir::Value tbuf) const;
};

template <class... Ts>
struct MultiDialectBuilder {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : builder(db.getBuilderPtr()), location(db.getLoc()) {}

  mlir::OpBuilder &getBuilder() const {
    assert(builder);
    return *builder;
  }
  mlir::OpBuilder *getBuilderPtr() const { return builder; }
  mlir::Location getLoc() const { return location; }

 private:
  mlir::OpBuilder *builder;
  mlir::Location location;
};

template <class... Ts>
struct MultiDialectBuilder<AscendCBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), ascendc(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), ascendc(db) {}
  AscendCBuilder ascendc;
};

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_CONVERSION_AFIRTOASCIR_DIALECT_BUILDER_H
