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

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCTypes.cpp.inc"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// BaseTensorImpl
//===----------------------------------------------------------------------===//

template <typename ConcreteT>
class BaseTensorImpl {
  using BaseT = typename ConcreteT::Base;

 public:
  static ConcreteT get(ArrayRef<int64_t> shape, Type elementType) {
    return BaseT::get(elementType.getContext(), shape, elementType);
  }

  static ConcreteT get(Type elementType) {
    return ConcreteT::get({}, elementType);
  }

  static ConcreteT get(BaseTensorType baseType) {
    return BaseT::get(baseType.getContext(), baseType.getShape(),
                      baseType.getElementType());
  }

  static Type parse(AsmParser &odsParser) {
    Type elementType;
    if (odsParser.parseLess()) return Type();
    SmallVector<int64_t> shape;
    if (odsParser.parseOptionalStar()) {
      // No '*' consumed => tensor is ranked (i.e. has shape)
      if (odsParser.parseDimensionList(shape)) {
        odsParser.emitError(
            odsParser.getNameLoc(),
            "either dimension list (for ranked tensor) or '*' symbol (for "
            "unranked tensor) must be declared");
        return Type();
      }
    } else if (odsParser.parseXInDimensionList()) {
      return Type();
    }
    if (odsParser.parseType(elementType) || odsParser.parseGreater()) {
      return Type();
    }
    return ConcreteT::get(shape, elementType);
  }

  static void print(const ConcreteT &tensor, AsmPrinter &odsPrinter) {
    odsPrinter << "<";
    ArrayRef<int64_t> shape = tensor.getShape();
    if (shape.empty()) {
      odsPrinter << "*x";
    } else {
      for (int64_t dim : tensor.getShape()) {
        if (ShapedType::isDynamic(dim))
          odsPrinter << "?";
        else
          odsPrinter << dim;
        odsPrinter << "x";
      }
    }
    odsPrinter << tensor.getElementType() << ">";
  }

  static ShapedType cloneWith(const ConcreteT &tensor,
                              std::optional<ArrayRef<int64_t>> shape,
                              Type elementType) {
    if (shape) return ConcreteT::get(*shape, elementType);
    return ConcreteT::get(tensor.getShape(), elementType);
  }

  static bool hasRank(const ConcreteT &tensor) {
    return !tensor.getShape().empty();
  }
};

//===----------------------------------------------------------------------===//
// BaseGlobalTensorType
//===----------------------------------------------------------------------===//

BaseGlobalTensorType BaseGlobalTensorType::get(ArrayRef<int64_t> shape,
                                               Type elementType) {
  return BaseTensorImpl<BaseGlobalTensorType>::get(shape, elementType);
}

BaseGlobalTensorType BaseGlobalTensorType::get(Type elementType) {
  return BaseTensorImpl<BaseGlobalTensorType>::get(elementType);
}

BaseGlobalTensorType BaseGlobalTensorType::get(BaseTensorType baseType) {
  return BaseTensorImpl<BaseGlobalTensorType>::get(baseType);
}

Type BaseGlobalTensorType::parse(AsmParser &odsParser) {
  return BaseTensorImpl<BaseGlobalTensorType>::parse(odsParser);
}

void BaseGlobalTensorType::print(AsmPrinter &odsPrinter) const {
  BaseTensorImpl<BaseGlobalTensorType>::print(*this, odsPrinter);
}

ShapedType BaseGlobalTensorType::cloneWith(
    std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  return BaseTensorImpl<BaseGlobalTensorType>::cloneWith(*this, shape,
                                                         elementType);
}

bool BaseGlobalTensorType::hasRank() const {
  return BaseTensorImpl<BaseGlobalTensorType>::hasRank(*this);
}

//===----------------------------------------------------------------------===//
// BaseLocalTensorType
//===----------------------------------------------------------------------===//

BaseLocalTensorType BaseLocalTensorType::get(ArrayRef<int64_t> shape,
                                             Type elementType) {
  return BaseTensorImpl<BaseLocalTensorType>::get(shape, elementType);
}

BaseLocalTensorType BaseLocalTensorType::get(Type elementType) {
  return BaseTensorImpl<BaseLocalTensorType>::get(elementType);
}

BaseLocalTensorType BaseLocalTensorType::get(BaseTensorType baseType) {
  return BaseTensorImpl<BaseLocalTensorType>::get(baseType);
}

Type BaseLocalTensorType::parse(AsmParser &odsParser) {
  return BaseTensorImpl<BaseLocalTensorType>::parse(odsParser);
}

void BaseLocalTensorType::print(AsmPrinter &odsPrinter) const {
  BaseTensorImpl<BaseLocalTensorType>::print(*this, odsPrinter);
}

ShapedType BaseLocalTensorType::cloneWith(
    std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  return BaseTensorImpl<BaseLocalTensorType>::cloneWith(*this, shape,
                                                        elementType);
}

bool BaseLocalTensorType::hasRank() const {
  return BaseTensorImpl<BaseLocalTensorType>::hasRank(*this);
}

//===----------------------------------------------------------------------===//
// GlobalTensorType
//===----------------------------------------------------------------------===//

GlobalTensorType GlobalTensorType::get(ArrayRef<int64_t> shape,
                                       Type elementType) {
  return BaseTensorImpl<GlobalTensorType>::get(shape, elementType);
}

GlobalTensorType GlobalTensorType::get(Type elementType) {
  return BaseTensorImpl<GlobalTensorType>::get(elementType);
}

GlobalTensorType GlobalTensorType::get(BaseTensorType baseType) {
  return BaseTensorImpl<GlobalTensorType>::get(baseType);
}

Type GlobalTensorType::parse(AsmParser &odsParser) {
  return BaseTensorImpl<GlobalTensorType>::parse(odsParser);
}

void GlobalTensorType::print(AsmPrinter &odsPrinter) const {
  BaseTensorImpl<GlobalTensorType>::print(*this, odsPrinter);
}

ShapedType GlobalTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  return BaseTensorImpl<GlobalTensorType>::cloneWith(*this, shape, elementType);
}

bool GlobalTensorType::hasRank() const {
  return BaseTensorImpl<GlobalTensorType>::hasRank(*this);
}

//===----------------------------------------------------------------------===//
// LocalTensorType
//===----------------------------------------------------------------------===//

LocalTensorType LocalTensorType::get(ArrayRef<int64_t> shape,
                                     Type elementType) {
  return BaseTensorImpl<LocalTensorType>::get(shape, elementType);
}

LocalTensorType LocalTensorType::get(Type elementType) {
  return BaseTensorImpl<LocalTensorType>::get(elementType);
}

LocalTensorType LocalTensorType::get(BaseTensorType baseType) {
  return BaseTensorImpl<LocalTensorType>::get(baseType);
}

Type LocalTensorType::parse(AsmParser &odsParser) {
  return BaseTensorImpl<LocalTensorType>::parse(odsParser);
}

void LocalTensorType::print(AsmPrinter &odsPrinter) const {
  BaseTensorImpl<LocalTensorType>::print(*this, odsPrinter);
}

ShapedType LocalTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                      Type elementType) const {
  return BaseTensorImpl<LocalTensorType>::cloneWith(*this, shape, elementType);
}

bool LocalTensorType::hasRank() const {
  return BaseTensorImpl<LocalTensorType>::hasRank(*this);
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void AscendCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ascir/Dialect/Asc/IR/AscendCTypes.cpp.inc"
      >();
}
