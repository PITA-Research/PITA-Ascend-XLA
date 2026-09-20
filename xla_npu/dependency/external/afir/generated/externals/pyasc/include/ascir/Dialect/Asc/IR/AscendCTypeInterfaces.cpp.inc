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

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Returns a clone of this type with the given shape and element type.
/// 
/// If no shape is provided, the shape of this type is used. In that case, if
/// this type is unranked, so is the resulting type.
/// 
/// If a shape is provided, the resulting type is always ranked, even if this
/// type is unranked.
::mlir::ShapedType mlir::ascendc::BaseTensorType::cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const {
      return getImpl()->implShapedType->cloneWith(getImpl()->implShapedType, *this, shape, elementType);
  }

/// Returns the element type of this shaped type.
::mlir::Type mlir::ascendc::BaseTensorType::getElementType() const {
      return getImpl()->implShapedType->getElementType(getImpl()->implShapedType, *this);
  }

/// Returns if this type is ranked, i.e. it has a known number of dimensions.
bool mlir::ascendc::BaseTensorType::hasRank() const {
      return getImpl()->implShapedType->hasRank(getImpl()->implShapedType, *this);
  }

/// Returns the shape of this type if it is ranked, otherwise asserts.
::llvm::ArrayRef<int64_t> mlir::ascendc::BaseTensorType::getShape() const {
      return getImpl()->implShapedType->getShape(getImpl()->implShapedType, *this);
  }
