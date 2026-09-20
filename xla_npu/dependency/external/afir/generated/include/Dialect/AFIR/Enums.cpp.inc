/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Enums.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::afir {

::llvm::StringRef stringifyAscGraphType(AscGraphType val) {
  switch (val) {
    case AscGraphType::Compute: return "Compute";
    case AscGraphType::Invalid: return "Invalid";
  }
  return "";
}

::std::optional<AscGraphType> symbolizeAscGraphType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AscGraphType>>(str)
      .Case("Compute", AscGraphType::Compute)
      .Case("Invalid", AscGraphType::Invalid)
      .Default(::std::nullopt);
}
::std::optional<AscGraphType> symbolizeAscGraphType(uint8_t value) {
  switch (value) {
  case 0: return AscGraphType::Compute;
  case 1: return AscGraphType::Invalid;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::afir
namespace mlir::afir {

::llvm::StringRef stringifyAxisType(AxisType val) {
  switch (val) {
    case AxisType::Original: return "Original";
    case AxisType::BlockOuter: return "BlockOuter";
    case AxisType::BlockInner: return "BlockInner";
    case AxisType::TileOuter: return "TileOuter";
    case AxisType::TileInner: return "TileInner";
    case AxisType::Merged: return "Merged";
    case AxisType::Invalid: return "Invalid";
  }
  return "";
}

::std::optional<AxisType> symbolizeAxisType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AxisType>>(str)
      .Case("Original", AxisType::Original)
      .Case("BlockOuter", AxisType::BlockOuter)
      .Case("BlockInner", AxisType::BlockInner)
      .Case("TileOuter", AxisType::TileOuter)
      .Case("TileInner", AxisType::TileInner)
      .Case("Merged", AxisType::Merged)
      .Case("Invalid", AxisType::Invalid)
      .Default(::std::nullopt);
}
::std::optional<AxisType> symbolizeAxisType(uint8_t value) {
  switch (value) {
  case 0: return AxisType::Original;
  case 1: return AxisType::BlockOuter;
  case 2: return AxisType::BlockInner;
  case 3: return AxisType::TileOuter;
  case 4: return AxisType::TileInner;
  case 5: return AxisType::Merged;
  case 6: return AxisType::Invalid;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::afir
namespace mlir::afir {

::llvm::StringRef stringifyComputeType(ComputeType val) {
  switch (val) {
    case ComputeType::Load: return "load";
    case ComputeType::Store: return "store";
    case ComputeType::ReduceStore: return "reduce_store";
    case ComputeType::Elementwise: return "elementwise";
    case ComputeType::Broadcast: return "broadcast";
    case ComputeType::Reduce: return "reduce";
    case ComputeType::Transpose: return "transpose";
    case ComputeType::Concat: return "concat";
    case ComputeType::Gather: return "gather";
    case ComputeType::Cube: return "cube";
    case ComputeType::Split: return "split";
    case ComputeType::Invalid: return "invalid";
  }
  return "";
}

::std::optional<ComputeType> symbolizeComputeType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComputeType>>(str)
      .Case("load", ComputeType::Load)
      .Case("store", ComputeType::Store)
      .Case("reduce_store", ComputeType::ReduceStore)
      .Case("elementwise", ComputeType::Elementwise)
      .Case("broadcast", ComputeType::Broadcast)
      .Case("reduce", ComputeType::Reduce)
      .Case("transpose", ComputeType::Transpose)
      .Case("concat", ComputeType::Concat)
      .Case("gather", ComputeType::Gather)
      .Case("cube", ComputeType::Cube)
      .Case("split", ComputeType::Split)
      .Case("invalid", ComputeType::Invalid)
      .Default(::std::nullopt);
}
::std::optional<ComputeType> symbolizeComputeType(uint8_t value) {
  switch (value) {
  case 0: return ComputeType::Load;
  case 1: return ComputeType::Store;
  case 2: return ComputeType::ReduceStore;
  case 3: return ComputeType::Elementwise;
  case 4: return ComputeType::Broadcast;
  case 5: return ComputeType::Reduce;
  case 6: return ComputeType::Transpose;
  case 7: return ComputeType::Concat;
  case 8: return ComputeType::Gather;
  case 9: return ComputeType::Cube;
  case 10: return ComputeType::Split;
  case 11: return ComputeType::Invalid;
  default: return ::std::nullopt;
  }
}

bool ComputeTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))));
}
ComputeTypeAttr ComputeTypeAttr::get(::mlir::MLIRContext *context, ComputeType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<ComputeTypeAttr>(baseAttr);
}
ComputeType ComputeTypeAttr::getValue() const {
  return static_cast<ComputeType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::afir
namespace mlir::afir {

::llvm::StringRef stringifyComputeUnit(ComputeUnit val) {
  switch (val) {
    case ComputeUnit::None: return "none";
    case ComputeUnit::MTE1: return "mte1";
    case ComputeUnit::MTE2: return "mte2";
    case ComputeUnit::MTE3: return "mte3";
    case ComputeUnit::Scalar: return "scalar";
    case ComputeUnit::Vector: return "vector";
    case ComputeUnit::Cube: return "cube";
    case ComputeUnit::Invalid: return "invalid";
  }
  return "";
}

::std::optional<ComputeUnit> symbolizeComputeUnit(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ComputeUnit>>(str)
      .Case("none", ComputeUnit::None)
      .Case("mte1", ComputeUnit::MTE1)
      .Case("mte2", ComputeUnit::MTE2)
      .Case("mte3", ComputeUnit::MTE3)
      .Case("scalar", ComputeUnit::Scalar)
      .Case("vector", ComputeUnit::Vector)
      .Case("cube", ComputeUnit::Cube)
      .Case("invalid", ComputeUnit::Invalid)
      .Default(::std::nullopt);
}
::std::optional<ComputeUnit> symbolizeComputeUnit(uint8_t value) {
  switch (value) {
  case 0: return ComputeUnit::None;
  case 1: return ComputeUnit::MTE1;
  case 2: return ComputeUnit::MTE2;
  case 3: return ComputeUnit::MTE3;
  case 4: return ComputeUnit::Scalar;
  case 5: return ComputeUnit::Vector;
  case 6: return ComputeUnit::Cube;
  case 7: return ComputeUnit::Invalid;
  default: return ::std::nullopt;
  }
}

bool ComputeUnitAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))));
}
ComputeUnitAttr ComputeUnitAttr::get(::mlir::MLIRContext *context, ComputeUnit val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<ComputeUnitAttr>(baseAttr);
}
ComputeUnit ComputeUnitAttr::getValue() const {
  return static_cast<ComputeUnit>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::afir
namespace mlir::afir {

::llvm::StringRef stringifyPosition(Position val) {
  switch (val) {
    case Position::GM: return "gm";
    case Position::VECTOR_IN: return "vector_in";
    case Position::VECTOR_OUT: return "vector_out";
    case Position::VECTOR_CALC: return "vector_calc";
    case Position::L1: return "l1";
    case Position::L2: return "l2";
    case Position::L0A: return "l0a";
    case Position::L0B: return "l0b";
    case Position::L0C: return "l0c";
  }
  return "";
}

::std::optional<Position> symbolizePosition(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Position>>(str)
      .Case("gm", Position::GM)
      .Case("vector_in", Position::VECTOR_IN)
      .Case("vector_out", Position::VECTOR_OUT)
      .Case("vector_calc", Position::VECTOR_CALC)
      .Case("l1", Position::L1)
      .Case("l2", Position::L2)
      .Case("l0a", Position::L0A)
      .Case("l0b", Position::L0B)
      .Case("l0c", Position::L0C)
      .Default(::std::nullopt);
}
::std::optional<Position> symbolizePosition(uint8_t value) {
  switch (value) {
  case 0: return Position::GM;
  case 1: return Position::VECTOR_IN;
  case 2: return Position::VECTOR_OUT;
  case 3: return Position::VECTOR_CALC;
  case 4: return Position::L1;
  case 5: return Position::L2;
  case 6: return Position::L0A;
  case 7: return Position::L0B;
  case 8: return Position::L0C;
  default: return ::std::nullopt;
  }
}

bool PositionAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))));
}
PositionAttr PositionAttr::get(::mlir::MLIRContext *context, Position val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<PositionAttr>(baseAttr);
}
Position PositionAttr::getValue() const {
  return static_cast<Position>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::afir
