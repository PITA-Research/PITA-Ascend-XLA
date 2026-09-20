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
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Enums.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::afir {

// Ascend graph type
enum class AscGraphType : uint8_t {
  Compute = 0,
  Invalid = 1,
};

::std::optional<AscGraphType> symbolizeAscGraphType(uint8_t);
::llvm::StringRef stringifyAscGraphType(AscGraphType);
::std::optional<AscGraphType> symbolizeAscGraphType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAscGraphType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(AscGraphType enumValue) {
  return stringifyAscGraphType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AscGraphType> symbolizeEnum<AscGraphType>(::llvm::StringRef str) {
  return symbolizeAscGraphType(str);
}

} // namespace mlir::afir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::afir::AscGraphType, ::mlir::afir::AscGraphType> {
  template <typename ParserT>
  static FailureOr<::mlir::afir::AscGraphType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Ascend graph type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::AscGraphType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::AscGraphType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [Compute, Invalid] for Ascend graph type, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::afir::AscGraphType>, std::optional<::mlir::afir::AscGraphType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::afir::AscGraphType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::afir::AscGraphType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::AscGraphType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::AscGraphType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [Compute, Invalid] for Ascend graph type, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::afir::AscGraphType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::afir::AscGraphType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::afir::AscGraphType getEmptyKey() {
    return static_cast<::mlir::afir::AscGraphType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::afir::AscGraphType getTombstoneKey() {
    return static_cast<::mlir::afir::AscGraphType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::afir::AscGraphType &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::afir::AscGraphType &lhs, const ::mlir::afir::AscGraphType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::afir {

// Axis type for computational loops
enum class AxisType : uint8_t {
  Original = 0,
  BlockOuter = 1,
  BlockInner = 2,
  TileOuter = 3,
  TileInner = 4,
  Merged = 5,
  Invalid = 6,
};

::std::optional<AxisType> symbolizeAxisType(uint8_t);
::llvm::StringRef stringifyAxisType(AxisType);
::std::optional<AxisType> symbolizeAxisType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAxisType() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(AxisType enumValue) {
  return stringifyAxisType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AxisType> symbolizeEnum<AxisType>(::llvm::StringRef str) {
  return symbolizeAxisType(str);
}

} // namespace mlir::afir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::afir::AxisType, ::mlir::afir::AxisType> {
  template <typename ParserT>
  static FailureOr<::mlir::afir::AxisType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Axis type for computational loops");

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::AxisType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::AxisType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [Original, BlockOuter, BlockInner, TileOuter, TileInner, Merged, Invalid] for Axis type for computational loops, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::afir::AxisType>, std::optional<::mlir::afir::AxisType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::afir::AxisType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::afir::AxisType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::AxisType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::AxisType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [Original, BlockOuter, BlockInner, TileOuter, TileInner, Merged, Invalid] for Axis type for computational loops, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::afir::AxisType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::afir::AxisType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::afir::AxisType getEmptyKey() {
    return static_cast<::mlir::afir::AxisType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::afir::AxisType getTombstoneKey() {
    return static_cast<::mlir::afir::AxisType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::afir::AxisType &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::afir::AxisType &lhs, const ::mlir::afir::AxisType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::afir {

// Compute type for node execution
enum class ComputeType : uint8_t {
  Load = 0,
  Store = 1,
  ReduceStore = 2,
  Elementwise = 3,
  Broadcast = 4,
  Reduce = 5,
  Transpose = 6,
  Concat = 7,
  Gather = 8,
  Cube = 9,
  Split = 10,
  Invalid = 11,
};

::std::optional<ComputeType> symbolizeComputeType(uint8_t);
::llvm::StringRef stringifyComputeType(ComputeType);
::std::optional<ComputeType> symbolizeComputeType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComputeType() {
  return 11;
}


inline ::llvm::StringRef stringifyEnum(ComputeType enumValue) {
  return stringifyComputeType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComputeType> symbolizeEnum<ComputeType>(::llvm::StringRef str) {
  return symbolizeComputeType(str);
}

class ComputeTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ComputeType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ComputeTypeAttr get(::mlir::MLIRContext *context, ComputeType val);
  ComputeType getValue() const;
};

} // namespace mlir::afir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::afir::ComputeType, ::mlir::afir::ComputeType> {
  template <typename ParserT>
  static FailureOr<::mlir::afir::ComputeType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Compute type for node execution");

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::ComputeType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::ComputeType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [load, store, reduce_store, elementwise, broadcast, reduce, transpose, concat, gather, cube, split, invalid] for Compute type for node execution, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::afir::ComputeType>, std::optional<::mlir::afir::ComputeType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::afir::ComputeType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::afir::ComputeType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::ComputeType> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::ComputeType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [load, store, reduce_store, elementwise, broadcast, reduce, transpose, concat, gather, cube, split, invalid] for Compute type for node execution, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::afir::ComputeType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::afir::ComputeType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::afir::ComputeType getEmptyKey() {
    return static_cast<::mlir::afir::ComputeType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::afir::ComputeType getTombstoneKey() {
    return static_cast<::mlir::afir::ComputeType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::afir::ComputeType &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::afir::ComputeType &lhs, const ::mlir::afir::ComputeType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::afir {

// Compute unit for node execution
enum class ComputeUnit : uint8_t {
  None = 0,
  MTE1 = 1,
  MTE2 = 2,
  MTE3 = 3,
  Scalar = 4,
  Vector = 5,
  Cube = 6,
  Invalid = 7,
};

::std::optional<ComputeUnit> symbolizeComputeUnit(uint8_t);
::llvm::StringRef stringifyComputeUnit(ComputeUnit);
::std::optional<ComputeUnit> symbolizeComputeUnit(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComputeUnit() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(ComputeUnit enumValue) {
  return stringifyComputeUnit(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ComputeUnit> symbolizeEnum<ComputeUnit>(::llvm::StringRef str) {
  return symbolizeComputeUnit(str);
}

class ComputeUnitAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ComputeUnit;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ComputeUnitAttr get(::mlir::MLIRContext *context, ComputeUnit val);
  ComputeUnit getValue() const;
};

} // namespace mlir::afir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::afir::ComputeUnit, ::mlir::afir::ComputeUnit> {
  template <typename ParserT>
  static FailureOr<::mlir::afir::ComputeUnit> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Compute unit for node execution");

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::ComputeUnit> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::ComputeUnit>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, mte1, mte2, mte3, scalar, vector, cube, invalid] for Compute unit for node execution, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::afir::ComputeUnit>, std::optional<::mlir::afir::ComputeUnit>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::afir::ComputeUnit>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::afir::ComputeUnit>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::ComputeUnit> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::ComputeUnit>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, mte1, mte2, mte3, scalar, vector, cube, invalid] for Compute unit for node execution, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::afir::ComputeUnit value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::afir::ComputeUnit> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::afir::ComputeUnit getEmptyKey() {
    return static_cast<::mlir::afir::ComputeUnit>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::afir::ComputeUnit getTombstoneKey() {
    return static_cast<::mlir::afir::ComputeUnit>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::afir::ComputeUnit &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::afir::ComputeUnit &lhs, const ::mlir::afir::ComputeUnit &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::afir {

// Memory position
enum class Position : uint8_t {
  GM = 0,
  VECTOR_IN = 1,
  VECTOR_OUT = 2,
  VECTOR_CALC = 3,
  L1 = 4,
  L2 = 5,
  L0A = 6,
  L0B = 7,
  L0C = 8,
};

::std::optional<Position> symbolizePosition(uint8_t);
::llvm::StringRef stringifyPosition(Position);
::std::optional<Position> symbolizePosition(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPosition() {
  return 8;
}


inline ::llvm::StringRef stringifyEnum(Position enumValue) {
  return stringifyPosition(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Position> symbolizeEnum<Position>(::llvm::StringRef str) {
  return symbolizePosition(str);
}

class PositionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Position;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static PositionAttr get(::mlir::MLIRContext *context, Position val);
  Position getValue() const;
};

} // namespace mlir::afir

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::afir::Position, ::mlir::afir::Position> {
  template <typename ParserT>
  static FailureOr<::mlir::afir::Position> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Memory position");

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::Position> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::Position>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [gm, vector_in, vector_out, vector_calc, l1, l2, l0a, l0b, l0c] for Memory position, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::afir::Position>, std::optional<::mlir::afir::Position>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::afir::Position>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::afir::Position>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::afir::Position> attr = ::mlir::afir::symbolizeEnum<::mlir::afir::Position>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [gm, vector_in, vector_out, vector_calc, l1, l2, l0a, l0b, l0c] for Memory position, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::afir::Position value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::afir::Position> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::afir::Position getEmptyKey() {
    return static_cast<::mlir::afir::Position>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::afir::Position getTombstoneKey() {
    return static_cast<::mlir::afir::Position>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::afir::Position &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::afir::Position &lhs, const ::mlir::afir::Position &rhs) {
    return lhs == rhs;
  }
};
}

