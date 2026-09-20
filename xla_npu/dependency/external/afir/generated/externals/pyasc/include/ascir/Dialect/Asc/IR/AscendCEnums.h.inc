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
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attributes.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 22, 23, 24, 25, 26, 27, 28
enum class AddressSpace : uint8_t {
  Default = 0,
  gm = 22,
  ca = 23,
  cb = 24,
  cc = 25,
  ubuf = 26,
  cbuf = 27,
  fbuf = 28,
};

::std::optional<AddressSpace> symbolizeAddressSpace(uint8_t);
::llvm::StringRef stringifyAddressSpace(AddressSpace);
::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddressSpace() {
  return 28;
}


inline ::llvm::StringRef stringifyEnum(AddressSpace enumValue) {
  return stringifyAddressSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddressSpace> symbolizeEnum<AddressSpace>(::llvm::StringRef str) {
  return symbolizeAddressSpace(str);
}

class AddressSpaceAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AddressSpace;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AddressSpaceAttr get(::mlir::MLIRContext *context, AddressSpace val);
  AddressSpace getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::AddressSpace, ::mlir::ascendc::AddressSpace> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::AddressSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 22, 23, 24, 25, 26, 27, 28");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::AddressSpace> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::AddressSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [Default, gm, ca, cb, cc, ubuf, cbuf, fbuf] for allowed 32-bit signless integer cases: 0, 22, 23, 24, 25, 26, 27, 28, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::AddressSpace>, std::optional<::mlir::ascendc::AddressSpace>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::AddressSpace>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::AddressSpace>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::AddressSpace> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::AddressSpace>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [Default, gm, ca, cb, cc, ubuf, cbuf, fbuf] for allowed 32-bit signless integer cases: 0, 22, 23, 24, 25, 26, 27, 28, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::AddressSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::AddressSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::AddressSpace getEmptyKey() {
    return static_cast<::mlir::ascendc::AddressSpace>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::AddressSpace getTombstoneKey() {
    return static_cast<::mlir::ascendc::AddressSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::AddressSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::AddressSpace &lhs, const ::mlir::ascendc::AddressSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 4, 9
enum class AippInputFormat : uint8_t {
  YUV420SP_U8 = 0,
  XRGB8888_U8 = 1,
  RGB888_U8 = 4,
  YUV400_U8 = 9,
};

::std::optional<AippInputFormat> symbolizeAippInputFormat(uint8_t);
::llvm::StringRef stringifyAippInputFormat(AippInputFormat);
::std::optional<AippInputFormat> symbolizeAippInputFormat(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAippInputFormat() {
  return 9;
}


inline ::llvm::StringRef stringifyEnum(AippInputFormat enumValue) {
  return stringifyAippInputFormat(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AippInputFormat> symbolizeEnum<AippInputFormat>(::llvm::StringRef str) {
  return symbolizeAippInputFormat(str);
}

class AippInputFormatAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = AippInputFormat;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static AippInputFormatAttr get(::mlir::MLIRContext *context, AippInputFormat val);
  AippInputFormat getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::AippInputFormat, ::mlir::ascendc::AippInputFormat> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::AippInputFormat> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 4, 9");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::AippInputFormat> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::AippInputFormat>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [YUV420SP_U8, XRGB8888_U8, RGB888_U8, YUV400_U8] for allowed 32-bit signless integer cases: 0, 1, 4, 9, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::AippInputFormat>, std::optional<::mlir::ascendc::AippInputFormat>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::AippInputFormat>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::AippInputFormat>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::AippInputFormat> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::AippInputFormat>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [YUV420SP_U8, XRGB8888_U8, RGB888_U8, YUV400_U8] for allowed 32-bit signless integer cases: 0, 1, 4, 9, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::AippInputFormat value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::AippInputFormat> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::AippInputFormat getEmptyKey() {
    return static_cast<::mlir::ascendc::AippInputFormat>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::AippInputFormat getTombstoneKey() {
    return static_cast<::mlir::ascendc::AippInputFormat>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::AippInputFormat &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::AippInputFormat &lhs, const ::mlir::ascendc::AippInputFormat &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3
enum class BatchMode : uint8_t {
  NONE = 0,
  BATCH_LESS_THAN_L1 = 1,
  BATCH_LARGE_THAN_L1 = 2,
  SINGLE_LARGE_THAN_L1 = 3,
};

::std::optional<BatchMode> symbolizeBatchMode(uint8_t);
::llvm::StringRef stringifyBatchMode(BatchMode);
::std::optional<BatchMode> symbolizeBatchMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBatchMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(BatchMode enumValue) {
  return stringifyBatchMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<BatchMode> symbolizeEnum<BatchMode>(::llvm::StringRef str) {
  return symbolizeBatchMode(str);
}

class BatchModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = BatchMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static BatchModeAttr get(::mlir::MLIRContext *context, BatchMode val);
  BatchMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::BatchMode, ::mlir::ascendc::BatchMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::BatchMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::BatchMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::BatchMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [NONE, BATCH_LESS_THAN_L1, BATCH_LARGE_THAN_L1, SINGLE_LARGE_THAN_L1] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::BatchMode>, std::optional<::mlir::ascendc::BatchMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::BatchMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::BatchMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::BatchMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::BatchMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [NONE, BATCH_LESS_THAN_L1, BATCH_LARGE_THAN_L1, SINGLE_LARGE_THAN_L1] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::BatchMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::BatchMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::BatchMode getEmptyKey() {
    return static_cast<::mlir::ascendc::BatchMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::BatchMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::BatchMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::BatchMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::BatchMode &lhs, const ::mlir::ascendc::BatchMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2
enum class BatchOutMode : uint8_t {
  SINGLE_BATCH = 0,
  MULTI_BATCH = 1,
  DYNAMIC = 2,
};

::std::optional<BatchOutMode> symbolizeBatchOutMode(uint8_t);
::llvm::StringRef stringifyBatchOutMode(BatchOutMode);
::std::optional<BatchOutMode> symbolizeBatchOutMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBatchOutMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(BatchOutMode enumValue) {
  return stringifyBatchOutMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<BatchOutMode> symbolizeEnum<BatchOutMode>(::llvm::StringRef str) {
  return symbolizeBatchOutMode(str);
}

class BatchOutModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = BatchOutMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static BatchOutModeAttr get(::mlir::MLIRContext *context, BatchOutMode val);
  BatchOutMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::BatchOutMode, ::mlir::ascendc::BatchOutMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::BatchOutMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::BatchOutMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::BatchOutMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [SINGLE_BATCH, MULTI_BATCH, DYNAMIC] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::BatchOutMode>, std::optional<::mlir::ascendc::BatchOutMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::BatchOutMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::BatchOutMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::BatchOutMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::BatchOutMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [SINGLE_BATCH, MULTI_BATCH, DYNAMIC] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::BatchOutMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::BatchOutMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::BatchOutMode getEmptyKey() {
    return static_cast<::mlir::ascendc::BatchOutMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::BatchOutMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::BatchOutMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::BatchOutMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::BatchOutMode &lhs, const ::mlir::ascendc::BatchOutMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5
enum class CMPMODE : uint64_t {
  LT = 0,
  GT = 1,
  EQ = 2,
  LE = 3,
  GE = 4,
  NE = 5,
};

::std::optional<CMPMODE> symbolizeCMPMODE(uint64_t);
::llvm::StringRef stringifyCMPMODE(CMPMODE);
::std::optional<CMPMODE> symbolizeCMPMODE(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCMPMODE() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(CMPMODE enumValue) {
  return stringifyCMPMODE(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CMPMODE> symbolizeEnum<CMPMODE>(::llvm::StringRef str) {
  return symbolizeCMPMODE(str);
}

class CMPMODEAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CMPMODE;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CMPMODEAttr get(::mlir::MLIRContext *context, CMPMODE val);
  CMPMODE getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CMPMODE, ::mlir::ascendc::CMPMODE> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CMPMODE> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CMPMODE> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CMPMODE>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [LT, GT, EQ, LE, GE, NE] for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CMPMODE>, std::optional<::mlir::ascendc::CMPMODE>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CMPMODE>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CMPMODE>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CMPMODE> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CMPMODE>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [LT, GT, EQ, LE, GE, NE] for allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CMPMODE value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CMPMODE> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::ascendc::CMPMODE getEmptyKey() {
    return static_cast<::mlir::ascendc::CMPMODE>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CMPMODE getTombstoneKey() {
    return static_cast<::mlir::ascendc::CMPMODE>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CMPMODE &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CMPMODE &lhs, const ::mlir::ascendc::CMPMODE &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2
enum class CO2Layout : uint8_t {
  NZ = 0,
  ROW_MAJOR = 1,
  COLUMN_MAJOR = 2,
};

::std::optional<CO2Layout> symbolizeCO2Layout(uint8_t);
::llvm::StringRef stringifyCO2Layout(CO2Layout);
::std::optional<CO2Layout> symbolizeCO2Layout(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCO2Layout() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(CO2Layout enumValue) {
  return stringifyCO2Layout(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CO2Layout> symbolizeEnum<CO2Layout>(::llvm::StringRef str) {
  return symbolizeCO2Layout(str);
}

class CO2LayoutAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CO2Layout;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CO2LayoutAttr get(::mlir::MLIRContext *context, CO2Layout val);
  CO2Layout getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CO2Layout, ::mlir::ascendc::CO2Layout> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CO2Layout> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CO2Layout> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CO2Layout>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [NZ, ROW_MAJOR, COLUMN_MAJOR] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CO2Layout>, std::optional<::mlir::ascendc::CO2Layout>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CO2Layout>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CO2Layout>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CO2Layout> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CO2Layout>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [NZ, ROW_MAJOR, COLUMN_MAJOR] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CO2Layout value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CO2Layout> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::CO2Layout getEmptyKey() {
    return static_cast<::mlir::ascendc::CO2Layout>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CO2Layout getTombstoneKey() {
    return static_cast<::mlir::ascendc::CO2Layout>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CO2Layout &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CO2Layout &lhs, const ::mlir::ascendc::CO2Layout &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1
enum class CacheLine : uint8_t {
  SINGLE_CACHE_LINE = 0,
  ENTIRE_DATA_CACHE = 1,
};

::std::optional<CacheLine> symbolizeCacheLine(uint8_t);
::llvm::StringRef stringifyCacheLine(CacheLine);
::std::optional<CacheLine> symbolizeCacheLine(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCacheLine() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(CacheLine enumValue) {
  return stringifyCacheLine(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CacheLine> symbolizeEnum<CacheLine>(::llvm::StringRef str) {
  return symbolizeCacheLine(str);
}

class CacheLineAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CacheLine;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CacheLineAttr get(::mlir::MLIRContext *context, CacheLine val);
  CacheLine getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CacheLine, ::mlir::ascendc::CacheLine> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CacheLine> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheLine> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheLine>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [single_cache_line, entire_data_cache] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CacheLine>, std::optional<::mlir::ascendc::CacheLine>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CacheLine>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CacheLine>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheLine> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheLine>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [single_cache_line, entire_data_cache] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CacheLine value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CacheLine> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::CacheLine getEmptyKey() {
    return static_cast<::mlir::ascendc::CacheLine>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CacheLine getTombstoneKey() {
    return static_cast<::mlir::ascendc::CacheLine>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CacheLine &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CacheLine &lhs, const ::mlir::ascendc::CacheLine &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1
enum class CacheMode : uint8_t {
  CACHE_MODE_DISABLE = 0,
  CACHE_MODE_NORMAL = 1,
};

::std::optional<CacheMode> symbolizeCacheMode(uint8_t);
::llvm::StringRef stringifyCacheMode(CacheMode);
::std::optional<CacheMode> symbolizeCacheMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCacheMode() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(CacheMode enumValue) {
  return stringifyCacheMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CacheMode> symbolizeEnum<CacheMode>(::llvm::StringRef str) {
  return symbolizeCacheMode(str);
}

class CacheModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CacheMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CacheModeAttr get(::mlir::MLIRContext *context, CacheMode val);
  CacheMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CacheMode, ::mlir::ascendc::CacheMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CacheMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [CACHE_MODE_DISABLE, CACHE_MODE_NORMAL] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CacheMode>, std::optional<::mlir::ascendc::CacheMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CacheMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CacheMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [CACHE_MODE_DISABLE, CACHE_MODE_NORMAL] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CacheMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CacheMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::CacheMode getEmptyKey() {
    return static_cast<::mlir::ascendc::CacheMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CacheMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::CacheMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CacheMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CacheMode &lhs, const ::mlir::ascendc::CacheMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 1, 2, 3
enum class CacheRwMode : uint8_t {
  READ = 1,
  WRITE = 2,
  RW = 3,
};

::std::optional<CacheRwMode> symbolizeCacheRwMode(uint8_t);
::llvm::StringRef stringifyCacheRwMode(CacheRwMode);
::std::optional<CacheRwMode> symbolizeCacheRwMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCacheRwMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(CacheRwMode enumValue) {
  return stringifyCacheRwMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CacheRwMode> symbolizeEnum<CacheRwMode>(::llvm::StringRef str) {
  return symbolizeCacheRwMode(str);
}

class CacheRwModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CacheRwMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CacheRwModeAttr get(::mlir::MLIRContext *context, CacheRwMode val);
  CacheRwMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CacheRwMode, ::mlir::ascendc::CacheRwMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CacheRwMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 1, 2, 3");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheRwMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheRwMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [READ, WRITE, RW] for allowed 32-bit signless integer cases: 1, 2, 3, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CacheRwMode>, std::optional<::mlir::ascendc::CacheRwMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CacheRwMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CacheRwMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CacheRwMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CacheRwMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [READ, WRITE, RW] for allowed 32-bit signless integer cases: 1, 2, 3, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CacheRwMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CacheRwMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::CacheRwMode getEmptyKey() {
    return static_cast<::mlir::ascendc::CacheRwMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CacheRwMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::CacheRwMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CacheRwMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CacheRwMode &lhs, const ::mlir::ascendc::CacheRwMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4
enum class CopyDirection : uint32_t {
  Unknown = 0,
  gm_ubuf = 1,
  ubuf_gm = 2,
  ubuf_ubuf = 3,
  gm_gm = 4,
};

::std::optional<CopyDirection> symbolizeCopyDirection(uint32_t);
::llvm::StringRef stringifyCopyDirection(CopyDirection);
::std::optional<CopyDirection> symbolizeCopyDirection(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCopyDirection() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(CopyDirection enumValue) {
  return stringifyCopyDirection(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CopyDirection> symbolizeEnum<CopyDirection>(::llvm::StringRef str) {
  return symbolizeCopyDirection(str);
}

class CopyDirectionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CopyDirection;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CopyDirectionAttr get(::mlir::MLIRContext *context, CopyDirection val);
  CopyDirection getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CopyDirection, ::mlir::ascendc::CopyDirection> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CopyDirection> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CopyDirection> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CopyDirection>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [Unknown, gm_ubuf, ubuf_gm, ubuf_ubuf, gm_gm] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CopyDirection>, std::optional<::mlir::ascendc::CopyDirection>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CopyDirection>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CopyDirection>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CopyDirection> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CopyDirection>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [Unknown, gm_ubuf, ubuf_gm, ubuf_ubuf, gm_gm] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CopyDirection value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CopyDirection> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::ascendc::CopyDirection getEmptyKey() {
    return static_cast<::mlir::ascendc::CopyDirection>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CopyDirection getTombstoneKey() {
    return static_cast<::mlir::ascendc::CopyDirection>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CopyDirection &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CopyDirection &lhs, const ::mlir::ascendc::CopyDirection &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// matmul tensor format
enum class CubeFormat : uint8_t {
  ND = 0,
  NZ = 1,
  ZN = 2,
  ZZ = 3,
  NN = 4,
  ND_ALIGN = 5,
  SCALAR = 6,
  VECTOR = 7,
};

::std::optional<CubeFormat> symbolizeCubeFormat(uint8_t);
::llvm::StringRef stringifyCubeFormat(CubeFormat);
::std::optional<CubeFormat> symbolizeCubeFormat(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCubeFormat() {
  return 7;
}


inline ::llvm::StringRef stringifyEnum(CubeFormat enumValue) {
  return stringifyCubeFormat(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CubeFormat> symbolizeEnum<CubeFormat>(::llvm::StringRef str) {
  return symbolizeCubeFormat(str);
}

class CubeFormatAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = CubeFormat;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static CubeFormatAttr get(::mlir::MLIRContext *context, CubeFormat val);
  CubeFormat getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::CubeFormat, ::mlir::ascendc::CubeFormat> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::CubeFormat> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for matmul tensor format");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CubeFormat> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CubeFormat>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [nd, nz, zn, zz, nn, nd_align, scalar, vector] for matmul tensor format, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::CubeFormat>, std::optional<::mlir::ascendc::CubeFormat>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::CubeFormat>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::CubeFormat>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::CubeFormat> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::CubeFormat>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [nd, nz, zn, zz, nn, nd_align, scalar, vector] for matmul tensor format, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::CubeFormat value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::CubeFormat> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::CubeFormat getEmptyKey() {
    return static_cast<::mlir::ascendc::CubeFormat>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::CubeFormat getTombstoneKey() {
    return static_cast<::mlir::ascendc::CubeFormat>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::CubeFormat &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::CubeFormat &lhs, const ::mlir::ascendc::CubeFormat &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3
enum class DcciDst : uint8_t {
  CACHELINE_ALL = 0,
  CACHELINE_UB = 1,
  CACHELINE_OUT = 2,
  CACHELINE_ATOMIC = 3,
};

::std::optional<DcciDst> symbolizeDcciDst(uint8_t);
::llvm::StringRef stringifyDcciDst(DcciDst);
::std::optional<DcciDst> symbolizeDcciDst(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDcciDst() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(DcciDst enumValue) {
  return stringifyDcciDst(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DcciDst> symbolizeEnum<DcciDst>(::llvm::StringRef str) {
  return symbolizeDcciDst(str);
}

class DcciDstAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DcciDst;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DcciDstAttr get(::mlir::MLIRContext *context, DcciDst val);
  DcciDst getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::DcciDst, ::mlir::ascendc::DcciDst> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::DcciDst> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::DcciDst> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::DcciDst>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [cacheline_all, cacheline_ub, cacheline_out, cacheline_atomic] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::DcciDst>, std::optional<::mlir::ascendc::DcciDst>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::DcciDst>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::DcciDst>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::DcciDst> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::DcciDst>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [cacheline_all, cacheline_ub, cacheline_out, cacheline_atomic] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::DcciDst value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::DcciDst> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::DcciDst getEmptyKey() {
    return static_cast<::mlir::ascendc::DcciDst>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::DcciDst getTombstoneKey() {
    return static_cast<::mlir::ascendc::DcciDst>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::DcciDst &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::DcciDst &lhs, const ::mlir::ascendc::DcciDst &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1
enum class FmatrixMode : uint8_t {
  FMATRIX_LEFT = 0,
  FMATRIX_RIGHT = 1,
};

::std::optional<FmatrixMode> symbolizeFmatrixMode(uint8_t);
::llvm::StringRef stringifyFmatrixMode(FmatrixMode);
::std::optional<FmatrixMode> symbolizeFmatrixMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFmatrixMode() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(FmatrixMode enumValue) {
  return stringifyFmatrixMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<FmatrixMode> symbolizeEnum<FmatrixMode>(::llvm::StringRef str) {
  return symbolizeFmatrixMode(str);
}

class FmatrixModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = FmatrixMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static FmatrixModeAttr get(::mlir::MLIRContext *context, FmatrixMode val);
  FmatrixMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::FmatrixMode, ::mlir::ascendc::FmatrixMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::FmatrixMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::FmatrixMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::FmatrixMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [fmatrix_left, fmatrix_right] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::FmatrixMode>, std::optional<::mlir::ascendc::FmatrixMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::FmatrixMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::FmatrixMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::FmatrixMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::FmatrixMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [fmatrix_left, fmatrix_right] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::FmatrixMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::FmatrixMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::FmatrixMode getEmptyKey() {
    return static_cast<::mlir::ascendc::FmatrixMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::FmatrixMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::FmatrixMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::FmatrixMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::FmatrixMode &lhs, const ::mlir::ascendc::FmatrixMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0
enum class GatherMaskMode : uint8_t {
  DEFAULT = 0,
};

::std::optional<GatherMaskMode> symbolizeGatherMaskMode(uint8_t);
::llvm::StringRef stringifyGatherMaskMode(GatherMaskMode);
::std::optional<GatherMaskMode> symbolizeGatherMaskMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForGatherMaskMode() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(GatherMaskMode enumValue) {
  return stringifyGatherMaskMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<GatherMaskMode> symbolizeEnum<GatherMaskMode>(::llvm::StringRef str) {
  return symbolizeGatherMaskMode(str);
}

class GatherMaskModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = GatherMaskMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static GatherMaskModeAttr get(::mlir::MLIRContext *context, GatherMaskMode val);
  GatherMaskMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::GatherMaskMode, ::mlir::ascendc::GatherMaskMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::GatherMaskMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::GatherMaskMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::GatherMaskMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [DEFAULT] for allowed 32-bit signless integer cases: 0, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::GatherMaskMode>, std::optional<::mlir::ascendc::GatherMaskMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::GatherMaskMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::GatherMaskMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::GatherMaskMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::GatherMaskMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [DEFAULT] for allowed 32-bit signless integer cases: 0, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::GatherMaskMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::GatherMaskMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::GatherMaskMode getEmptyKey() {
    return static_cast<::mlir::ascendc::GatherMaskMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::GatherMaskMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::GatherMaskMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::GatherMaskMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::GatherMaskMode &lhs, const ::mlir::ascendc::GatherMaskMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
enum class HardEvent : uint8_t {
  MTE2_MTE1 = 0,
  MTE1_MTE2 = 1,
  MTE1_M = 2,
  M_MTE1 = 3,
  MTE2_V = 4,
  V_MTE2 = 5,
  MTE3_V = 6,
  V_MTE3 = 7,
  M_V = 8,
  V_M = 9,
  V_V = 10,
  MTE3_MTE1 = 11,
  MTE1_MTE3 = 12,
  MTE1_V = 13,
  MTE2_M = 14,
  M_MTE2 = 15,
  V_MTE1 = 16,
  M_FIX = 17,
  FIX_M = 18,
  MTE3_MTE2 = 19,
  MTE2_MTE3 = 20,
  S_V = 21,
  V_S = 22,
  S_MTE2 = 23,
  MTE2_S = 24,
  S_MTE3 = 25,
  MTE3_S = 26,
  MTE2_FIX = 27,
  FIX_MTE2 = 28,
  FIX_S = 29,
  M_S = 30,
  FIX_MTE3 = 31,
  MTE1_FIX = 32,
  FIX_MTE1 = 33,
  FIX_FIX = 34,
  MAX = 35,
};

::std::optional<HardEvent> symbolizeHardEvent(uint8_t);
::llvm::StringRef stringifyHardEvent(HardEvent);
::std::optional<HardEvent> symbolizeHardEvent(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForHardEvent() {
  return 35;
}


inline ::llvm::StringRef stringifyEnum(HardEvent enumValue) {
  return stringifyHardEvent(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<HardEvent> symbolizeEnum<HardEvent>(::llvm::StringRef str) {
  return symbolizeHardEvent(str);
}

class HardEventAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = HardEvent;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static HardEventAttr get(::mlir::MLIRContext *context, HardEvent val);
  HardEvent getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::HardEvent, ::mlir::ascendc::HardEvent> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::HardEvent> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::HardEvent> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::HardEvent>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [mte2_mte1, mte1_mte2, mte1_m, m_mte1, mte2_v, v_mte2, mte3_v, v_mte3, m_v, v_m, v_v, mte3_mte1, mte1_mte3, mte1_v, mte2_m, m_mte2, v_mte1, m_fix, fix_m, mte3_mte2, mte2_mte3, s_v, v_s, s_mte2, mte2_s, s_mte3, mte3_s, mte2_fix, fix_mte2, fix_s, m_s, fix_mte3, mte1_fix, fix_mte1, fix_fix, max] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::HardEvent>, std::optional<::mlir::ascendc::HardEvent>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::HardEvent>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::HardEvent>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::HardEvent> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::HardEvent>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [mte2_mte1, mte1_mte2, mte1_m, m_mte1, mte2_v, v_mte2, mte3_v, v_mte3, m_v, v_m, v_v, mte3_mte1, mte1_mte3, mte1_v, mte2_m, m_mte2, v_mte1, m_fix, fix_m, mte3_mte2, mte2_mte3, s_v, v_s, s_mte2, mte2_s, s_mte3, mte3_s, mte2_fix, fix_mte2, fix_s, m_s, fix_mte3, mte1_fix, fix_mte1, fix_fix, max] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::HardEvent value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::HardEvent> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::HardEvent getEmptyKey() {
    return static_cast<::mlir::ascendc::HardEvent>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::HardEvent getTombstoneKey() {
    return static_cast<::mlir::ascendc::HardEvent>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::HardEvent &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::HardEvent &lhs, const ::mlir::ascendc::HardEvent &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 1, 2, 4, 8, 255
enum class IterateMode : uint8_t {
  ITERATE_MODE_NORMAL = 1,
  ITERATE_MODE_ALL = 2,
  ITERATE_MODE_BATCH = 4,
  ITERATE_MODE_N_BATCH = 8,
  ITERATE_MODE_DEFAULT = 255,
};

::std::optional<IterateMode> symbolizeIterateMode(uint8_t);
::llvm::StringRef stringifyIterateMode(IterateMode);
::std::optional<IterateMode> symbolizeIterateMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForIterateMode() {
  return 255;
}


inline ::llvm::StringRef stringifyEnum(IterateMode enumValue) {
  return stringifyIterateMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<IterateMode> symbolizeEnum<IterateMode>(::llvm::StringRef str) {
  return symbolizeIterateMode(str);
}

class IterateModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = IterateMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static IterateModeAttr get(::mlir::MLIRContext *context, IterateMode val);
  IterateMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::IterateMode, ::mlir::ascendc::IterateMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::IterateMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 1, 2, 4, 8, 255");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::IterateMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::IterateMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [ITERATE_MODE_NORMAL, ITERATE_MODE_ALL, ITERATE_MODE_BATCH, ITERATE_MODE_N_BATCH, ITERATE_MODE_DEFAULT] for allowed 32-bit signless integer cases: 1, 2, 4, 8, 255, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::IterateMode>, std::optional<::mlir::ascendc::IterateMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::IterateMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::IterateMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::IterateMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::IterateMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [ITERATE_MODE_NORMAL, ITERATE_MODE_ALL, ITERATE_MODE_BATCH, ITERATE_MODE_N_BATCH, ITERATE_MODE_DEFAULT] for allowed 32-bit signless integer cases: 1, 2, 4, 8, 255, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::IterateMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::IterateMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::IterateMode getEmptyKey() {
    return static_cast<::mlir::ascendc::IterateMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::IterateMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::IterateMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::IterateMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::IterateMode &lhs, const ::mlir::ascendc::IterateMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2
enum class IterateOrder : uint8_t {
  ORDER_M = 0,
  ORDER_N = 1,
  UNDEF = 2,
};

::std::optional<IterateOrder> symbolizeIterateOrder(uint8_t);
::llvm::StringRef stringifyIterateOrder(IterateOrder);
::std::optional<IterateOrder> symbolizeIterateOrder(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForIterateOrder() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(IterateOrder enumValue) {
  return stringifyIterateOrder(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<IterateOrder> symbolizeEnum<IterateOrder>(::llvm::StringRef str) {
  return symbolizeIterateOrder(str);
}

class IterateOrderAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = IterateOrder;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static IterateOrderAttr get(::mlir::MLIRContext *context, IterateOrder val);
  IterateOrder getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::IterateOrder, ::mlir::ascendc::IterateOrder> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::IterateOrder> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::IterateOrder> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::IterateOrder>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [ORDER_M, ORDER_N, UNDEF] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::IterateOrder>, std::optional<::mlir::ascendc::IterateOrder>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::IterateOrder>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::IterateOrder>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::IterateOrder> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::IterateOrder>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [ORDER_M, ORDER_N, UNDEF] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::IterateOrder value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::IterateOrder> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::IterateOrder getEmptyKey() {
    return static_cast<::mlir::ascendc::IterateOrder>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::IterateOrder getTombstoneKey() {
    return static_cast<::mlir::ascendc::IterateOrder>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::IterateOrder &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::IterateOrder &lhs, const ::mlir::ascendc::IterateOrder &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// matmul tensor layout
enum class LayoutMode : uint8_t {
  NONE = 0,
  NORMAL = 1,
  BSNGD = 2,
  SBNGD = 3,
  BNGS1S2 = 4,
};

::std::optional<LayoutMode> symbolizeLayoutMode(uint8_t);
::llvm::StringRef stringifyLayoutMode(LayoutMode);
::std::optional<LayoutMode> symbolizeLayoutMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLayoutMode() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(LayoutMode enumValue) {
  return stringifyLayoutMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LayoutMode> symbolizeEnum<LayoutMode>(::llvm::StringRef str) {
  return symbolizeLayoutMode(str);
}

class LayoutModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = LayoutMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static LayoutModeAttr get(::mlir::MLIRContext *context, LayoutMode val);
  LayoutMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::LayoutMode, ::mlir::ascendc::LayoutMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::LayoutMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for matmul tensor layout");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::LayoutMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::LayoutMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, normal, bsngd, sbngd, bngs1s2] for matmul tensor layout, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::LayoutMode>, std::optional<::mlir::ascendc::LayoutMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::LayoutMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::LayoutMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::LayoutMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::LayoutMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, normal, bsngd, sbngd, bngs1s2] for matmul tensor layout, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::LayoutMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::LayoutMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::LayoutMode getEmptyKey() {
    return static_cast<::mlir::ascendc::LayoutMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::LayoutMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::LayoutMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::LayoutMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::LayoutMode &lhs, const ::mlir::ascendc::LayoutMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1
enum class MaskMode : uint8_t {
  NORMAL = 0,
  COUNTER = 1,
};

::std::optional<MaskMode> symbolizeMaskMode(uint8_t);
::llvm::StringRef stringifyMaskMode(MaskMode);
::std::optional<MaskMode> symbolizeMaskMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMaskMode() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MaskMode enumValue) {
  return stringifyMaskMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MaskMode> symbolizeEnum<MaskMode>(::llvm::StringRef str) {
  return symbolizeMaskMode(str);
}

class MaskModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MaskMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MaskModeAttr get(::mlir::MLIRContext *context, MaskMode val);
  MaskMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::MaskMode, ::mlir::ascendc::MaskMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::MaskMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::MaskMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::MaskMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [NORMAL, COUNTER] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::MaskMode>, std::optional<::mlir::ascendc::MaskMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::MaskMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::MaskMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::MaskMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::MaskMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [NORMAL, COUNTER] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::MaskMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::MaskMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::MaskMode getEmptyKey() {
    return static_cast<::mlir::ascendc::MaskMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::MaskMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::MaskMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::MaskMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::MaskMode &lhs, const ::mlir::ascendc::MaskMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace Detail::Impl::AscendC {

// allowed 32-bit signless integer cases: 0, 1, 2, 3
enum class MatmulPolicy : uint8_t {
  MatmulPolicy = 0,
  TrianUpperMatmulPolicy = 1,
  TrianLowerMatmulPolicy = 2,
  NBuffer33MatmulPolicy = 3,
};

::std::optional<MatmulPolicy> symbolizeMatmulPolicy(uint8_t);
::llvm::StringRef stringifyMatmulPolicy(MatmulPolicy);
::std::optional<MatmulPolicy> symbolizeMatmulPolicy(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMatmulPolicy() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(MatmulPolicy enumValue) {
  return stringifyMatmulPolicy(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MatmulPolicy> symbolizeEnum<MatmulPolicy>(::llvm::StringRef str) {
  return symbolizeMatmulPolicy(str);
}

class MatmulPolicyAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MatmulPolicy;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MatmulPolicyAttr get(::mlir::MLIRContext *context, MatmulPolicy val);
  MatmulPolicy getValue() const;
};

} // namespace Detail::Impl::AscendC

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::Detail::Impl::AscendC::MatmulPolicy, ::Detail::Impl::AscendC::MatmulPolicy> {
  template <typename ParserT>
  static FailureOr<::Detail::Impl::AscendC::MatmulPolicy> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3");

    // Symbolize the keyword.
    if (::std::optional<::Detail::Impl::AscendC::MatmulPolicy> attr = ::Detail::Impl::AscendC::symbolizeEnum<::Detail::Impl::AscendC::MatmulPolicy>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [MatmulPolicy, TrianUpperMatmulPolicy, TrianLowerMatmulPolicy, NBuffer33MatmulPolicy] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::Detail::Impl::AscendC::MatmulPolicy>, std::optional<::Detail::Impl::AscendC::MatmulPolicy>> {
  template <typename ParserT>
  static FailureOr<std::optional<::Detail::Impl::AscendC::MatmulPolicy>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::Detail::Impl::AscendC::MatmulPolicy>{};

    // Symbolize the keyword.
    if (::std::optional<::Detail::Impl::AscendC::MatmulPolicy> attr = ::Detail::Impl::AscendC::symbolizeEnum<::Detail::Impl::AscendC::MatmulPolicy>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [MatmulPolicy, TrianUpperMatmulPolicy, TrianLowerMatmulPolicy, NBuffer33MatmulPolicy] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::Detail::Impl::AscendC::MatmulPolicy value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::Detail::Impl::AscendC::MatmulPolicy> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::Detail::Impl::AscendC::MatmulPolicy getEmptyKey() {
    return static_cast<::Detail::Impl::AscendC::MatmulPolicy>(StorageInfo::getEmptyKey());
  }

  static inline ::Detail::Impl::AscendC::MatmulPolicy getTombstoneKey() {
    return static_cast<::Detail::Impl::AscendC::MatmulPolicy>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::Detail::Impl::AscendC::MatmulPolicy &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::Detail::Impl::AscendC::MatmulPolicy &lhs, const ::Detail::Impl::AscendC::MatmulPolicy &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3
enum class MemDsbT : uint8_t {
  ALL = 0,
  DDR = 1,
  UB = 2,
  SEQ = 3,
};

::std::optional<MemDsbT> symbolizeMemDsbT(uint8_t);
::llvm::StringRef stringifyMemDsbT(MemDsbT);
::std::optional<MemDsbT> symbolizeMemDsbT(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemDsbT() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(MemDsbT enumValue) {
  return stringifyMemDsbT(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemDsbT> symbolizeEnum<MemDsbT>(::llvm::StringRef str) {
  return symbolizeMemDsbT(str);
}

class MemDsbTAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MemDsbT;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MemDsbTAttr get(::mlir::MLIRContext *context, MemDsbT val);
  MemDsbT getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::MemDsbT, ::mlir::ascendc::MemDsbT> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::MemDsbT> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::MemDsbT> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::MemDsbT>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [all, ddr, ub, seq] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::MemDsbT>, std::optional<::mlir::ascendc::MemDsbT>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::MemDsbT>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::MemDsbT>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::MemDsbT> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::MemDsbT>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [all, ddr, ub, seq] for allowed 32-bit signless integer cases: 0, 1, 2, 3, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::MemDsbT value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::MemDsbT> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::MemDsbT getEmptyKey() {
    return static_cast<::mlir::ascendc::MemDsbT>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::MemDsbT getTombstoneKey() {
    return static_cast<::mlir::ascendc::MemDsbT>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::MemDsbT &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::MemDsbT &lhs, const ::mlir::ascendc::MemDsbT &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
enum class Pipe : uint8_t {
  PIPE_S = 0,
  PIPE_V = 1,
  PIPE_M = 2,
  PIPE_MTE1 = 3,
  PIPE_MTE2 = 4,
  PIPE_MTE3 = 5,
  PIPE_ALL = 6,
  PIPE_MTE4 = 7,
  PIPE_MTE5 = 8,
  PIPE_V2 = 9,
  PIPE_FIX = 10,
};

::std::optional<Pipe> symbolizePipe(uint8_t);
::llvm::StringRef stringifyPipe(Pipe);
::std::optional<Pipe> symbolizePipe(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForPipe() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(Pipe enumValue) {
  return stringifyPipe(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Pipe> symbolizeEnum<Pipe>(::llvm::StringRef str) {
  return symbolizePipe(str);
}

class PipeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Pipe;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static PipeAttr get(::mlir::MLIRContext *context, Pipe val);
  Pipe getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::Pipe, ::mlir::ascendc::Pipe> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::Pipe> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::Pipe> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::Pipe>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [pipe_s, pipe_v, pipe_m, pipe_mte1, pipe_mte2, pipe_mte3, pipe_all, pipe_mte4, pipe_mte5, pipe_v2, pipe_fix] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::Pipe>, std::optional<::mlir::ascendc::Pipe>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::Pipe>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::Pipe>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::Pipe> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::Pipe>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [pipe_s, pipe_v, pipe_m, pipe_mte1, pipe_mte2, pipe_mte3, pipe_all, pipe_mte4, pipe_mte5, pipe_v2, pipe_fix] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::Pipe value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::Pipe> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::Pipe getEmptyKey() {
    return static_cast<::mlir::ascendc::Pipe>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::Pipe getTombstoneKey() {
    return static_cast<::mlir::ascendc::Pipe>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::Pipe &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::Pipe &lhs, const ::mlir::ascendc::Pipe &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
enum class QuantMode : uint32_t {
  NoQuant = 0,
  F322F16 = 1,
  VQF322HIF8_PRE = 2,
  QF322HIF8_PRE = 3,
  VQF322HIF8_PRE_HYBRID = 4,
  QF322HIF8_PRE_HYBRID = 5,
  AttachF16Mul = 6,
  VREQ8 = 8,
  REQ8 = 9,
  VDEQF16 = 10,
  DEQF16 = 11,
  VQF322FP8_PRE = 12,
  QF322FP8_PRE = 13,
  VQF322F32_PRE = 14,
  QF322F32_PRE = 15,
  F322BF16 = 16,
  VQF162B8_PRE = 17,
  QF162B8_PRE = 18,
  VQF162S4_PRE = 19,
  QF162S4_PRE = 20,
  VREQ4 = 21,
  REQ4 = 22,
  VQF322B8_PRE = 23,
  QF322B8_PRE = 24,
  VQF322S4_PRE = 25,
  QF322S4_PRE = 26,
  VDEQS16 = 27,
  DEQS16 = 28,
  VQF162S16_PRE = 29,
  QF162S16_PRE = 30,
  VQF322F16_PRE = 31,
  QF322F16_PRE = 32,
  VQF322BF16_PRE = 33,
  QF322BF16_PRE = 34,
  VQS322BF16_PRE = 35,
  QS322BF16_PRE = 36,
};

::std::optional<QuantMode> symbolizeQuantMode(uint32_t);
::llvm::StringRef stringifyQuantMode(QuantMode);
::std::optional<QuantMode> symbolizeQuantMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForQuantMode() {
  return 36;
}


inline ::llvm::StringRef stringifyEnum(QuantMode enumValue) {
  return stringifyQuantMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<QuantMode> symbolizeEnum<QuantMode>(::llvm::StringRef str) {
  return symbolizeQuantMode(str);
}

class QuantModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = QuantMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static QuantModeAttr get(::mlir::MLIRContext *context, QuantMode val);
  QuantMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::QuantMode, ::mlir::ascendc::QuantMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::QuantMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::QuantMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::QuantMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [no_quant, f322f16, vqf322hif8_pre, qf322hif8_pre, vqf322hif8_pre_hybrid, qf322hif8_pre_hybrid, attachf16mul, vreq8, req8, vdeqf16, deqf16, vqf322fp8_pre, qf322fp8_pre, vqf322f32_pre, qf322f32_pre, f322bf16, vqf162b8_pre, qf162b8_pre, vqf162s4_pre, qf162s4_pre, vreq4, req4, vqf322b8_pre, qf322b8_pre, vqf322s4_pre, qf322s4_pre, vdeqs16, deqs16, vqf162s16_pre, qf162s16_pre, vqf322f16_pre, qf322f16_pre, vqf322bf16_pre, qf322bf16_pre, vqs322bf16_pre, qs322bf16_pre] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::QuantMode>, std::optional<::mlir::ascendc::QuantMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::QuantMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::QuantMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::QuantMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::QuantMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [no_quant, f322f16, vqf322hif8_pre, qf322hif8_pre, vqf322hif8_pre_hybrid, qf322hif8_pre_hybrid, attachf16mul, vreq8, req8, vdeqf16, deqf16, vqf322fp8_pre, qf322fp8_pre, vqf322f32_pre, qf322f32_pre, f322bf16, vqf162b8_pre, qf162b8_pre, vqf162s4_pre, qf162s4_pre, vreq4, req4, vqf322b8_pre, qf322b8_pre, vqf322s4_pre, qf322s4_pre, vdeqs16, deqs16, vqf162s16_pre, qf162s16_pre, vqf322f16_pre, qf322f16_pre, vqf322bf16_pre, qf322bf16_pre, vqs322bf16_pre, qs322bf16_pre] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::QuantMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::QuantMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::ascendc::QuantMode getEmptyKey() {
    return static_cast<::mlir::ascendc::QuantMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::QuantMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::QuantMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::QuantMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::QuantMode &lhs, const ::mlir::ascendc::QuantMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// reduce result order and behavior
enum class ReduceOrder : uint8_t {
  ORDER_VALUE_INDEX = 0,
  ORDER_INDEX_VALUE = 1,
  ORDER_ONLY_VALUE = 2,
  ORDER_ONLY_INDEX = 3,
};

::std::optional<ReduceOrder> symbolizeReduceOrder(uint8_t);
::llvm::StringRef stringifyReduceOrder(ReduceOrder);
::std::optional<ReduceOrder> symbolizeReduceOrder(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReduceOrder() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(ReduceOrder enumValue) {
  return stringifyReduceOrder(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReduceOrder> symbolizeEnum<ReduceOrder>(::llvm::StringRef str) {
  return symbolizeReduceOrder(str);
}

class ReduceOrderAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ReduceOrder;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ReduceOrderAttr get(::mlir::MLIRContext *context, ReduceOrder val);
  ReduceOrder getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::ReduceOrder, ::mlir::ascendc::ReduceOrder> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::ReduceOrder> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for reduce result order and behavior");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::ReduceOrder> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::ReduceOrder>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [order_value_index, order_index_value, order_only_value, order_only_index] for reduce result order and behavior, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::ReduceOrder>, std::optional<::mlir::ascendc::ReduceOrder>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::ReduceOrder>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::ReduceOrder>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::ReduceOrder> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::ReduceOrder>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [order_value_index, order_index_value, order_only_value, order_only_index] for reduce result order and behavior, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::ReduceOrder value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::ReduceOrder> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::ReduceOrder getEmptyKey() {
    return static_cast<::mlir::ascendc::ReduceOrder>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::ReduceOrder getTombstoneKey() {
    return static_cast<::mlir::ascendc::ReduceOrder>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::ReduceOrder &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::ReduceOrder &lhs, const ::mlir::ascendc::ReduceOrder &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6
enum class RoundMode : uint8_t {
  CAST_NONE = 0,
  CAST_RINT = 1,
  CAST_FLOOR = 2,
  CAST_CEIL = 3,
  CAST_ROUND = 4,
  CAST_TRUNC = 5,
  CAST_ODD = 6,
};

::std::optional<RoundMode> symbolizeRoundMode(uint8_t);
::llvm::StringRef stringifyRoundMode(RoundMode);
::std::optional<RoundMode> symbolizeRoundMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRoundMode() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(RoundMode enumValue) {
  return stringifyRoundMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RoundMode> symbolizeEnum<RoundMode>(::llvm::StringRef str) {
  return symbolizeRoundMode(str);
}

class RoundModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = RoundMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static RoundModeAttr get(::mlir::MLIRContext *context, RoundMode val);
  RoundMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::RoundMode, ::mlir::ascendc::RoundMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::RoundMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::RoundMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::RoundMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [CAST_NONE, CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_ODD] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::RoundMode>, std::optional<::mlir::ascendc::RoundMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::RoundMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::RoundMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::RoundMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::RoundMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [CAST_NONE, CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_ODD] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::RoundMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::RoundMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::RoundMode getEmptyKey() {
    return static_cast<::mlir::ascendc::RoundMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::RoundMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::RoundMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::RoundMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::RoundMode &lhs, const ::mlir::ascendc::RoundMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1
enum class ScheduleType : uint8_t {
  INNER_PRODUCT = 0,
  OUTER_PRODUCT = 1,
};

::std::optional<ScheduleType> symbolizeScheduleType(uint8_t);
::llvm::StringRef stringifyScheduleType(ScheduleType);
::std::optional<ScheduleType> symbolizeScheduleType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForScheduleType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ScheduleType enumValue) {
  return stringifyScheduleType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ScheduleType> symbolizeEnum<ScheduleType>(::llvm::StringRef str) {
  return symbolizeScheduleType(str);
}

class ScheduleTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ScheduleType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ScheduleTypeAttr get(::mlir::MLIRContext *context, ScheduleType val);
  ScheduleType getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::ScheduleType, ::mlir::ascendc::ScheduleType> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::ScheduleType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::ScheduleType> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::ScheduleType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [INNER_PRODUCT, OUTER_PRODUCT] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::ScheduleType>, std::optional<::mlir::ascendc::ScheduleType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::ScheduleType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::ScheduleType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::ScheduleType> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::ScheduleType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [INNER_PRODUCT, OUTER_PRODUCT] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::ScheduleType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::ScheduleType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::ScheduleType getEmptyKey() {
    return static_cast<::mlir::ascendc::ScheduleType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::ScheduleType getTombstoneKey() {
    return static_cast<::mlir::ascendc::ScheduleType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::ScheduleType &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::ScheduleType &lhs, const ::mlir::ascendc::ScheduleType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// allowed 32-bit signless integer cases: 0, 1, 2
enum class SelMode : uint8_t {
  VSEL_CMPMASK_SPR = 0,
  VSEL_TENSOR_SCALAR_MODE = 1,
  VSEL_TENSOR_TENSOR_MODE = 2,
};

::std::optional<SelMode> symbolizeSelMode(uint8_t);
::llvm::StringRef stringifySelMode(SelMode);
::std::optional<SelMode> symbolizeSelMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSelMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(SelMode enumValue) {
  return stringifySelMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SelMode> symbolizeEnum<SelMode>(::llvm::StringRef str) {
  return symbolizeSelMode(str);
}

class SelModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = SelMode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static SelModeAttr get(::mlir::MLIRContext *context, SelMode val);
  SelMode getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::SelMode, ::mlir::ascendc::SelMode> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::SelMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::SelMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::SelMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [vsel_cmpmask_spr, vsel_tensor_scalar_mode, vsel_tensor_tensor_mode] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::SelMode>, std::optional<::mlir::ascendc::SelMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::SelMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::SelMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::SelMode> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::SelMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [vsel_cmpmask_spr, vsel_tensor_scalar_mode, vsel_tensor_tensor_mode] for allowed 32-bit signless integer cases: 0, 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::SelMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::SelMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::SelMode getEmptyKey() {
    return static_cast<::mlir::ascendc::SelMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::SelMode getTombstoneKey() {
    return static_cast<::mlir::ascendc::SelMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::SelMode &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::SelMode &lhs, const ::mlir::ascendc::SelMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// queue/buffer position
enum class TPosition : uint8_t {
  GM = 0,
  A1 = 1,
  A2 = 2,
  B1 = 3,
  B2 = 4,
  C1 = 5,
  C2 = 6,
  CO1 = 7,
  CO2 = 8,
  VECIN = 9,
  VECOUT = 10,
  VECCALC = 11,
};

::std::optional<TPosition> symbolizeTPosition(uint8_t);
::llvm::StringRef stringifyTPosition(TPosition);
::std::optional<TPosition> symbolizeTPosition(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTPosition() {
  return 11;
}


inline ::llvm::StringRef stringifyEnum(TPosition enumValue) {
  return stringifyTPosition(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TPosition> symbolizeEnum<TPosition>(::llvm::StringRef str) {
  return symbolizeTPosition(str);
}

class TPositionAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TPosition;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TPositionAttr get(::mlir::MLIRContext *context, TPosition val);
  TPosition getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::TPosition, ::mlir::ascendc::TPosition> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::TPosition> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for queue/buffer position");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::TPosition> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::TPosition>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [gm, a1, a2, b1, b2, c1, c2, co1, co2, vecin, vecout, veccalc] for queue/buffer position, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::TPosition>, std::optional<::mlir::ascendc::TPosition>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::TPosition>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::TPosition>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::TPosition> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::TPosition>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [gm, a1, a2, b1, b2, c1, c2, co1, co2, vecin, vecout, veccalc] for queue/buffer position, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::TPosition value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::TPosition> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::TPosition getEmptyKey() {
    return static_cast<::mlir::ascendc::TPosition>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::TPosition getTombstoneKey() {
    return static_cast<::mlir::ascendc::TPosition>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::TPosition &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::TPosition &lhs, const ::mlir::ascendc::TPosition &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::ascendc {

// Transpose type.
enum class TransposeType : uint8_t {
  TRANSPOSE_TYPE_NONE = 0,
  TRANSPOSE_NZ2ND_0213 = 1,
  TRANSPOSE_NZ2NZ_0213 = 2,
  TRANSPOSE_NZ2NZ_012_WITH_N = 3,
  TRANSPOSE_NZ2ND_012_WITH_N = 4,
  TRANSPOSE_NZ2ND_012_WITHOUT_N = 5,
  TRANSPOSE_NZ2NZ_012_WITHOUT_N = 6,
  TRANSPOSE_ND2ND_ONLY = 7,
  TRANSPOSE_ND_UB_GM = 8,
  TRANSPOSE_GRAD_ND_UB_GM = 9,
  TRANSPOSE_ND2ND_B16 = 10,
  TRANSPOSE_NCHW2NHWC = 11,
  TRANSPOSE_NHWC2NCHW = 12,
};

::std::optional<TransposeType> symbolizeTransposeType(uint8_t);
::llvm::StringRef stringifyTransposeType(TransposeType);
::std::optional<TransposeType> symbolizeTransposeType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTransposeType() {
  return 12;
}


inline ::llvm::StringRef stringifyEnum(TransposeType enumValue) {
  return stringifyTransposeType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TransposeType> symbolizeEnum<TransposeType>(::llvm::StringRef str) {
  return symbolizeTransposeType(str);
}

class TransposeTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TransposeType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TransposeTypeAttr get(::mlir::MLIRContext *context, TransposeType val);
  TransposeType getValue() const;
};

} // namespace mlir::ascendc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::ascendc::TransposeType, ::mlir::ascendc::TransposeType> {
  template <typename ParserT>
  static FailureOr<::mlir::ascendc::TransposeType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Transpose type.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::TransposeType> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::TransposeType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [TRANSPOSE_TYPE_NONE, TRANSPOSE_NZ2ND_0213, TRANSPOSE_NZ2NZ_0213, TRANSPOSE_NZ2NZ_012_WITH_N, TRANSPOSE_NZ2ND_012_WITH_N, TRANSPOSE_NZ2ND_012_WITHOUT_N, TRANSPOSE_NZ2NZ_012_WITHOUT_N, TRANSPOSE_ND2ND_ONLY, TRANSPOSE_ND_UB_GM, TRANSPOSE_GRAD_ND_UB_GM, TRANSPOSE_ND2ND_B16, TRANSPOSE_NCHW2NHWC, TRANSPOSE_NHWC2NCHW] for Transpose type., got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::ascendc::TransposeType>, std::optional<::mlir::ascendc::TransposeType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::ascendc::TransposeType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::ascendc::TransposeType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::ascendc::TransposeType> attr = ::mlir::ascendc::symbolizeEnum<::mlir::ascendc::TransposeType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [TRANSPOSE_TYPE_NONE, TRANSPOSE_NZ2ND_0213, TRANSPOSE_NZ2NZ_0213, TRANSPOSE_NZ2NZ_012_WITH_N, TRANSPOSE_NZ2ND_012_WITH_N, TRANSPOSE_NZ2ND_012_WITHOUT_N, TRANSPOSE_NZ2NZ_012_WITHOUT_N, TRANSPOSE_ND2ND_ONLY, TRANSPOSE_ND_UB_GM, TRANSPOSE_GRAD_ND_UB_GM, TRANSPOSE_ND2ND_B16, TRANSPOSE_NCHW2NHWC, TRANSPOSE_NHWC2NCHW] for Transpose type., got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::ascendc::TransposeType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::ascendc::TransposeType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint8_t>;

  static inline ::mlir::ascendc::TransposeType getEmptyKey() {
    return static_cast<::mlir::ascendc::TransposeType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::ascendc::TransposeType getTombstoneKey() {
    return static_cast<::mlir::ascendc::TransposeType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::ascendc::TransposeType &val) {
    return StorageInfo::getHashValue(static_cast<uint8_t>(val));
  }

  static bool isEqual(const ::mlir::ascendc::TransposeType &lhs, const ::mlir::ascendc::TransposeType &rhs) {
    return lhs == rhs;
  }
};
}

