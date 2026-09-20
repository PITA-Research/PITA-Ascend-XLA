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
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::afir::PositionConfigAttr,
::mlir::afir::AscTensorGroupsAttr,
::mlir::afir::AxisAttr,
::mlir::afir::TmpBufDescAttr,
::mlir::afir::AscGraphAttrGroupsAttr,
::mlir::afir::ScalarValueAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::afir::PositionConfigAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::PositionConfigAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::afir::AscTensorGroupsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::AscTensorGroupsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::afir::AxisAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::AxisAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::afir::TmpBufDescAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::TmpBufDescAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::afir::AscGraphAttrGroupsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::AscGraphAttrGroupsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::afir::ScalarValueAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::afir::ScalarValueAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::afir::PositionConfigAttr>([&](auto t) {
      printer << ::mlir::afir::PositionConfigAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::afir::AscTensorGroupsAttr>([&](auto t) {
      printer << ::mlir::afir::AscTensorGroupsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::afir::AxisAttr>([&](auto t) {
      printer << ::mlir::afir::AxisAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::afir::TmpBufDescAttr>([&](auto t) {
      printer << ::mlir::afir::TmpBufDescAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::afir::AscGraphAttrGroupsAttr>([&](auto t) {
      printer << ::mlir::afir::AscGraphAttrGroupsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::afir::ScalarValueAttr>([&](auto t) {
      printer << ::mlir::afir::ScalarValueAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::afir {

namespace detail {

struct PositionConfigAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::afir::Position, uint32_t, bool>;
  PositionConfigAttrStorage(::mlir::afir::Position position, uint32_t depth, bool is_double_buffer) : position(std::move(position)), depth(std::move(depth)), is_double_buffer(std::move(is_double_buffer)) {}

  KeyTy getAsKey() const {
    return KeyTy(position, depth, is_double_buffer);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (position == std::get<0>(tblgenKey)) && (depth == std::get<1>(tblgenKey)) && (is_double_buffer == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static PositionConfigAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto position = std::move(std::get<0>(tblgenKey));
    auto depth = std::move(std::get<1>(tblgenKey));
    auto is_double_buffer = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<PositionConfigAttrStorage>()) PositionConfigAttrStorage(std::move(position), std::move(depth), std::move(is_double_buffer));
  }

  ::mlir::afir::Position position;
  uint32_t depth;
  bool is_double_buffer;
};

} // namespace detail
PositionConfigAttr PositionConfigAttr::get(::mlir::MLIRContext *context, ::mlir::afir::Position position, uint32_t depth, bool is_double_buffer) {
  return Base::get(context, std::move(position), std::move(depth), std::move(is_double_buffer));
}

::mlir::Attribute PositionConfigAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::afir::Position> _result_position;
  ::mlir::FailureOr<uint32_t> _result_depth;
  ::mlir::FailureOr<bool> _result_is_double_buffer;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'position'
  _result_position = ::mlir::FieldParser<::mlir::afir::Position>::parse(odsParser);
  if (::mlir::failed(_result_position)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PositionConfigAttr parameter 'position' which is to be a `::mlir::afir::Position`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'depth'
    if (odsParser.parseKeyword("depth")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'depth'
    _result_depth = ::mlir::FieldParser<uint32_t>::parse(odsParser);
    if (::mlir::failed(_result_depth)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PositionConfigAttr parameter 'depth' which is to be a `uint32_t`");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'is_double_buffer'
    if (odsParser.parseKeyword("is_double_buffer")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'is_double_buffer'
    _result_is_double_buffer = ::mlir::FieldParser<bool>::parse(odsParser);
    if (::mlir::failed(_result_is_double_buffer)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse PositionConfigAttr parameter 'is_double_buffer' which is to be a `bool`");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_position));
  return PositionConfigAttr::get(odsParser.getContext(),
      ::mlir::afir::Position((*_result_position)),
      uint32_t((_result_depth.value_or(0))),
      bool((_result_is_double_buffer.value_or(false))));
}

void PositionConfigAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getPosition());
  if (!(getDepth() == 0)) {
    odsPrinter << ",";
    odsPrinter << ' ' << "depth";
    odsPrinter << ' ' << "=";
    if (!(getDepth() == 0)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getDepth());
    }
  } else {
  }
  if (!(getIsDoubleBuffer() == false)) {
    odsPrinter << ",";
    odsPrinter << ' ' << "is_double_buffer";
    odsPrinter << ' ' << "=";
    if (!(getIsDoubleBuffer() == false)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getIsDoubleBuffer());
    }
  } else {
  }
  odsPrinter << ">";
}

::mlir::afir::Position PositionConfigAttr::getPosition() const {
  return getImpl()->position;
}

uint32_t PositionConfigAttr::getDepth() const {
  return getImpl()->depth;
}

bool PositionConfigAttr::getIsDoubleBuffer() const {
  return getImpl()->is_double_buffer;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::PositionConfigAttr)
namespace mlir::afir {

namespace detail {

struct AscTensorGroupsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<Attribute>, int64_t, int64_t, PositionConfigAttr, int64_t>;
  AscTensorGroupsAttrStorage(::llvm::ArrayRef<int64_t> vectorized_axis, ::llvm::ArrayRef<Attribute> vectorized_strides, int64_t tensor_id, int64_t reuse_id, PositionConfigAttr position_config, int64_t position_id) : vectorized_axis(std::move(vectorized_axis)), vectorized_strides(std::move(vectorized_strides)), tensor_id(std::move(tensor_id)), reuse_id(std::move(reuse_id)), position_config(std::move(position_config)), position_id(std::move(position_id)) {}

  KeyTy getAsKey() const {
    return KeyTy(vectorized_axis, vectorized_strides, tensor_id, reuse_id, position_config, position_id);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (::llvm::ArrayRef<int64_t>(vectorized_axis) == ::llvm::ArrayRef<int64_t>(std::get<0>(tblgenKey))) && (::llvm::ArrayRef<Attribute>(vectorized_strides) == ::llvm::ArrayRef<Attribute>(std::get<1>(tblgenKey))) && (tensor_id == std::get<2>(tblgenKey)) && (reuse_id == std::get<3>(tblgenKey)) && (position_config == std::get<4>(tblgenKey)) && (position_id == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static AscTensorGroupsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto vectorized_axis = std::move(std::get<0>(tblgenKey));
    auto vectorized_strides = std::move(std::get<1>(tblgenKey));
    auto tensor_id = std::move(std::get<2>(tblgenKey));
    auto reuse_id = std::move(std::get<3>(tblgenKey));
    auto position_config = std::move(std::get<4>(tblgenKey));
    auto position_id = std::move(std::get<5>(tblgenKey));
    vectorized_axis = allocator.copyInto(vectorized_axis);
    vectorized_strides = allocator.copyInto(vectorized_strides);
    return new (allocator.allocate<AscTensorGroupsAttrStorage>()) AscTensorGroupsAttrStorage(std::move(vectorized_axis), std::move(vectorized_strides), std::move(tensor_id), std::move(reuse_id), std::move(position_config), std::move(position_id));
  }

  ::llvm::ArrayRef<int64_t> vectorized_axis;
  ::llvm::ArrayRef<Attribute> vectorized_strides;
  int64_t tensor_id;
  int64_t reuse_id;
  PositionConfigAttr position_config;
  int64_t position_id;
};

} // namespace detail
AscTensorGroupsAttr AscTensorGroupsAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> vectorized_axis, ::llvm::ArrayRef<Attribute> vectorized_strides, int64_t tensor_id, int64_t reuse_id, PositionConfigAttr position_config, int64_t position_id) {
  return Base::get(context, std::move(vectorized_axis), std::move(vectorized_strides), std::move(tensor_id), std::move(reuse_id), std::move(position_config), std::move(position_id));
}

::mlir::Attribute AscTensorGroupsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_vectorized_axis;
  ::mlir::FailureOr<::llvm::SmallVector<Attribute>> _result_vectorized_strides;
  ::mlir::FailureOr<int64_t> _result_tensor_id;
  ::mlir::FailureOr<int64_t> _result_reuse_id;
  ::mlir::FailureOr<PositionConfigAttr> _result_position_config;
  ::mlir::FailureOr<int64_t> _result_position_id;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'vectorized_axis'
  if (odsParser.parseOptionalKeyword("vectorized_axis")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};
    // Parse literal '['
    if (odsParser.parseLSquare()) return {};

    // Parse variable 'vectorized_axis'
    _result_vectorized_axis = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
    if (::mlir::failed(_result_vectorized_axis)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'vectorized_axis' which is to be a `::llvm::ArrayRef<int64_t>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
    // Parse literal ','
    if (odsParser.parseComma()) return {};
    // Parse literal 'vectorized_strides'
    if (odsParser.parseKeyword("vectorized_strides")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};
    // Parse literal '['
    if (odsParser.parseLSquare()) return {};

    // Parse variable 'vectorized_strides'
    _result_vectorized_strides = ::mlir::FieldParser<::llvm::SmallVector<Attribute>>::parse(odsParser);
    if (::mlir::failed(_result_vectorized_strides)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'vectorized_strides' which is to be a `::llvm::ArrayRef<Attribute>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  // Parse literal 'tensor_id'
  if (odsParser.parseKeyword("tensor_id")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'tensor_id'
  _result_tensor_id = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_tensor_id)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'tensor_id' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'reuse_id'
  if (odsParser.parseOptionalKeyword("reuse_id")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'reuse_id'
    _result_reuse_id = ::mlir::FieldParser<int64_t>::parse(odsParser);
    if (::mlir::failed(_result_reuse_id)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'reuse_id' which is to be a `int64_t`");
      return {};
    }
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  // Parse literal 'position'
  if (odsParser.parseKeyword("position")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'position_config'
  _result_position_config = ::mlir::FieldParser<PositionConfigAttr>::parse(odsParser);
  if (::mlir::failed(_result_position_config)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'position_config' which is to be a `PositionConfigAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'position_id'
    if (odsParser.parseKeyword("position_id")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'position_id'
    _result_position_id = ::mlir::FieldParser<int64_t>::parse(odsParser);
    if (::mlir::failed(_result_position_id)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscTensorGroupsAttr parameter 'position_id' which is to be a `int64_t`");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tensor_id));
  assert(::mlir::succeeded(_result_position_config));
  return AscTensorGroupsAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((_result_vectorized_axis.value_or(::llvm::SmallVector<int64_t>()))),
      ::llvm::ArrayRef<Attribute>((_result_vectorized_strides.value_or(::llvm::SmallVector<Attribute>()))),
      int64_t((*_result_tensor_id)),
      int64_t((_result_reuse_id.value_or(-1))),
      PositionConfigAttr((*_result_position_config)),
      int64_t((_result_position_id.value_or(-1))));
}

void AscTensorGroupsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  if (!(::llvm::ArrayRef<int64_t>(getVectorizedAxis()) == ::llvm::ArrayRef<int64_t>(::llvm::SmallVector<int64_t>()))) {
    odsPrinter << "vectorized_axis";
    odsPrinter << ' ' << "=";
    odsPrinter << ' ' << "[";
    if (!(::llvm::ArrayRef<int64_t>(getVectorizedAxis()) == ::llvm::ArrayRef<int64_t>(::llvm::SmallVector<int64_t>()))) {
      odsPrinter.printStrippedAttrOrType(getVectorizedAxis());
    }
    odsPrinter << "]";
    odsPrinter << ",";
    odsPrinter << ' ' << "vectorized_strides";
    odsPrinter << ' ' << "=";
    odsPrinter << ' ' << "[";
    if (!(::llvm::ArrayRef<Attribute>(getVectorizedStrides()) == ::llvm::ArrayRef<Attribute>(::llvm::SmallVector<Attribute>()))) {
      odsPrinter.printStrippedAttrOrType(getVectorizedStrides());
    }
    odsPrinter << "]";
    odsPrinter << ",";
  } else {
  }
  odsPrinter << "tensor_id";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTensorId());
  odsPrinter << ",";
  if (!(getReuseId() == -1)) {
    odsPrinter << ' ' << "reuse_id";
    odsPrinter << ' ' << "=";
    if (!(getReuseId() == -1)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getReuseId());
    }
    odsPrinter << ",";
  } else {
  }
  odsPrinter << ' ' << "position";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getPositionConfig());
  if (!(getPositionId() == -1)) {
    odsPrinter << ",";
    odsPrinter << ' ' << "position_id";
    odsPrinter << ' ' << "=";
    if (!(getPositionId() == -1)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getPositionId());
    }
  } else {
  }
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> AscTensorGroupsAttr::getVectorizedAxis() const {
  return getImpl()->vectorized_axis;
}

::llvm::ArrayRef<Attribute> AscTensorGroupsAttr::getVectorizedStrides() const {
  return getImpl()->vectorized_strides;
}

int64_t AscTensorGroupsAttr::getTensorId() const {
  return getImpl()->tensor_id;
}

int64_t AscTensorGroupsAttr::getReuseId() const {
  return getImpl()->reuse_id;
}

PositionConfigAttr AscTensorGroupsAttr::getPositionConfig() const {
  return getImpl()->position_config;
}

int64_t AscTensorGroupsAttr::getPositionId() const {
  return getImpl()->position_id;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::AscTensorGroupsAttr)
namespace mlir::afir {

namespace detail {

struct AxisAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, StringAttr, ::mlir::afir::AxisType, bool, StringAttr, StringAttr, ::llvm::ArrayRef<int64_t>>;
  AxisAttrStorage(int64_t id, StringAttr name, ::mlir::afir::AxisType axis_type, bool bind_block, StringAttr size, StringAttr align, ::llvm::ArrayRef<int64_t> from) : id(std::move(id)), name(std::move(name)), axis_type(std::move(axis_type)), bind_block(std::move(bind_block)), size(std::move(size)), align(std::move(align)), from(std::move(from)) {}

  KeyTy getAsKey() const {
    return KeyTy(id, name, axis_type, bind_block, size, align, from);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (id == std::get<0>(tblgenKey)) && (name == std::get<1>(tblgenKey)) && (axis_type == std::get<2>(tblgenKey)) && (bind_block == std::get<3>(tblgenKey)) && (size == std::get<4>(tblgenKey)) && (align == std::get<5>(tblgenKey)) && (::llvm::ArrayRef<int64_t>(from) == ::llvm::ArrayRef<int64_t>(std::get<6>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static AxisAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto id = std::move(std::get<0>(tblgenKey));
    auto name = std::move(std::get<1>(tblgenKey));
    auto axis_type = std::move(std::get<2>(tblgenKey));
    auto bind_block = std::move(std::get<3>(tblgenKey));
    auto size = std::move(std::get<4>(tblgenKey));
    auto align = std::move(std::get<5>(tblgenKey));
    auto from = std::move(std::get<6>(tblgenKey));
    from = allocator.copyInto(from);
    return new (allocator.allocate<AxisAttrStorage>()) AxisAttrStorage(std::move(id), std::move(name), std::move(axis_type), std::move(bind_block), std::move(size), std::move(align), std::move(from));
  }

  int64_t id;
  StringAttr name;
  ::mlir::afir::AxisType axis_type;
  bool bind_block;
  StringAttr size;
  StringAttr align;
  ::llvm::ArrayRef<int64_t> from;
};

} // namespace detail
AxisAttr AxisAttr::get(::mlir::MLIRContext *context, int64_t id, StringAttr name, ::mlir::afir::AxisType axis_type, bool bind_block, StringAttr size, StringAttr align, ::llvm::ArrayRef<int64_t> from) {
  return Base::get(context, std::move(id), std::move(name), std::move(axis_type), std::move(bind_block), std::move(size), std::move(align), std::move(from));
}

::mlir::Attribute AxisAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_id;
  ::mlir::FailureOr<StringAttr> _result_name;
  ::mlir::FailureOr<::mlir::afir::AxisType> _result_axis_type;
  ::mlir::FailureOr<bool> _result_bind_block;
  ::mlir::FailureOr<StringAttr> _result_size;
  ::mlir::FailureOr<StringAttr> _result_align;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_from;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'id'
  if (odsParser.parseKeyword("id")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'id'
  _result_id = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_id)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'id' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'name'
  if (odsParser.parseKeyword("name")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'axis_type'
  if (odsParser.parseKeyword("axis_type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'axis_type'
  _result_axis_type = ::mlir::FieldParser<::mlir::afir::AxisType>::parse(odsParser);
  if (::mlir::failed(_result_axis_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'axis_type' which is to be a `::mlir::afir::AxisType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'bind_block'
  if (odsParser.parseOptionalKeyword("bind_block")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'bind_block'
    _result_bind_block = ::mlir::FieldParser<bool>::parse(odsParser);
    if (::mlir::failed(_result_bind_block)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'bind_block' which is to be a `bool`");
      return {};
    }
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  // Parse literal 'size'
  if (odsParser.parseKeyword("size")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'size'
  _result_size = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_size)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'size' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'align'
    if (odsParser.parseKeyword("align")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'align'
    _result_align = ::mlir::FieldParser<StringAttr>::parse(odsParser);
    if (::mlir::failed(_result_align)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'align' which is to be a `StringAttr`");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'from'
    if (odsParser.parseKeyword("from")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};
    // Parse literal '['
    if (odsParser.parseLSquare()) return {};

    // Parse variable 'from'
    _result_from = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
    if (::mlir::failed(_result_from)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AxisAttr parameter 'from' which is to be a `::llvm::ArrayRef<int64_t>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  assert(::mlir::succeeded(_result_axis_type));
  assert(::mlir::succeeded(_result_size));
  return AxisAttr::get(odsParser.getContext(),
      int64_t((_result_id.value_or(-1))),
      StringAttr((*_result_name)),
      ::mlir::afir::AxisType((*_result_axis_type)),
      bool((_result_bind_block.value_or(false))),
      StringAttr((*_result_size)),
      StringAttr((_result_align.value_or(StringAttr()))),
      ::llvm::ArrayRef<int64_t>((_result_from.value_or(::llvm::SmallVector<int64_t>()))));
}

void AxisAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "id";
  odsPrinter << ' ' << "=";
  if (!(getId() == -1)) {
    odsPrinter << ' ';
    odsPrinter.printStrippedAttrOrType(getId());
  }
  odsPrinter << ",";
  odsPrinter << ' ' << "name";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ",";
  odsPrinter << ' ' << "axis_type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getAxisType());
  odsPrinter << ",";
  if (!(getBindBlock() == false)) {
    odsPrinter << ' ' << "bind_block";
    odsPrinter << ' ' << "=";
    if (!(getBindBlock() == false)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getBindBlock());
    }
    odsPrinter << ",";
  } else {
  }
  odsPrinter << ' ' << "size";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getSize());
  if (!(getAlign() == StringAttr())) {
    odsPrinter << ",";
    odsPrinter << ' ' << "align";
    odsPrinter << ' ' << "=";
    if (!(getAlign() == StringAttr())) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getAlign());
    }
  } else {
  }
  if (!(::llvm::ArrayRef<int64_t>(getFrom()) == ::llvm::ArrayRef<int64_t>(::llvm::SmallVector<int64_t>()))) {
    odsPrinter << ",";
    odsPrinter << ' ' << "from";
    odsPrinter << ' ' << "=";
    odsPrinter << ' ' << "[";
    if (!(::llvm::ArrayRef<int64_t>(getFrom()) == ::llvm::ArrayRef<int64_t>(::llvm::SmallVector<int64_t>()))) {
      odsPrinter.printStrippedAttrOrType(getFrom());
    }
    odsPrinter << "]";
  } else {
  }
  odsPrinter << ">";
}

int64_t AxisAttr::getId() const {
  return getImpl()->id;
}

StringAttr AxisAttr::getName() const {
  return getImpl()->name;
}

::mlir::afir::AxisType AxisAttr::getAxisType() const {
  return getImpl()->axis_type;
}

bool AxisAttr::getBindBlock() const {
  return getImpl()->bind_block;
}

StringAttr AxisAttr::getSize() const {
  return getImpl()->size;
}

StringAttr AxisAttr::getAlign() const {
  return getImpl()->align;
}

::llvm::ArrayRef<int64_t> AxisAttr::getFrom() const {
  return getImpl()->from;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::AxisAttr)
namespace mlir::afir {

namespace detail {

struct TmpBufDescAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, int64_t>;
  TmpBufDescAttrStorage(StringAttr size, int64_t life_time_axis_id) : size(std::move(size)), life_time_axis_id(std::move(life_time_axis_id)) {}

  KeyTy getAsKey() const {
    return KeyTy(size, life_time_axis_id);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (size == std::get<0>(tblgenKey)) && (life_time_axis_id == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TmpBufDescAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto size = std::move(std::get<0>(tblgenKey));
    auto life_time_axis_id = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<TmpBufDescAttrStorage>()) TmpBufDescAttrStorage(std::move(size), std::move(life_time_axis_id));
  }

  StringAttr size;
  int64_t life_time_axis_id;
};

} // namespace detail
TmpBufDescAttr TmpBufDescAttr::get(::mlir::MLIRContext *context, StringAttr size, int64_t life_time_axis_id) {
  return Base::get(context, std::move(size), std::move(life_time_axis_id));
}

::mlir::Attribute TmpBufDescAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_size;
  ::mlir::FailureOr<int64_t> _result_life_time_axis_id;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_size = false;
  bool _seen_life_time_axis_id = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_size && _paramKey == "size") {
        _seen_size = true;

        // Parse variable 'size'
        _result_size = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_size)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TmpBufDescAttr parameter 'size' which is to be a `StringAttr`");
          return {};
        }
      } else if (!_seen_life_time_axis_id && _paramKey == "life_time_axis_id") {
        _seen_life_time_axis_id = true;

        // Parse variable 'life_time_axis_id'
        _result_life_time_axis_id = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_life_time_axis_id)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TmpBufDescAttr parameter 'life_time_axis_id' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    do {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
    } while(!odsParser.parseOptionalComma());
    if (!_seen_size) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "size";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_size));
  return TmpBufDescAttr::get(odsParser.getContext(),
      StringAttr((*_result_size)),
      int64_t((_result_life_time_axis_id.value_or(-1))));
}

void TmpBufDescAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "size = ";
    odsPrinter.printStrippedAttrOrType(getSize());
    if (!(getLifeTimeAxisId() == -1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "life_time_axis_id = ";
      if (!(getLifeTimeAxisId() == -1)) {
        odsPrinter.printStrippedAttrOrType(getLifeTimeAxisId());
      }
    }
  }
  odsPrinter << ">";
}

StringAttr TmpBufDescAttr::getSize() const {
  return getImpl()->size;
}

int64_t TmpBufDescAttr::getLifeTimeAxisId() const {
  return getImpl()->life_time_axis_id;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::TmpBufDescAttr)
namespace mlir::afir {

namespace detail {

struct AscGraphAttrGroupsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, ::llvm::ArrayRef<AxisAttr>, ::mlir::afir::AscGraphType, ::llvm::ArrayRef<StringAttr>>;
  AscGraphAttrGroupsAttrStorage(int64_t tiling_key, ::llvm::ArrayRef<AxisAttr> axes, ::mlir::afir::AscGraphType type, ::llvm::ArrayRef<StringAttr> size_var) : tiling_key(std::move(tiling_key)), axes(std::move(axes)), type(std::move(type)), size_var(std::move(size_var)) {}

  KeyTy getAsKey() const {
    return KeyTy(tiling_key, axes, type, size_var);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tiling_key == std::get<0>(tblgenKey)) && (axes == std::get<1>(tblgenKey)) && (type == std::get<2>(tblgenKey)) && (::llvm::ArrayRef<StringAttr>(size_var) == ::llvm::ArrayRef<StringAttr>(std::get<3>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static AscGraphAttrGroupsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tiling_key = std::move(std::get<0>(tblgenKey));
    auto axes = std::move(std::get<1>(tblgenKey));
    auto type = std::move(std::get<2>(tblgenKey));
    auto size_var = std::move(std::get<3>(tblgenKey));
    axes = allocator.copyInto(axes);
    size_var = allocator.copyInto(size_var);
    return new (allocator.allocate<AscGraphAttrGroupsAttrStorage>()) AscGraphAttrGroupsAttrStorage(std::move(tiling_key), std::move(axes), std::move(type), std::move(size_var));
  }

  int64_t tiling_key;
  ::llvm::ArrayRef<AxisAttr> axes;
  ::mlir::afir::AscGraphType type;
  ::llvm::ArrayRef<StringAttr> size_var;
};

} // namespace detail
AscGraphAttrGroupsAttr AscGraphAttrGroupsAttr::get(::mlir::MLIRContext *context, int64_t tiling_key, ::llvm::ArrayRef<AxisAttr> axes, ::mlir::afir::AscGraphType type, ::llvm::ArrayRef<StringAttr> size_var) {
  return Base::get(context, std::move(tiling_key), std::move(axes), std::move(type), std::move(size_var));
}

::mlir::Attribute AscGraphAttrGroupsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_tiling_key;
  ::mlir::FailureOr<::llvm::SmallVector<AxisAttr>> _result_axes;
  ::mlir::FailureOr<::mlir::afir::AscGraphType> _result_type;
  ::mlir::FailureOr<::llvm::SmallVector<StringAttr>> _result_size_var;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'tiling_key'
  if (odsParser.parseOptionalKeyword("tiling_key")) {
  } else {
    // Parse literal '='
    if (odsParser.parseEqual()) return {};

    // Parse variable 'tiling_key'
    _result_tiling_key = ::mlir::FieldParser<int64_t>::parse(odsParser);
    if (::mlir::failed(_result_tiling_key)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscGraphAttrGroupsAttr parameter 'tiling_key' which is to be a `int64_t`");
      return {};
    }
    // Parse literal ','
    if (odsParser.parseComma()) return {};
  }
  // Parse literal 'axes'
  if (odsParser.parseKeyword("axes")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'axes'
  _result_axes = ::mlir::FieldParser<::llvm::SmallVector<AxisAttr>>::parse(odsParser);
  if (::mlir::failed(_result_axes)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscGraphAttrGroupsAttr parameter 'axes' which is to be a `::llvm::ArrayRef<AxisAttr>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'type'
  if (odsParser.parseKeyword("type")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<::mlir::afir::AscGraphType>::parse(odsParser);
  if (::mlir::failed(_result_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscGraphAttrGroupsAttr parameter 'type' which is to be a `::mlir::afir::AscGraphType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    // Parse literal 'size_var'
    if (odsParser.parseKeyword("size_var")) return {};
    // Parse literal '='
    if (odsParser.parseEqual()) return {};
    // Parse literal '['
    if (odsParser.parseLSquare()) return {};

    // Parse variable 'size_var'
    _result_size_var = ::mlir::FieldParser<::llvm::SmallVector<StringAttr>>::parse(odsParser);
    if (::mlir::failed(_result_size_var)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscGraphAttrGroupsAttr parameter 'size_var' which is to be a `::llvm::ArrayRef<StringAttr>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_axes));
  assert(::mlir::succeeded(_result_type));
  return AscGraphAttrGroupsAttr::get(odsParser.getContext(),
      int64_t((_result_tiling_key.value_or(-1))),
      ::llvm::ArrayRef<AxisAttr>((*_result_axes)),
      ::mlir::afir::AscGraphType((*_result_type)),
      ::llvm::ArrayRef<StringAttr>((_result_size_var.value_or(::llvm::SmallVector<StringAttr>()))));
}

void AscGraphAttrGroupsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  if (!(getTilingKey() == -1)) {
    odsPrinter << "tiling_key";
    odsPrinter << ' ' << "=";
    if (!(getTilingKey() == -1)) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getTilingKey());
    }
    odsPrinter << ",";
  } else {
  }
  odsPrinter << "axes";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getAxes());
  odsPrinter << "]";
  odsPrinter << ",";
  odsPrinter << ' ' << "type";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getType());
  if (!(::llvm::ArrayRef<StringAttr>(getSizeVar()) == ::llvm::ArrayRef<StringAttr>(::llvm::SmallVector<StringAttr>()))) {
    odsPrinter << ",";
    odsPrinter << ' ' << "size_var";
    odsPrinter << ' ' << "=";
    odsPrinter << ' ' << "[";
    if (!(::llvm::ArrayRef<StringAttr>(getSizeVar()) == ::llvm::ArrayRef<StringAttr>(::llvm::SmallVector<StringAttr>()))) {
      odsPrinter.printStrippedAttrOrType(getSizeVar());
    }
    odsPrinter << "]";
  } else {
  }
  odsPrinter << ">";
}

int64_t AscGraphAttrGroupsAttr::getTilingKey() const {
  return getImpl()->tiling_key;
}

::llvm::ArrayRef<AxisAttr> AscGraphAttrGroupsAttr::getAxes() const {
  return getImpl()->axes;
}

::mlir::afir::AscGraphType AscGraphAttrGroupsAttr::getType() const {
  return getImpl()->type;
}

::llvm::ArrayRef<StringAttr> AscGraphAttrGroupsAttr::getSizeVar() const {
  return getImpl()->size_var;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::AscGraphAttrGroupsAttr)
namespace mlir::afir {

namespace detail {

struct ScalarValueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr>;
  ScalarValueAttrStorage(StringAttr value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ScalarValueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ScalarValueAttrStorage>()) ScalarValueAttrStorage(std::move(value));
  }

  StringAttr value;
};

} // namespace detail
ScalarValueAttr ScalarValueAttr::get(::mlir::MLIRContext *context, StringAttr value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ScalarValueAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_value = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_value && _paramKey == "value") {
        _seen_value = true;

        // Parse variable 'value'
        _result_value = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_value)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScalarValueAttr parameter 'value' which is to be a `StringAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ScalarValueAttr::get(odsParser.getContext(),
      StringAttr((*_result_value)));
}

void ScalarValueAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "value = ";
    odsPrinter.printStrippedAttrOrType(getValue());
  }
  odsPrinter << ">";
}

StringAttr ScalarValueAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::afir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::afir::ScalarValueAttr)

#endif // GET_ATTRDEF_CLASSES

