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

::mlir::ascendc::APIAttr,
::mlir::ascendc::MatmulConfigAttr,
::mlir::ascendc::SampleAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::ascendc::APIAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::APIAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MatmulConfigAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MatmulConfigAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::SampleAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::SampleAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::ascendc::APIAttr>([&](auto t) {
      printer << ::mlir::ascendc::APIAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MatmulConfigAttr>([&](auto t) {
      printer << ::mlir::ascendc::MatmulConfigAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::SampleAttr>([&](auto t) {
      printer << ::mlir::ascendc::SampleAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::ascendc {

namespace detail {

struct APIAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, StringAttr, StringAttr, StringAttr, ArrayAttr>;
  APIAttrStorage(StringAttr group, StringAttr name, StringAttr pipe_in, StringAttr pipe_out, ArrayAttr args) : group(std::move(group)), name(std::move(name)), pipe_in(std::move(pipe_in)), pipe_out(std::move(pipe_out)), args(std::move(args)) {}

  KeyTy getAsKey() const {
    return KeyTy(group, name, pipe_in, pipe_out, args);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (group == std::get<0>(tblgenKey)) && (name == std::get<1>(tblgenKey)) && (pipe_in == std::get<2>(tblgenKey)) && (pipe_out == std::get<3>(tblgenKey)) && (args == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static APIAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto group = std::move(std::get<0>(tblgenKey));
    auto name = std::move(std::get<1>(tblgenKey));
    auto pipe_in = std::move(std::get<2>(tblgenKey));
    auto pipe_out = std::move(std::get<3>(tblgenKey));
    auto args = std::move(std::get<4>(tblgenKey));
    return new (allocator.allocate<APIAttrStorage>()) APIAttrStorage(std::move(group), std::move(name), std::move(pipe_in), std::move(pipe_out), std::move(args));
  }

  StringAttr group;
  StringAttr name;
  StringAttr pipe_in;
  StringAttr pipe_out;
  ArrayAttr args;
};

} // namespace detail
APIAttr APIAttr::get(::mlir::MLIRContext *context, StringAttr group, StringAttr name, StringAttr pipe_in, StringAttr pipe_out, ArrayAttr args) {
  return Base::get(context, std::move(group), std::move(name), std::move(pipe_in), std::move(pipe_out), std::move(args));
}

APIAttr APIAttr::get(StringAttr group, StringAttr name) {
  return Base::get(group.getContext(), group, name, StringAttr{},
               StringAttr{}, ArrayAttr{});
}

::mlir::Attribute APIAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_group;
  ::mlir::FailureOr<StringAttr> _result_name;
  ::mlir::FailureOr<StringAttr> _result_pipe_in;
  ::mlir::FailureOr<StringAttr> _result_pipe_out;
  ::mlir::FailureOr<ArrayAttr> _result_args;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_group = false;
  bool _seen_name = false;
  bool _seen_pipe_in = false;
  bool _seen_pipe_out = false;
  bool _seen_args = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_group && _paramKey == "group") {
        _seen_group = true;

        // Parse variable 'group'
        _result_group = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_group)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_APIAttr parameter 'group' which is to be a `StringAttr`");
          return {};
        }
      } else if (!_seen_name && _paramKey == "name") {
        _seen_name = true;

        // Parse variable 'name'
        _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_name)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_APIAttr parameter 'name' which is to be a `StringAttr`");
          return {};
        }
      } else if (!_seen_pipe_in && _paramKey == "pipe_in") {
        _seen_pipe_in = true;

        // Parse variable 'pipe_in'
        _result_pipe_in = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_pipe_in)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_APIAttr parameter 'pipe_in' which is to be a `StringAttr`");
          return {};
        }
      } else if (!_seen_pipe_out && _paramKey == "pipe_out") {
        _seen_pipe_out = true;

        // Parse variable 'pipe_out'
        _result_pipe_out = ::mlir::FieldParser<StringAttr>::parse(odsParser);
        if (::mlir::failed(_result_pipe_out)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_APIAttr parameter 'pipe_out' which is to be a `StringAttr`");
          return {};
        }
      } else if (!_seen_args && _paramKey == "args") {
        _seen_args = true;

        // Parse variable 'args'
        _result_args = ::mlir::FieldParser<ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_args)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_APIAttr parameter 'args' which is to be a `ArrayAttr`");
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
    if (!_seen_name) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "name";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  return APIAttr::get(odsParser.getContext(),
      StringAttr((_result_group.value_or(StringAttr()))),
      StringAttr((*_result_name)),
      StringAttr((_result_pipe_in.value_or(StringAttr()))),
      StringAttr((_result_pipe_out.value_or(StringAttr()))),
      ArrayAttr((_result_args.value_or(ArrayAttr()))));
}

void APIAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getGroup() == StringAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "group = ";
      if (!(getGroup() == StringAttr())) {
        odsPrinter.printStrippedAttrOrType(getGroup());
      }
    }
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "name = ";
    odsPrinter.printStrippedAttrOrType(getName());
    if (!(getPipeIn() == StringAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "pipe_in = ";
      if (!(getPipeIn() == StringAttr())) {
        odsPrinter.printStrippedAttrOrType(getPipeIn());
      }
    }
    if (!(getPipeOut() == StringAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "pipe_out = ";
      if (!(getPipeOut() == StringAttr())) {
        odsPrinter.printStrippedAttrOrType(getPipeOut());
      }
    }
    if (!(getArgs() == ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "args = ";
      if (!(getArgs() == ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getArgs());
      }
    }
  }
  odsPrinter << ">";
}

StringAttr APIAttr::getGroup() const {
  return getImpl()->group;
}

StringAttr APIAttr::getName() const {
  return getImpl()->name;
}

StringAttr APIAttr::getPipeIn() const {
  return getImpl()->pipe_in;
}

StringAttr APIAttr::getPipeOut() const {
  return getImpl()->pipe_out;
}

ArrayAttr APIAttr::getArgs() const {
  return getImpl()->args;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::APIAttr)
namespace mlir::ascendc {

namespace detail {

struct MatmulConfigAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<BoolAttr, BoolAttr, BoolAttr, IntegerAttr, IntegerAttr, IntegerAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, IntegerAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, IntegerAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, IntegerAttr, IntegerAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, BoolAttr, IntegerAttr, IntegerAttr>;
  MatmulConfigAttrStorage(BoolAttr do_norm, BoolAttr do_basic_block, BoolAttr do_multi_data_load, IntegerAttr basic_m, IntegerAttr basic_n, IntegerAttr basic_k, BoolAttr intrinsics_check, BoolAttr is_n_batch, BoolAttr en_vec_nd2nz, BoolAttr do_special_basic_block, IntegerAttr do_mte2_preload, IntegerAttr single_core_m, IntegerAttr single_core_n, IntegerAttr single_core_k, IntegerAttr step_m, IntegerAttr step_n, IntegerAttr base_mn, IntegerAttr single_core_mn, BoolAttr en_unit_flag, BoolAttr is_per_tensor, BoolAttr has_anti_quant_offset, BoolAttr do_ib_share_norm, BoolAttr do_special_mdl, BoolAttr enable_init, IntegerAttr batch_mode, BoolAttr enable_end, BoolAttr enable_get_tensor_c, BoolAttr enable_set_org_shape, BoolAttr enable_set_bias, BoolAttr enable_set_tail, BoolAttr enable_quant_vector, BoolAttr enable_set_define_data, IntegerAttr iterate_mode, BoolAttr enable_reuse, BoolAttr enable_ub_reuse, BoolAttr enable_l1_cache_ub, BoolAttr intra_block_part_sum, IntegerAttr iterate_order, IntegerAttr schedule_type, BoolAttr enable_double_cache, BoolAttr is_bias_batch, BoolAttr enable_static_pad_zeros, BoolAttr is_partial_output, BoolAttr enable_mix_dual_master, BoolAttr is_a2b2_shared, BoolAttr is_enable_channel_split, BoolAttr enable_kdim_reorder_load, BoolAttr is_co1_shared, IntegerAttr shared_co1_buffer_size, IntegerAttr batch_out_mode) : do_norm(std::move(do_norm)), do_basic_block(std::move(do_basic_block)), do_multi_data_load(std::move(do_multi_data_load)), basic_m(std::move(basic_m)), basic_n(std::move(basic_n)), basic_k(std::move(basic_k)), intrinsics_check(std::move(intrinsics_check)), is_n_batch(std::move(is_n_batch)), en_vec_nd2nz(std::move(en_vec_nd2nz)), do_special_basic_block(std::move(do_special_basic_block)), do_mte2_preload(std::move(do_mte2_preload)), single_core_m(std::move(single_core_m)), single_core_n(std::move(single_core_n)), single_core_k(std::move(single_core_k)), step_m(std::move(step_m)), step_n(std::move(step_n)), base_mn(std::move(base_mn)), single_core_mn(std::move(single_core_mn)), en_unit_flag(std::move(en_unit_flag)), is_per_tensor(std::move(is_per_tensor)), has_anti_quant_offset(std::move(has_anti_quant_offset)), do_ib_share_norm(std::move(do_ib_share_norm)), do_special_mdl(std::move(do_special_mdl)), enable_init(std::move(enable_init)), batch_mode(std::move(batch_mode)), enable_end(std::move(enable_end)), enable_get_tensor_c(std::move(enable_get_tensor_c)), enable_set_org_shape(std::move(enable_set_org_shape)), enable_set_bias(std::move(enable_set_bias)), enable_set_tail(std::move(enable_set_tail)), enable_quant_vector(std::move(enable_quant_vector)), enable_set_define_data(std::move(enable_set_define_data)), iterate_mode(std::move(iterate_mode)), enable_reuse(std::move(enable_reuse)), enable_ub_reuse(std::move(enable_ub_reuse)), enable_l1_cache_ub(std::move(enable_l1_cache_ub)), intra_block_part_sum(std::move(intra_block_part_sum)), iterate_order(std::move(iterate_order)), schedule_type(std::move(schedule_type)), enable_double_cache(std::move(enable_double_cache)), is_bias_batch(std::move(is_bias_batch)), enable_static_pad_zeros(std::move(enable_static_pad_zeros)), is_partial_output(std::move(is_partial_output)), enable_mix_dual_master(std::move(enable_mix_dual_master)), is_a2b2_shared(std::move(is_a2b2_shared)), is_enable_channel_split(std::move(is_enable_channel_split)), enable_kdim_reorder_load(std::move(enable_kdim_reorder_load)), is_co1_shared(std::move(is_co1_shared)), shared_co1_buffer_size(std::move(shared_co1_buffer_size)), batch_out_mode(std::move(batch_out_mode)) {}

  KeyTy getAsKey() const {
    return KeyTy(do_norm, do_basic_block, do_multi_data_load, basic_m, basic_n, basic_k, intrinsics_check, is_n_batch, en_vec_nd2nz, do_special_basic_block, do_mte2_preload, single_core_m, single_core_n, single_core_k, step_m, step_n, base_mn, single_core_mn, en_unit_flag, is_per_tensor, has_anti_quant_offset, do_ib_share_norm, do_special_mdl, enable_init, batch_mode, enable_end, enable_get_tensor_c, enable_set_org_shape, enable_set_bias, enable_set_tail, enable_quant_vector, enable_set_define_data, iterate_mode, enable_reuse, enable_ub_reuse, enable_l1_cache_ub, intra_block_part_sum, iterate_order, schedule_type, enable_double_cache, is_bias_batch, enable_static_pad_zeros, is_partial_output, enable_mix_dual_master, is_a2b2_shared, is_enable_channel_split, enable_kdim_reorder_load, is_co1_shared, shared_co1_buffer_size, batch_out_mode);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (do_norm == std::get<0>(tblgenKey)) && (do_basic_block == std::get<1>(tblgenKey)) && (do_multi_data_load == std::get<2>(tblgenKey)) && (basic_m == std::get<3>(tblgenKey)) && (basic_n == std::get<4>(tblgenKey)) && (basic_k == std::get<5>(tblgenKey)) && (intrinsics_check == std::get<6>(tblgenKey)) && (is_n_batch == std::get<7>(tblgenKey)) && (en_vec_nd2nz == std::get<8>(tblgenKey)) && (do_special_basic_block == std::get<9>(tblgenKey)) && (do_mte2_preload == std::get<10>(tblgenKey)) && (single_core_m == std::get<11>(tblgenKey)) && (single_core_n == std::get<12>(tblgenKey)) && (single_core_k == std::get<13>(tblgenKey)) && (step_m == std::get<14>(tblgenKey)) && (step_n == std::get<15>(tblgenKey)) && (base_mn == std::get<16>(tblgenKey)) && (single_core_mn == std::get<17>(tblgenKey)) && (en_unit_flag == std::get<18>(tblgenKey)) && (is_per_tensor == std::get<19>(tblgenKey)) && (has_anti_quant_offset == std::get<20>(tblgenKey)) && (do_ib_share_norm == std::get<21>(tblgenKey)) && (do_special_mdl == std::get<22>(tblgenKey)) && (enable_init == std::get<23>(tblgenKey)) && (batch_mode == std::get<24>(tblgenKey)) && (enable_end == std::get<25>(tblgenKey)) && (enable_get_tensor_c == std::get<26>(tblgenKey)) && (enable_set_org_shape == std::get<27>(tblgenKey)) && (enable_set_bias == std::get<28>(tblgenKey)) && (enable_set_tail == std::get<29>(tblgenKey)) && (enable_quant_vector == std::get<30>(tblgenKey)) && (enable_set_define_data == std::get<31>(tblgenKey)) && (iterate_mode == std::get<32>(tblgenKey)) && (enable_reuse == std::get<33>(tblgenKey)) && (enable_ub_reuse == std::get<34>(tblgenKey)) && (enable_l1_cache_ub == std::get<35>(tblgenKey)) && (intra_block_part_sum == std::get<36>(tblgenKey)) && (iterate_order == std::get<37>(tblgenKey)) && (schedule_type == std::get<38>(tblgenKey)) && (enable_double_cache == std::get<39>(tblgenKey)) && (is_bias_batch == std::get<40>(tblgenKey)) && (enable_static_pad_zeros == std::get<41>(tblgenKey)) && (is_partial_output == std::get<42>(tblgenKey)) && (enable_mix_dual_master == std::get<43>(tblgenKey)) && (is_a2b2_shared == std::get<44>(tblgenKey)) && (is_enable_channel_split == std::get<45>(tblgenKey)) && (enable_kdim_reorder_load == std::get<46>(tblgenKey)) && (is_co1_shared == std::get<47>(tblgenKey)) && (shared_co1_buffer_size == std::get<48>(tblgenKey)) && (batch_out_mode == std::get<49>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey), std::get<8>(tblgenKey), std::get<9>(tblgenKey), std::get<10>(tblgenKey), std::get<11>(tblgenKey), std::get<12>(tblgenKey), std::get<13>(tblgenKey), std::get<14>(tblgenKey), std::get<15>(tblgenKey), std::get<16>(tblgenKey), std::get<17>(tblgenKey), std::get<18>(tblgenKey), std::get<19>(tblgenKey), std::get<20>(tblgenKey), std::get<21>(tblgenKey), std::get<22>(tblgenKey), std::get<23>(tblgenKey), std::get<24>(tblgenKey), std::get<25>(tblgenKey), std::get<26>(tblgenKey), std::get<27>(tblgenKey), std::get<28>(tblgenKey), std::get<29>(tblgenKey), std::get<30>(tblgenKey), std::get<31>(tblgenKey), std::get<32>(tblgenKey), std::get<33>(tblgenKey), std::get<34>(tblgenKey), std::get<35>(tblgenKey), std::get<36>(tblgenKey), std::get<37>(tblgenKey), std::get<38>(tblgenKey), std::get<39>(tblgenKey), std::get<40>(tblgenKey), std::get<41>(tblgenKey), std::get<42>(tblgenKey), std::get<43>(tblgenKey), std::get<44>(tblgenKey), std::get<45>(tblgenKey), std::get<46>(tblgenKey), std::get<47>(tblgenKey), std::get<48>(tblgenKey), std::get<49>(tblgenKey));
  }

  static MatmulConfigAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto do_norm = std::move(std::get<0>(tblgenKey));
    auto do_basic_block = std::move(std::get<1>(tblgenKey));
    auto do_multi_data_load = std::move(std::get<2>(tblgenKey));
    auto basic_m = std::move(std::get<3>(tblgenKey));
    auto basic_n = std::move(std::get<4>(tblgenKey));
    auto basic_k = std::move(std::get<5>(tblgenKey));
    auto intrinsics_check = std::move(std::get<6>(tblgenKey));
    auto is_n_batch = std::move(std::get<7>(tblgenKey));
    auto en_vec_nd2nz = std::move(std::get<8>(tblgenKey));
    auto do_special_basic_block = std::move(std::get<9>(tblgenKey));
    auto do_mte2_preload = std::move(std::get<10>(tblgenKey));
    auto single_core_m = std::move(std::get<11>(tblgenKey));
    auto single_core_n = std::move(std::get<12>(tblgenKey));
    auto single_core_k = std::move(std::get<13>(tblgenKey));
    auto step_m = std::move(std::get<14>(tblgenKey));
    auto step_n = std::move(std::get<15>(tblgenKey));
    auto base_mn = std::move(std::get<16>(tblgenKey));
    auto single_core_mn = std::move(std::get<17>(tblgenKey));
    auto en_unit_flag = std::move(std::get<18>(tblgenKey));
    auto is_per_tensor = std::move(std::get<19>(tblgenKey));
    auto has_anti_quant_offset = std::move(std::get<20>(tblgenKey));
    auto do_ib_share_norm = std::move(std::get<21>(tblgenKey));
    auto do_special_mdl = std::move(std::get<22>(tblgenKey));
    auto enable_init = std::move(std::get<23>(tblgenKey));
    auto batch_mode = std::move(std::get<24>(tblgenKey));
    auto enable_end = std::move(std::get<25>(tblgenKey));
    auto enable_get_tensor_c = std::move(std::get<26>(tblgenKey));
    auto enable_set_org_shape = std::move(std::get<27>(tblgenKey));
    auto enable_set_bias = std::move(std::get<28>(tblgenKey));
    auto enable_set_tail = std::move(std::get<29>(tblgenKey));
    auto enable_quant_vector = std::move(std::get<30>(tblgenKey));
    auto enable_set_define_data = std::move(std::get<31>(tblgenKey));
    auto iterate_mode = std::move(std::get<32>(tblgenKey));
    auto enable_reuse = std::move(std::get<33>(tblgenKey));
    auto enable_ub_reuse = std::move(std::get<34>(tblgenKey));
    auto enable_l1_cache_ub = std::move(std::get<35>(tblgenKey));
    auto intra_block_part_sum = std::move(std::get<36>(tblgenKey));
    auto iterate_order = std::move(std::get<37>(tblgenKey));
    auto schedule_type = std::move(std::get<38>(tblgenKey));
    auto enable_double_cache = std::move(std::get<39>(tblgenKey));
    auto is_bias_batch = std::move(std::get<40>(tblgenKey));
    auto enable_static_pad_zeros = std::move(std::get<41>(tblgenKey));
    auto is_partial_output = std::move(std::get<42>(tblgenKey));
    auto enable_mix_dual_master = std::move(std::get<43>(tblgenKey));
    auto is_a2b2_shared = std::move(std::get<44>(tblgenKey));
    auto is_enable_channel_split = std::move(std::get<45>(tblgenKey));
    auto enable_kdim_reorder_load = std::move(std::get<46>(tblgenKey));
    auto is_co1_shared = std::move(std::get<47>(tblgenKey));
    auto shared_co1_buffer_size = std::move(std::get<48>(tblgenKey));
    auto batch_out_mode = std::move(std::get<49>(tblgenKey));
    return new (allocator.allocate<MatmulConfigAttrStorage>()) MatmulConfigAttrStorage(std::move(do_norm), std::move(do_basic_block), std::move(do_multi_data_load), std::move(basic_m), std::move(basic_n), std::move(basic_k), std::move(intrinsics_check), std::move(is_n_batch), std::move(en_vec_nd2nz), std::move(do_special_basic_block), std::move(do_mte2_preload), std::move(single_core_m), std::move(single_core_n), std::move(single_core_k), std::move(step_m), std::move(step_n), std::move(base_mn), std::move(single_core_mn), std::move(en_unit_flag), std::move(is_per_tensor), std::move(has_anti_quant_offset), std::move(do_ib_share_norm), std::move(do_special_mdl), std::move(enable_init), std::move(batch_mode), std::move(enable_end), std::move(enable_get_tensor_c), std::move(enable_set_org_shape), std::move(enable_set_bias), std::move(enable_set_tail), std::move(enable_quant_vector), std::move(enable_set_define_data), std::move(iterate_mode), std::move(enable_reuse), std::move(enable_ub_reuse), std::move(enable_l1_cache_ub), std::move(intra_block_part_sum), std::move(iterate_order), std::move(schedule_type), std::move(enable_double_cache), std::move(is_bias_batch), std::move(enable_static_pad_zeros), std::move(is_partial_output), std::move(enable_mix_dual_master), std::move(is_a2b2_shared), std::move(is_enable_channel_split), std::move(enable_kdim_reorder_load), std::move(is_co1_shared), std::move(shared_co1_buffer_size), std::move(batch_out_mode));
  }

  BoolAttr do_norm;
  BoolAttr do_basic_block;
  BoolAttr do_multi_data_load;
  IntegerAttr basic_m;
  IntegerAttr basic_n;
  IntegerAttr basic_k;
  BoolAttr intrinsics_check;
  BoolAttr is_n_batch;
  BoolAttr en_vec_nd2nz;
  BoolAttr do_special_basic_block;
  IntegerAttr do_mte2_preload;
  IntegerAttr single_core_m;
  IntegerAttr single_core_n;
  IntegerAttr single_core_k;
  IntegerAttr step_m;
  IntegerAttr step_n;
  IntegerAttr base_mn;
  IntegerAttr single_core_mn;
  BoolAttr en_unit_flag;
  BoolAttr is_per_tensor;
  BoolAttr has_anti_quant_offset;
  BoolAttr do_ib_share_norm;
  BoolAttr do_special_mdl;
  BoolAttr enable_init;
  IntegerAttr batch_mode;
  BoolAttr enable_end;
  BoolAttr enable_get_tensor_c;
  BoolAttr enable_set_org_shape;
  BoolAttr enable_set_bias;
  BoolAttr enable_set_tail;
  BoolAttr enable_quant_vector;
  BoolAttr enable_set_define_data;
  IntegerAttr iterate_mode;
  BoolAttr enable_reuse;
  BoolAttr enable_ub_reuse;
  BoolAttr enable_l1_cache_ub;
  BoolAttr intra_block_part_sum;
  IntegerAttr iterate_order;
  IntegerAttr schedule_type;
  BoolAttr enable_double_cache;
  BoolAttr is_bias_batch;
  BoolAttr enable_static_pad_zeros;
  BoolAttr is_partial_output;
  BoolAttr enable_mix_dual_master;
  BoolAttr is_a2b2_shared;
  BoolAttr is_enable_channel_split;
  BoolAttr enable_kdim_reorder_load;
  BoolAttr is_co1_shared;
  IntegerAttr shared_co1_buffer_size;
  IntegerAttr batch_out_mode;
};

} // namespace detail
MatmulConfigAttr MatmulConfigAttr::get(::mlir::MLIRContext *context, BoolAttr do_norm, BoolAttr do_basic_block, BoolAttr do_multi_data_load, IntegerAttr basic_m, IntegerAttr basic_n, IntegerAttr basic_k, BoolAttr intrinsics_check, BoolAttr is_n_batch, BoolAttr en_vec_nd2nz, BoolAttr do_special_basic_block, IntegerAttr do_mte2_preload, IntegerAttr single_core_m, IntegerAttr single_core_n, IntegerAttr single_core_k, IntegerAttr step_m, IntegerAttr step_n, IntegerAttr base_mn, IntegerAttr single_core_mn, BoolAttr en_unit_flag, BoolAttr is_per_tensor, BoolAttr has_anti_quant_offset, BoolAttr do_ib_share_norm, BoolAttr do_special_mdl, BoolAttr enable_init, IntegerAttr batch_mode, BoolAttr enable_end, BoolAttr enable_get_tensor_c, BoolAttr enable_set_org_shape, BoolAttr enable_set_bias, BoolAttr enable_set_tail, BoolAttr enable_quant_vector, BoolAttr enable_set_define_data, IntegerAttr iterate_mode, BoolAttr enable_reuse, BoolAttr enable_ub_reuse, BoolAttr enable_l1_cache_ub, BoolAttr intra_block_part_sum, IntegerAttr iterate_order, IntegerAttr schedule_type, BoolAttr enable_double_cache, BoolAttr is_bias_batch, BoolAttr enable_static_pad_zeros, BoolAttr is_partial_output, BoolAttr enable_mix_dual_master, BoolAttr is_a2b2_shared, BoolAttr is_enable_channel_split, BoolAttr enable_kdim_reorder_load, BoolAttr is_co1_shared, IntegerAttr shared_co1_buffer_size, IntegerAttr batch_out_mode) {
  return Base::get(context, std::move(do_norm), std::move(do_basic_block), std::move(do_multi_data_load), std::move(basic_m), std::move(basic_n), std::move(basic_k), std::move(intrinsics_check), std::move(is_n_batch), std::move(en_vec_nd2nz), std::move(do_special_basic_block), std::move(do_mte2_preload), std::move(single_core_m), std::move(single_core_n), std::move(single_core_k), std::move(step_m), std::move(step_n), std::move(base_mn), std::move(single_core_mn), std::move(en_unit_flag), std::move(is_per_tensor), std::move(has_anti_quant_offset), std::move(do_ib_share_norm), std::move(do_special_mdl), std::move(enable_init), std::move(batch_mode), std::move(enable_end), std::move(enable_get_tensor_c), std::move(enable_set_org_shape), std::move(enable_set_bias), std::move(enable_set_tail), std::move(enable_quant_vector), std::move(enable_set_define_data), std::move(iterate_mode), std::move(enable_reuse), std::move(enable_ub_reuse), std::move(enable_l1_cache_ub), std::move(intra_block_part_sum), std::move(iterate_order), std::move(schedule_type), std::move(enable_double_cache), std::move(is_bias_batch), std::move(enable_static_pad_zeros), std::move(is_partial_output), std::move(enable_mix_dual_master), std::move(is_a2b2_shared), std::move(is_enable_channel_split), std::move(enable_kdim_reorder_load), std::move(is_co1_shared), std::move(shared_co1_buffer_size), std::move(batch_out_mode));
}

::mlir::Attribute MatmulConfigAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<BoolAttr> _result_do_norm;
  ::mlir::FailureOr<BoolAttr> _result_do_basic_block;
  ::mlir::FailureOr<BoolAttr> _result_do_multi_data_load;
  ::mlir::FailureOr<IntegerAttr> _result_basic_m;
  ::mlir::FailureOr<IntegerAttr> _result_basic_n;
  ::mlir::FailureOr<IntegerAttr> _result_basic_k;
  ::mlir::FailureOr<BoolAttr> _result_intrinsics_check;
  ::mlir::FailureOr<BoolAttr> _result_is_n_batch;
  ::mlir::FailureOr<BoolAttr> _result_en_vec_nd2nz;
  ::mlir::FailureOr<BoolAttr> _result_do_special_basic_block;
  ::mlir::FailureOr<IntegerAttr> _result_do_mte2_preload;
  ::mlir::FailureOr<IntegerAttr> _result_single_core_m;
  ::mlir::FailureOr<IntegerAttr> _result_single_core_n;
  ::mlir::FailureOr<IntegerAttr> _result_single_core_k;
  ::mlir::FailureOr<IntegerAttr> _result_step_m;
  ::mlir::FailureOr<IntegerAttr> _result_step_n;
  ::mlir::FailureOr<IntegerAttr> _result_base_mn;
  ::mlir::FailureOr<IntegerAttr> _result_single_core_mn;
  ::mlir::FailureOr<BoolAttr> _result_en_unit_flag;
  ::mlir::FailureOr<BoolAttr> _result_is_per_tensor;
  ::mlir::FailureOr<BoolAttr> _result_has_anti_quant_offset;
  ::mlir::FailureOr<BoolAttr> _result_do_ib_share_norm;
  ::mlir::FailureOr<BoolAttr> _result_do_special_mdl;
  ::mlir::FailureOr<BoolAttr> _result_enable_init;
  ::mlir::FailureOr<IntegerAttr> _result_batch_mode;
  ::mlir::FailureOr<BoolAttr> _result_enable_end;
  ::mlir::FailureOr<BoolAttr> _result_enable_get_tensor_c;
  ::mlir::FailureOr<BoolAttr> _result_enable_set_org_shape;
  ::mlir::FailureOr<BoolAttr> _result_enable_set_bias;
  ::mlir::FailureOr<BoolAttr> _result_enable_set_tail;
  ::mlir::FailureOr<BoolAttr> _result_enable_quant_vector;
  ::mlir::FailureOr<BoolAttr> _result_enable_set_define_data;
  ::mlir::FailureOr<IntegerAttr> _result_iterate_mode;
  ::mlir::FailureOr<BoolAttr> _result_enable_reuse;
  ::mlir::FailureOr<BoolAttr> _result_enable_ub_reuse;
  ::mlir::FailureOr<BoolAttr> _result_enable_l1_cache_ub;
  ::mlir::FailureOr<BoolAttr> _result_intra_block_part_sum;
  ::mlir::FailureOr<IntegerAttr> _result_iterate_order;
  ::mlir::FailureOr<IntegerAttr> _result_schedule_type;
  ::mlir::FailureOr<BoolAttr> _result_enable_double_cache;
  ::mlir::FailureOr<BoolAttr> _result_is_bias_batch;
  ::mlir::FailureOr<BoolAttr> _result_enable_static_pad_zeros;
  ::mlir::FailureOr<BoolAttr> _result_is_partial_output;
  ::mlir::FailureOr<BoolAttr> _result_enable_mix_dual_master;
  ::mlir::FailureOr<BoolAttr> _result_is_a2b2_shared;
  ::mlir::FailureOr<BoolAttr> _result_is_enable_channel_split;
  ::mlir::FailureOr<BoolAttr> _result_enable_kdim_reorder_load;
  ::mlir::FailureOr<BoolAttr> _result_is_co1_shared;
  ::mlir::FailureOr<IntegerAttr> _result_shared_co1_buffer_size;
  ::mlir::FailureOr<IntegerAttr> _result_batch_out_mode;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_do_norm = false;
  bool _seen_do_basic_block = false;
  bool _seen_do_multi_data_load = false;
  bool _seen_basic_m = false;
  bool _seen_basic_n = false;
  bool _seen_basic_k = false;
  bool _seen_intrinsics_check = false;
  bool _seen_is_n_batch = false;
  bool _seen_en_vec_nd2nz = false;
  bool _seen_do_special_basic_block = false;
  bool _seen_do_mte2_preload = false;
  bool _seen_single_core_m = false;
  bool _seen_single_core_n = false;
  bool _seen_single_core_k = false;
  bool _seen_step_m = false;
  bool _seen_step_n = false;
  bool _seen_base_mn = false;
  bool _seen_single_core_mn = false;
  bool _seen_en_unit_flag = false;
  bool _seen_is_per_tensor = false;
  bool _seen_has_anti_quant_offset = false;
  bool _seen_do_ib_share_norm = false;
  bool _seen_do_special_mdl = false;
  bool _seen_enable_init = false;
  bool _seen_batch_mode = false;
  bool _seen_enable_end = false;
  bool _seen_enable_get_tensor_c = false;
  bool _seen_enable_set_org_shape = false;
  bool _seen_enable_set_bias = false;
  bool _seen_enable_set_tail = false;
  bool _seen_enable_quant_vector = false;
  bool _seen_enable_set_define_data = false;
  bool _seen_iterate_mode = false;
  bool _seen_enable_reuse = false;
  bool _seen_enable_ub_reuse = false;
  bool _seen_enable_l1_cache_ub = false;
  bool _seen_intra_block_part_sum = false;
  bool _seen_iterate_order = false;
  bool _seen_schedule_type = false;
  bool _seen_enable_double_cache = false;
  bool _seen_is_bias_batch = false;
  bool _seen_enable_static_pad_zeros = false;
  bool _seen_is_partial_output = false;
  bool _seen_enable_mix_dual_master = false;
  bool _seen_is_a2b2_shared = false;
  bool _seen_is_enable_channel_split = false;
  bool _seen_enable_kdim_reorder_load = false;
  bool _seen_is_co1_shared = false;
  bool _seen_shared_co1_buffer_size = false;
  bool _seen_batch_out_mode = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_do_norm && _paramKey == "do_norm") {
        _seen_do_norm = true;

        // Parse variable 'do_norm'
        _result_do_norm = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_norm)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_norm' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_basic_block && _paramKey == "do_basic_block") {
        _seen_do_basic_block = true;

        // Parse variable 'do_basic_block'
        _result_do_basic_block = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_basic_block)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_basic_block' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_multi_data_load && _paramKey == "do_multi_data_load") {
        _seen_do_multi_data_load = true;

        // Parse variable 'do_multi_data_load'
        _result_do_multi_data_load = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_multi_data_load)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_multi_data_load' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_basic_m && _paramKey == "basic_m") {
        _seen_basic_m = true;

        // Parse variable 'basic_m'
        _result_basic_m = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_basic_m)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'basic_m' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_basic_n && _paramKey == "basic_n") {
        _seen_basic_n = true;

        // Parse variable 'basic_n'
        _result_basic_n = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_basic_n)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'basic_n' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_basic_k && _paramKey == "basic_k") {
        _seen_basic_k = true;

        // Parse variable 'basic_k'
        _result_basic_k = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_basic_k)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'basic_k' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_intrinsics_check && _paramKey == "intrinsics_check") {
        _seen_intrinsics_check = true;

        // Parse variable 'intrinsics_check'
        _result_intrinsics_check = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_intrinsics_check)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'intrinsics_check' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_n_batch && _paramKey == "is_n_batch") {
        _seen_is_n_batch = true;

        // Parse variable 'is_n_batch'
        _result_is_n_batch = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_n_batch)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_n_batch' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_en_vec_nd2nz && _paramKey == "en_vec_nd2nz") {
        _seen_en_vec_nd2nz = true;

        // Parse variable 'en_vec_nd2nz'
        _result_en_vec_nd2nz = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_en_vec_nd2nz)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'en_vec_nd2nz' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_special_basic_block && _paramKey == "do_special_basic_block") {
        _seen_do_special_basic_block = true;

        // Parse variable 'do_special_basic_block'
        _result_do_special_basic_block = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_special_basic_block)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_special_basic_block' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_mte2_preload && _paramKey == "do_mte2_preload") {
        _seen_do_mte2_preload = true;

        // Parse variable 'do_mte2_preload'
        _result_do_mte2_preload = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_mte2_preload)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_mte2_preload' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_single_core_m && _paramKey == "single_core_m") {
        _seen_single_core_m = true;

        // Parse variable 'single_core_m'
        _result_single_core_m = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_single_core_m)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'single_core_m' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_single_core_n && _paramKey == "single_core_n") {
        _seen_single_core_n = true;

        // Parse variable 'single_core_n'
        _result_single_core_n = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_single_core_n)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'single_core_n' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_single_core_k && _paramKey == "single_core_k") {
        _seen_single_core_k = true;

        // Parse variable 'single_core_k'
        _result_single_core_k = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_single_core_k)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'single_core_k' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_step_m && _paramKey == "step_m") {
        _seen_step_m = true;

        // Parse variable 'step_m'
        _result_step_m = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_step_m)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'step_m' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_step_n && _paramKey == "step_n") {
        _seen_step_n = true;

        // Parse variable 'step_n'
        _result_step_n = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_step_n)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'step_n' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_base_mn && _paramKey == "base_mn") {
        _seen_base_mn = true;

        // Parse variable 'base_mn'
        _result_base_mn = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_base_mn)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'base_mn' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_single_core_mn && _paramKey == "single_core_mn") {
        _seen_single_core_mn = true;

        // Parse variable 'single_core_mn'
        _result_single_core_mn = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_single_core_mn)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'single_core_mn' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_en_unit_flag && _paramKey == "en_unit_flag") {
        _seen_en_unit_flag = true;

        // Parse variable 'en_unit_flag'
        _result_en_unit_flag = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_en_unit_flag)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'en_unit_flag' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_per_tensor && _paramKey == "is_per_tensor") {
        _seen_is_per_tensor = true;

        // Parse variable 'is_per_tensor'
        _result_is_per_tensor = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_per_tensor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_per_tensor' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_has_anti_quant_offset && _paramKey == "has_anti_quant_offset") {
        _seen_has_anti_quant_offset = true;

        // Parse variable 'has_anti_quant_offset'
        _result_has_anti_quant_offset = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_has_anti_quant_offset)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'has_anti_quant_offset' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_ib_share_norm && _paramKey == "do_ib_share_norm") {
        _seen_do_ib_share_norm = true;

        // Parse variable 'do_ib_share_norm'
        _result_do_ib_share_norm = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_ib_share_norm)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_ib_share_norm' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_do_special_mdl && _paramKey == "do_special_mdl") {
        _seen_do_special_mdl = true;

        // Parse variable 'do_special_mdl'
        _result_do_special_mdl = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_do_special_mdl)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'do_special_mdl' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_init && _paramKey == "enable_init") {
        _seen_enable_init = true;

        // Parse variable 'enable_init'
        _result_enable_init = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_init)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_init' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_batch_mode && _paramKey == "batch_mode") {
        _seen_batch_mode = true;

        // Parse variable 'batch_mode'
        _result_batch_mode = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_batch_mode)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'batch_mode' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_enable_end && _paramKey == "enable_end") {
        _seen_enable_end = true;

        // Parse variable 'enable_end'
        _result_enable_end = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_end)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_end' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_get_tensor_c && _paramKey == "enable_get_tensor_c") {
        _seen_enable_get_tensor_c = true;

        // Parse variable 'enable_get_tensor_c'
        _result_enable_get_tensor_c = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_get_tensor_c)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_get_tensor_c' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_set_org_shape && _paramKey == "enable_set_org_shape") {
        _seen_enable_set_org_shape = true;

        // Parse variable 'enable_set_org_shape'
        _result_enable_set_org_shape = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_set_org_shape)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_set_org_shape' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_set_bias && _paramKey == "enable_set_bias") {
        _seen_enable_set_bias = true;

        // Parse variable 'enable_set_bias'
        _result_enable_set_bias = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_set_bias)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_set_bias' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_set_tail && _paramKey == "enable_set_tail") {
        _seen_enable_set_tail = true;

        // Parse variable 'enable_set_tail'
        _result_enable_set_tail = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_set_tail)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_set_tail' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_quant_vector && _paramKey == "enable_quant_vector") {
        _seen_enable_quant_vector = true;

        // Parse variable 'enable_quant_vector'
        _result_enable_quant_vector = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_quant_vector)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_quant_vector' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_set_define_data && _paramKey == "enable_set_define_data") {
        _seen_enable_set_define_data = true;

        // Parse variable 'enable_set_define_data'
        _result_enable_set_define_data = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_set_define_data)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_set_define_data' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_iterate_mode && _paramKey == "iterate_mode") {
        _seen_iterate_mode = true;

        // Parse variable 'iterate_mode'
        _result_iterate_mode = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_iterate_mode)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'iterate_mode' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_enable_reuse && _paramKey == "enable_reuse") {
        _seen_enable_reuse = true;

        // Parse variable 'enable_reuse'
        _result_enable_reuse = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_reuse)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_reuse' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_ub_reuse && _paramKey == "enable_ub_reuse") {
        _seen_enable_ub_reuse = true;

        // Parse variable 'enable_ub_reuse'
        _result_enable_ub_reuse = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_ub_reuse)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_ub_reuse' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_l1_cache_ub && _paramKey == "enable_l1_cache_ub") {
        _seen_enable_l1_cache_ub = true;

        // Parse variable 'enable_l1_cache_ub'
        _result_enable_l1_cache_ub = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_l1_cache_ub)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_l1_cache_ub' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_intra_block_part_sum && _paramKey == "intra_block_part_sum") {
        _seen_intra_block_part_sum = true;

        // Parse variable 'intra_block_part_sum'
        _result_intra_block_part_sum = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_intra_block_part_sum)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'intra_block_part_sum' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_iterate_order && _paramKey == "iterate_order") {
        _seen_iterate_order = true;

        // Parse variable 'iterate_order'
        _result_iterate_order = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_iterate_order)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'iterate_order' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_schedule_type && _paramKey == "schedule_type") {
        _seen_schedule_type = true;

        // Parse variable 'schedule_type'
        _result_schedule_type = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_schedule_type)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'schedule_type' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_enable_double_cache && _paramKey == "enable_double_cache") {
        _seen_enable_double_cache = true;

        // Parse variable 'enable_double_cache'
        _result_enable_double_cache = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_double_cache)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_double_cache' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_bias_batch && _paramKey == "is_bias_batch") {
        _seen_is_bias_batch = true;

        // Parse variable 'is_bias_batch'
        _result_is_bias_batch = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_bias_batch)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_bias_batch' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_static_pad_zeros && _paramKey == "enable_static_pad_zeros") {
        _seen_enable_static_pad_zeros = true;

        // Parse variable 'enable_static_pad_zeros'
        _result_enable_static_pad_zeros = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_static_pad_zeros)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_static_pad_zeros' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_partial_output && _paramKey == "is_partial_output") {
        _seen_is_partial_output = true;

        // Parse variable 'is_partial_output'
        _result_is_partial_output = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_partial_output)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_partial_output' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_mix_dual_master && _paramKey == "enable_mix_dual_master") {
        _seen_enable_mix_dual_master = true;

        // Parse variable 'enable_mix_dual_master'
        _result_enable_mix_dual_master = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_mix_dual_master)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_mix_dual_master' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_a2b2_shared && _paramKey == "is_a2b2_shared") {
        _seen_is_a2b2_shared = true;

        // Parse variable 'is_a2b2_shared'
        _result_is_a2b2_shared = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_a2b2_shared)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_a2b2_shared' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_enable_channel_split && _paramKey == "is_enable_channel_split") {
        _seen_is_enable_channel_split = true;

        // Parse variable 'is_enable_channel_split'
        _result_is_enable_channel_split = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_enable_channel_split)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_enable_channel_split' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_enable_kdim_reorder_load && _paramKey == "enable_kdim_reorder_load") {
        _seen_enable_kdim_reorder_load = true;

        // Parse variable 'enable_kdim_reorder_load'
        _result_enable_kdim_reorder_load = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_enable_kdim_reorder_load)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'enable_kdim_reorder_load' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_is_co1_shared && _paramKey == "is_co1_shared") {
        _seen_is_co1_shared = true;

        // Parse variable 'is_co1_shared'
        _result_is_co1_shared = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_is_co1_shared)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'is_co1_shared' which is to be a `BoolAttr`");
          return {};
        }
      } else if (!_seen_shared_co1_buffer_size && _paramKey == "shared_co1_buffer_size") {
        _seen_shared_co1_buffer_size = true;

        // Parse variable 'shared_co1_buffer_size'
        _result_shared_co1_buffer_size = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_shared_co1_buffer_size)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'shared_co1_buffer_size' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_batch_out_mode && _paramKey == "batch_out_mode") {
        _seen_batch_out_mode = true;

        // Parse variable 'batch_out_mode'
        _result_batch_out_mode = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_batch_out_mode)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_MatmulConfigAttr parameter 'batch_out_mode' which is to be a `IntegerAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return MatmulConfigAttr::get(odsParser.getContext(),
      BoolAttr((_result_do_norm.value_or(BoolAttr()))),
      BoolAttr((_result_do_basic_block.value_or(BoolAttr()))),
      BoolAttr((_result_do_multi_data_load.value_or(BoolAttr()))),
      IntegerAttr((_result_basic_m.value_or(IntegerAttr()))),
      IntegerAttr((_result_basic_n.value_or(IntegerAttr()))),
      IntegerAttr((_result_basic_k.value_or(IntegerAttr()))),
      BoolAttr((_result_intrinsics_check.value_or(BoolAttr()))),
      BoolAttr((_result_is_n_batch.value_or(BoolAttr()))),
      BoolAttr((_result_en_vec_nd2nz.value_or(BoolAttr()))),
      BoolAttr((_result_do_special_basic_block.value_or(BoolAttr()))),
      IntegerAttr((_result_do_mte2_preload.value_or(IntegerAttr()))),
      IntegerAttr((_result_single_core_m.value_or(IntegerAttr()))),
      IntegerAttr((_result_single_core_n.value_or(IntegerAttr()))),
      IntegerAttr((_result_single_core_k.value_or(IntegerAttr()))),
      IntegerAttr((_result_step_m.value_or(IntegerAttr()))),
      IntegerAttr((_result_step_n.value_or(IntegerAttr()))),
      IntegerAttr((_result_base_mn.value_or(IntegerAttr()))),
      IntegerAttr((_result_single_core_mn.value_or(IntegerAttr()))),
      BoolAttr((_result_en_unit_flag.value_or(BoolAttr()))),
      BoolAttr((_result_is_per_tensor.value_or(BoolAttr()))),
      BoolAttr((_result_has_anti_quant_offset.value_or(BoolAttr()))),
      BoolAttr((_result_do_ib_share_norm.value_or(BoolAttr()))),
      BoolAttr((_result_do_special_mdl.value_or(BoolAttr()))),
      BoolAttr((_result_enable_init.value_or(BoolAttr()))),
      IntegerAttr((_result_batch_mode.value_or(IntegerAttr()))),
      BoolAttr((_result_enable_end.value_or(BoolAttr()))),
      BoolAttr((_result_enable_get_tensor_c.value_or(BoolAttr()))),
      BoolAttr((_result_enable_set_org_shape.value_or(BoolAttr()))),
      BoolAttr((_result_enable_set_bias.value_or(BoolAttr()))),
      BoolAttr((_result_enable_set_tail.value_or(BoolAttr()))),
      BoolAttr((_result_enable_quant_vector.value_or(BoolAttr()))),
      BoolAttr((_result_enable_set_define_data.value_or(BoolAttr()))),
      IntegerAttr((_result_iterate_mode.value_or(IntegerAttr()))),
      BoolAttr((_result_enable_reuse.value_or(BoolAttr()))),
      BoolAttr((_result_enable_ub_reuse.value_or(BoolAttr()))),
      BoolAttr((_result_enable_l1_cache_ub.value_or(BoolAttr()))),
      BoolAttr((_result_intra_block_part_sum.value_or(BoolAttr()))),
      IntegerAttr((_result_iterate_order.value_or(IntegerAttr()))),
      IntegerAttr((_result_schedule_type.value_or(IntegerAttr()))),
      BoolAttr((_result_enable_double_cache.value_or(BoolAttr()))),
      BoolAttr((_result_is_bias_batch.value_or(BoolAttr()))),
      BoolAttr((_result_enable_static_pad_zeros.value_or(BoolAttr()))),
      BoolAttr((_result_is_partial_output.value_or(BoolAttr()))),
      BoolAttr((_result_enable_mix_dual_master.value_or(BoolAttr()))),
      BoolAttr((_result_is_a2b2_shared.value_or(BoolAttr()))),
      BoolAttr((_result_is_enable_channel_split.value_or(BoolAttr()))),
      BoolAttr((_result_enable_kdim_reorder_load.value_or(BoolAttr()))),
      BoolAttr((_result_is_co1_shared.value_or(BoolAttr()))),
      IntegerAttr((_result_shared_co1_buffer_size.value_or(IntegerAttr()))),
      IntegerAttr((_result_batch_out_mode.value_or(IntegerAttr()))));
}

void MatmulConfigAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getDoNorm() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_norm = ";
      if (!(getDoNorm() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoNorm());
      }
    }
    if (!(getDoBasicBlock() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_basic_block = ";
      if (!(getDoBasicBlock() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoBasicBlock());
      }
    }
    if (!(getDoMultiDataLoad() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_multi_data_load = ";
      if (!(getDoMultiDataLoad() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoMultiDataLoad());
      }
    }
    if (!(getBasicM() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "basic_m = ";
      if (!(getBasicM() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBasicM());
      }
    }
    if (!(getBasicN() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "basic_n = ";
      if (!(getBasicN() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBasicN());
      }
    }
    if (!(getBasicK() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "basic_k = ";
      if (!(getBasicK() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBasicK());
      }
    }
    if (!(getIntrinsicsCheck() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "intrinsics_check = ";
      if (!(getIntrinsicsCheck() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIntrinsicsCheck());
      }
    }
    if (!(getIsNBatch() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_n_batch = ";
      if (!(getIsNBatch() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsNBatch());
      }
    }
    if (!(getEnVecNd2nz() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "en_vec_nd2nz = ";
      if (!(getEnVecNd2nz() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnVecNd2nz());
      }
    }
    if (!(getDoSpecialBasicBlock() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_special_basic_block = ";
      if (!(getDoSpecialBasicBlock() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoSpecialBasicBlock());
      }
    }
    if (!(getDoMte2Preload() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_mte2_preload = ";
      if (!(getDoMte2Preload() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoMte2Preload());
      }
    }
    if (!(getSingleCoreM() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "single_core_m = ";
      if (!(getSingleCoreM() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getSingleCoreM());
      }
    }
    if (!(getSingleCoreN() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "single_core_n = ";
      if (!(getSingleCoreN() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getSingleCoreN());
      }
    }
    if (!(getSingleCoreK() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "single_core_k = ";
      if (!(getSingleCoreK() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getSingleCoreK());
      }
    }
    if (!(getStepM() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "step_m = ";
      if (!(getStepM() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getStepM());
      }
    }
    if (!(getStepN() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "step_n = ";
      if (!(getStepN() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getStepN());
      }
    }
    if (!(getBaseMn() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "base_mn = ";
      if (!(getBaseMn() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBaseMn());
      }
    }
    if (!(getSingleCoreMn() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "single_core_mn = ";
      if (!(getSingleCoreMn() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getSingleCoreMn());
      }
    }
    if (!(getEnUnitFlag() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "en_unit_flag = ";
      if (!(getEnUnitFlag() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnUnitFlag());
      }
    }
    if (!(getIsPerTensor() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_per_tensor = ";
      if (!(getIsPerTensor() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsPerTensor());
      }
    }
    if (!(getHasAntiQuantOffset() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "has_anti_quant_offset = ";
      if (!(getHasAntiQuantOffset() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getHasAntiQuantOffset());
      }
    }
    if (!(getDoIbShareNorm() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_ib_share_norm = ";
      if (!(getDoIbShareNorm() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoIbShareNorm());
      }
    }
    if (!(getDoSpecialMdl() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "do_special_mdl = ";
      if (!(getDoSpecialMdl() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getDoSpecialMdl());
      }
    }
    if (!(getEnableInit() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_init = ";
      if (!(getEnableInit() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableInit());
      }
    }
    if (!(getBatchMode() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "batch_mode = ";
      if (!(getBatchMode() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBatchMode());
      }
    }
    if (!(getEnableEnd() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_end = ";
      if (!(getEnableEnd() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableEnd());
      }
    }
    if (!(getEnableGetTensorC() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_get_tensor_c = ";
      if (!(getEnableGetTensorC() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableGetTensorC());
      }
    }
    if (!(getEnableSetOrgShape() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_set_org_shape = ";
      if (!(getEnableSetOrgShape() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableSetOrgShape());
      }
    }
    if (!(getEnableSetBias() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_set_bias = ";
      if (!(getEnableSetBias() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableSetBias());
      }
    }
    if (!(getEnableSetTail() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_set_tail = ";
      if (!(getEnableSetTail() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableSetTail());
      }
    }
    if (!(getEnableQuantVector() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_quant_vector = ";
      if (!(getEnableQuantVector() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableQuantVector());
      }
    }
    if (!(getEnableSetDefineData() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_set_define_data = ";
      if (!(getEnableSetDefineData() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableSetDefineData());
      }
    }
    if (!(getIterateMode() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "iterate_mode = ";
      if (!(getIterateMode() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getIterateMode());
      }
    }
    if (!(getEnableReuse() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_reuse = ";
      if (!(getEnableReuse() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableReuse());
      }
    }
    if (!(getEnableUbReuse() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_ub_reuse = ";
      if (!(getEnableUbReuse() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableUbReuse());
      }
    }
    if (!(getEnableL1CacheUb() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_l1_cache_ub = ";
      if (!(getEnableL1CacheUb() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableL1CacheUb());
      }
    }
    if (!(getIntraBlockPartSum() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "intra_block_part_sum = ";
      if (!(getIntraBlockPartSum() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIntraBlockPartSum());
      }
    }
    if (!(getIterateOrder() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "iterate_order = ";
      if (!(getIterateOrder() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getIterateOrder());
      }
    }
    if (!(getScheduleType() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "schedule_type = ";
      if (!(getScheduleType() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getScheduleType());
      }
    }
    if (!(getEnableDoubleCache() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_double_cache = ";
      if (!(getEnableDoubleCache() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableDoubleCache());
      }
    }
    if (!(getIsBiasBatch() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_bias_batch = ";
      if (!(getIsBiasBatch() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsBiasBatch());
      }
    }
    if (!(getEnableStaticPadZeros() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_static_pad_zeros = ";
      if (!(getEnableStaticPadZeros() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableStaticPadZeros());
      }
    }
    if (!(getIsPartialOutput() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_partial_output = ";
      if (!(getIsPartialOutput() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsPartialOutput());
      }
    }
    if (!(getEnableMixDualMaster() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_mix_dual_master = ";
      if (!(getEnableMixDualMaster() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableMixDualMaster());
      }
    }
    if (!(getIsA2b2Shared() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_a2b2_shared = ";
      if (!(getIsA2b2Shared() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsA2b2Shared());
      }
    }
    if (!(getIsEnableChannelSplit() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_enable_channel_split = ";
      if (!(getIsEnableChannelSplit() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsEnableChannelSplit());
      }
    }
    if (!(getEnableKdimReorderLoad() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "enable_kdim_reorder_load = ";
      if (!(getEnableKdimReorderLoad() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getEnableKdimReorderLoad());
      }
    }
    if (!(getIsCo1Shared() == BoolAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "is_co1_shared = ";
      if (!(getIsCo1Shared() == BoolAttr())) {
        odsPrinter.printStrippedAttrOrType(getIsCo1Shared());
      }
    }
    if (!(getSharedCo1BufferSize() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "shared_co1_buffer_size = ";
      if (!(getSharedCo1BufferSize() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getSharedCo1BufferSize());
      }
    }
    if (!(getBatchOutMode() == IntegerAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "batch_out_mode = ";
      if (!(getBatchOutMode() == IntegerAttr())) {
        odsPrinter.printStrippedAttrOrType(getBatchOutMode());
      }
    }
  }
  odsPrinter << ">";
}

BoolAttr MatmulConfigAttr::getDoNorm() const {
  return getImpl()->do_norm;
}

BoolAttr MatmulConfigAttr::getDoBasicBlock() const {
  return getImpl()->do_basic_block;
}

BoolAttr MatmulConfigAttr::getDoMultiDataLoad() const {
  return getImpl()->do_multi_data_load;
}

IntegerAttr MatmulConfigAttr::getBasicM() const {
  return getImpl()->basic_m;
}

IntegerAttr MatmulConfigAttr::getBasicN() const {
  return getImpl()->basic_n;
}

IntegerAttr MatmulConfigAttr::getBasicK() const {
  return getImpl()->basic_k;
}

BoolAttr MatmulConfigAttr::getIntrinsicsCheck() const {
  return getImpl()->intrinsics_check;
}

BoolAttr MatmulConfigAttr::getIsNBatch() const {
  return getImpl()->is_n_batch;
}

BoolAttr MatmulConfigAttr::getEnVecNd2nz() const {
  return getImpl()->en_vec_nd2nz;
}

BoolAttr MatmulConfigAttr::getDoSpecialBasicBlock() const {
  return getImpl()->do_special_basic_block;
}

IntegerAttr MatmulConfigAttr::getDoMte2Preload() const {
  return getImpl()->do_mte2_preload;
}

IntegerAttr MatmulConfigAttr::getSingleCoreM() const {
  return getImpl()->single_core_m;
}

IntegerAttr MatmulConfigAttr::getSingleCoreN() const {
  return getImpl()->single_core_n;
}

IntegerAttr MatmulConfigAttr::getSingleCoreK() const {
  return getImpl()->single_core_k;
}

IntegerAttr MatmulConfigAttr::getStepM() const {
  return getImpl()->step_m;
}

IntegerAttr MatmulConfigAttr::getStepN() const {
  return getImpl()->step_n;
}

IntegerAttr MatmulConfigAttr::getBaseMn() const {
  return getImpl()->base_mn;
}

IntegerAttr MatmulConfigAttr::getSingleCoreMn() const {
  return getImpl()->single_core_mn;
}

BoolAttr MatmulConfigAttr::getEnUnitFlag() const {
  return getImpl()->en_unit_flag;
}

BoolAttr MatmulConfigAttr::getIsPerTensor() const {
  return getImpl()->is_per_tensor;
}

BoolAttr MatmulConfigAttr::getHasAntiQuantOffset() const {
  return getImpl()->has_anti_quant_offset;
}

BoolAttr MatmulConfigAttr::getDoIbShareNorm() const {
  return getImpl()->do_ib_share_norm;
}

BoolAttr MatmulConfigAttr::getDoSpecialMdl() const {
  return getImpl()->do_special_mdl;
}

BoolAttr MatmulConfigAttr::getEnableInit() const {
  return getImpl()->enable_init;
}

IntegerAttr MatmulConfigAttr::getBatchMode() const {
  return getImpl()->batch_mode;
}

BoolAttr MatmulConfigAttr::getEnableEnd() const {
  return getImpl()->enable_end;
}

BoolAttr MatmulConfigAttr::getEnableGetTensorC() const {
  return getImpl()->enable_get_tensor_c;
}

BoolAttr MatmulConfigAttr::getEnableSetOrgShape() const {
  return getImpl()->enable_set_org_shape;
}

BoolAttr MatmulConfigAttr::getEnableSetBias() const {
  return getImpl()->enable_set_bias;
}

BoolAttr MatmulConfigAttr::getEnableSetTail() const {
  return getImpl()->enable_set_tail;
}

BoolAttr MatmulConfigAttr::getEnableQuantVector() const {
  return getImpl()->enable_quant_vector;
}

BoolAttr MatmulConfigAttr::getEnableSetDefineData() const {
  return getImpl()->enable_set_define_data;
}

IntegerAttr MatmulConfigAttr::getIterateMode() const {
  return getImpl()->iterate_mode;
}

BoolAttr MatmulConfigAttr::getEnableReuse() const {
  return getImpl()->enable_reuse;
}

BoolAttr MatmulConfigAttr::getEnableUbReuse() const {
  return getImpl()->enable_ub_reuse;
}

BoolAttr MatmulConfigAttr::getEnableL1CacheUb() const {
  return getImpl()->enable_l1_cache_ub;
}

BoolAttr MatmulConfigAttr::getIntraBlockPartSum() const {
  return getImpl()->intra_block_part_sum;
}

IntegerAttr MatmulConfigAttr::getIterateOrder() const {
  return getImpl()->iterate_order;
}

IntegerAttr MatmulConfigAttr::getScheduleType() const {
  return getImpl()->schedule_type;
}

BoolAttr MatmulConfigAttr::getEnableDoubleCache() const {
  return getImpl()->enable_double_cache;
}

BoolAttr MatmulConfigAttr::getIsBiasBatch() const {
  return getImpl()->is_bias_batch;
}

BoolAttr MatmulConfigAttr::getEnableStaticPadZeros() const {
  return getImpl()->enable_static_pad_zeros;
}

BoolAttr MatmulConfigAttr::getIsPartialOutput() const {
  return getImpl()->is_partial_output;
}

BoolAttr MatmulConfigAttr::getEnableMixDualMaster() const {
  return getImpl()->enable_mix_dual_master;
}

BoolAttr MatmulConfigAttr::getIsA2b2Shared() const {
  return getImpl()->is_a2b2_shared;
}

BoolAttr MatmulConfigAttr::getIsEnableChannelSplit() const {
  return getImpl()->is_enable_channel_split;
}

BoolAttr MatmulConfigAttr::getEnableKdimReorderLoad() const {
  return getImpl()->enable_kdim_reorder_load;
}

BoolAttr MatmulConfigAttr::getIsCo1Shared() const {
  return getImpl()->is_co1_shared;
}

IntegerAttr MatmulConfigAttr::getSharedCo1BufferSize() const {
  return getImpl()->shared_co1_buffer_size;
}

IntegerAttr MatmulConfigAttr::getBatchOutMode() const {
  return getImpl()->batch_out_mode;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulConfigAttr)
namespace mlir::ascendc {

namespace detail {

struct SampleAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<IntegerAttr>;
  SampleAttrStorage(IntegerAttr number) : number(std::move(number)) {}

  KeyTy getAsKey() const {
    return KeyTy(number);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (number == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SampleAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto number = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SampleAttrStorage>()) SampleAttrStorage(std::move(number));
  }

  IntegerAttr number;
};

} // namespace detail
SampleAttr SampleAttr::get(::mlir::MLIRContext *context, IntegerAttr number) {
  return Base::get(context, std::move(number));
}

::mlir::Attribute SampleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<IntegerAttr> _result_number;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'number'
  _result_number = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_number)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_SampleAttr parameter 'number' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_number));
  return SampleAttr::get(odsParser.getContext(),
      IntegerAttr((*_result_number)));
}

void SampleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumber());
  odsPrinter << ">";
}

IntegerAttr SampleAttr::getNumber() const {
  return getImpl()->number;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::SampleAttr)
namespace mlir::ascendc {


/// Parse an attribute registered to this dialect.
::mlir::Attribute AscendCDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void AscendCDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::ascendc

#endif // GET_ATTRDEF_CLASSES

