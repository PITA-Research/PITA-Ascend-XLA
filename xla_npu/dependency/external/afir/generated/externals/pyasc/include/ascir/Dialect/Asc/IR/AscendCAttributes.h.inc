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
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::ascendc {

class APIAttr;
class MatmulConfigAttr;
class SampleAttr;
namespace detail {

struct APIAttrStorage;

} // namespace detail
class APIAttr : public ::mlir::Attribute::AttrBase<APIAttr, ::mlir::Attribute, detail::APIAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.api";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static APIAttr get(::mlir::MLIRContext *context, StringAttr group, StringAttr name, StringAttr pipe_in, StringAttr pipe_out, ArrayAttr args);
  static APIAttr get(StringAttr group, StringAttr name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"api"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getGroup() const;
  StringAttr getName() const;
  StringAttr getPipeIn() const;
  StringAttr getPipeOut() const;
  ArrayAttr getArgs() const;
};
namespace detail {

struct MatmulConfigAttrStorage;

} // namespace detail
class MatmulConfigAttr : public ::mlir::Attribute::AttrBase<MatmulConfigAttr, ::mlir::Attribute, detail::MatmulConfigAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.matmul_config";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static MatmulConfigAttr get(::mlir::MLIRContext *context, BoolAttr do_norm, BoolAttr do_basic_block, BoolAttr do_multi_data_load, IntegerAttr basic_m, IntegerAttr basic_n, IntegerAttr basic_k, BoolAttr intrinsics_check, BoolAttr is_n_batch, BoolAttr en_vec_nd2nz, BoolAttr do_special_basic_block, IntegerAttr do_mte2_preload, IntegerAttr single_core_m, IntegerAttr single_core_n, IntegerAttr single_core_k, IntegerAttr step_m, IntegerAttr step_n, IntegerAttr base_mn, IntegerAttr single_core_mn, BoolAttr en_unit_flag, BoolAttr is_per_tensor, BoolAttr has_anti_quant_offset, BoolAttr do_ib_share_norm, BoolAttr do_special_mdl, BoolAttr enable_init, IntegerAttr batch_mode, BoolAttr enable_end, BoolAttr enable_get_tensor_c, BoolAttr enable_set_org_shape, BoolAttr enable_set_bias, BoolAttr enable_set_tail, BoolAttr enable_quant_vector, BoolAttr enable_set_define_data, IntegerAttr iterate_mode, BoolAttr enable_reuse, BoolAttr enable_ub_reuse, BoolAttr enable_l1_cache_ub, BoolAttr intra_block_part_sum, IntegerAttr iterate_order, IntegerAttr schedule_type, BoolAttr enable_double_cache, BoolAttr is_bias_batch, BoolAttr enable_static_pad_zeros, BoolAttr is_partial_output, BoolAttr enable_mix_dual_master, BoolAttr is_a2b2_shared, BoolAttr is_enable_channel_split, BoolAttr enable_kdim_reorder_load, BoolAttr is_co1_shared, IntegerAttr shared_co1_buffer_size, IntegerAttr batch_out_mode);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul_config"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  BoolAttr getDoNorm() const;
  BoolAttr getDoBasicBlock() const;
  BoolAttr getDoMultiDataLoad() const;
  IntegerAttr getBasicM() const;
  IntegerAttr getBasicN() const;
  IntegerAttr getBasicK() const;
  BoolAttr getIntrinsicsCheck() const;
  BoolAttr getIsNBatch() const;
  BoolAttr getEnVecNd2nz() const;
  BoolAttr getDoSpecialBasicBlock() const;
  IntegerAttr getDoMte2Preload() const;
  IntegerAttr getSingleCoreM() const;
  IntegerAttr getSingleCoreN() const;
  IntegerAttr getSingleCoreK() const;
  IntegerAttr getStepM() const;
  IntegerAttr getStepN() const;
  IntegerAttr getBaseMn() const;
  IntegerAttr getSingleCoreMn() const;
  BoolAttr getEnUnitFlag() const;
  BoolAttr getIsPerTensor() const;
  BoolAttr getHasAntiQuantOffset() const;
  BoolAttr getDoIbShareNorm() const;
  BoolAttr getDoSpecialMdl() const;
  BoolAttr getEnableInit() const;
  IntegerAttr getBatchMode() const;
  BoolAttr getEnableEnd() const;
  BoolAttr getEnableGetTensorC() const;
  BoolAttr getEnableSetOrgShape() const;
  BoolAttr getEnableSetBias() const;
  BoolAttr getEnableSetTail() const;
  BoolAttr getEnableQuantVector() const;
  BoolAttr getEnableSetDefineData() const;
  IntegerAttr getIterateMode() const;
  BoolAttr getEnableReuse() const;
  BoolAttr getEnableUbReuse() const;
  BoolAttr getEnableL1CacheUb() const;
  BoolAttr getIntraBlockPartSum() const;
  IntegerAttr getIterateOrder() const;
  IntegerAttr getScheduleType() const;
  BoolAttr getEnableDoubleCache() const;
  BoolAttr getIsBiasBatch() const;
  BoolAttr getEnableStaticPadZeros() const;
  BoolAttr getIsPartialOutput() const;
  BoolAttr getEnableMixDualMaster() const;
  BoolAttr getIsA2b2Shared() const;
  BoolAttr getIsEnableChannelSplit() const;
  BoolAttr getEnableKdimReorderLoad() const;
  BoolAttr getIsCo1Shared() const;
  IntegerAttr getSharedCo1BufferSize() const;
  IntegerAttr getBatchOutMode() const;
};
namespace detail {

struct SampleAttrStorage;

} // namespace detail
class SampleAttr : public ::mlir::Attribute::AttrBase<SampleAttr, ::mlir::Attribute, detail::SampleAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.sample";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static SampleAttr get(::mlir::MLIRContext *context, IntegerAttr number);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sample"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  IntegerAttr getNumber() const;
};

} // namespace mlir::ascendc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::APIAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulConfigAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::SampleAttr)

#endif // GET_ATTRDEF_CLASSES

