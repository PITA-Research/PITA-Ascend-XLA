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
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::ascendc {

/// Represents AscendC::BaseGlobalTensor
class BaseGlobalTensorType;
/// Represents AscendC::BaseLocalTensor
class BaseLocalTensorType;
/// Represents AscendC::TBuf
class TBufType;
/// Represents AscendC::DataCopyPadExtParams
class DataCopyPadExtParamsType;
/// Represents AscendC::FixpipeParams
class FixpipeParamsType;
/// Global tensor from GM (OUT)
class GlobalTensorType;
/// Represents AscendC::LoadData2DMxParams
class LoadData2DMxParamsType;
/// Represents AscendC::LoadData3DParamsV2
class LoadData3DParamsV2Type;
/// Local tensor from UB
class LocalTensorType;
/// Represents vector mask (bit mode)
class MaskType;
/// Represents matmul::Matmul
class MatmulType;
/// Represents AscendC::TQueBind
class QueBindType;
/// Represents AscendC::TQue
class QueueType;
/// Represents AscendC::TBufPool
class TBufPoolType;
/// Represents AscendC::AippChannelPaddingParams
class AippChannelPaddingParamsType;
/// Represents AscendC::AippColorSpaceConvParams
class AippColorSpaceConvParamsType;
/// Represents AscendC::AippDataTypeConvParams
class AippDataTypeConvParamsType;
/// Represents AscendC::AippInputFormat
class AippInputFormatType;
/// Represents AscendC::AippPaddingParams
class AippPaddingParamsType;
/// Represents AscendC::AippParams
class AippParamsType;
/// Represents AscendC::AippSingleLineParams
class AippSingleLineParamsType;
/// Represents AscendC::AippSwapParams
class AippSwapParamsType;
/// Represents AscendC::AscendQuantConfig
class AscendQuantConfigType;
/// Represents BatchMode
class BatchModeType;
/// Represents BatchOutMode
class BatchOutModeType;
/// Represents AscendC::BinaryRepeatParams
class BinaryRepeatParamsType;
/// Represents AscendC::BlockMode
class BlockModeType;
/// Represents AscendC::BrcbRepeatParams
class BrcbRepeatParamsType;
/// Represents AscendC::CopyRepeatParams
class CopyRepeatParamsType;
/// Represents AscendC::DataCopyCO12DstParams
class DataCopyCO12DstParamsType;
/// Represents AscendC::DataCopyEnhancedParams
class DataCopyEnhancedParamsType;
/// Represents AscendC::DataCopyExtParams
class DataCopyExtParamsType;
/// Represents AscendC::DataCopyPadParams
class DataCopyPadParamsType;
/// Represents AscendC::DataCopyParams
class DataCopyParamsType;
/// Represents AscendC::DataFormat
class DataFormatType;
/// Represents AscendC::DeqScale
class DeqScaleType;
/// Represents AscendC::FixpipeConfig
class FixpipeConfigType;
/// Represents AscendC::FmatrixMode
class FmatrixModeType;
/// Represents AscendC::GatherMaskParams
class GatherMaskParamsType;
/// Represents AscendC::GatherRepeatParams
class GatherRepeatParamsType;
/// Represents AscendC::IsResetLoad3dConfig
class IsResetLoad3dConfigType;
/// Represents IterateMode
class IterateModeType;
/// Represents IterateOrder
class IterateOrderType;
/// Represents AscendC::KfcServer
class KfcServerType;
/// Represents AscendC::LoadData2DParams
class LoadData2DParamsType;
/// Represents AscendC::LoadData2dTransposeParams
class LoadData2dTransposeParamsType;
/// Represents AscendC::LoadDataRepeatParam
class LoadDataRepeatParamType;
/// Represents AscendC::LoadImageToLocalParams
class LoadImageToLocalParamsType;
/// Represents AscendC::MaskMode
class MaskModeType;
/// Represents MatmulApiStaticTiling
class MatmulApiStaticTilingType;
/// Represents MatmulConfig
class MatmulConfigType;
/// Represents MatrixOffset
class MatrixOffsetType;
/// Represents AscendC::MmadParams
class MmadParamsType;
/// Represents AscendC::Nd2NzParams
class Nd2NzParamsType;
/// Represents AscendC::Nz2NdParams
class Nz2NdParamsType;
/// Represents AscendC::Nz2NdParamsFull
class Nz2NdParamsFullType;
/// Represents AscendC::pad_t
class pad_tType;
/// Represents QuantMode_t
class QuantModesType;
/// Represents AscendC::QuantParams
class QuantParamsType;
/// Represents AscendC::ReduceOrder
class ReduceOrderType;
/// Represents RmsNormTiling
class RmsNormTilingType;
/// Represents ScheduleType
class ScheduleTypeType;
/// Represents AscendC::ShapeInfo
class ShapeInfoType;
/// Represents AscendC::SliceInfo
class SliceInfoType;
/// Represents AscendC::SoftMaxShapeInfo
class SoftMaxShapeInfoType;
/// Represents SoftMaxTiling
class SoftMaxTilingType;
/// Represents AscendC::TBuffAddr
class BufAddrType;
/// Represents TCubeTiling
class TCubeTilingType;
/// Represents AscendC::TPipe
class PipeType;
/// Represents AscendC::TransDataTo5HDParams
class TransDataTo5HDParamsType;
/// Represents AscendC::TransposeParamsExt
class TransposeParamsExtType;
/// Represents AscendC::TransposeType
class TransposeTypeType;
/// Represents AscendC::UnaryRepeatParams
class UnaryRepeatParamsType;
namespace detail {

struct BaseGlobalTensorTypeStorage;

} // namespace detail
class BaseGlobalTensorType : public ::mlir::Type::TypeBase<BaseGlobalTensorType, ::mlir::Type, detail::BaseGlobalTensorTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ShapedType::Trait, ::mlir::ascendc::BaseTensorType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.base_global_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static BaseGlobalTensorType get(ArrayRef<int64_t> shape, Type elementType);
  static BaseGlobalTensorType get(Type elementType);
  static BaseGlobalTensorType get(::mlir::ascendc::BaseTensorType baseType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"base_global_tensor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  bool hasRank() const;
};
namespace detail {

struct BaseLocalTensorTypeStorage;

} // namespace detail
class BaseLocalTensorType : public ::mlir::Type::TypeBase<BaseLocalTensorType, ::mlir::Type, detail::BaseLocalTensorTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ShapedType::Trait, ::mlir::ascendc::BaseTensorType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.base_local_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static BaseLocalTensorType get(ArrayRef<int64_t> shape, Type elementType);
  static BaseLocalTensorType get(Type elementType);
  static BaseLocalTensorType get(::mlir::ascendc::BaseTensorType baseType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"base_local_tensor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  bool hasRank() const;
};
namespace detail {

struct TBufTypeStorage;

} // namespace detail
class TBufType : public ::mlir::Type::TypeBase<TBufType, ::mlir::Type, detail::TBufTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ascendc::BaseQueueType::Trait> {
public:
  using Base::Base;
  TPosition getTPosition() { return getTPositionAttr().getValue(); }
  static constexpr ::llvm::StringLiteral name = "ascendc.tbuf";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static TBufType get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr);
  static TBufType get(::mlir::MLIRContext *context, TPosition position);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tbuf"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TPositionAttr getTPositionAttr() const;
};
namespace detail {

struct DataCopyPadExtParamsTypeStorage;

} // namespace detail
class DataCopyPadExtParamsType : public ::mlir::Type::TypeBase<DataCopyPadExtParamsType, ::mlir::Type, detail::DataCopyPadExtParamsTypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_pad_ext_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static DataCopyPadExtParamsType get(::mlir::MLIRContext *context, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_pad_ext_params"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
};
namespace detail {

struct FixpipeParamsTypeStorage;

} // namespace detail
class FixpipeParamsType : public ::mlir::Type::TypeBase<FixpipeParamsType, ::mlir::Type, detail::FixpipeParamsTypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.fixpipe_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static FixpipeParamsType get(::mlir::MLIRContext *context, Type type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fixpipe_params"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getType() const;
};
namespace detail {

struct GlobalTensorTypeStorage;

} // namespace detail
class GlobalTensorType : public ::mlir::Type::TypeBase<GlobalTensorType, ::mlir::Type, detail::GlobalTensorTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ShapedType::Trait, ::mlir::ascendc::BaseTensorType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.global_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static GlobalTensorType get(ArrayRef<int64_t> shape, Type elementType);
  static GlobalTensorType get(Type elementType);
  static GlobalTensorType get(::mlir::ascendc::BaseTensorType baseType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"global_tensor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  bool hasRank() const;
};
class LoadData2DMxParamsType : public ::mlir::Type::TypeBase<LoadData2DMxParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ascendc::BaseLoadDataParamsType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_data_2d_mx_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_data_2d_mx_params"};
  }

};
namespace detail {

struct LoadData3DParamsV2TypeStorage;

} // namespace detail
class LoadData3DParamsV2Type : public ::mlir::Type::TypeBase<LoadData3DParamsV2Type, ::mlir::Type, detail::LoadData3DParamsV2TypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ascendc::BaseLoadDataParamsType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_data_3d_params_v2";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static LoadData3DParamsV2Type get(::mlir::MLIRContext *context, Type type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_data_3d_params_v2"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getType() const;
};
namespace detail {

struct LocalTensorTypeStorage;

} // namespace detail
class LocalTensorType : public ::mlir::Type::TypeBase<LocalTensorType, ::mlir::Type, detail::LocalTensorTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ShapedType::Trait, ::mlir::ascendc::BaseTensorType::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.local_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static LocalTensorType get(ArrayRef<int64_t> shape, Type elementType);
  static LocalTensorType get(Type elementType);
  static LocalTensorType get(::mlir::ascendc::BaseTensorType baseType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"local_tensor"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  bool hasRank() const;
};
class MaskType : public ::mlir::Type::TypeBase<MaskType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.mask";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mask"};
  }

};
namespace detail {

struct MatmulTypeStorage;

} // namespace detail
class MatmulType : public ::mlir::Type::TypeBase<MatmulType, ::mlir::Type, detail::MatmulTypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  TPosition getSrcAPosition() { return getSrcAPositionAttr().getValue(); }
  TPosition getSrcBPosition() { return getSrcBPositionAttr().getValue(); }
  TPosition getSrcCPosition() { return getSrcCPositionAttr().getValue(); }
  TPosition getSrcBiasPosition() { return getSrcBiasPositionAttr().getValue(); }
  CubeFormat getCubeFormatA() { return getCubeFormatAAttr().getValue(); }
  CubeFormat getCubeFormatB() { return getCubeFormatBAttr().getValue(); }
  CubeFormat getCubeFormatC() { return getCubeFormatCAttr().getValue(); }
  LayoutMode getLayoutModeA() { return getLayoutModeAAttr().getValue(); }
  LayoutMode getLayoutModeB() { return getLayoutModeBAttr().getValue(); }
  LayoutMode getLayoutModeC() { return getLayoutModeCAttr().getValue(); }
  CubeFormat getCubeFormatBias() { return getCubeFormatBiasAttr().getValue(); }
  static constexpr ::llvm::StringLiteral name = "ascendc.matmul";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static MatmulType get(::mlir::MLIRContext *context, TPositionAttr srcAPositionAttr, CubeFormatAttr cubeFormatAAttr, Type typeA, bool isTransA, LayoutModeAttr layoutModeAAttr, TPositionAttr srcBPositionAttr, CubeFormatAttr cubeFormatBAttr, Type typeB, bool isTransB, LayoutModeAttr layoutModeBAttr, TPositionAttr srcCPositionAttr, CubeFormatAttr cubeFormatCAttr, Type typeC, bool isTransC, LayoutModeAttr layoutModeCAttr, TPositionAttr srcBiasPositionAttr, CubeFormatAttr cubeFormatBiasAttr, Type typeBias, MatmulConfigAttr matmulConfig);
  static MatmulType get(::mlir::MLIRContext *context, TPosition srcAPosition, CubeFormat cubeFormatA, Type typeA, bool isTransA, LayoutMode layoutModeA, TPosition srcBPosition, CubeFormat cubeFormatB, Type typeB, bool isTransB, LayoutMode layoutModeB, TPosition srcCPosition, CubeFormat cubeFormatC, Type typeC, bool isTransC, LayoutMode layoutModeC, TPosition srcBiasPosition, CubeFormat cubeFormatBias, Type typeBias, MatmulConfigAttr matmulConfig);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TPositionAttr getSrcAPositionAttr() const;
  CubeFormatAttr getCubeFormatAAttr() const;
  Type getTypeA() const;
  bool getIsTransA() const;
  LayoutModeAttr getLayoutModeAAttr() const;
  TPositionAttr getSrcBPositionAttr() const;
  CubeFormatAttr getCubeFormatBAttr() const;
  Type getTypeB() const;
  bool getIsTransB() const;
  LayoutModeAttr getLayoutModeBAttr() const;
  TPositionAttr getSrcCPositionAttr() const;
  CubeFormatAttr getCubeFormatCAttr() const;
  Type getTypeC() const;
  bool getIsTransC() const;
  LayoutModeAttr getLayoutModeCAttr() const;
  TPositionAttr getSrcBiasPositionAttr() const;
  CubeFormatAttr getCubeFormatBiasAttr() const;
  Type getTypeBias() const;
  MatmulConfigAttr getMatmulConfig() const;
};
namespace detail {

struct QueBindTypeStorage;

} // namespace detail
class QueBindType : public ::mlir::Type::TypeBase<QueBindType, ::mlir::Type, detail::QueBindTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ascendc::BaseQueueType::Trait> {
public:
  using Base::Base;
  TPosition getSrcPosition() { return getSrcPositionAttr().getValue(); }
  TPosition getDstPosition() { return getDstPositionAttr().getValue(); }
  static constexpr ::llvm::StringLiteral name = "ascendc.que_bind";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static QueBindType get(::mlir::MLIRContext *context, TPositionAttr srcPositionAttr, TPositionAttr dstPositionAttr, int64_t depth);
  static QueBindType get(::mlir::MLIRContext *context, TPosition srcPosition, TPosition dstPosition, int64_t depth);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"que_bind"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TPositionAttr getSrcPositionAttr() const;
  TPositionAttr getDstPositionAttr() const;
  int64_t getDepth() const;
};
namespace detail {

struct QueueTypeStorage;

} // namespace detail
class QueueType : public ::mlir::Type::TypeBase<QueueType, ::mlir::Type, detail::QueueTypeStorage, ::mlir::MemRefElementTypeInterface::Trait, ::mlir::ascendc::BaseQueueType::Trait> {
public:
  using Base::Base;
  TPosition getPosition() { return getTPositionAttr().getValue(); }
  static constexpr ::llvm::StringLiteral name = "ascendc.queue";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static QueueType get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr, int64_t depth);
  static QueueType get(::mlir::MLIRContext *context, TPosition position, int64_t depth);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"queue"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TPositionAttr getTPositionAttr() const;
  int64_t getDepth() const;
};
namespace detail {

struct TBufPoolTypeStorage;

} // namespace detail
class TBufPoolType : public ::mlir::Type::TypeBase<TBufPoolType, ::mlir::Type, detail::TBufPoolTypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  TPosition getTPosition() { return getTPositionAttr().getValue(); }
  static constexpr ::llvm::StringLiteral name = "ascendc.tbuf_pool";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static TBufPoolType get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr, uint32_t bufIDSize);
  static TBufPoolType get(::mlir::MLIRContext *context, TPosition position, uint32_t bufIDSize);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tbuf_pool"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TPositionAttr getTPositionAttr() const;
  uint32_t getBufIDSize() const;
};
class AippChannelPaddingParamsType : public ::mlir::Type::TypeBase<AippChannelPaddingParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_cpadding_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_cpadding_params"};
  }

};
class AippColorSpaceConvParamsType : public ::mlir::Type::TypeBase<AippColorSpaceConvParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_csc_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_csc_params"};
  }

};
class AippDataTypeConvParamsType : public ::mlir::Type::TypeBase<AippDataTypeConvParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_dtc_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_dtc_params"};
  }

};
class AippInputFormatType : public ::mlir::Type::TypeBase<AippInputFormatType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_input_format";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_input_format"};
  }

};
class AippPaddingParamsType : public ::mlir::Type::TypeBase<AippPaddingParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_padding_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_padding_params"};
  }

};
class AippParamsType : public ::mlir::Type::TypeBase<AippParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_params"};
  }

};
class AippSingleLineParamsType : public ::mlir::Type::TypeBase<AippSingleLineParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_single_line_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_single_line_params"};
  }

};
class AippSwapParamsType : public ::mlir::Type::TypeBase<AippSwapParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.aipp_swap_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aipp_swap_params"};
  }

};
class AscendQuantConfigType : public ::mlir::Type::TypeBase<AscendQuantConfigType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.ascend_quant_config";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ascend_quant_config"};
  }

};
class BatchModeType : public ::mlir::Type::TypeBase<BatchModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.batch_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"batch_mode"};
  }

};
class BatchOutModeType : public ::mlir::Type::TypeBase<BatchOutModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.batch_out_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"batch_out_mode"};
  }

};
class BinaryRepeatParamsType : public ::mlir::Type::TypeBase<BinaryRepeatParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.binary_repeat_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"binary_repeat_params"};
  }

};
class BlockModeType : public ::mlir::Type::TypeBase<BlockModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.block_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block_mode"};
  }

};
class BrcbRepeatParamsType : public ::mlir::Type::TypeBase<BrcbRepeatParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.brcb_repeat_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"brcb_repeat_params"};
  }

};
class CopyRepeatParamsType : public ::mlir::Type::TypeBase<CopyRepeatParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.copy_repeat_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"copy_repeat_params"};
  }

};
class DataCopyCO12DstParamsType : public ::mlir::Type::TypeBase<DataCopyCO12DstParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_co12dst_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_co12dst_params"};
  }

};
class DataCopyEnhancedParamsType : public ::mlir::Type::TypeBase<DataCopyEnhancedParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_enhanced_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_enhanced_params"};
  }

};
class DataCopyExtParamsType : public ::mlir::Type::TypeBase<DataCopyExtParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_ext_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_ext_params"};
  }

};
class DataCopyPadParamsType : public ::mlir::Type::TypeBase<DataCopyPadParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_pad_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_pad_params"};
  }

};
class DataCopyParamsType : public ::mlir::Type::TypeBase<DataCopyParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_copy_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_copy_params"};
  }

};
class DataFormatType : public ::mlir::Type::TypeBase<DataFormatType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.data_format";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_format"};
  }

};
class DeqScaleType : public ::mlir::Type::TypeBase<DeqScaleType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.deq_scale";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"deq_scale"};
  }

};
class FixpipeConfigType : public ::mlir::Type::TypeBase<FixpipeConfigType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.fixpipe_config";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fixpipe_config"};
  }

};
class FmatrixModeType : public ::mlir::Type::TypeBase<FmatrixModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.fmatrix_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fmatrix_mode"};
  }

};
class GatherMaskParamsType : public ::mlir::Type::TypeBase<GatherMaskParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.gather_mask_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"gather_mask_params"};
  }

};
class GatherRepeatParamsType : public ::mlir::Type::TypeBase<GatherRepeatParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.gather_repeat_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"gather_repeat_params"};
  }

};
class IsResetLoad3dConfigType : public ::mlir::Type::TypeBase<IsResetLoad3dConfigType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.is_reset_load_3d_config";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"is_reset_load_3d_config"};
  }

};
class IterateModeType : public ::mlir::Type::TypeBase<IterateModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.iterate_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"iterate_mode"};
  }

};
class IterateOrderType : public ::mlir::Type::TypeBase<IterateOrderType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.iterate_order";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"iterate_order"};
  }

};
class KfcServerType : public ::mlir::Type::TypeBase<KfcServerType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.kfc_server";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kfc_server"};
  }

};
class LoadData2DParamsType : public ::mlir::Type::TypeBase<LoadData2DParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_data_2d_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_data_2d_params"};
  }

};
class LoadData2dTransposeParamsType : public ::mlir::Type::TypeBase<LoadData2dTransposeParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_data_2d_transpose_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_data_2d_transpose_params"};
  }

};
class LoadDataRepeatParamType : public ::mlir::Type::TypeBase<LoadDataRepeatParamType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_data_repeat_param";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_data_repeat_param"};
  }

};
class LoadImageToLocalParamsType : public ::mlir::Type::TypeBase<LoadImageToLocalParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.load_image_to_local_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_image_to_local_params"};
  }

};
class MaskModeType : public ::mlir::Type::TypeBase<MaskModeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.mask_mode";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mask_mode"};
  }

};
class MatmulApiStaticTilingType : public ::mlir::Type::TypeBase<MatmulApiStaticTilingType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.matmul_api_static_tiling";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul_api_static_tiling"};
  }

};
class MatmulConfigType : public ::mlir::Type::TypeBase<MatmulConfigType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.matmul_config";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul_config"};
  }

};
class MatrixOffsetType : public ::mlir::Type::TypeBase<MatrixOffsetType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.matrix_offset";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matrix_offset"};
  }

};
class MmadParamsType : public ::mlir::Type::TypeBase<MmadParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.mmad_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mmad_params"};
  }

};
class Nd2NzParamsType : public ::mlir::Type::TypeBase<Nd2NzParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.nd2nz_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nd2nz_params"};
  }

};
class Nz2NdParamsType : public ::mlir::Type::TypeBase<Nz2NdParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.nz2nd_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nz2nd_params"};
  }

};
class Nz2NdParamsFullType : public ::mlir::Type::TypeBase<Nz2NdParamsFullType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.nz2nd_params_full";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nz2nd_params_full"};
  }

};
class pad_tType : public ::mlir::Type::TypeBase<pad_tType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.pad_t";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pad_t"};
  }

};
class QuantModesType : public ::mlir::Type::TypeBase<QuantModesType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.quant_mode_t";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"quant_mode_t"};
  }

};
class QuantParamsType : public ::mlir::Type::TypeBase<QuantParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.quant_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"quant_params"};
  }

};
class ReduceOrderType : public ::mlir::Type::TypeBase<ReduceOrderType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.reduce_order";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"reduce_order"};
  }

};
class RmsNormTilingType : public ::mlir::Type::TypeBase<RmsNormTilingType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.rmsnorm_tiling";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rmsnorm_tiling"};
  }

};
class ScheduleTypeType : public ::mlir::Type::TypeBase<ScheduleTypeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.schedule_type";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"schedule_type"};
  }

};
class ShapeInfoType : public ::mlir::Type::TypeBase<ShapeInfoType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.shape_info";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape_info"};
  }

};
class SliceInfoType : public ::mlir::Type::TypeBase<SliceInfoType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.slice_info";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"slice_info"};
  }

};
class SoftMaxShapeInfoType : public ::mlir::Type::TypeBase<SoftMaxShapeInfoType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.softmax_shape_info";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"softmax_shape_info"};
  }

};
class SoftMaxTilingType : public ::mlir::Type::TypeBase<SoftMaxTilingType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.softmax_tiling";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"softmax_tiling"};
  }

};
class BufAddrType : public ::mlir::Type::TypeBase<BufAddrType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.buf_addr";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"buf_addr"};
  }

};
class TCubeTilingType : public ::mlir::Type::TypeBase<TCubeTilingType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.cube_tiling";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cube_tiling"};
  }

};
class PipeType : public ::mlir::Type::TypeBase<PipeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.pipe";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pipe"};
  }

};
class TransDataTo5HDParamsType : public ::mlir::Type::TypeBase<TransDataTo5HDParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.trans_data_to_5hd_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"trans_data_to_5hd_params"};
  }

};
class TransposeParamsExtType : public ::mlir::Type::TypeBase<TransposeParamsExtType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.transpose_params_ext";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"transpose_params_ext"};
  }

};
class TransposeTypeType : public ::mlir::Type::TypeBase<TransposeTypeType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.transpose_type";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"transpose_type"};
  }

};
class UnaryRepeatParamsType : public ::mlir::Type::TypeBase<UnaryRepeatParamsType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ascendc.unary_repeat_params";
  static constexpr ::llvm::StringLiteral dialectName = "ascendc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"unary_repeat_params"};
  }

};

} // namespace mlir::ascendc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BaseGlobalTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BaseLocalTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TBufType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyPadExtParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::FixpipeParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::GlobalTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2DMxParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData3DParamsV2Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LocalTensorType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MaskType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::QueBindType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::QueueType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TBufPoolType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippChannelPaddingParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippColorSpaceConvParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippDataTypeConvParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippInputFormatType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippPaddingParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippSingleLineParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippSwapParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::AscendQuantConfigType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BatchModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BatchOutModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BinaryRepeatParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BlockModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BrcbRepeatParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::CopyRepeatParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyCO12DstParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyEnhancedParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyExtParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyPadParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataFormatType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::DeqScaleType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::FixpipeConfigType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::FmatrixModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::GatherMaskParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::GatherRepeatParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::IsResetLoad3dConfigType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::IterateModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::IterateOrderType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::KfcServerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2DParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2dTransposeParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadDataRepeatParamType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadImageToLocalParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MaskModeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulApiStaticTilingType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulConfigType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatrixOffsetType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::MmadParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nd2NzParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nz2NdParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nz2NdParamsFullType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::pad_tType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::QuantModesType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::QuantParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::ReduceOrderType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::RmsNormTilingType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::ScheduleTypeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::ShapeInfoType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::SliceInfoType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::SoftMaxShapeInfoType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::SoftMaxTilingType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::BufAddrType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TCubeTilingType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::PipeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransDataTo5HDParamsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransposeParamsExtType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransposeTypeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ascendc::UnaryRepeatParamsType)

#endif // GET_TYPEDEF_CLASSES

