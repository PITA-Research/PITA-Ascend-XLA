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
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::ascendc::BaseGlobalTensorType,
::mlir::ascendc::BaseLocalTensorType,
::mlir::ascendc::TBufType,
::mlir::ascendc::DataCopyPadExtParamsType,
::mlir::ascendc::FixpipeParamsType,
::mlir::ascendc::GlobalTensorType,
::mlir::ascendc::LoadData2DMxParamsType,
::mlir::ascendc::LoadData3DParamsV2Type,
::mlir::ascendc::LocalTensorType,
::mlir::ascendc::MaskType,
::mlir::ascendc::MatmulType,
::mlir::ascendc::QueBindType,
::mlir::ascendc::QueueType,
::mlir::ascendc::TBufPoolType,
::mlir::ascendc::AippChannelPaddingParamsType,
::mlir::ascendc::AippColorSpaceConvParamsType,
::mlir::ascendc::AippDataTypeConvParamsType,
::mlir::ascendc::AippInputFormatType,
::mlir::ascendc::AippPaddingParamsType,
::mlir::ascendc::AippParamsType,
::mlir::ascendc::AippSingleLineParamsType,
::mlir::ascendc::AippSwapParamsType,
::mlir::ascendc::AscendQuantConfigType,
::mlir::ascendc::BatchModeType,
::mlir::ascendc::BatchOutModeType,
::mlir::ascendc::BinaryRepeatParamsType,
::mlir::ascendc::BlockModeType,
::mlir::ascendc::BrcbRepeatParamsType,
::mlir::ascendc::CopyRepeatParamsType,
::mlir::ascendc::DataCopyCO12DstParamsType,
::mlir::ascendc::DataCopyEnhancedParamsType,
::mlir::ascendc::DataCopyExtParamsType,
::mlir::ascendc::DataCopyPadParamsType,
::mlir::ascendc::DataCopyParamsType,
::mlir::ascendc::DataFormatType,
::mlir::ascendc::DeqScaleType,
::mlir::ascendc::FixpipeConfigType,
::mlir::ascendc::FmatrixModeType,
::mlir::ascendc::GatherMaskParamsType,
::mlir::ascendc::GatherRepeatParamsType,
::mlir::ascendc::IsResetLoad3dConfigType,
::mlir::ascendc::IterateModeType,
::mlir::ascendc::IterateOrderType,
::mlir::ascendc::KfcServerType,
::mlir::ascendc::LoadData2DParamsType,
::mlir::ascendc::LoadData2dTransposeParamsType,
::mlir::ascendc::LoadDataRepeatParamType,
::mlir::ascendc::LoadImageToLocalParamsType,
::mlir::ascendc::MaskModeType,
::mlir::ascendc::MatmulApiStaticTilingType,
::mlir::ascendc::MatmulConfigType,
::mlir::ascendc::MatrixOffsetType,
::mlir::ascendc::MmadParamsType,
::mlir::ascendc::Nd2NzParamsType,
::mlir::ascendc::Nz2NdParamsType,
::mlir::ascendc::Nz2NdParamsFullType,
::mlir::ascendc::pad_tType,
::mlir::ascendc::QuantModesType,
::mlir::ascendc::QuantParamsType,
::mlir::ascendc::ReduceOrderType,
::mlir::ascendc::RmsNormTilingType,
::mlir::ascendc::ScheduleTypeType,
::mlir::ascendc::ShapeInfoType,
::mlir::ascendc::SliceInfoType,
::mlir::ascendc::SoftMaxShapeInfoType,
::mlir::ascendc::SoftMaxTilingType,
::mlir::ascendc::BufAddrType,
::mlir::ascendc::TCubeTilingType,
::mlir::ascendc::PipeType,
::mlir::ascendc::TransDataTo5HDParamsType,
::mlir::ascendc::TransposeParamsExtType,
::mlir::ascendc::TransposeTypeType,
::mlir::ascendc::UnaryRepeatParamsType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::ascendc::BaseGlobalTensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BaseGlobalTensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BaseLocalTensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BaseLocalTensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TBufType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TBufType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyPadExtParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyPadExtParamsType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::FixpipeParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::FixpipeParamsType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::GlobalTensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::GlobalTensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadData2DMxParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadData2DMxParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadData3DParamsV2Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadData3DParamsV2Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LocalTensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LocalTensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MaskType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MaskType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MatmulType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MatmulType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::QueBindType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::QueBindType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::QueueType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::QueueType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TBufPoolType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TBufPoolType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippChannelPaddingParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippChannelPaddingParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippColorSpaceConvParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippColorSpaceConvParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippDataTypeConvParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippDataTypeConvParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippInputFormatType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippInputFormatType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippPaddingParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippPaddingParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippSingleLineParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippSingleLineParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AippSwapParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AippSwapParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::AscendQuantConfigType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::AscendQuantConfigType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BatchModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BatchModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BatchOutModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BatchOutModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BinaryRepeatParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BinaryRepeatParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BlockModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BlockModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BrcbRepeatParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BrcbRepeatParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::CopyRepeatParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::CopyRepeatParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyCO12DstParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyCO12DstParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyEnhancedParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyEnhancedParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyExtParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyExtParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyPadParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyPadParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataCopyParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataCopyParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DataFormatType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DataFormatType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::DeqScaleType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::DeqScaleType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::FixpipeConfigType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::FixpipeConfigType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::FmatrixModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::FmatrixModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::GatherMaskParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::GatherMaskParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::GatherRepeatParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::GatherRepeatParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::IsResetLoad3dConfigType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::IsResetLoad3dConfigType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::IterateModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::IterateModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::IterateOrderType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::IterateOrderType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::KfcServerType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::KfcServerType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadData2DParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadData2DParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadData2dTransposeParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadData2dTransposeParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadDataRepeatParamType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadDataRepeatParamType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::LoadImageToLocalParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::LoadImageToLocalParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MaskModeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MaskModeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MatmulApiStaticTilingType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MatmulApiStaticTilingType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MatmulConfigType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MatmulConfigType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MatrixOffsetType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MatrixOffsetType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::MmadParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::MmadParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::Nd2NzParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::Nd2NzParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::Nz2NdParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::Nz2NdParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::Nz2NdParamsFullType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::Nz2NdParamsFullType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::pad_tType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::pad_tType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::QuantModesType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::QuantModesType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::QuantParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::QuantParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::ReduceOrderType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::ReduceOrderType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::RmsNormTilingType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::RmsNormTilingType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::ScheduleTypeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::ScheduleTypeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::ShapeInfoType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::ShapeInfoType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::SliceInfoType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::SliceInfoType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::SoftMaxShapeInfoType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::SoftMaxShapeInfoType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::SoftMaxTilingType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::SoftMaxTilingType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::BufAddrType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::BufAddrType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TCubeTilingType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TCubeTilingType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::PipeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::PipeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TransDataTo5HDParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TransDataTo5HDParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TransposeParamsExtType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TransposeParamsExtType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::TransposeTypeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::TransposeTypeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::ascendc::UnaryRepeatParamsType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::ascendc::UnaryRepeatParamsType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::ascendc::BaseGlobalTensorType>([&](auto t) {
      printer << ::mlir::ascendc::BaseGlobalTensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BaseLocalTensorType>([&](auto t) {
      printer << ::mlir::ascendc::BaseLocalTensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TBufType>([&](auto t) {
      printer << ::mlir::ascendc::TBufType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyPadExtParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyPadExtParamsType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::FixpipeParamsType>([&](auto t) {
      printer << ::mlir::ascendc::FixpipeParamsType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::GlobalTensorType>([&](auto t) {
      printer << ::mlir::ascendc::GlobalTensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadData2DMxParamsType>([&](auto t) {
      printer << ::mlir::ascendc::LoadData2DMxParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadData3DParamsV2Type>([&](auto t) {
      printer << ::mlir::ascendc::LoadData3DParamsV2Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LocalTensorType>([&](auto t) {
      printer << ::mlir::ascendc::LocalTensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MaskType>([&](auto t) {
      printer << ::mlir::ascendc::MaskType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MatmulType>([&](auto t) {
      printer << ::mlir::ascendc::MatmulType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::QueBindType>([&](auto t) {
      printer << ::mlir::ascendc::QueBindType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::QueueType>([&](auto t) {
      printer << ::mlir::ascendc::QueueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TBufPoolType>([&](auto t) {
      printer << ::mlir::ascendc::TBufPoolType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippChannelPaddingParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippChannelPaddingParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippColorSpaceConvParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippColorSpaceConvParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippDataTypeConvParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippDataTypeConvParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippInputFormatType>([&](auto t) {
      printer << ::mlir::ascendc::AippInputFormatType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippPaddingParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippPaddingParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippSingleLineParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippSingleLineParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AippSwapParamsType>([&](auto t) {
      printer << ::mlir::ascendc::AippSwapParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::AscendQuantConfigType>([&](auto t) {
      printer << ::mlir::ascendc::AscendQuantConfigType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BatchModeType>([&](auto t) {
      printer << ::mlir::ascendc::BatchModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BatchOutModeType>([&](auto t) {
      printer << ::mlir::ascendc::BatchOutModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BinaryRepeatParamsType>([&](auto t) {
      printer << ::mlir::ascendc::BinaryRepeatParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BlockModeType>([&](auto t) {
      printer << ::mlir::ascendc::BlockModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BrcbRepeatParamsType>([&](auto t) {
      printer << ::mlir::ascendc::BrcbRepeatParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::CopyRepeatParamsType>([&](auto t) {
      printer << ::mlir::ascendc::CopyRepeatParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyCO12DstParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyCO12DstParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyEnhancedParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyEnhancedParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyExtParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyExtParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyPadParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyPadParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataCopyParamsType>([&](auto t) {
      printer << ::mlir::ascendc::DataCopyParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DataFormatType>([&](auto t) {
      printer << ::mlir::ascendc::DataFormatType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::DeqScaleType>([&](auto t) {
      printer << ::mlir::ascendc::DeqScaleType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::FixpipeConfigType>([&](auto t) {
      printer << ::mlir::ascendc::FixpipeConfigType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::FmatrixModeType>([&](auto t) {
      printer << ::mlir::ascendc::FmatrixModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::GatherMaskParamsType>([&](auto t) {
      printer << ::mlir::ascendc::GatherMaskParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::GatherRepeatParamsType>([&](auto t) {
      printer << ::mlir::ascendc::GatherRepeatParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::IsResetLoad3dConfigType>([&](auto t) {
      printer << ::mlir::ascendc::IsResetLoad3dConfigType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::IterateModeType>([&](auto t) {
      printer << ::mlir::ascendc::IterateModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::IterateOrderType>([&](auto t) {
      printer << ::mlir::ascendc::IterateOrderType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::KfcServerType>([&](auto t) {
      printer << ::mlir::ascendc::KfcServerType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadData2DParamsType>([&](auto t) {
      printer << ::mlir::ascendc::LoadData2DParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadData2dTransposeParamsType>([&](auto t) {
      printer << ::mlir::ascendc::LoadData2dTransposeParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadDataRepeatParamType>([&](auto t) {
      printer << ::mlir::ascendc::LoadDataRepeatParamType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::LoadImageToLocalParamsType>([&](auto t) {
      printer << ::mlir::ascendc::LoadImageToLocalParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MaskModeType>([&](auto t) {
      printer << ::mlir::ascendc::MaskModeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MatmulApiStaticTilingType>([&](auto t) {
      printer << ::mlir::ascendc::MatmulApiStaticTilingType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MatmulConfigType>([&](auto t) {
      printer << ::mlir::ascendc::MatmulConfigType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MatrixOffsetType>([&](auto t) {
      printer << ::mlir::ascendc::MatrixOffsetType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::MmadParamsType>([&](auto t) {
      printer << ::mlir::ascendc::MmadParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::Nd2NzParamsType>([&](auto t) {
      printer << ::mlir::ascendc::Nd2NzParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::Nz2NdParamsType>([&](auto t) {
      printer << ::mlir::ascendc::Nz2NdParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::Nz2NdParamsFullType>([&](auto t) {
      printer << ::mlir::ascendc::Nz2NdParamsFullType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::pad_tType>([&](auto t) {
      printer << ::mlir::ascendc::pad_tType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::QuantModesType>([&](auto t) {
      printer << ::mlir::ascendc::QuantModesType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::QuantParamsType>([&](auto t) {
      printer << ::mlir::ascendc::QuantParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::ReduceOrderType>([&](auto t) {
      printer << ::mlir::ascendc::ReduceOrderType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::RmsNormTilingType>([&](auto t) {
      printer << ::mlir::ascendc::RmsNormTilingType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::ScheduleTypeType>([&](auto t) {
      printer << ::mlir::ascendc::ScheduleTypeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::ShapeInfoType>([&](auto t) {
      printer << ::mlir::ascendc::ShapeInfoType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::SliceInfoType>([&](auto t) {
      printer << ::mlir::ascendc::SliceInfoType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::SoftMaxShapeInfoType>([&](auto t) {
      printer << ::mlir::ascendc::SoftMaxShapeInfoType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::SoftMaxTilingType>([&](auto t) {
      printer << ::mlir::ascendc::SoftMaxTilingType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::BufAddrType>([&](auto t) {
      printer << ::mlir::ascendc::BufAddrType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TCubeTilingType>([&](auto t) {
      printer << ::mlir::ascendc::TCubeTilingType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::PipeType>([&](auto t) {
      printer << ::mlir::ascendc::PipeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TransDataTo5HDParamsType>([&](auto t) {
      printer << ::mlir::ascendc::TransDataTo5HDParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TransposeParamsExtType>([&](auto t) {
      printer << ::mlir::ascendc::TransposeParamsExtType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::TransposeTypeType>([&](auto t) {
      printer << ::mlir::ascendc::TransposeTypeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::ascendc::UnaryRepeatParamsType>([&](auto t) {
      printer << ::mlir::ascendc::UnaryRepeatParamsType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::ascendc {

namespace detail {

struct BaseGlobalTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type>;
  BaseGlobalTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static BaseGlobalTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<BaseGlobalTensorTypeStorage>()) BaseGlobalTensorTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
::llvm::ArrayRef<int64_t> BaseGlobalTensorType::getShape() const {
  return getImpl()->shape;
}

Type BaseGlobalTensorType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BaseGlobalTensorType)
namespace mlir::ascendc {

namespace detail {

struct BaseLocalTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type>;
  BaseLocalTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static BaseLocalTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<BaseLocalTensorTypeStorage>()) BaseLocalTensorTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
::llvm::ArrayRef<int64_t> BaseLocalTensorType::getShape() const {
  return getImpl()->shape;
}

Type BaseLocalTensorType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BaseLocalTensorType)
namespace mlir::ascendc {

namespace detail {

struct TBufTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TPositionAttr>;
  TBufTypeStorage(TPositionAttr tPositionAttr) : tPositionAttr(std::move(tPositionAttr)) {}

  KeyTy getAsKey() const {
    return KeyTy(tPositionAttr);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tPositionAttr == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TBufTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tPositionAttr = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TBufTypeStorage>()) TBufTypeStorage(std::move(tPositionAttr));
  }

  TPositionAttr tPositionAttr;
};

} // namespace detail
TBufType TBufType::get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr) {
  return Base::get(context, std::move(tPositionAttr));
}

TBufType TBufType::get(::mlir::MLIRContext *context, TPosition position) {
  return Base::get(context, TPositionAttr::get(context, position));
}

::mlir::Type TBufType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TPositionAttr> _result_tPositionAttr;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_tPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_tPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'tPositionAttr'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tPositionAttr));
  return TBufType::get(odsParser.getContext(),
      TPositionAttr((*_result_tPositionAttr)));
}

void TBufType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyTPosition(odsPrinter,
    getTPositionAttr());
  odsPrinter << ">";
}

TPositionAttr TBufType::getTPositionAttr() const {
  return getImpl()->tPositionAttr;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TBufType)
namespace mlir::ascendc {

namespace detail {

struct DataCopyPadExtParamsTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  DataCopyPadExtParamsTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DataCopyPadExtParamsTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DataCopyPadExtParamsTypeStorage>()) DataCopyPadExtParamsTypeStorage(std::move(elementType));
  }

  Type elementType;
};

} // namespace detail
DataCopyPadExtParamsType DataCopyPadExtParamsType::get(::mlir::MLIRContext *context, Type elementType) {
  return Base::get(context, std::move(elementType));
}

::mlir::Type DataCopyPadExtParamsType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_DataCopyPadExtParams parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return DataCopyPadExtParamsType::get(odsParser.getContext(),
      Type((*_result_elementType)));
}

void DataCopyPadExtParamsType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

Type DataCopyPadExtParamsType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyPadExtParamsType)
namespace mlir::ascendc {

namespace detail {

struct FixpipeParamsTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  FixpipeParamsTypeStorage(Type type) : type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FixpipeParamsTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FixpipeParamsTypeStorage>()) FixpipeParamsTypeStorage(std::move(type));
  }

  Type type;
};

} // namespace detail
FixpipeParamsType FixpipeParamsType::get(::mlir::MLIRContext *context, Type type) {
  return Base::get(context, std::move(type));
}

::mlir::Type FixpipeParamsType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_FixpipeParams parameter 'type' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_type));
  return FixpipeParamsType::get(odsParser.getContext(),
      Type((*_result_type)));
}

void FixpipeParamsType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getType());
  odsPrinter << ">";
}

Type FixpipeParamsType::getType() const {
  return getImpl()->type;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::FixpipeParamsType)
namespace mlir::ascendc {

namespace detail {

struct GlobalTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type>;
  GlobalTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static GlobalTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<GlobalTensorTypeStorage>()) GlobalTensorTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
::llvm::ArrayRef<int64_t> GlobalTensorType::getShape() const {
  return getImpl()->shape;
}

Type GlobalTensorType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::GlobalTensorType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2DMxParamsType)
namespace mlir::ascendc {

namespace detail {

struct LoadData3DParamsV2TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  LoadData3DParamsV2TypeStorage(Type type) : type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LoadData3DParamsV2TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LoadData3DParamsV2TypeStorage>()) LoadData3DParamsV2TypeStorage(std::move(type));
  }

  Type type;
};

} // namespace detail
LoadData3DParamsV2Type LoadData3DParamsV2Type::get(::mlir::MLIRContext *context, Type type) {
  return Base::get(context, std::move(type));
}

::mlir::Type LoadData3DParamsV2Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_LoadData3DParamsV2 parameter 'type' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_type));
  return LoadData3DParamsV2Type::get(odsParser.getContext(),
      Type((*_result_type)));
}

void LoadData3DParamsV2Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getType());
  odsPrinter << ">";
}

Type LoadData3DParamsV2Type::getType() const {
  return getImpl()->type;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData3DParamsV2Type)
namespace mlir::ascendc {

namespace detail {

struct LocalTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type>;
  LocalTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType) : shape(std::move(shape)), elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LocalTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<LocalTensorTypeStorage>()) LocalTensorTypeStorage(std::move(shape), std::move(elementType));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace detail
::llvm::ArrayRef<int64_t> LocalTensorType::getShape() const {
  return getImpl()->shape;
}

Type LocalTensorType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LocalTensorType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MaskType)
namespace mlir::ascendc {

namespace detail {

struct MatmulTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TPositionAttr, CubeFormatAttr, Type, bool, LayoutModeAttr, TPositionAttr, CubeFormatAttr, Type, bool, LayoutModeAttr, TPositionAttr, CubeFormatAttr, Type, bool, LayoutModeAttr, TPositionAttr, CubeFormatAttr, Type, MatmulConfigAttr>;
  MatmulTypeStorage(TPositionAttr srcAPositionAttr, CubeFormatAttr cubeFormatAAttr, Type typeA, bool isTransA, LayoutModeAttr layoutModeAAttr, TPositionAttr srcBPositionAttr, CubeFormatAttr cubeFormatBAttr, Type typeB, bool isTransB, LayoutModeAttr layoutModeBAttr, TPositionAttr srcCPositionAttr, CubeFormatAttr cubeFormatCAttr, Type typeC, bool isTransC, LayoutModeAttr layoutModeCAttr, TPositionAttr srcBiasPositionAttr, CubeFormatAttr cubeFormatBiasAttr, Type typeBias, MatmulConfigAttr matmulConfig) : srcAPositionAttr(std::move(srcAPositionAttr)), cubeFormatAAttr(std::move(cubeFormatAAttr)), typeA(std::move(typeA)), isTransA(std::move(isTransA)), layoutModeAAttr(std::move(layoutModeAAttr)), srcBPositionAttr(std::move(srcBPositionAttr)), cubeFormatBAttr(std::move(cubeFormatBAttr)), typeB(std::move(typeB)), isTransB(std::move(isTransB)), layoutModeBAttr(std::move(layoutModeBAttr)), srcCPositionAttr(std::move(srcCPositionAttr)), cubeFormatCAttr(std::move(cubeFormatCAttr)), typeC(std::move(typeC)), isTransC(std::move(isTransC)), layoutModeCAttr(std::move(layoutModeCAttr)), srcBiasPositionAttr(std::move(srcBiasPositionAttr)), cubeFormatBiasAttr(std::move(cubeFormatBiasAttr)), typeBias(std::move(typeBias)), matmulConfig(std::move(matmulConfig)) {}

  KeyTy getAsKey() const {
    return KeyTy(srcAPositionAttr, cubeFormatAAttr, typeA, isTransA, layoutModeAAttr, srcBPositionAttr, cubeFormatBAttr, typeB, isTransB, layoutModeBAttr, srcCPositionAttr, cubeFormatCAttr, typeC, isTransC, layoutModeCAttr, srcBiasPositionAttr, cubeFormatBiasAttr, typeBias, matmulConfig);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (srcAPositionAttr == std::get<0>(tblgenKey)) && (cubeFormatAAttr == std::get<1>(tblgenKey)) && (typeA == std::get<2>(tblgenKey)) && (isTransA == std::get<3>(tblgenKey)) && (layoutModeAAttr == std::get<4>(tblgenKey)) && (srcBPositionAttr == std::get<5>(tblgenKey)) && (cubeFormatBAttr == std::get<6>(tblgenKey)) && (typeB == std::get<7>(tblgenKey)) && (isTransB == std::get<8>(tblgenKey)) && (layoutModeBAttr == std::get<9>(tblgenKey)) && (srcCPositionAttr == std::get<10>(tblgenKey)) && (cubeFormatCAttr == std::get<11>(tblgenKey)) && (typeC == std::get<12>(tblgenKey)) && (isTransC == std::get<13>(tblgenKey)) && (layoutModeCAttr == std::get<14>(tblgenKey)) && (srcBiasPositionAttr == std::get<15>(tblgenKey)) && (cubeFormatBiasAttr == std::get<16>(tblgenKey)) && (typeBias == std::get<17>(tblgenKey)) && (matmulConfig == std::get<18>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey), std::get<8>(tblgenKey), std::get<9>(tblgenKey), std::get<10>(tblgenKey), std::get<11>(tblgenKey), std::get<12>(tblgenKey), std::get<13>(tblgenKey), std::get<14>(tblgenKey), std::get<15>(tblgenKey), std::get<16>(tblgenKey), std::get<17>(tblgenKey), std::get<18>(tblgenKey));
  }

  static MatmulTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto srcAPositionAttr = std::move(std::get<0>(tblgenKey));
    auto cubeFormatAAttr = std::move(std::get<1>(tblgenKey));
    auto typeA = std::move(std::get<2>(tblgenKey));
    auto isTransA = std::move(std::get<3>(tblgenKey));
    auto layoutModeAAttr = std::move(std::get<4>(tblgenKey));
    auto srcBPositionAttr = std::move(std::get<5>(tblgenKey));
    auto cubeFormatBAttr = std::move(std::get<6>(tblgenKey));
    auto typeB = std::move(std::get<7>(tblgenKey));
    auto isTransB = std::move(std::get<8>(tblgenKey));
    auto layoutModeBAttr = std::move(std::get<9>(tblgenKey));
    auto srcCPositionAttr = std::move(std::get<10>(tblgenKey));
    auto cubeFormatCAttr = std::move(std::get<11>(tblgenKey));
    auto typeC = std::move(std::get<12>(tblgenKey));
    auto isTransC = std::move(std::get<13>(tblgenKey));
    auto layoutModeCAttr = std::move(std::get<14>(tblgenKey));
    auto srcBiasPositionAttr = std::move(std::get<15>(tblgenKey));
    auto cubeFormatBiasAttr = std::move(std::get<16>(tblgenKey));
    auto typeBias = std::move(std::get<17>(tblgenKey));
    auto matmulConfig = std::move(std::get<18>(tblgenKey));
    return new (allocator.allocate<MatmulTypeStorage>()) MatmulTypeStorage(std::move(srcAPositionAttr), std::move(cubeFormatAAttr), std::move(typeA), std::move(isTransA), std::move(layoutModeAAttr), std::move(srcBPositionAttr), std::move(cubeFormatBAttr), std::move(typeB), std::move(isTransB), std::move(layoutModeBAttr), std::move(srcCPositionAttr), std::move(cubeFormatCAttr), std::move(typeC), std::move(isTransC), std::move(layoutModeCAttr), std::move(srcBiasPositionAttr), std::move(cubeFormatBiasAttr), std::move(typeBias), std::move(matmulConfig));
  }

  TPositionAttr srcAPositionAttr;
  CubeFormatAttr cubeFormatAAttr;
  Type typeA;
  bool isTransA;
  LayoutModeAttr layoutModeAAttr;
  TPositionAttr srcBPositionAttr;
  CubeFormatAttr cubeFormatBAttr;
  Type typeB;
  bool isTransB;
  LayoutModeAttr layoutModeBAttr;
  TPositionAttr srcCPositionAttr;
  CubeFormatAttr cubeFormatCAttr;
  Type typeC;
  bool isTransC;
  LayoutModeAttr layoutModeCAttr;
  TPositionAttr srcBiasPositionAttr;
  CubeFormatAttr cubeFormatBiasAttr;
  Type typeBias;
  MatmulConfigAttr matmulConfig;
};

} // namespace detail
MatmulType MatmulType::get(::mlir::MLIRContext *context, TPositionAttr srcAPositionAttr, CubeFormatAttr cubeFormatAAttr, Type typeA, bool isTransA, LayoutModeAttr layoutModeAAttr, TPositionAttr srcBPositionAttr, CubeFormatAttr cubeFormatBAttr, Type typeB, bool isTransB, LayoutModeAttr layoutModeBAttr, TPositionAttr srcCPositionAttr, CubeFormatAttr cubeFormatCAttr, Type typeC, bool isTransC, LayoutModeAttr layoutModeCAttr, TPositionAttr srcBiasPositionAttr, CubeFormatAttr cubeFormatBiasAttr, Type typeBias, MatmulConfigAttr matmulConfig) {
  return Base::get(context, std::move(srcAPositionAttr), std::move(cubeFormatAAttr), std::move(typeA), std::move(isTransA), std::move(layoutModeAAttr), std::move(srcBPositionAttr), std::move(cubeFormatBAttr), std::move(typeB), std::move(isTransB), std::move(layoutModeBAttr), std::move(srcCPositionAttr), std::move(cubeFormatCAttr), std::move(typeC), std::move(isTransC), std::move(layoutModeCAttr), std::move(srcBiasPositionAttr), std::move(cubeFormatBiasAttr), std::move(typeBias), std::move(matmulConfig));
}

MatmulType MatmulType::get(::mlir::MLIRContext *context, TPosition srcAPosition, CubeFormat cubeFormatA, Type typeA, bool isTransA, LayoutMode layoutModeA, TPosition srcBPosition, CubeFormat cubeFormatB, Type typeB, bool isTransB, LayoutMode layoutModeB, TPosition srcCPosition, CubeFormat cubeFormatC, Type typeC, bool isTransC, LayoutMode layoutModeC, TPosition srcBiasPosition, CubeFormat cubeFormatBias, Type typeBias, MatmulConfigAttr matmulConfig) {
  auto srcA = TPositionAttr::get(context, srcAPosition);
  auto formatA = CubeFormatAttr::get(context, cubeFormatA);
  auto layoutA = LayoutModeAttr::get(context, layoutModeA);
  auto srcB = TPositionAttr::get(context, srcBPosition);
  auto formatB = CubeFormatAttr::get(context, cubeFormatB);
  auto layoutB = LayoutModeAttr::get(context, layoutModeB);
  auto srcC = TPositionAttr::get(context, srcCPosition);
  auto formatC = CubeFormatAttr::get(context, cubeFormatC);
  auto layoutC = LayoutModeAttr::get(context, layoutModeC);
  auto srcBias = TPositionAttr::get(context, srcBiasPosition);
  auto formatBias = CubeFormatAttr::get(context, cubeFormatBias);
  return Base::get(context, srcA, formatA, typeA, isTransA, layoutA,
                srcB, formatB, typeB, isTransB, layoutB,
                srcC, formatC, typeC, isTransC, layoutC,
                srcBias, formatBias, typeBias, matmulConfig);
}

::mlir::Type MatmulType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TPositionAttr> _result_srcAPositionAttr;
  ::mlir::FailureOr<CubeFormatAttr> _result_cubeFormatAAttr;
  ::mlir::FailureOr<Type> _result_typeA;
  ::mlir::FailureOr<bool> _result_isTransA;
  ::mlir::FailureOr<LayoutModeAttr> _result_layoutModeAAttr;
  ::mlir::FailureOr<TPositionAttr> _result_srcBPositionAttr;
  ::mlir::FailureOr<CubeFormatAttr> _result_cubeFormatBAttr;
  ::mlir::FailureOr<Type> _result_typeB;
  ::mlir::FailureOr<bool> _result_isTransB;
  ::mlir::FailureOr<LayoutModeAttr> _result_layoutModeBAttr;
  ::mlir::FailureOr<TPositionAttr> _result_srcCPositionAttr;
  ::mlir::FailureOr<CubeFormatAttr> _result_cubeFormatCAttr;
  ::mlir::FailureOr<Type> _result_typeC;
  ::mlir::FailureOr<bool> _result_isTransC;
  ::mlir::FailureOr<LayoutModeAttr> _result_layoutModeCAttr;
  ::mlir::FailureOr<TPositionAttr> _result_srcBiasPositionAttr;
  ::mlir::FailureOr<CubeFormatAttr> _result_cubeFormatBiasAttr;
  ::mlir::FailureOr<Type> _result_typeBias;
  ::mlir::FailureOr<MatmulConfigAttr> _result_matmulConfig;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_srcAPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_srcAPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'srcAPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'cubeFormatAAttr'
  _result_cubeFormatAAttr = ::mlir::FieldParser<CubeFormatAttr>::parse(odsParser);
  if (::mlir::failed(_result_cubeFormatAAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'cubeFormatAAttr' which is to be a `CubeFormatAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'typeA'
  _result_typeA = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_typeA)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'typeA' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'isTransA'
  _result_isTransA = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_isTransA)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'isTransA' which is to be a `bool`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'layoutModeAAttr'
  _result_layoutModeAAttr = ::mlir::FieldParser<LayoutModeAttr>::parse(odsParser);
  if (::mlir::failed(_result_layoutModeAAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'layoutModeAAttr' which is to be a `LayoutModeAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_srcBPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_srcBPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'srcBPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'cubeFormatBAttr'
  _result_cubeFormatBAttr = ::mlir::FieldParser<CubeFormatAttr>::parse(odsParser);
  if (::mlir::failed(_result_cubeFormatBAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'cubeFormatBAttr' which is to be a `CubeFormatAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'typeB'
  _result_typeB = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_typeB)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'typeB' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'isTransB'
  _result_isTransB = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_isTransB)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'isTransB' which is to be a `bool`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'layoutModeBAttr'
  _result_layoutModeBAttr = ::mlir::FieldParser<LayoutModeAttr>::parse(odsParser);
  if (::mlir::failed(_result_layoutModeBAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'layoutModeBAttr' which is to be a `LayoutModeAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_srcCPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_srcCPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'srcCPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'cubeFormatCAttr'
  _result_cubeFormatCAttr = ::mlir::FieldParser<CubeFormatAttr>::parse(odsParser);
  if (::mlir::failed(_result_cubeFormatCAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'cubeFormatCAttr' which is to be a `CubeFormatAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'typeC'
  _result_typeC = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_typeC)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'typeC' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'isTransC'
  _result_isTransC = ::mlir::FieldParser<bool>::parse(odsParser);
  if (::mlir::failed(_result_isTransC)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'isTransC' which is to be a `bool`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'layoutModeCAttr'
  _result_layoutModeCAttr = ::mlir::FieldParser<LayoutModeAttr>::parse(odsParser);
  if (::mlir::failed(_result_layoutModeCAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'layoutModeCAttr' which is to be a `LayoutModeAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_srcBiasPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_srcBiasPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'srcBiasPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'cubeFormatBiasAttr'
  _result_cubeFormatBiasAttr = ::mlir::FieldParser<CubeFormatAttr>::parse(odsParser);
  if (::mlir::failed(_result_cubeFormatBiasAttr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'cubeFormatBiasAttr' which is to be a `CubeFormatAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'typeBias'
  _result_typeBias = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_typeBias)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'typeBias' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'matmulConfig'
  _result_matmulConfig = ::mlir::FieldParser<MatmulConfigAttr>::parse(odsParser);
  if (::mlir::failed(_result_matmulConfig)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Matmul parameter 'matmulConfig' which is to be a `MatmulConfigAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_srcAPositionAttr));
  assert(::mlir::succeeded(_result_cubeFormatAAttr));
  assert(::mlir::succeeded(_result_typeA));
  assert(::mlir::succeeded(_result_isTransA));
  assert(::mlir::succeeded(_result_layoutModeAAttr));
  assert(::mlir::succeeded(_result_srcBPositionAttr));
  assert(::mlir::succeeded(_result_cubeFormatBAttr));
  assert(::mlir::succeeded(_result_typeB));
  assert(::mlir::succeeded(_result_isTransB));
  assert(::mlir::succeeded(_result_layoutModeBAttr));
  assert(::mlir::succeeded(_result_srcCPositionAttr));
  assert(::mlir::succeeded(_result_cubeFormatCAttr));
  assert(::mlir::succeeded(_result_typeC));
  assert(::mlir::succeeded(_result_isTransC));
  assert(::mlir::succeeded(_result_layoutModeCAttr));
  assert(::mlir::succeeded(_result_srcBiasPositionAttr));
  assert(::mlir::succeeded(_result_cubeFormatBiasAttr));
  assert(::mlir::succeeded(_result_typeBias));
  assert(::mlir::succeeded(_result_matmulConfig));
  return MatmulType::get(odsParser.getContext(),
      TPositionAttr((*_result_srcAPositionAttr)),
      CubeFormatAttr((*_result_cubeFormatAAttr)),
      Type((*_result_typeA)),
      bool((*_result_isTransA)),
      LayoutModeAttr((*_result_layoutModeAAttr)),
      TPositionAttr((*_result_srcBPositionAttr)),
      CubeFormatAttr((*_result_cubeFormatBAttr)),
      Type((*_result_typeB)),
      bool((*_result_isTransB)),
      LayoutModeAttr((*_result_layoutModeBAttr)),
      TPositionAttr((*_result_srcCPositionAttr)),
      CubeFormatAttr((*_result_cubeFormatCAttr)),
      Type((*_result_typeC)),
      bool((*_result_isTransC)),
      LayoutModeAttr((*_result_layoutModeCAttr)),
      TPositionAttr((*_result_srcBiasPositionAttr)),
      CubeFormatAttr((*_result_cubeFormatBiasAttr)),
      Type((*_result_typeBias)),
      MatmulConfigAttr((*_result_matmulConfig)));
}

void MatmulType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyTPosition(odsPrinter,
    getSrcAPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getCubeFormatAAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTypeA());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getIsTransA());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLayoutModeAAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  printPrettyTPosition(odsPrinter,
    getSrcBPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getCubeFormatBAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTypeB());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getIsTransB());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLayoutModeBAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  printPrettyTPosition(odsPrinter,
    getSrcCPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getCubeFormatCAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTypeC());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getIsTransC());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLayoutModeCAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  printPrettyTPosition(odsPrinter,
    getSrcBiasPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getCubeFormatBiasAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getTypeBias());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getMatmulConfig());
  odsPrinter << ">";
}

TPositionAttr MatmulType::getSrcAPositionAttr() const {
  return getImpl()->srcAPositionAttr;
}

CubeFormatAttr MatmulType::getCubeFormatAAttr() const {
  return getImpl()->cubeFormatAAttr;
}

Type MatmulType::getTypeA() const {
  return getImpl()->typeA;
}

bool MatmulType::getIsTransA() const {
  return getImpl()->isTransA;
}

LayoutModeAttr MatmulType::getLayoutModeAAttr() const {
  return getImpl()->layoutModeAAttr;
}

TPositionAttr MatmulType::getSrcBPositionAttr() const {
  return getImpl()->srcBPositionAttr;
}

CubeFormatAttr MatmulType::getCubeFormatBAttr() const {
  return getImpl()->cubeFormatBAttr;
}

Type MatmulType::getTypeB() const {
  return getImpl()->typeB;
}

bool MatmulType::getIsTransB() const {
  return getImpl()->isTransB;
}

LayoutModeAttr MatmulType::getLayoutModeBAttr() const {
  return getImpl()->layoutModeBAttr;
}

TPositionAttr MatmulType::getSrcCPositionAttr() const {
  return getImpl()->srcCPositionAttr;
}

CubeFormatAttr MatmulType::getCubeFormatCAttr() const {
  return getImpl()->cubeFormatCAttr;
}

Type MatmulType::getTypeC() const {
  return getImpl()->typeC;
}

bool MatmulType::getIsTransC() const {
  return getImpl()->isTransC;
}

LayoutModeAttr MatmulType::getLayoutModeCAttr() const {
  return getImpl()->layoutModeCAttr;
}

TPositionAttr MatmulType::getSrcBiasPositionAttr() const {
  return getImpl()->srcBiasPositionAttr;
}

CubeFormatAttr MatmulType::getCubeFormatBiasAttr() const {
  return getImpl()->cubeFormatBiasAttr;
}

Type MatmulType::getTypeBias() const {
  return getImpl()->typeBias;
}

MatmulConfigAttr MatmulType::getMatmulConfig() const {
  return getImpl()->matmulConfig;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulType)
namespace mlir::ascendc {

namespace detail {

struct QueBindTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TPositionAttr, TPositionAttr, int64_t>;
  QueBindTypeStorage(TPositionAttr srcPositionAttr, TPositionAttr dstPositionAttr, int64_t depth) : srcPositionAttr(std::move(srcPositionAttr)), dstPositionAttr(std::move(dstPositionAttr)), depth(std::move(depth)) {}

  KeyTy getAsKey() const {
    return KeyTy(srcPositionAttr, dstPositionAttr, depth);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (srcPositionAttr == std::get<0>(tblgenKey)) && (dstPositionAttr == std::get<1>(tblgenKey)) && (depth == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static QueBindTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto srcPositionAttr = std::move(std::get<0>(tblgenKey));
    auto dstPositionAttr = std::move(std::get<1>(tblgenKey));
    auto depth = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<QueBindTypeStorage>()) QueBindTypeStorage(std::move(srcPositionAttr), std::move(dstPositionAttr), std::move(depth));
  }

  TPositionAttr srcPositionAttr;
  TPositionAttr dstPositionAttr;
  int64_t depth;
};

} // namespace detail
QueBindType QueBindType::get(::mlir::MLIRContext *context, TPositionAttr srcPositionAttr, TPositionAttr dstPositionAttr, int64_t depth) {
  return Base::get(context, std::move(srcPositionAttr), std::move(dstPositionAttr), std::move(depth));
}

QueBindType QueBindType::get(::mlir::MLIRContext *context, TPosition srcPosition, TPosition dstPosition, int64_t depth) {
  auto src = TPositionAttr::get(context, srcPosition);
  auto dst = TPositionAttr::get(context, dstPosition);
  return Base::get(context, src, dst, depth);
}

::mlir::Type QueBindType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TPositionAttr> _result_srcPositionAttr;
  ::mlir::FailureOr<TPositionAttr> _result_dstPositionAttr;
  ::mlir::FailureOr<int64_t> _result_depth;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_srcPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_srcPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'srcPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_dstPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_dstPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'dstPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'depth'
  _result_depth = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_depth)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_QueBind parameter 'depth' which is to be a `int64_t`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_srcPositionAttr));
  assert(::mlir::succeeded(_result_dstPositionAttr));
  assert(::mlir::succeeded(_result_depth));
  return QueBindType::get(odsParser.getContext(),
      TPositionAttr((*_result_srcPositionAttr)),
      TPositionAttr((*_result_dstPositionAttr)),
      int64_t((*_result_depth)));
}

void QueBindType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyTPosition(odsPrinter,
    getSrcPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  printPrettyTPosition(odsPrinter,
    getDstPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getDepth());
  odsPrinter << ">";
}

TPositionAttr QueBindType::getSrcPositionAttr() const {
  return getImpl()->srcPositionAttr;
}

TPositionAttr QueBindType::getDstPositionAttr() const {
  return getImpl()->dstPositionAttr;
}

int64_t QueBindType::getDepth() const {
  return getImpl()->depth;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::QueBindType)
namespace mlir::ascendc {

namespace detail {

struct QueueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TPositionAttr, int64_t>;
  QueueTypeStorage(TPositionAttr tPositionAttr, int64_t depth) : tPositionAttr(std::move(tPositionAttr)), depth(std::move(depth)) {}

  KeyTy getAsKey() const {
    return KeyTy(tPositionAttr, depth);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tPositionAttr == std::get<0>(tblgenKey)) && (depth == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static QueueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tPositionAttr = std::move(std::get<0>(tblgenKey));
    auto depth = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<QueueTypeStorage>()) QueueTypeStorage(std::move(tPositionAttr), std::move(depth));
  }

  TPositionAttr tPositionAttr;
  int64_t depth;
};

} // namespace detail
QueueType QueueType::get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr, int64_t depth) {
  return Base::get(context, std::move(tPositionAttr), std::move(depth));
}

QueueType QueueType::get(::mlir::MLIRContext *context, TPosition position, int64_t depth) {
  return Base::get(context, TPositionAttr::get(context, position), depth);
}

::mlir::Type QueueType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TPositionAttr> _result_tPositionAttr;
  ::mlir::FailureOr<int64_t> _result_depth;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_tPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_tPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'tPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'depth'
  _result_depth = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_depth)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_Queue parameter 'depth' which is to be a `int64_t`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tPositionAttr));
  assert(::mlir::succeeded(_result_depth));
  return QueueType::get(odsParser.getContext(),
      TPositionAttr((*_result_tPositionAttr)),
      int64_t((*_result_depth)));
}

void QueueType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyTPosition(odsPrinter,
    getTPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getDepth());
  odsPrinter << ">";
}

TPositionAttr QueueType::getTPositionAttr() const {
  return getImpl()->tPositionAttr;
}

int64_t QueueType::getDepth() const {
  return getImpl()->depth;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::QueueType)
namespace mlir::ascendc {

namespace detail {

struct TBufPoolTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TPositionAttr, uint32_t>;
  TBufPoolTypeStorage(TPositionAttr tPositionAttr, uint32_t bufIDSize) : tPositionAttr(std::move(tPositionAttr)), bufIDSize(std::move(bufIDSize)) {}

  KeyTy getAsKey() const {
    return KeyTy(tPositionAttr, bufIDSize);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (tPositionAttr == std::get<0>(tblgenKey)) && (bufIDSize == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TBufPoolTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto tPositionAttr = std::move(std::get<0>(tblgenKey));
    auto bufIDSize = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<TBufPoolTypeStorage>()) TBufPoolTypeStorage(std::move(tPositionAttr), std::move(bufIDSize));
  }

  TPositionAttr tPositionAttr;
  uint32_t bufIDSize;
};

} // namespace detail
TBufPoolType TBufPoolType::get(::mlir::MLIRContext *context, TPositionAttr tPositionAttr, uint32_t bufIDSize) {
  return Base::get(context, std::move(tPositionAttr), std::move(bufIDSize));
}

TBufPoolType TBufPoolType::get(::mlir::MLIRContext *context, TPosition position, uint32_t bufIDSize) {
  return Base::get(context, TPositionAttr::get(context, position), bufIDSize);
}

::mlir::Type TBufPoolType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TPositionAttr> _result_tPositionAttr;
  ::mlir::FailureOr<uint32_t> _result_bufIDSize;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyTPosition(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_tPositionAttr));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_tPositionAttr)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'tPositionAttr'");
      return {};
    }
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'bufIDSize'
  _result_bufIDSize = ::mlir::FieldParser<uint32_t>::parse(odsParser);
  if (::mlir::failed(_result_bufIDSize)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AscendC_TBufPool parameter 'bufIDSize' which is to be a `uint32_t`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_tPositionAttr));
  assert(::mlir::succeeded(_result_bufIDSize));
  return TBufPoolType::get(odsParser.getContext(),
      TPositionAttr((*_result_tPositionAttr)),
      uint32_t((*_result_bufIDSize)));
}

void TBufPoolType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyTPosition(odsPrinter,
    getTPositionAttr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getBufIDSize());
  odsPrinter << ">";
}

TPositionAttr TBufPoolType::getTPositionAttr() const {
  return getImpl()->tPositionAttr;
}

uint32_t TBufPoolType::getBufIDSize() const {
  return getImpl()->bufIDSize;
}


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TBufPoolType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippChannelPaddingParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippColorSpaceConvParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippDataTypeConvParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippInputFormatType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippPaddingParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippSingleLineParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AippSwapParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::AscendQuantConfigType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BatchModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BatchOutModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BinaryRepeatParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BlockModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BrcbRepeatParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::CopyRepeatParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyCO12DstParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyEnhancedParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyExtParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyPadParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataCopyParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DataFormatType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::DeqScaleType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::FixpipeConfigType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::FmatrixModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::GatherMaskParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::GatherRepeatParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::IsResetLoad3dConfigType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::IterateModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::IterateOrderType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::KfcServerType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2DParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadData2dTransposeParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadDataRepeatParamType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::LoadImageToLocalParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MaskModeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulApiStaticTilingType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatmulConfigType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MatrixOffsetType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::MmadParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nd2NzParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nz2NdParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::Nz2NdParamsFullType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::pad_tType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::QuantModesType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::QuantParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::ReduceOrderType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::RmsNormTilingType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::ScheduleTypeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::ShapeInfoType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::SliceInfoType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::SoftMaxShapeInfoType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::SoftMaxTilingType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::BufAddrType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TCubeTilingType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::PipeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransDataTo5HDParamsType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransposeParamsExtType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::TransposeTypeType)
namespace mlir::ascendc {


} // namespace mlir::ascendc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ascendc::UnaryRepeatParamsType)
namespace mlir::ascendc {


/// Parse a type registered to this dialect.
::mlir::Type AscendCDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void AscendCDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::ascendc

#endif // GET_TYPEDEF_CLASSES

