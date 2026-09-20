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

#ifdef GEN_EMITTER
if (auto concrete = dyn_cast<::mlir::ascendc::AippChannelPaddingParamsType>(type)) {
  os << "AscendC::AippChannelPaddingParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippColorSpaceConvParamsType>(type)) {
  os << "AscendC::AippColorSpaceConvParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippDataTypeConvParamsType>(type)) {
  os << "AscendC::AippDataTypeConvParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippInputFormatType>(type)) {
  os << "AscendC::AippInputFormat";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippPaddingParamsType>(type)) {
  os << "AscendC::AippPaddingParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippParamsType>(type)) {
  os << "AscendC::AippParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippSingleLineParamsType>(type)) {
  os << "AscendC::AippSingleLineParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AippSwapParamsType>(type)) {
  os << "AscendC::AippSwapParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::AscendQuantConfigType>(type)) {
  os << "AscendC::AscendQuantConfig";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BatchModeType>(type)) {
  os << "BatchMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BatchOutModeType>(type)) {
  os << "BatchOutMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BinaryRepeatParamsType>(type)) {
  os << "AscendC::BinaryRepeatParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BlockModeType>(type)) {
  os << "AscendC::BlockMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BrcbRepeatParamsType>(type)) {
  os << "AscendC::BrcbRepeatParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::CopyRepeatParamsType>(type)) {
  os << "AscendC::CopyRepeatParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataCopyCO12DstParamsType>(type)) {
  os << "AscendC::DataCopyCO12DstParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataCopyEnhancedParamsType>(type)) {
  os << "AscendC::DataCopyEnhancedParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataCopyExtParamsType>(type)) {
  os << "AscendC::DataCopyExtParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataCopyPadParamsType>(type)) {
  os << "AscendC::DataCopyPadParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataCopyParamsType>(type)) {
  os << "AscendC::DataCopyParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DataFormatType>(type)) {
  os << "AscendC::DataFormat";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::DeqScaleType>(type)) {
  os << "AscendC::DeqScale";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::FixpipeConfigType>(type)) {
  os << "AscendC::FixpipeConfig";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::FmatrixModeType>(type)) {
  os << "AscendC::FmatrixMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::GatherMaskParamsType>(type)) {
  os << "AscendC::GatherMaskParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::GatherRepeatParamsType>(type)) {
  os << "AscendC::GatherRepeatParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::IsResetLoad3dConfigType>(type)) {
  os << "AscendC::IsResetLoad3dConfig";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::IterateModeType>(type)) {
  os << "IterateMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::IterateOrderType>(type)) {
  os << "IterateOrder";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::KfcServerType>(type)) {
  os << "AscendC::KfcServer";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadData2DMxParamsType>(type)) {
  os << "AscendC::LoadData2DMxParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadData2DParamsType>(type)) {
  os << "AscendC::LoadData2DParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadData2dTransposeParamsType>(type)) {
  os << "AscendC::LoadData2dTransposeParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadData3DParamsV2Type>(type)) {
  os << "AscendC::LoadData3DParamsV2";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadDataRepeatParamType>(type)) {
  os << "AscendC::LoadDataRepeatParam";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::LoadImageToLocalParamsType>(type)) {
  os << "AscendC::LoadImageToLocalParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::MaskModeType>(type)) {
  os << "AscendC::MaskMode";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::MatmulApiStaticTilingType>(type)) {
  os << "MatmulApiStaticTiling";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::MatmulConfigType>(type)) {
  os << "MatmulConfig";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::MatrixOffsetType>(type)) {
  os << "MatrixOffset";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::MmadParamsType>(type)) {
  os << "AscendC::MmadParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::Nd2NzParamsType>(type)) {
  os << "AscendC::Nd2NzParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::Nz2NdParamsType>(type)) {
  os << "AscendC::Nz2NdParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::Nz2NdParamsFullType>(type)) {
  os << "AscendC::Nz2NdParamsFull";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::pad_tType>(type)) {
  os << "AscendC::pad_t";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::QuantModesType>(type)) {
  os << "QuantMode_t";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::QuantParamsType>(type)) {
  os << "AscendC::QuantParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::ReduceOrderType>(type)) {
  os << "AscendC::ReduceOrder";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::RmsNormTilingType>(type)) {
  os << "RmsNormTiling";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::ScheduleTypeType>(type)) {
  os << "ScheduleType";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::ShapeInfoType>(type)) {
  os << "AscendC::ShapeInfo";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::SliceInfoType>(type)) {
  os << "AscendC::SliceInfo";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::SoftMaxShapeInfoType>(type)) {
  os << "AscendC::SoftMaxShapeInfo";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::SoftMaxTilingType>(type)) {
  os << "SoftMaxTiling";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::BufAddrType>(type)) {
  os << "AscendC::TBuffAddr";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::TCubeTilingType>(type)) {
  os << "TCubeTiling";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::PipeType>(type)) {
  os << "AscendC::TPipe";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::TransDataTo5HDParamsType>(type)) {
  os << "AscendC::TransDataTo5HDParams";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::TransposeParamsExtType>(type)) {
  os << "AscendC::TransposeParamsExt";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::TransposeTypeType>(type)) {
  os << "AscendC::TransposeType";
  return success();
}
if (auto concrete = dyn_cast<::mlir::ascendc::UnaryRepeatParamsType>(type)) {
  os << "AscendC::UnaryRepeatParams";
  return success();
}
#undef GEN_EMITTER
#endif // GEN_EMITTER
