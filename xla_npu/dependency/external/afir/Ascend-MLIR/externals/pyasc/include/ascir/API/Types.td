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

#ifndef API_TYPES_TD
#define API_TYPES_TD

// Ascend C API type
class APIType<string mlirTypeName = ""> {
  // IR mnemonic
  string mnemonic = ?;

  // Ascend C API class name
  string apiName = ?;

  // IR type class name
  string typeName = mlirTypeName;

  // Generate MLIR type definition for this record
  bit genTypedef = !not(!empty(mlirTypeName));

  // Generate MLIR type emitter for this record
  bit genEmitter = !not(!empty(mlirTypeName));
}

def AippChannelPaddingParams : APIType<"AippChannelPaddingParams"> {
  let mnemonic = "aipp_cpadding_params";
  let apiName = "AscendC::AippChannelPaddingParams";
}

def AippColorSpaceConvParams : APIType<"AippColorSpaceConvParams"> {
  let mnemonic = "aipp_csc_params";
  let apiName = "AscendC::AippColorSpaceConvParams";
}

def AippDataTypeConvParams : APIType<"AippDataTypeConvParams"> {
  let mnemonic = "aipp_dtc_params";
  let apiName = "AscendC::AippDataTypeConvParams";
}

def AippInputFormat : APIType<"AippInputFormat"> {
  let mnemonic = "aipp_input_format";
  let apiName = "AscendC::AippInputFormat";
}

def AippPaddingParams : APIType<"AippPaddingParams"> {
  let mnemonic = "aipp_padding_params";
  let apiName = "AscendC::AippPaddingParams";
}

def AippParams : APIType<"AippParams"> {
  let mnemonic = "aipp_params";
  let apiName = "AscendC::AippParams";
}

def AippSingleLineParams : APIType<"AippSingleLineParams"> {
  let mnemonic = "aipp_single_line_params";
  let apiName = "AscendC::AippSingleLineParams";
}

def AippSwapParams : APIType<"AippSwapParams"> {
  let mnemonic = "aipp_swap_params";
  let apiName = "AscendC::AippSwapParams";
}

def AscendQuantConfig : APIType<"AscendQuantConfig"> {
  let mnemonic = "ascend_quant_config";
  let apiName = "AscendC::AscendQuantConfig";
}

def BaseGlobalTensor : APIType {
  let mnemonic = "base_global_tensor";
  let apiName = "AscendC::BaseGlobalTensor";
}

def BaseLocalTensor : APIType {
  let mnemonic = "base_local_tensor";
  let apiName = "AscendC::BaseLocalTensor";
}

def BatchMode : APIType<"BatchMode"> {
  let mnemonic = "batch_mode";
  let apiName = "BatchMode";
}

def BatchOutMode : APIType<"BatchOutMode"> {
  let mnemonic = "batch_out_mode";
  let apiName = "BatchOutMode";
}

def BinaryRepeatParams : APIType<"BinaryRepeatParams"> {
  let mnemonic = "binary_repeat_params";
  let apiName = "AscendC::BinaryRepeatParams";
}

def BlockMode : APIType<"BlockMode"> {
  let mnemonic = "block_mode";
  let apiName = "AscendC::BlockMode";
}

def BrcbRepeatParams : APIType<"BrcbRepeatParams"> {
  let mnemonic = "brcb_repeat_params";
  let apiName = "AscendC::BrcbRepeatParams";
}

def CopyRepeatParams : APIType<"CopyRepeatParams"> {
  let mnemonic = "copy_repeat_params";
  let apiName = "AscendC::CopyRepeatParams";
}

def DataCopyCO12DstParams : APIType<"DataCopyCO12DstParams"> {
  let mnemonic = "data_copy_co12dst_params";
  let apiName = "AscendC::DataCopyCO12DstParams";
}

def DataCopyEnhancedParams : APIType<"DataCopyEnhancedParams"> {
  let mnemonic = "data_copy_enhanced_params";
  let apiName = "AscendC::DataCopyEnhancedParams";
}

def DataCopyParams : APIType<"DataCopyParams"> {
  let mnemonic = "data_copy_params";
  let apiName = "AscendC::DataCopyParams";
}

def DataCopyExtParams : APIType<"DataCopyExtParams"> {
  let mnemonic = "data_copy_ext_params";
  let apiName = "AscendC::DataCopyExtParams";
}

def DataCopyPadExtParams : APIType {
  let mnemonic = "data_copy_pad_ext_params";
  let apiName = "AscendC::DataCopyPadExtParams";
}

def DataCopyPadParams : APIType<"DataCopyPadParams"> {
  let mnemonic = "data_copy_pad_params";
  let apiName = "AscendC::DataCopyPadParams";
}

def DataFormat : APIType<"DataFormat"> {
  let mnemonic = "data_format";
  let apiName = "AscendC::DataFormat";
}

def MatrixOffset : APIType<"MatrixOffset"> {
  let mnemonic = "matrix_offset";
  let apiName = "MatrixOffset";
}

def DeqScale : APIType<"DeqScale"> {
  let mnemonic = "deq_scale";
  let apiName = "AscendC::DeqScale";
}

def FixpipeConfig : APIType<"FixpipeConfig"> {
  let mnemonic = "fixpipe_config";
  let apiName = "AscendC::FixpipeConfig";
}

def FixpipeParams : APIType {
  let mnemonic = "fixpipe_params";
  let apiName = "AscendC::FixpipeParams";
}

def FmatrixMode : APIType<"FmatrixMode"> {
  let mnemonic = "fmatrix_mode";
  let apiName = "AscendC::FmatrixMode";
}

def GatherMaskParams : APIType<"GatherMaskParams"> {
  let mnemonic = "gather_mask_params";
  let apiName = "AscendC::GatherMaskParams";
}

def GatherRepeatParams : APIType<"GatherRepeatParams"> {
  let mnemonic = "gather_repeat_params";
  let apiName = "AscendC::GatherRepeatParams";
}

def GlobalTensor : APIType {
  let mnemonic = "global_tensor";
  let apiName = "AscendC::GlobalTensor";
}

def IsResetLoad3dConfig : APIType<"IsResetLoad3dConfig"> {
  let mnemonic = "is_reset_load_3d_config";
  let apiName = "AscendC::IsResetLoad3dConfig";
}

def IterateMode : APIType<"IterateMode"> {
  let mnemonic = "iterate_mode";
  let apiName = "IterateMode";
}

def IterateOrder : APIType<"IterateOrder"> {
  let mnemonic = "iterate_order";
  let apiName = "IterateOrder";
}

def KfcServer : APIType<"KfcServer"> {
  let mnemonic = "kfc_server";
  let apiName = "AscendC::KfcServer";
}

def LoadData2DParams : APIType<"LoadData2DParams"> {
  let mnemonic = "load_data_2d_params";
  let apiName = "AscendC::LoadData2DParams";
}

def LoadData2dTransposeParams : APIType<"LoadData2dTransposeParams"> {
  let mnemonic = "load_data_2d_transpose_params";
  let apiName = "AscendC::LoadData2dTransposeParams";
}

def LoadData2DMxParams : APIType<"LoadData2DMxParams"> {
  let mnemonic = "load_data_2d_mx_params";
  let apiName = "AscendC::LoadData2DMxParams";
  let genTypedef = false;
}

def LoadData3DParamsV2 : APIType<"LoadData3DParamsV2"> {
  let mnemonic = "load_data_3d_params_v2";
  let apiName = "AscendC::LoadData3DParamsV2";
  let genTypedef = false;
}

def LoadDataRepeatParam : APIType<"LoadDataRepeatParam"> {
  let mnemonic = "load_data_repeat_param";
  let apiName = "AscendC::LoadDataRepeatParam";
}

def LoadImageToLocalParams : APIType<"LoadImageToLocalParams"> {
  let mnemonic = "load_image_to_local_params";
  let apiName = "AscendC::LoadImageToLocalParams";
}

def LocalTensor : APIType {
  let mnemonic = "local_tensor";
  let apiName = "AscendC::LocalTensor";
}

def MaskMode : APIType<"MaskMode"> {
  let mnemonic = "mask_mode";
  let apiName = "AscendC::MaskMode";
}

def Matmul : APIType {
  let mnemonic = "matmul";
  let apiName = "matmul::Matmul";
}

def MatmulApiStaticTiling : APIType<"MatmulApiStaticTiling"> {
  let mnemonic = "matmul_api_static_tiling";
  let apiName = "MatmulApiStaticTiling";
}

def MatmulConfig : APIType<"MatmulConfig"> {
  let mnemonic = "matmul_config";
  let apiName = "MatmulConfig";
}

def MmadParams : APIType<"MmadParams"> {
  let mnemonic = "mmad_params";
  let apiName = "AscendC::MmadParams";
}

def Nd2NzParams : APIType<"Nd2NzParams"> {
  let mnemonic = "nd2nz_params";
  let apiName = "AscendC::Nd2NzParams";
}

def Nz2NdParams : APIType<"Nz2NdParams"> {
  let mnemonic = "nz2nd_params";
  let apiName = "AscendC::Nz2NdParams";
}

def Nz2NdParamsFull : APIType<"Nz2NdParamsFull"> {
  let mnemonic = "nz2nd_params_full";
  let apiName = "AscendC::Nz2NdParamsFull";
}

def PadT : APIType<"pad_t"> {
  let mnemonic = "pad_t";
  let apiName = "AscendC::pad_t";
}

def QuantModes : APIType<"QuantModes"> {
  let mnemonic = "quant_mode_t";
  let apiName = "QuantMode_t";
}

def QuantParams : APIType<"QuantParams"> {
  let mnemonic = "quant_params";
  let apiName = "AscendC::QuantParams";
}

def ReduceOrder : APIType<"ReduceOrder"> {
  let mnemonic = "reduce_order";
  let apiName = "AscendC::ReduceOrder";
}

def RmsNormTiling : APIType<"RmsNormTiling"> {
  let mnemonic = "rmsnorm_tiling";
  let apiName = "RmsNormTiling";
}

def ScheduleType : APIType<"ScheduleType"> {
  let mnemonic = "schedule_type";
  let apiName = "ScheduleType";
}

def ShapeInfo : APIType<"ShapeInfo"> {
  let mnemonic = "shape_info";
  let apiName = "AscendC::ShapeInfo";
}

def SliceInfo : APIType<"SliceInfo"> {
  let mnemonic = "slice_info";
  let apiName = "AscendC::SliceInfo";
}

def SoftMaxTiling : APIType<"SoftMaxTiling"> {
  let mnemonic = "softmax_tiling";
  let apiName = "SoftMaxTiling";
}

def SoftMaxShapeInfo : APIType<"SoftMaxShapeInfo"> {
  let mnemonic = "softmax_shape_info";
  let apiName = "AscendC::SoftMaxShapeInfo";
}

def TBuf : APIType {
  let mnemonic = "tbuf";
  let apiName = "AscendC::TBuf";
}

def TBuffAddr : APIType<"BufAddr"> {
  let mnemonic = "buf_addr";
  let apiName = "AscendC::TBuffAddr";
}

def TCubeTiling : APIType<"TCubeTiling"> {
  let mnemonic = "cube_tiling";
  let apiName = "TCubeTiling";
}

def TPipe : APIType<"Pipe"> {
  let mnemonic = "pipe";
  let apiName = "AscendC::TPipe";
}

def TQue : APIType {
  let mnemonic = "queue";
  let apiName = "AscendC::TQue";
}

def TQueBind : APIType {
  let mnemonic = "que_bind";
  let apiName = "AscendC::TQueBind";
}

def TransDataTo5HDParams : APIType<"TransDataTo5HDParams"> {
  let mnemonic = "trans_data_to_5hd_params";
  let apiName = "AscendC::TransDataTo5HDParams";
}

def TransposeParamsExt : APIType<"TransposeParamsExt"> {
  let mnemonic = "transpose_params_ext";
  let apiName = "AscendC::TransposeParamsExt";
}

def TransposeType : APIType<"TransposeType"> {
  let mnemonic = "transpose_type";
  let apiName = "AscendC::TransposeType";
}

def UnaryRepeatParams : APIType<"UnaryRepeatParams"> {
  let mnemonic = "unary_repeat_params";
  let apiName = "AscendC::UnaryRepeatParams";
}
#endif // API_TYPES_TD
