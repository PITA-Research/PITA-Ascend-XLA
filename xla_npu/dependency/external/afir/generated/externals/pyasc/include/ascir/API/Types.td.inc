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

def AscendC_AippChannelPaddingParams : AscendC_Type<"AippChannelPaddingParams", "aipp_cpadding_params"> {
  let description = "Represents AscendC::AippChannelPaddingParams";
}
def AscendC_AippColorSpaceConvParams : AscendC_Type<"AippColorSpaceConvParams", "aipp_csc_params"> {
  let description = "Represents AscendC::AippColorSpaceConvParams";
}
def AscendC_AippDataTypeConvParams : AscendC_Type<"AippDataTypeConvParams", "aipp_dtc_params"> {
  let description = "Represents AscendC::AippDataTypeConvParams";
}
def AscendC_AippInputFormat : AscendC_Type<"AippInputFormat", "aipp_input_format"> {
  let description = "Represents AscendC::AippInputFormat";
}
def AscendC_AippPaddingParams : AscendC_Type<"AippPaddingParams", "aipp_padding_params"> {
  let description = "Represents AscendC::AippPaddingParams";
}
def AscendC_AippParams : AscendC_Type<"AippParams", "aipp_params"> {
  let description = "Represents AscendC::AippParams";
}
def AscendC_AippSingleLineParams : AscendC_Type<"AippSingleLineParams", "aipp_single_line_params"> {
  let description = "Represents AscendC::AippSingleLineParams";
}
def AscendC_AippSwapParams : AscendC_Type<"AippSwapParams", "aipp_swap_params"> {
  let description = "Represents AscendC::AippSwapParams";
}
def AscendC_AscendQuantConfig : AscendC_Type<"AscendQuantConfig", "ascend_quant_config"> {
  let description = "Represents AscendC::AscendQuantConfig";
}
def AscendC_BatchMode : AscendC_Type<"BatchMode", "batch_mode"> {
  let description = "Represents BatchMode";
}
def AscendC_BatchOutMode : AscendC_Type<"BatchOutMode", "batch_out_mode"> {
  let description = "Represents BatchOutMode";
}
def AscendC_BinaryRepeatParams : AscendC_Type<"BinaryRepeatParams", "binary_repeat_params"> {
  let description = "Represents AscendC::BinaryRepeatParams";
}
def AscendC_BlockMode : AscendC_Type<"BlockMode", "block_mode"> {
  let description = "Represents AscendC::BlockMode";
}
def AscendC_BrcbRepeatParams : AscendC_Type<"BrcbRepeatParams", "brcb_repeat_params"> {
  let description = "Represents AscendC::BrcbRepeatParams";
}
def AscendC_CopyRepeatParams : AscendC_Type<"CopyRepeatParams", "copy_repeat_params"> {
  let description = "Represents AscendC::CopyRepeatParams";
}
def AscendC_DataCopyCO12DstParams : AscendC_Type<"DataCopyCO12DstParams", "data_copy_co12dst_params"> {
  let description = "Represents AscendC::DataCopyCO12DstParams";
}
def AscendC_DataCopyEnhancedParams : AscendC_Type<"DataCopyEnhancedParams", "data_copy_enhanced_params"> {
  let description = "Represents AscendC::DataCopyEnhancedParams";
}
def AscendC_DataCopyExtParams : AscendC_Type<"DataCopyExtParams", "data_copy_ext_params"> {
  let description = "Represents AscendC::DataCopyExtParams";
}
def AscendC_DataCopyPadParams : AscendC_Type<"DataCopyPadParams", "data_copy_pad_params"> {
  let description = "Represents AscendC::DataCopyPadParams";
}
def AscendC_DataCopyParams : AscendC_Type<"DataCopyParams", "data_copy_params"> {
  let description = "Represents AscendC::DataCopyParams";
}
def AscendC_DataFormat : AscendC_Type<"DataFormat", "data_format"> {
  let description = "Represents AscendC::DataFormat";
}
def AscendC_DeqScale : AscendC_Type<"DeqScale", "deq_scale"> {
  let description = "Represents AscendC::DeqScale";
}
def AscendC_FixpipeConfig : AscendC_Type<"FixpipeConfig", "fixpipe_config"> {
  let description = "Represents AscendC::FixpipeConfig";
}
def AscendC_FmatrixMode : AscendC_Type<"FmatrixMode", "fmatrix_mode"> {
  let description = "Represents AscendC::FmatrixMode";
}
def AscendC_GatherMaskParams : AscendC_Type<"GatherMaskParams", "gather_mask_params"> {
  let description = "Represents AscendC::GatherMaskParams";
}
def AscendC_GatherRepeatParams : AscendC_Type<"GatherRepeatParams", "gather_repeat_params"> {
  let description = "Represents AscendC::GatherRepeatParams";
}
def AscendC_IsResetLoad3dConfig : AscendC_Type<"IsResetLoad3dConfig", "is_reset_load_3d_config"> {
  let description = "Represents AscendC::IsResetLoad3dConfig";
}
def AscendC_IterateMode : AscendC_Type<"IterateMode", "iterate_mode"> {
  let description = "Represents IterateMode";
}
def AscendC_IterateOrder : AscendC_Type<"IterateOrder", "iterate_order"> {
  let description = "Represents IterateOrder";
}
def AscendC_KfcServer : AscendC_Type<"KfcServer", "kfc_server"> {
  let description = "Represents AscendC::KfcServer";
}
def AscendC_LoadData2DParams : AscendC_Type<"LoadData2DParams", "load_data_2d_params"> {
  let description = "Represents AscendC::LoadData2DParams";
}
def AscendC_LoadData2dTransposeParams : AscendC_Type<"LoadData2dTransposeParams", "load_data_2d_transpose_params"> {
  let description = "Represents AscendC::LoadData2dTransposeParams";
}
def AscendC_LoadDataRepeatParam : AscendC_Type<"LoadDataRepeatParam", "load_data_repeat_param"> {
  let description = "Represents AscendC::LoadDataRepeatParam";
}
def AscendC_LoadImageToLocalParams : AscendC_Type<"LoadImageToLocalParams", "load_image_to_local_params"> {
  let description = "Represents AscendC::LoadImageToLocalParams";
}
def AscendC_MaskMode : AscendC_Type<"MaskMode", "mask_mode"> {
  let description = "Represents AscendC::MaskMode";
}
def AscendC_MatmulApiStaticTiling : AscendC_Type<"MatmulApiStaticTiling", "matmul_api_static_tiling"> {
  let description = "Represents MatmulApiStaticTiling";
}
def AscendC_MatmulConfig : AscendC_Type<"MatmulConfig", "matmul_config"> {
  let description = "Represents MatmulConfig";
}
def AscendC_MatrixOffset : AscendC_Type<"MatrixOffset", "matrix_offset"> {
  let description = "Represents MatrixOffset";
}
def AscendC_MmadParams : AscendC_Type<"MmadParams", "mmad_params"> {
  let description = "Represents AscendC::MmadParams";
}
def AscendC_Nd2NzParams : AscendC_Type<"Nd2NzParams", "nd2nz_params"> {
  let description = "Represents AscendC::Nd2NzParams";
}
def AscendC_Nz2NdParams : AscendC_Type<"Nz2NdParams", "nz2nd_params"> {
  let description = "Represents AscendC::Nz2NdParams";
}
def AscendC_Nz2NdParamsFull : AscendC_Type<"Nz2NdParamsFull", "nz2nd_params_full"> {
  let description = "Represents AscendC::Nz2NdParamsFull";
}
def AscendC_pad_t : AscendC_Type<"pad_t", "pad_t"> {
  let description = "Represents AscendC::pad_t";
}
def AscendC_QuantModes : AscendC_Type<"QuantModes", "quant_mode_t"> {
  let description = "Represents QuantMode_t";
}
def AscendC_QuantParams : AscendC_Type<"QuantParams", "quant_params"> {
  let description = "Represents AscendC::QuantParams";
}
def AscendC_ReduceOrder : AscendC_Type<"ReduceOrder", "reduce_order"> {
  let description = "Represents AscendC::ReduceOrder";
}
def AscendC_RmsNormTiling : AscendC_Type<"RmsNormTiling", "rmsnorm_tiling"> {
  let description = "Represents RmsNormTiling";
}
def AscendC_ScheduleType : AscendC_Type<"ScheduleType", "schedule_type"> {
  let description = "Represents ScheduleType";
}
def AscendC_ShapeInfo : AscendC_Type<"ShapeInfo", "shape_info"> {
  let description = "Represents AscendC::ShapeInfo";
}
def AscendC_SliceInfo : AscendC_Type<"SliceInfo", "slice_info"> {
  let description = "Represents AscendC::SliceInfo";
}
def AscendC_SoftMaxShapeInfo : AscendC_Type<"SoftMaxShapeInfo", "softmax_shape_info"> {
  let description = "Represents AscendC::SoftMaxShapeInfo";
}
def AscendC_SoftMaxTiling : AscendC_Type<"SoftMaxTiling", "softmax_tiling"> {
  let description = "Represents SoftMaxTiling";
}
def AscendC_BufAddr : AscendC_Type<"BufAddr", "buf_addr"> {
  let description = "Represents AscendC::TBuffAddr";
}
def AscendC_TCubeTiling : AscendC_Type<"TCubeTiling", "cube_tiling"> {
  let description = "Represents TCubeTiling";
}
def AscendC_Pipe : AscendC_Type<"Pipe", "pipe"> {
  let description = "Represents AscendC::TPipe";
}
def AscendC_TransDataTo5HDParams : AscendC_Type<"TransDataTo5HDParams", "trans_data_to_5hd_params"> {
  let description = "Represents AscendC::TransDataTo5HDParams";
}
def AscendC_TransposeParamsExt : AscendC_Type<"TransposeParamsExt", "transpose_params_ext"> {
  let description = "Represents AscendC::TransposeParamsExt";
}
def AscendC_TransposeType : AscendC_Type<"TransposeType", "transpose_type"> {
  let description = "Represents AscendC::TransposeType";
}
def AscendC_UnaryRepeatParams : AscendC_Type<"UnaryRepeatParams", "unary_repeat_params"> {
  let description = "Represents AscendC::UnaryRepeatParams";
}
