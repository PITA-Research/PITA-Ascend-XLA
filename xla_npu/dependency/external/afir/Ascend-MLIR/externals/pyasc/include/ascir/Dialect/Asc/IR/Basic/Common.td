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

#ifndef ASC_BASIC_COMMON_TD
#define ASC_BASIC_COMMON_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_ResetMaskOp : APIOp<"reset_mask", "ResetMask", [AscFunc]> 
{
  let summary = "Reset mask ";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetHF32ModeOp : APIOp<"set_hf32_mode", "SetHF32Mode", [AscFunc]> {
  let summary = "Set HF32 compute mode";
  let arguments = (ins I1:$mode);
  let assemblyFormat = "$mode attr-dict `:` type($mode)";
}

def AscendC_SetHF32TransModeOp : APIOp<"set_hf32_trans_mode", "SetHF32TransMode", [AscFunc]> {
  let summary = "Set HF32 transform mode";
  let arguments = (ins I1:$trans_mode);
  let assemblyFormat = "$trans_mode attr-dict `:` type($trans_mode)";
}

def AscendC_SetMaskCountOp : APIOp<"set_mask_count", "SetMaskCount", [AscFunc]> {
  let summary = "Set mask count";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetMaskNormOp : APIOp<"set_mask_norm", "SetMaskNorm", [AscFunc]> {
  let summary = "Set mask normalization";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetMMLayoutTransformOp : APIOp<"set_mm_layout_transform", "SetMMLayoutTransform", [AscFunc]> {
  let summary = "Set MM layout transform mode";
  let arguments = (ins I1:$mm_layout_mode);
  let assemblyFormat = "$mm_layout_mode attr-dict `:` type($mm_layout_mode)";
}

def AscendC_SetVectorMaskL0Op : VectorOp<"set_vector_mask_l0", "SetVectorMask", []> {
  let summary = "Set vector mask in continuous mode (L0 API, Normal or Counter mode)";
  let arguments = (ins 
    AnyType:$len,
    TypeAttr:$dtype,
    AscendC_MaskModeAttr:$mode);
}

def AscendC_SetVectorMaskL1Op : VectorOp<"set_vector_mask_l1", "SetVectorMask", []> {
  let summary = "Set vector mask in bit mode (L1 API, Normal or Counter mode)";
  let arguments = (ins 
      AnyType:$maskHigh,   
      AnyType:$maskLow,
      TypeAttr:$dtype,  
      AscendC_MaskModeAttr:$mode
  ); 
}

#endif // ASC_BASIC_OP_COMMON_TD
