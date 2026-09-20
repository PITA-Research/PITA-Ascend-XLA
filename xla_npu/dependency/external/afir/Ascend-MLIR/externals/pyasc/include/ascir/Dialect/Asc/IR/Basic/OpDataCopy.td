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

#ifndef ASC_BASIC_OP_DATACOPY_TD
#define ASC_BASIC_OP_DATACOPY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data transfer operations
//===----------------------------------------------------------------------===//

def AscendC_SliceInfoSubIndexOp
    : AscendC_Op<"slice_info.subindex", [Pure]> {
  let summary = "Call `AscendC::SliceInfo::operator[]` method";
  let arguments = (ins AscendC_SliceInfo:$sliceInfo, AnySignlessIntegerOrIndex:$index);
  let results = (outs AscendC_SliceInfo:$result);
  let assemblyFormat = [{
    $sliceInfo `[` $index `]`  attr-dict `:` qualified(type($sliceInfo)) `,`
    type($index) `,` qualified(type($result))
  }];
}

def AscendC_DataCopyCO12DstOp : DataCopyOp<"data_copy_co12dst", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (with DataCopyCO12DstParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyCO12DstParams:$intriParams);
}

def AscendC_DataCopyEnhancedOp : DataCopyOp<"data_copy_enhanced", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (with DataCopyEnhancedParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$intriParams,
                       AscendC_DataCopyEnhancedParams:$enhancedParams);
}

def AscendC_DataCopyNd2NzOp : DataCopyOp<"data_copy_nd2nz", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (with Nd2NzParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_Nd2NzParams:$intriParams);
}

def AscendC_DataCopyNz2NdOp : DataCopyOp<"data_copy_nz2nd", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (with Nz2NdParamsFull)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_Nz2NdParamsFull:$intriParams);
}

def AscendC_DataCopySliceOp : DataCopyOp<"data_copy_slice", "DataCopy", [AttrSizedOperandSegments]> {
  let description = "Perform copying between tensors (with SliceInfo)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       Variadic<AscendC_SliceInfo>:$dstSliceInfo,
                       Variadic<AscendC_SliceInfo>:$srcSliceInfo,
                       Optional<UI32>:$dimValue);
}

def AscendC_DataCopyL0Op : DataCopyOp<"data_copy_l0", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (L0 API)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$repeatParams);
}

def AscendC_DataCopyL2Op : DataCopyOp<"data_copy_l2", "DataCopy", [AscFunc]> {
  let description = "Perform copying between tensors (L2 API)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AnyType:$calCount);
}

def AscendC_DataCopyPadExtParamsOp : AscendC_Op<"data_copy_pad_ext_params", [AscConstructor]> {
  let summary = "Instantiate DataCopyPadExtParams";
  let results = (outs AscendC_DataCopyPadExtParams:$params);
  let assemblyFormat = "attr-dict `:` qualified(type($params))";
}

def AscendC_DataCopyPadExtL0Op : DataCopyOp<"data_copy_pad_l0_ext", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (L0 API with DataCopyExtParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyExtParams:$dataCopyParams,
                       AscendC_DataCopyPadExtParams:$padParams);
}

def AscendC_DataCopyPadExtL2Op : DataCopyOp<"data_copy_pad_l2_ext", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (L2 API with DataCopyExtParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyExtParams:$dataCopyParams);
}

def AscendC_DataCopyPadL0Op : DataCopyOp<"data_copy_pad_l0", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (L0 API with DataCopyParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$dataCopyParams,
                       AscendC_DataCopyPadParams:$padParams);
}

def AscendC_DataCopyPadL2Op : DataCopyOp<"data_copy_pad_l2", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (L2 API with DataCopyParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$dataCopyParams);
}

def AscendC_DataCopyPadExtNd2NzOp : DataCopyOp<"data_copy_pad_nd2nz_ext", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (Nd2Nz with DataCopyExtParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyExtParams:$dataCopyParams,
                       AscendC_Nd2NzParams:$nd2nzParams);
}

def AscendC_DataCopyPadNd2NzOp : DataCopyOp<"data_copy_pad_nd2nz", "DataCopyPad",[AscFunc]> {
  let description = "Perform copying between tensors (Nd2Nz with DataCopyParams)";
  let arguments = (ins AscendC_BaseTensorTypeInterface:$dst,
                       AscendC_BaseTensorTypeInterface:$src,
                       AscendC_DataCopyParams:$dataCopyParams,
                       AscendC_Nd2NzParams:$nd2nzParams);
}

def AscendC_LoadImageToLocalOp : APIOp<"load_image_to_local", "LoadImageToLocal", [AscFunc]> {
  let description = "Load image data from Global Memory to LocalTensor (supports TPosition A1/B1). ";
  let arguments = (ins 
    AscendC_LocalTensor:$dst,  
    AscendC_LoadImageToLocalParams:$loadDataParams 
  );
}

#endif // ASC_BASIC_OP_DATACOPY_TD
