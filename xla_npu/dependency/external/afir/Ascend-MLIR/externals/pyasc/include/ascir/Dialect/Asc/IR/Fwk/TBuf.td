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

#ifndef ASC_FWK_TBUF_TD
#define ASC_FWK_TBUF_TD
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// Buffer operations (TBuf)
//===----------------------------------------------------------------------===//

def AscendC_TBufOp : AscendC_Op<"tbuf", [AscConstructor]> {
  let summary = "Instantiate buffer";
  let results = (outs AscendC_TBuf:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

def AscendC_TBufGetTensorOp : APIOp<"tbuf.get_tensor", "Get"> {
  let summary = "Obtain tensor from initialized buffer";
  let arguments = (ins AscendC_TBuf:$buffer, Optional<AnyType>:$len);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $buffer attr-dict (`,` $len^)? `:` qualified(type($buffer)) `,` qualified(type($tensor)) (`,` type($len)^)?
  }];
}

def AscendC_TBufGetWithOffsetOp : APIOp<"tbuf.get_with_offset", "GetWithOffset"> {
  let summary = "Obtain tensor from initialized buffer with offset";
  let arguments = (ins AscendC_TBuf:$buffer, AnyType:$size, AnyType:$bufOffset);
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = [{
    $buffer `,` $size `,` $bufOffset attr-dict `:` 
    qualified(type($buffer)) `,` qualified(type($size)) `,` qualified(type($bufOffset)) `,`
    qualified(type($tensor))
  }];
}

#endif
