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

#ifndef ASC_BASIC_OP_COPY_TD
#define ASC_BASIC_OP_COPY_TD

include "Base.td"

//===----------------------------------------------------------------------===//
// Copy operations
//===----------------------------------------------------------------------===//

class CopyBaseOp<string mnemonic, list<Trait> traits = []>
    : APIOp<mnemonic, "Copy", traits> {
  let arguments = (ins 
    AscendC_LocalTensor:$dst, 
    AscendC_LocalTensor:$src,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}

def AscendC_CopyL0Op 
    : CopyBaseOp<"copy_l0"> {
  let summary = "Copy with bitwise mask";
  let arguments = (ins 
    AscendC_LocalTensor:$dst, 
    AscendC_LocalTensor:$src,
    Variadic<AnyInteger>:$mask,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}

def AscendC_CopyL1Op : CopyBaseOp<"copy_l1"> {
  let summary = "Copy with continuous mask";
  let arguments = (ins 
    AscendC_LocalTensor:$dst, 
    AscendC_LocalTensor:$src,
    AnyInteger:$mask,
    AnyInteger:$repeatTime,
    AscendC_CopyRepeatParams:$repeatParams,
    AnyInteger:$isSetMask
  );
}
#endif // ASC_BASIC_OP_COPY_TD