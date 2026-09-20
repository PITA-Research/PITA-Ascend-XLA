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

#ifndef ASC_BASIC_OP_AIPP_TD
#define ASC_BASIC_OP_AIPP_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// AIPP operations
//===----------------------------------------------------------------------===//

def AscendC_SetAippFunctionsOp
    : APIOp<"set_aipp_functions", "SetAippFunctions"> {
  let summary = "Call `AscendC::SetAippFunctions` method for AIPP configuration";
  let arguments = (ins
    AscendC_GlobalTensor:$src0,
    Optional<AscendC_GlobalTensor>:$src1,
    AscendC_AippInputFormatAttr:$format,
    AscendC_AippParams:$config
  );
  let assemblyFormat = [{
    $src0 (`,` $src1^)? `,` $config `,` `format` `=` $format
    attr-dict
    `:` functional-type(operands, results)
  }];
}

#endif // ASC_BASIC_OP_AIPP_TD