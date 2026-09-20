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

#ifndef ASC_BASIC_OP_DUMPTENSOR_TD
#define ASC_BASIC_OP_DUMPTENSOR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DebugAPI operations
//===----------------------------------------------------------------------===//

def AscendC_PrintfOp : APIOp<"printf", "printf"> {
  let arguments = (ins StrAttr:$desc, Variadic<AnyType>:$vars);
}

def AscendC_DumpTensorOp : APIOp<"dump_tensor", "DumpTensor", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$tensor, AnyType:$desc,
    AnyType:$dumpSize, Optional<AscendC_ShapeInfo>:$shapeInfo);
}

def AscendC_PrintTimeStampOp : APIOp<"print_time_stamp", "PrintTimeStamp", [AscFunc]> {
  let arguments = (ins AnyType:$descId);
}

def AscendC_DumpAccChkPointOp : APIOp<"dump_acc_chk_point", "DumpAccChkPoint", [AscFunc]> {
  let arguments = (ins AscendC_BaseTensorTypeInterface:$tensor, AnyType:$index,
    AnyType:$countOff, AnyType:$dumpSize);
}

def AscendC_MetricsProfStartOp : APIOp<"metrics_prof_start","MetricsProfStart", [AscFunc]> {
  let description = "Start performance metrics collection for a code region; pair with MetricsProfStop.";
  let assemblyFormat = "attr-dict";
}

def AscendC_MetricsProfStopOp : APIOp<"metrics_prof_stop","MetricsProfStop", [AscFunc]> {
  let description = "Stop performance metrics collection started by MetricsProfStart.";
  let assemblyFormat = "attr-dict";
}
#endif // ASC_BASIC_OP_DUMPTNESOR_TD
