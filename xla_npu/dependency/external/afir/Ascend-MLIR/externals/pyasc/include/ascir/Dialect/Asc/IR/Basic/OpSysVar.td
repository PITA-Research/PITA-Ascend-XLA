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

#ifndef ASC_BASIC_OP_SYSVAR_TD
#define ASC_BASIC_OP_SYSVAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_GetArchVersionOp : APIOp<"get_arch_version", "GetArchVersion", [AscFunc]> {
  let description = "Obtain the architecture version";
  let arguments = (ins AnyType:$coreVersion);
  let paramTypeLists = [0];
}

def AscendC_GetBlockIdxOp : APIOp<"get_block_idx", "GetBlockIdx", [Pure]> {
  let description = "Obtain an index of the current block";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetBlockNumOp : APIOp<"get_block_num", "GetBlockNum", [Pure]> {
  let description = "Obtain the number of active blocks";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetDataBlockSizeInBytesOp : APIOp<"get_data_block_size_in_bytes", "GetDataBlockSizeInBytes", [AscFunc]> {
  let description = "Obtain the size of a datablock in bytes for the current chip version";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetProgramCounterOp : APIOp<"get_program_counter", "GetProgramCounter", [AscFunc]> {
  let description = "Get the program counter pointer";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetSystemCycleOp : APIOp<"get_system_cycle", "GetSystemCycle", [AscFunc]> {
  let description = "Get the current system cycle count";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetSubBlockNumOp : APIOp<"get_sub_block_num", "GetSubBlockNum", [AscFunc]> {
  let description = "Get the num of vector core in aicore";
  let results = (outs UI64:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}


def AscendC_TrapOp : APIOp<"trap", "Trap", [AscFunc]> {
  let description = "Interrupt AI Core execution in NPU mode, equivalent to assert in CPU mode";
  let assemblyFormat = "attr-dict";
}
#endif // ASC_BASIC_OP_SYSVAR_TD
