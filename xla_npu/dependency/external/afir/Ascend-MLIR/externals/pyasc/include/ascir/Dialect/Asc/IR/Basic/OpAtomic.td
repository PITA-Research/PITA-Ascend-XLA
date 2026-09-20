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

#ifndef ASC_BASIC_OP_ATOMIC_TD
#define ASC_BASIC_OP_ATOMIC_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_SetAtomicTypeOp : APIOp<"set_atomic_type", "SetAtomicType", [AscFunc]> {
    let summary = "Set atomic operation data type by template type parameter";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicAddOp : APIOp<"set_atomic_add", "SetAtomicAdd", [AscFunc]> {
    let summary = "Enable atomic addition for subsequent data transfers using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicNoneOp : APIOp<"set_atomic_none", "SetAtomicNone", [AscFunc]> {
    let summary = "Disable all atomic operations for subsequent data transfers";
}

def AscendC_SetAtomicMaxOp : APIOp<"set_atomic_max", "SetAtomicMax", [AscFunc]> {
    let summary = "Enable atomic maximum comparison using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

def AscendC_SetAtomicMinOp : APIOp<"set_atomic_min", "SetAtomicMin", [AscFunc]> {
    let summary = "Enable atomic minimum comparison using the specified data type";
    let arguments = (ins TypeAttr:$dtype);
    let paramTypeLists = [5];
}

#endif //ASC_BASIC_OP_ATOMIC_TD
