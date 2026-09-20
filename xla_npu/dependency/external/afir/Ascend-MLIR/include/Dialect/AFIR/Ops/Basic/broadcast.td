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

//===- broadcast.td - AFIR broadcast operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the broadcast operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BROADCAST_OPS
#define AFIR_BROADCAST_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AFIR_BroadcastOp : AFIR_UnaryOp<"broadcast", [Pure], [I8, UI8, I16, UI16, I32, UI32, UI64, F16, F32], "Broadcast", "Vector", 1>;

#endif // AFIR_BROADCAST_OPS