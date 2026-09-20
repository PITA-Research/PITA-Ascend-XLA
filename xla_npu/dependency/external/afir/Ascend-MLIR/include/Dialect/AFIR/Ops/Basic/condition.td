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

//===- condition.td - AFIR condition operations -------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the condition operations for the AFIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONDITION_OPS
#define AFIR_CONDITION_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class AFIR_TernaryConditionOp<string mnemonic, list<Trait> traits = [], list<Type> conditionTypes, list<Type> types> :
    AFIR_TernaryOpBase<mnemonic, traits, conditionTypes, types, types, types, "Elementwise", "Vector", 1>;

def AFIR_SelectOp : AFIR_TernaryConditionOp<"select", [Pure], [UI8], [I16, I32, I64, F16, F32]>;
def AFIR_WhereOp : AFIR_TernaryConditionOp<"where", [Pure], [UI8], [I16, I32, I64, F16, F32]>;

#endif // AFIR_CONDITION_OPS