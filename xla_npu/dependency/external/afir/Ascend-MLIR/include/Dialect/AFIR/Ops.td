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

//===- AFIROps.td - AFIR operation definitions -------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the AFIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "Dialect/AFIR/Dialect.td"
include "Dialect/AFIR/Base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

include "Dialect/AFIR/Ops/Basic/broadcast.td"
include "Dialect/AFIR/Ops/Basic/cast.td"
include "Dialect/AFIR/Ops/Basic/condition.td"
include "Dialect/AFIR/Ops/Basic/data.td"
include "Dialect/AFIR/Ops/Basic/math.td"

include "Dialect/AFIR/Ops/Adv/concat.td"
include "Dialect/AFIR/Ops/Adv/reduce.td"
include "Dialect/AFIR/Ops/Adv/transpose.td"

#endif // AFIR_OPS