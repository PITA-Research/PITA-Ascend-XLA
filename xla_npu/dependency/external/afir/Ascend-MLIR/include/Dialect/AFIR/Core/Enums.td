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

//===- AFIREnums.td - AFIR dialect enum definitions -------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file defines enumerations for the AFIR dialect. All enum definitions
// should be placed here to avoid redefinition issues.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ENUMS
#define AFIR_ENUMS

include "Dialect/AFIR/Core/Base.td"
include "mlir/IR/EnumAttr.td"


//===----------------------------------------------------------------------===//
// Position Enumeration
//===----------------------------------------------------------------------===//

def AFIR_PositionAttr : I32EnumAttr<"Position",
    "Memory position",
    [
      I32EnumAttrCase<"GM", 0, "gm">,
      I32EnumAttrCase<"VECTOR_IN", 1, "vector_in">,
      I32EnumAttrCase<"VECTOR_OUT", 2, "vector_out">,
      I32EnumAttrCase<"VECTOR_CALC", 3, "vector_calc">,
      I32EnumAttrCase<"L1", 4, "l1">,
      I32EnumAttrCase<"L2", 5, "l2">,
      I32EnumAttrCase<"L0A", 6, "l0a">,
      I32EnumAttrCase<"L0B", 7, "l0b">,
      I32EnumAttrCase<"L0C", 8, "l0c">
    ]> {
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Compute Unit Enumeration
//===----------------------------------------------------------------------===//

def AFIR_ComputeUnitAttr : I32EnumAttr<"ComputeUnit",
    "Compute unit for node execution",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"MTE1", 1, "mte1">,
      I32EnumAttrCase<"MTE2", 2, "mte2">,
      I32EnumAttrCase<"MTE3", 3, "mte3">,
      I32EnumAttrCase<"Scalar", 4, "scalar">,
      I32EnumAttrCase<"Vector", 5, "vector">,
      I32EnumAttrCase<"Cube", 6, "cube">,
      I32EnumAttrCase<"Invalid", 7, "invalid">
    ]> {
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Compute Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_ComputeTypeAttr : I32EnumAttr<"ComputeType",
    "Compute type for node execution",
    [
      I32EnumAttrCase<"Load", 0, "load">,
      I32EnumAttrCase<"Store", 1, "store">,
      I32EnumAttrCase<"ReduceStore", 2, "reduce_store">,
      I32EnumAttrCase<"Elementwise", 3, "elementwise">,
      I32EnumAttrCase<"Broadcast", 4, "broadcast">,
      I32EnumAttrCase<"Reduce", 5, "reduce">,
      I32EnumAttrCase<"Transpose", 6, "transpose">,
      I32EnumAttrCase<"Concat", 7, "concat">,
      I32EnumAttrCase<"Gather", 8, "gather">,
      I32EnumAttrCase<"Cube", 9, "cube">,
      I32EnumAttrCase<"Split", 10, "split">,
      I32EnumAttrCase<"Invalid", 11, "invalid">
    ]> {
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Axis Type Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AxisTypeAttr : I32EnumAttr<"AxisType",
    "Axis type for computational loops",
    [
      I32EnumAttrCase<"Original", 0>,
      I32EnumAttrCase<"BlockOuter", 1>,
      I32EnumAttrCase<"BlockInner", 2>,
      I32EnumAttrCase<"TileOuter", 3>,
      I32EnumAttrCase<"TileInner", 4>,
      I32EnumAttrCase<"Merged", 5>,
      I32EnumAttrCase<"Invalid", 6>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// AscGraphType Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AscGraphTypeAttr : I32EnumAttr<"AscGraphType",
    "Ascend graph type",
    [
      I32EnumAttrCase<"Compute", 0>,
      I32EnumAttrCase<"Invalid", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

#endif // AFIR_ENUMS