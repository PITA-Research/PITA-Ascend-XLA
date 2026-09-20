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

//===- Passes.td - AFIR dialect passes ---------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TRANSFORMS_PASSES
#define AFIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AFIRCanonicalizePass : Pass<"afir-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize AFIR operations";
  let description = [{
    This pass performs canonicalization transformations on AFIR operations,
    including constant folding and algebraic simplifications.
  }];
  let constructor = "mlir::afir::createAFIRCanonicalizePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def AFIRShapeInferencePass : Pass<"afir-shape-inference", "mlir::ModuleOp"> {
  let summary = "Infer shapes for AFIR operations";
  let description = [{
    This pass infers and propagates shape information through AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRShapeInferencePass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}

def AFIRAddAxisPass : Pass<"afir-add-axis", "mlir::ModuleOp"> {
  let summary = "Add Axis for AFIR operations";
  let description = [{
    This pass add axis fpr AFIR operations.
  }];
  let constructor = "mlir::afir::createAFIRAddAxisPass()";
  let dependentDialects = ["mlir::afir::AFIRDialect"];
}


#endif // AFIR_TRANSFORMS_PASSES
