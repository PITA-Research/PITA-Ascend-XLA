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

//===- AFIRCanonicalize.cpp - AFIR canonicalization pass --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRCANONICALIZEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

// Pattern: add(x, 0) -> x (when 0 is a constant)
// Pattern: mul(x, 1) -> x (when 1 is a constant)
// These patterns can be expanded based on needs

struct AFIRCanonicalizePass
    : public impl::AFIRCanonicalizePassBase<AFIRCanonicalizePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());

    // Add canonicalization patterns here
    // patterns.add<...>(patterns.getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRCanonicalizePass() {
  return std::make_unique<AFIRCanonicalizePass>();
}

}  // namespace afir
}  // namespace mlir
