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

//===- AFIRToASCIRText.h - AFIR to ASCIR text conversion ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H
#define ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass();

}  // namespace mlir::afir

#endif  // ASCEND_MLIR_CONVERSION_AFIRTOASCIR_AFIRTOASCIRTEXT_H
