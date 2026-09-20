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

#ifndef AIR_UTILS_CONVERT_TENSOR_H_
#define AIR_UTILS_CONVERT_TENSOR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"    // from @llvm-project

#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "./status.h"

using namespace ge;

namespace mlir::air {

// Converts an MLIR elements attribute to a ascend tensor.
Status ConvertToTensor(mlir::ElementsAttr attr, Tensor* output_tensor);

}  // namespace mlir::air

#endif  // AIR_UTILS_CONVERT_TENSOR_H_
