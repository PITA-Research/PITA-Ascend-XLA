/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_COMMON_INC_OP_DEV_TENSOR_VIEW_UTILS_H
#define OP_API_COMMON_INC_OP_DEV_TENSOR_VIEW_UTILS_H

#include "common_types.h"

namespace op {
/**
 * @brief Check whether tensor is contiguous. Do not used to check whether the pointer is null.
 * Contiguous tensor must meet one of the following conditions:
 * 1. Private formats tensor must be contiguous.
 * 2. Empty tensor must be contiguous.
 * 3. The stride and view shape match contiguous features.
 * 4. nullptr return true by default. 
 * @param tensor The input tensor
 * @return bool True/false
 */
bool IsContiguous(const aclTensor *tensor);

/**
 * @brief Check whether all input tensors can be regarded as a view of contiguous tensors
 * and all tensors have the same view feature.
 * @param tensorList The input tensors.
 * @return
 */
bool CanPickViewAsContiguous(
    std::initializer_list<const aclTensor *> tensorList);

/**
 * @brief Check whether the input tensor can be regarded as a view of a contiguous tensor.
 * @param tensor The input tensor
 * @return True/false
 */
bool CanPickViewAsContiguous(const aclTensor *tensor);

/**
 * @brief Check whether the input tensor is valid.
 * @param tensor The input tensor
 * @return bool True/false
 */
bool Validate(const aclTensor *tensor);
}  // namespace op

#endif  // OP_API_COMMON_INC_OP_DEV_TENSOR_VIEW_UTILS_H
