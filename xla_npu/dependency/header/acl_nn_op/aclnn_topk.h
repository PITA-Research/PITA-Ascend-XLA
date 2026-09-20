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
#ifndef OP_API_INC_TOPK_H_
#define OP_API_INC_TOPK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTopk calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete calculation of k extreme values ​​and subscripts of the input.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the NPU device side supports INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT32, and DOUBLE.
 * Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [in] k:
 * int64_t type integer. Indicates the number of extreme values ​​output in the calculation dimension. The value range is [0, self.size(dim)].
 * @param [in] dim:
 * int64_t type integer. Represents the calculated dimension. The value range is [-self.dim(), self.dim()).
 * @param [in] largest:
 * bool type data. True means that the results in the calculation dimension should be output from large to small, False means that the results in the calculation dimension should be output from small to large.
 * @param [in] sorted:
 * bool type data. True indicates that the output results need to be sorted (if largest is True, the results are sorted from large to small, if largest is False, the results are sorted from small to large),
 * False means that the output results are not sorted and are output in the order of input data.
 * @param [in] valuesOut:
 * aclTensor on the dnpu device side, the data type supports INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT32, DOUBLE,
 * The data type is consistent with self, supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [in] indicesOut:
 * The aclTensor on the NPU device side supports INT64 as the data type. Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTopkGetWorkspaceSize(
    const aclTensor* self, int64_t k, int64_t dim, bool largest, bool sorted,
    aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnTopk is used to perform calculations.
 *
 * Operator function: Complete calculation of k extreme values ​​and subscripts of the input.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTopkGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTopk(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TOPK_H_
