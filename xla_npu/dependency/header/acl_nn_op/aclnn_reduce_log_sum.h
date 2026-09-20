/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_REDUCE_LOG_SUM_H_
#define OP_API_INC_REDUCE_LOG_SUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnReduceLogSum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * * Operator function: fill the input tensor using the reflection of the input boundary.
 * @param [in] data: Represents the target tensor participating in the calculation, with dimensions less than 8 dimensions. The aclTensor on the Device side supports [non-continuous Tensor](../../../../docs/zh/context/non-continuous Tensor.md),
 * The data type supports FLOAT16 and FLOAT32, and [data format](../../../../docs/zh/context/data format.md) supports ND.
 * @param [in] axes: Specifies the calculation dimension, aclIntArray on the Host side, the data type supports INT64, and the value range is [-self.dim(), self.dim()-1].
 * @param [in] keepDims: Specifies whether to retain the dimensions of the input tensor in the output tensor, BOOL value on the Host side.
 * @param [in] noopWithEmptyAxes: Specify the behavior when axes is empty: false means all axes are calculated; true means no calculation is performed, the output tensor is equal to the input tensor, and the BOOL value on the Host side.
 * @param [in] reduce: represents the calculated result, with dimensions less than 8 dimensions. The aclTensor on the Device side supports [non-continuous Tensor](../../../../docs/zh/context/non-continuous Tensor.md),
 * The data type supports FLOAT16 and FLOAT32, which must be consistent with data. [Data format] (../../../../docs/zh/context/data format.md) supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReduceLogSumGetWorkspaceSize(
    const aclTensor* data, const aclIntArray* axes, bool keepDims,
    bool noopWithEmptyAxes, aclTensor* reduce, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnReduceLogSum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceRandom.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReduceLogSum(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REDUCE_LOG_SUM_H_
