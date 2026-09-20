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

#ifndef OP_API_INC_DIAG_FLAT_H_
#define OP_API_INC_DIAG_FLAT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnDiagFlat calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports all data types and non-continuous Tensors. The data format supports ND, and the data format needs to be consistent with out.
 * @param [in] diagonal: The basic data type on the host side. The data type supports INT64 and represents the offset of the matrix.
 * @param [out] out: aclTensor on the NPU device side, the data type supports all data types, supports non-continuous Tensor, the data type supports ND,
 * Represents the output tensor.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDiagFlatGetWorkspaceSize(const aclTensor* self,
                                                    int64_t diagonal,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDiagFlat is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDiagFlatGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDiagFlat(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DIAG_FLAT_H_
