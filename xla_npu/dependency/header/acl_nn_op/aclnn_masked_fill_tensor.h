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
#ifndef OP_API_INC_MASKEF_FILL_TENSOR_H_
#define OP_API_INC_MASKEF_FILL_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceMaskedFillTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Use value to fill the elements in selfRef corresponding to the position where the value in the mask matrix is true.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A1[(mask)] -->B1([Contiguous])
 *  B1  -->C1([Cast])
 *  C1  -->D
 *
 *  A2[(value)]-->B2[(Cast)]
 *  B2-->D
 * ```
 *
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT,
 *                      FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] mask: aclTensor on the NPU device side, the data type supports BOOL. And shape and selfRef satisfy the broadcast relationship. The data format supports ND.
 * @param [in] value: aclTensor on the NPU device side. And shape and selfRef satisfy the broadcast relationship. The data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* value,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnInplaceMaskedFillTensor's second interface is used to perform calculations.
 *
 * Operator function: Use value to fill the elements in selfRef corresponding to the position where the value in the mask matrix is true.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A2[(mask)] -->B2([Contiguous])
 *  B2 --> C2([Cast])
 *  C2  -->D
 *
 *  A1[(value)]-->B1([Contiguous])
 *  B1-->C1([Cast])
 *  C1-->D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is determined by the first interface aclnnInplaceMaskedFillTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedFillTensor(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKEF_FILL_TENSOR_H_