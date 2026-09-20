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
#ifndef OP_API_INC_TRANS_MATMUL_WEIGHT_H_
#define OP_API_INC_TRANS_MATMUL_WEIGHT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * aclnnCalculateMatmulWeightSizeV2 is used to calculate and call aclnnMatMul, aclnnMm, aclnnWeightQuantBatchMatmulV2,
 * The element size that the weight tensor passed in aclnnWeightQuantBatchMatmulV3 and aclnnQuantMatmulV3 needs to occupy
 * @domain aclnn_ops_infer
 *
 * @param [in] tensorShape: Shape used to express the weight of the matrix loaded by Matmul this time
 * @param [in] dataType: Datatype of input Weight, supports INT8 and Float16
 * @param [out] weightTensorSize: According to MatMul's internal processing logic, calculate how many elements of data are needed for Weight under this input
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCalculateMatmulWeightSizeV2(
    const aclIntArray* tensorShape, aclDataType dataType,
    uint64_t* weightTensorSize);

/**
 * @brief aclnnCalculateMatmulWeightSize is used to calculate the element size required to be occupied by the weight tensor passed in when calling aclnnMatMul or aclnnMm
 *
 *
 * @param [in] tensorShape: Shape used to express the weight of the matrix loaded by Matmul this time
 * @param [in] weightTensorSize: According to MatMul's internal processing logic, calculate how many elements of data are needed for Weight under this input
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCalculateMatmulWeightSize(
    const aclIntArray* tensorShape, uint64_t* weightTensorSize);

/**
 * The first interface of @brief aclnnTransMatmulWeight calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] mmWeightRef: The input is a weightTensor of matmul to be processed, the format is normal ND input, the data type supports float16, int8, bfloat16
 * After being processed by this interface, the tensor is refreshed into the preprocessed matmul weightTensor format and converted to ND or private format according to the affinity.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(
    aclTensor* mmWeightRef, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnTransMatmulWeight is used to perform calculations.
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnTransMatmulWeightGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTransMatmulWeight(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANS_MATMUL_WEIGHT_H_
