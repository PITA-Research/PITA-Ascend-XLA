/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ACLNN_DYNAMIC_QUANT_V3_H_
#define ACLNN_DYNAMIC_QUANT_V3_H_

#include "acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnDynamicQuantV3GetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * The shape dimension of aclTensor on the device side and the Tensor input by the operator must be greater than 1. The data type of aclTensor on the device side supports FLOAT16 and BFLOAT16.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] smoothScalesOptional: npu
 * The smoothScales input by the operator. When there is no groupIndexOptional, the shape dimension is the last dimension of x; when there is groupIndexOptional, the shape is two dimensions.
 * The size of the first dimension is the number of experts, not exceeding 1024, and the size of the second dimension is the last dimension of x. aclTensor on Device side
 * The data type supports FLOAT16 and BFLOAT16, and the data type must be consistent with x. Non-continuous Tensors are supported, and the data format supports ND.
 * @param [in] groupIndexOptional: npu
 * The groupIndex of the operator input, the aclTensor on the Device side, the data type supports INT32, supports non-continuous Tensor, and the data format supports ND.
 * Shape only supports one dimension, and the dimension size is equal to the first dimension of smoothScalesOptional. When groupIndexOptional is non-nullptr, smoothScalesOptional must be non-nullptr.
 * @param [in]
 * dstType: Outputs the enumeration value corresponding to the data type of yOut, int on the Host side. Support DT_INT8, DT_INT32, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M4FN, DT_HIFLOAT8.
 * @param [in]
 * isSymmetrical: specifies whether to quantize symmetrically, bool on the Host side.
 * @param [in]
 * quantMode: used to specify the quantization mode, char* on the Host side. Currently "pertoken" and "pertensor" are supported. When quantMode takes "pertensor", groupIndexOptional must be nullptr.
 * @param [in] yOut: npu
 * The shape of the quantized output Tensor is consistent with x, and the type is specified by dstType. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scaleOut: npu
 * The scale used for quantification, the aclTensor on the Device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] offsetOut: npu
 * The offset used for quantification, the aclTensor on the Device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnDynamicQuantV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional,
    const aclTensor* groupIndexOptional, int64_t dstType, bool isSymmetrical,
    const char* quantMode, const aclTensor* yOut, const aclTensor* scaleOut,
    const aclTensor* offsetOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDynamicQuantV3 is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDynamicQuantV3GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DYNAMIC_QUANT_V3_H_