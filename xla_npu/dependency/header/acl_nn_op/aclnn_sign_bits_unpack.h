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
#ifndef OP_API_INC_SIGN_BITS_UNPACK_H_
#define OP_API_INC_SIGN_BITS_UNPACK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSignBitsUnpack calculates the workspace size according to the specific calculation process.
 * @domain aclnnop_ops_train
 *
 * Operator function: Unpack the uint8 type 1-bit Adam into float32 or float16.
 *
 * @param [in] self: 
 * The aclTensor on the device side supports UINT8 data type and empty Tensor. Supports non-continuous Tensor, data type supports UINT8, and data format supports ND.
 * @param [in] size: int64_t on the host side, the first dimension of the output tensor when reshaping.
 * @param [in] dtype: aclDataType on the host side, indicating the data type of the quantized output Tensor. The data type supports ACL_FLOAT16 and ACL_FLOAT.
 * @param [in] out: 
 * The data type of aclTensor on the device side supports FLOAT16 and FLOAT. The data type is determined by dtype, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(
    const aclTensor* self, int64_t size, aclDataType dtype, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnSignBitsUnpack is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSignBitsUnpackGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSignBitsUnpack(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SIGN_BITS_UNPACK_H_
