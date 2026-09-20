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

#ifndef OP_API_INC_DIV_H_
#define OP_API_INC_DIV_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnDiv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in]
 * mode: an integer constant of the remainder processing method. The enumeration value is as follows: 0 - rounding is not performed by default. 1 - Round the fractional part of the division to zero. 2- Round down the result of division.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to be a convertible data type after self and other are deduced, and the shape needs to be self and other.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDivGetWorkspaceSize(const aclTensor* self,
                                               const aclTensor* other,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnDivs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivsGetWorkspaceSize(const aclTensor* self,
                                                const aclScalar* other,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnDivMod calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor* self,
                                                  const aclTensor* other,
                                                  int mode, aclTensor* out,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnDivMods calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor* self,
                                                   const aclScalar* other,
                                                   int mode, aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceDiv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivGetWorkspaceSize(aclTensor* selfRef,
                                                      const aclTensor* other,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceDivs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnInplaceDivMod calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivModGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, int mode,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnInplaceDivMods calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, int mode,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDiv is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDivGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDiv(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnDivs(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnDivMod(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnDivMods(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceDiv(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceDivs(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceDivMod(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceDivMods(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DIV_H_
