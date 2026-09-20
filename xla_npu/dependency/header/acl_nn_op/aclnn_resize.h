/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef OP_API_INC_RESIZE_H_
#define OP_API_INC_RESIZE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnResize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: adjust the size of the input tensor
 * @param [in] self Input Tensor, data type supports FLOAT, FLOAT16. Supports non-continuous Tensor, and the data format supports NCHW.
 * @param [in] scales input aclFloatArray, the data type supports INT32.
 * @param [in] mode input attribute, data type supports const char*
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnResizeGetWorkspaceSize(
    const aclTensor *self, const aclFloatArray *scales, const char *mode,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnResize is used to perform calculations.
 * @domain aclnn_ops_infer
 *
 * Operator function: adjust the size of the input tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnResizeGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnResize(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RESIZE_H_