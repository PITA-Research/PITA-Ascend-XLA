/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_CUSTOM_OP_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_CUSTOM_OP_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_CUST_OPAPI(func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libcust_opapi, aclnnStatus, func_name, args_type, args_name)

#define DECLARE_ACL_EXEC_FUNC_IN_CUST_OPAPI(func_name)                  \
  DECLARE_ACL_FUNC_IN_CUST_OPAPI(                                       \
      func_name,                                                        \
      (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, \
       const aclrtStream stream),                                       \
      (workspace, workspaceSize, executor, stream))

namespace xla {
namespace dep {

DECLARE_ACL_FUNC_IN_CUST_OPAPI(
    aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize,
    (const aclTensor* lhs, const aclTensor* rhs, const aclTensor* out,
     uint64_t* workspaceSize, aclOpExecutor** executor),
    (lhs, rhs, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_CUST_OPAPI(aclnnDotFp32LeftOrderNoNeumaier)

DECLARE_ACL_FUNC_IN_CUST_OPAPI(
    aclnnFmaAddcmulGetWorkspaceSize,
    (const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
     const aclScalar* value, const aclTensor* out, uint64_t* workspaceSize,
     aclOpExecutor** executor),
    (self, tensor1, tensor2, value, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_CUST_OPAPI(aclnnFmaAddcmul)

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_CUSTOM_OP_INTERFACE_H_
