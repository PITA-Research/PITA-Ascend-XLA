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

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_NNOPBASE(ret, func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libnnopbase, ret, func_name, args_type, args_name)

namespace xla {
namespace dep {

DECLARE_ACL_FUNC_IN_NNOPBASE(aclTensor*, aclCreateTensor,
                             (const int64_t* viewDims, uint64_t viewDimsNum,
                              aclDataType dataType, const int64_t* stride,
                              int64_t offset, aclFormat format,
                              const int64_t* storageDims,
                              uint64_t storageDimsNum, void* tensorData),
                             (viewDims, viewDimsNum, dataType, stride, offset,
                              format, storageDims, storageDimsNum, tensorData))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclScalar*, aclCreateScalar,
                             (void* value, aclDataType dataType),
                             (value, dataType))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclDestroyScalar,
                             (const aclScalar* scalar), (scalar))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclIntArray*, aclCreateIntArray,
                             (const int64_t* value, uint64_t size),
                             (value, size))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclTensorList*, aclCreateTensorList,
                             (const aclTensor* const* value, uint64_t size),
                             (value, size))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclDestroyTensorList,
                             (const aclTensorList* array), (array))

DECLARE_ACL_FUNC_IN_NNOPBASE(void, aclDestroyIntArray,
                             (const aclIntArray* array), (array))

DECLARE_ACL_FUNC_IN_NNOPBASE(void, aclDestroyTensor, (const aclTensor* tensor),
                             (tensor))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclDestroyAclOpExecutor,
                             (aclOpExecutor* executor), (executor))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclSetAclOpExecutorRepeatable,
                             (aclOpExecutor* executor), (executor))

DECLARE_ACL_FUNC_IN_NNOPBASE(
    aclnnStatus, aclSetDynamicInputTensorAddr,
    (aclOpExecutor* executor, size_t ir_index, size_t relative_index,
     aclTensorList* tensors, void* addr),
    (executor, ir_index, relative_index, tensors, addr))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclnnStatus, aclSetOutputTensorAddr,
                             (aclOpExecutor* executor, size_t index,
                              aclTensor* tensor, void* addr),
                             (executor, index, tensor, addr))

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_INTERFACE_H_
