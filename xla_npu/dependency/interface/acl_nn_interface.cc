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

#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace xla {
namespace dep {

REGISTER_LIBRARY(libnnopbase)
REGISTER_FUNCTION(libnnopbase, aclCreateTensor)
REGISTER_FUNCTION(libnnopbase, aclCreateScalar)
REGISTER_FUNCTION(libnnopbase, aclCreateIntArray)
REGISTER_FUNCTION(libnnopbase, aclCreateTensorList)
REGISTER_FUNCTION(libnnopbase, aclDestroyTensorList)
REGISTER_FUNCTION(libnnopbase, aclDestroyScalar)
REGISTER_FUNCTION(libnnopbase, aclDestroyIntArray)
REGISTER_FUNCTION(libnnopbase, aclDestroyTensor)
REGISTER_FUNCTION(libnnopbase, aclDestroyAclOpExecutor)
REGISTER_FUNCTION(libnnopbase, aclSetAclOpExecutorRepeatable)
REGISTER_FUNCTION(libnnopbase, aclSetDynamicInputTensorAddr)
REGISTER_FUNCTION(libnnopbase, aclSetOutputTensorAddr)

}  // namespace dep
}  // namespace xla
