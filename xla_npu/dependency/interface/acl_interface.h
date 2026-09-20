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

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl/acl.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_ASCENDCL(func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libascendcl, aclError, func_name, args_type, args_name)

namespace xla {
namespace dep {

DECLARE_ACL_FUNC_IN_ASCENDCL(aclInit, (const char* configPath), (configPath))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclFinalize, (), ())

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtGetDeviceCount, (uint32_t * count), (count))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtGetDevice, (uint32_t * device_id),
                             (device_id))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtSetDevice, (uint32_t device_id), (device_id))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtCreateStream, (aclrtStream * stream),
                             (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtDestroyStream, (aclrtStream stream), (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtSynchronizeStream, (aclrtStream stream),
                             (stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtCreateEvent, (aclrtEvent * event), (event))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtDestroyEvent, (aclrtEvent event), (event))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtRecordEvent,
                             (aclrtEvent event, aclrtStream stream),
                             (event, stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtSynchronizeEvent, (aclrtEvent event),
                             (event))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMalloc,
                             (void** devPtr, size_t size,
                              aclrtMemMallocPolicy policy),
                             (devPtr, size, policy))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtFree, (void* devPtr), (devPtr))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMemcpy,
                             (void* dst, size_t destMax, const void* src,
                              size_t count, aclrtMemcpyKind kind),
                             (dst, destMax, src, count, kind))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMemcpyAsync,
                             (void* dst, size_t destMax, const void* src,
                              size_t count, aclrtMemcpyKind kind,
                              aclrtStream stream),
                             (dst, destMax, src, count, kind, stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMemset,
                             (void* devPtr, size_t maxCount, int32_t value,
                              size_t count),
                             (devPtr, maxCount, value, count))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtMallocHost, (void** hostPtr, size_t size),
                             (hostPtr, size))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtFreeHost, (void* hostPtr), (hostPtr))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtBinaryLoadFromFile,
                             (const char* binPath,
                              aclrtBinaryLoadOptions* options,
                              aclrtBinHandle* binHandle),
                             (binPath, options, binHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtBinaryLoadFromData,
                             (const void* data, size_t length,
                              const aclrtBinaryLoadOptions* options,
                              aclrtBinHandle* binHandle),
                             (data, length, options, binHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtBinaryGetFunction,
                             (const aclrtBinHandle binHandle,
                              const char* kernelName,
                              aclrtFuncHandle* funcHandle),
                             (binHandle, kernelName, funcHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtKernelArgsInit,
                             (aclrtFuncHandle funcHandle,
                              aclrtArgsHandle* argsHandle),
                             (funcHandle, argsHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtKernelArgsAppend,
                             (aclrtArgsHandle argsHandle, void* param,
                              size_t paramSize, aclrtParamHandle* paramHandle),
                             (argsHandle, param, paramSize, paramHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtKernelArgsAppendPlaceHolder,
                             (aclrtArgsHandle argsHandle,
                              aclrtParamHandle* paramHandle),
                             (argsHandle, paramHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtKernelArgsGetPlaceHolderBuffer,
                             (aclrtArgsHandle argsHandle,
                              aclrtParamHandle paramHandle, size_t dataSize,
                              void** bufferAddr),
                             (argsHandle, paramHandle, dataSize, bufferAddr))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtKernelArgsFinalize,
                             (aclrtArgsHandle argsHandle), (argsHandle))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtLaunchKernel,
                             (aclrtFuncHandle funcHandle, uint32_t blockDim,
                              const void* argsData, size_t argsSize,
                              aclrtStream stream),
                             (funcHandle, blockDim, argsData, argsSize, stream))

DECLARE_ACL_FUNC_IN_ASCENDCL(aclrtLaunchKernelWithConfig,
                             (aclrtFuncHandle funcHandle, uint32_t blockDim,
                              aclrtStream stream, aclrtLaunchKernelCfg* cfg,
                              aclrtArgsHandle argsHandle, void* reserve),
                             (funcHandle, blockDim, stream, cfg, argsHandle,
                              reserve))

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_INTERFACE_H_
