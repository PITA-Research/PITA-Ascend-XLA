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

#include "xla/xla_npu/common/npu_operation.h"

#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::Status NpuOperation::SetDevice(int device_id) {
  aclError ret = dep::aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Set device %d for '%s' failed, error code: %d",
                         device_id, NpuConstant::platformName, ret);
    return absl::InternalError(
        absl::StrFormat("Set device %d failed.", device_id));
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Set device %d for '%s' success.", device_id,
                       NpuConstant::platformName);

  return absl::OkStatus();
}

absl::Status NpuOperation::CreateStream(aclrtStream* stream) {
  aclError ret = dep::aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Create stream for '%s' failed. Error code: %d",
                         NpuConstant::platformName, ret);
    return absl::InternalError(absl::StrFormat("Create stream failed."));
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Create stream for '%s' success.",
                       NpuConstant::platformName);

  return absl::OkStatus();
}

absl::Status NpuOperation::Malloc(void** new_device_addr, size_t size_bytes) {
  // Ascend aclrtMalloc does not support zero-size allocations.
  if (size_bytes == 0) {
    *new_device_addr = nullptr;
    return absl::OkStatus();
  }

  aclError ret =
      dep::aclrtMalloc(new_device_addr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR,
                         "Allocate %d byte on npu failed, error code: %d",
                         size_bytes, ret);
    return absl::InternalError(absl::StrFormat(
        "Allocate %d byte on npu failed, error code: %d", size_bytes, ret));
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory allocated, addr %p, size %d",
                       *new_device_addr, size_bytes);

  return absl::OkStatus();
}

absl::Status NpuOperation::Memcpy(void* dst, void* src, size_t size_bytes,
                                  MemcpyType memcpyType) {
  // Zero-size memcpy is a no-op.
  if (size_bytes == 0) {
    return absl::OkStatus();
  }

  aclrtMemcpyKind memcpyKind = aclrtMemcpyKind::ACL_MEMCPY_DEFAULT;
  std::string descStr = "";

  switch (memcpyType) {
    case MemcpyType::H2D:
      memcpyKind = ACL_MEMCPY_HOST_TO_DEVICE;
      descStr = "Copy date from cpu to npu";
      break;
    case MemcpyType::D2H:
      memcpyKind = ACL_MEMCPY_DEVICE_TO_HOST;
      descStr = "Copy date from npu to cpu";
      break;
  };

  aclError ret = dep::aclrtMemcpy(dst, size_bytes, src, size_bytes, memcpyKind);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(
        ERROR,
        "%s failed, dst addr %p, src addr %p, size bytes %d, error code: %d",
        descStr, dst, src, size_bytes, ret);
    return absl::InternalError("Acl memory copy failed.");
  }

  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "%s success, dst addr %p, src addr %p, size bytes %d",
                       descStr, dst, src, size_bytes);

  return absl::OkStatus();
}

}  // namespace xla