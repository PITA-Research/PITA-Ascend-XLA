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

#ifndef XLA_XLA_NPU_COMMON_NPU_OPERATION_H_
#define XLA_XLA_NPU_COMMON_NPU_OPERATION_H_

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {

// NPU hardware interactive interface encapsulation class
class NpuOperation {
 public:
  enum class MemcpyType { H2D, D2H };

  static absl::Status SetDevice(int device_id);
  static absl::Status CreateStream(aclrtStream* stream);
  static absl::Status Malloc(void** new_device_addr, size_t size_bytes);
  static absl::Status Memcpy(void* dst, void* src, size_t size_bytes,
                             MemcpyType memcpyType);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_OPERATION_H_