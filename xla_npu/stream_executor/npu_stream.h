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

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/common/npu_operation.h"

namespace stream_executor {

// NpuStream for launching work on the host NPU. In contrast to the device
// streams, NpuStream is fully synchronous and launches all operations in the
// caller thread.
class NpuStream : public StreamCommon {
 public:
  explicit NpuStream(StreamExecutor* executor);
  ~NpuStream() override;

  absl::Status Init();

  PlatformSpecificHandle platform_specific_handle() const override;

  absl::Status BlockHostUntilDone() override;

  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override;
  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* npu_dst,
                      const DeviceMemoryBase& npu_src, uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* npu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceMemoryBase& npu_src,
                      uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

 private:
  int device_ordinal_;
  aclrtStream stream_;  // aclrtStream = void*
  bool stream_init_ = false;
};

}  // namespace stream_executor

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_H_
