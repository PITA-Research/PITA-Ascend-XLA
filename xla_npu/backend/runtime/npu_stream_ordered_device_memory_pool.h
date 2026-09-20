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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_STREAM_ORDERED_DEVICE_MEMORY_POOL_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_STREAM_ORDERED_DEVICE_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"

namespace xla {

// A single-stream device arena for short-lived ACLNN intermediate buffers.
// Slices are never reused while work can still be in flight. All chunks become
// reusable only after ResetForStream is called following a successful stream
// drain. This removes per-allocation events and aclrtMalloc/aclrtFree traffic.
class NpuStreamOrderedDeviceMemoryPool {
 public:
  struct RetiredAclResources {
    // Device addresses referenced by this metadata group. Multiple metadata
    // groups and multiple tensor views may refer to the same address.
    std::vector<void*> device_ptrs;
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;
    std::vector<aclIntArray*> int_arrays;
  };

  static NpuStreamOrderedDeviceMemoryPool* GetInstance();

  absl::StatusOr<void*> AllocateForStream(aclrtStream stream, size_t size);

  // Keeps all host-side ACLNN metadata alive until the next successful stream
  // drain. No event is created for individual descriptors.
  absl::Status RetireAclResources(aclrtStream stream,
                                  RetiredAclResources&& resources);

  // Marks all slices reusable. The caller must have already synchronized the
  // stream successfully.
  void ResetForStream(aclrtStream stream);
  static void ResetForStreamIfInitialized(aclrtStream stream);

  NpuStreamOrderedDeviceMemoryPool(
      const NpuStreamOrderedDeviceMemoryPool&) = delete;
  NpuStreamOrderedDeviceMemoryPool& operator=(
      const NpuStreamOrderedDeviceMemoryPool&) = delete;

 private:
  struct Chunk {
    void* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
  };

  NpuStreamOrderedDeviceMemoryPool() = default;
  ~NpuStreamOrderedDeviceMemoryPool();

  absl::StatusOr<void*> AllocateLocked(aclrtStream stream, size_t size);
  absl::Status AddChunkLocked(size_t minimum_size);
  void DestroyRetiredAclResourcesLocked();
  void ResetLocked();

  static constexpr size_t kAlignment = 512;
  static constexpr size_t kInitialChunkSize = 64ULL << 20;
  static constexpr size_t kMaximumChunkSize = 512ULL << 20;
  static constexpr size_t kMaximumTotalCapacity = 2ULL << 30;

  std::mutex mutex_;
  std::vector<Chunk> chunks_;
  aclrtStream active_stream_ = nullptr;
  size_t current_chunk_ = 0;
  size_t total_capacity_ = 0;
  size_t current_bytes_ = 0;
  size_t high_watermark_ = 0;
  size_t drain_count_ = 0;
  std::vector<RetiredAclResources> retired_acl_resource_groups_;
  std::unordered_map<void*, std::vector<size_t>> metadata_groups_by_device_ptr_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_STREAM_ORDERED_DEVICE_MEMORY_POOL_H_
