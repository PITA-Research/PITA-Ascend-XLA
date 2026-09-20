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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {

// Workspace memory pool constant definition
// All sizes are in bytes

// Total memory pool size: 1GB
inline constexpr size_t kWorkspaceMemoryPoolTotalSize =
    1ULL * 1024 * 1024 * 1024;

// Memory block size allocated per stream: 1GB
inline constexpr size_t kWorkspaceMemoryPerStream = 1ULL * 1024 * 1024 * 1024;

// Maximum number of concurrent streams supported (only one stream is currently supported)
inline constexpr size_t kMaxStreamCount =
    kWorkspaceMemoryPoolTotalSize / kWorkspaceMemoryPerStream;

// Memory pool alignment requirement: 32-byte alignment
inline constexpr size_t kWorkspaceMemoryAlignment = 32;

/**
 * NpuWorkspaceMemoryPool - Workspace memory pool manager
 *
 * 设计目标:
 * 1. 预分配 1GB 设备内存作为 workspace 专用内存池
 * 2. 按流划分内存块,每个流分配 1GB
 * 3. 同一个流内的算子从 bump allocator 获取独立切片；只有 stream
 *    drain 后才复用先前切片
 * 4. 线程安全,支持多流并发访问
 *
 * How to use:
 *   auto* pool = NpuWorkspaceMemoryPool::GetInstance();
 *   TF_ASSIGN_OR_RETURN(void* workspace, pool->AllocateForStream(stream, size));
 * //Use workspace to execute the operator
 * pool->ReleaseForStream(stream); // Optional, called when the stream ends
 */
class NpuWorkspaceMemoryPool {
 public:
  // Get singleton instance
  static NpuWorkspaceMemoryPool* GetInstance();

  // Copying and moving prohibited
  NpuWorkspaceMemoryPool(const NpuWorkspaceMemoryPool&) = delete;
  NpuWorkspaceMemoryPool& operator=(const NpuWorkspaceMemoryPool&) = delete;

  /**
   * Allocate workspace memory for the specified stream
   *
   * @param stream ACL stream handle
   * @param size required memory size (bytes)
   * @return The allocated memory address. If the memory block size of the stream is exceeded, an error will be returned.
   */
  absl::StatusOr<void*> AllocateForStream(aclrtStream stream, size_t size);

  /**
   * Get the memory block base address of the specified stream
   * Scenarios for directly managing memory offsets
   *
   * @param stream ACL stream handle
   * @return the memory block base address of the stream
   */
  absl::StatusOr<void*> GetStreamMemoryBase(aclrtStream stream);

  // Marks all workspace slices previously handed to this stream reusable.
  // The caller must have drained `stream` before invoking this method.
  void ResetForStream(aclrtStream stream);

  // Same as ResetForStream, but does not instantiate the pool solely for a
  // stream synchronization that did not use any ACLNN workspace.
  static void ResetForStreamIfInitialized(aclrtStream stream);

  /**
   * Get the memory block size of the specified stream
   *
   * @param stream ACL stream handle
   * @return The memory block size of the stream (bytes)
   */
  size_t GetStreamMemorySize(aclrtStream stream) const;

  /**
   * Release the memory block occupied by the specified stream
   * Usually called at the end of the stream, the memory block will be marked as reusable.
   *
   * @param stream ACL stream handle
   */
  void ReleaseForStream(aclrtStream stream);

  /**
   * Get the total size of the memory pool
   */
  size_t GetTotalSize() const { return kWorkspaceMemoryPoolTotalSize; }

  /**
   * Get the memory block size of each stream
   */
  size_t GetPerStreamSize() const { return kWorkspaceMemoryPerStream; }

  /**
   * Check if the memory pool has been initialized
   */
  bool IsInitialized() const { return initialized_; }

  /**
   * Destroy the memory pool and release all device memory
   * Usually called when the program exits
   */
  static void DestroyInstance();

 private:
  NpuWorkspaceMemoryPool() = default;
  ~NpuWorkspaceMemoryPool();

  // Initialize the memory pool and allocate device memory
  absl::Status Initialize();

  // Get the unique identifier of the stream
  uint64_t GetStreamId(aclrtStream stream) const;

  // Gets or allocates the memory block index of the stream
  absl::StatusOr<size_t> GetOrAssignStreamIndex(aclrtStream stream);

  // Whether the memory pool has been initialized
  bool initialized_ = false;

  // Device memory base address
  void* device_memory_base_ = nullptr;

  // Stream index mapping: stream id -> memory block index
  // Use fixed size arrays to avoid dynamic allocation
  struct StreamMapping {
    uint64_t stream_id;
    size_t block_index;
    size_t next_offset;
    bool active;
  };
  StreamMapping stream_mappings_[kMaxStreamCount];
  size_t stream_mapping_count_ = 0;

  // Mutex lock protecting stream index mapping
  mutable std::mutex mutex_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WORKSPACE_MEMORY_POOL_H_
