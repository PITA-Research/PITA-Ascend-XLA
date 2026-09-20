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

#include "xla/xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "absl/strings/str_format.h"
#include "xla/status_macros.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace {

NpuStreamOrderedDeviceMemoryPool* g_pool = nullptr;
std::mutex g_pool_mutex;

size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

NpuStreamOrderedDeviceMemoryPool*
NpuStreamOrderedDeviceMemoryPool::GetInstance() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (g_pool == nullptr) {
    g_pool = new NpuStreamOrderedDeviceMemoryPool();
  }
  return g_pool;
}

void NpuStreamOrderedDeviceMemoryPool::ResetForStreamIfInitialized(
    aclrtStream stream) {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (g_pool != nullptr) g_pool->ResetForStream(stream);
}

NpuStreamOrderedDeviceMemoryPool::~NpuStreamOrderedDeviceMemoryPool() {
  DestroyRetiredAclResourcesLocked();
  for (const Chunk& chunk : chunks_) {
    if (chunk.base != nullptr) {
      aclError ret = dep::aclrtFree(chunk.base);
      if (ret != ACL_SUCCESS) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Failed to free ACLNN temporary pool chunk %p: %d",
                             chunk.base, ret);
      }
    }
  }
}

absl::Status NpuStreamOrderedDeviceMemoryPool::RetireAclResources(
    aclrtStream stream, RetiredAclResources&& resources) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "ACLNN metadata retirement requires a non-null stream");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_stream_ != nullptr && active_stream_ != stream) {
    return absl::FailedPreconditionError(
        "ACLNN metadata retirement used an unrelated stream");
  }
  active_stream_ = stream;
  const size_t group_index = retired_acl_resource_groups_.size();
  for (void* device_ptr : resources.device_ptrs) {
    if (device_ptr != nullptr) {
      metadata_groups_by_device_ptr_[device_ptr].push_back(group_index);
    }
  }
  retired_acl_resource_groups_.push_back(std::move(resources));
  return absl::OkStatus();
}

absl::StatusOr<void*> NpuStreamOrderedDeviceMemoryPool::AllocateForStream(
    aclrtStream stream, size_t size) {
  if (size == 0) return nullptr;
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "ACLNN temporary device allocation requires a non-null stream");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  return AllocateLocked(stream, size);
}

absl::StatusOr<void*> NpuStreamOrderedDeviceMemoryPool::AllocateLocked(
    aclrtStream stream, size_t size) {
  if (size > kMaximumTotalCapacity) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "ACLNN temporary allocation %zu exceeds pool maximum %zu", size,
        kMaximumTotalCapacity));
  }
  const size_t aligned_size = AlignUp(size, kAlignment);

  if (active_stream_ != nullptr && active_stream_ != stream) {
    return absl::FailedPreconditionError(
        "ACLNN temporary pool currently supports one active stream");
  }
  active_stream_ = stream;

  for (size_t i = current_chunk_; i < chunks_.size(); ++i) {
    Chunk& chunk = chunks_[i];
    if (aligned_size <= chunk.capacity - chunk.offset) {
      current_chunk_ = i;
      void* result = static_cast<char*>(chunk.base) + chunk.offset;
      chunk.offset += aligned_size;
      current_bytes_ += aligned_size;
      high_watermark_ = std::max(high_watermark_, current_bytes_);
      return result;
    }
  }

  if (total_capacity_ + aligned_size <= kMaximumTotalCapacity) {
    absl::Status add_status = AddChunkLocked(aligned_size);
    if (add_status.ok()) return AllocateLocked(stream, size);
    NPU_LOG_FORMAT_PRINT(
        WARN, "Growing ACLNN temporary pool failed (%s); draining stream",
        add_status.ToString().c_str());
  }

  // No slice can be reused safely while earlier ACLNN work is in flight.
  // At the configured cap (or on allocation pressure), drain once and reuse
  // all existing chunks instead of creating per-allocation events.
  aclError sync_ret = dep::aclrtSynchronizeStream(stream);
  if (sync_ret != ACL_SUCCESS) {
    return absl::InternalError(absl::StrFormat(
        "Failed to drain ACLNN stream before temporary pool reuse: %d",
        sync_ret));
  }
  ++drain_count_;
  ResetLocked();
  const bool has_fitting_chunk =
      std::any_of(chunks_.begin(), chunks_.end(),
                  [aligned_size](const Chunk& chunk) {
                    return aligned_size <= chunk.capacity;
                  });
  if (!has_fitting_chunk) {
    TF_RETURN_IF_ERROR(AddChunkLocked(aligned_size));
  }
  return AllocateLocked(stream, size);
}

absl::Status NpuStreamOrderedDeviceMemoryPool::AddChunkLocked(
    size_t minimum_size) {
  size_t preferred = chunks_.empty()
                         ? kInitialChunkSize
                         : std::min(chunks_.back().capacity * 2,
                                    kMaximumChunkSize);
  size_t capacity = AlignUp(std::max(preferred, minimum_size), kAlignment);
  if (capacity > kMaximumTotalCapacity - total_capacity_) {
    capacity = kMaximumTotalCapacity - total_capacity_;
  }
  if (capacity < minimum_size) {
    return absl::ResourceExhaustedError("ACLNN temporary pool reached its cap");
  }

  void* base = nullptr;
  aclError ret =
      dep::aclrtMalloc(&base, capacity, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS || base == nullptr) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "aclrtMalloc for ACLNN temporary pool chunk failed: size=%zu error=%d",
        capacity, ret));
  }
  chunks_.push_back(Chunk{base, capacity, 0});
  total_capacity_ += capacity;
  current_chunk_ = chunks_.size() - 1;
  NPU_LOG_FORMAT_PRINT(INFO,
                       "Added ACLNN temporary pool chunk: addr=%p size=%zu "
                       "total_capacity=%zu",
                       base, capacity, total_capacity_);
  return absl::OkStatus();
}

void NpuStreamOrderedDeviceMemoryPool::ResetForStream(aclrtStream stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_stream_ == nullptr) return;
  if (active_stream_ != stream) {
    NPU_LOG_FORMAT_PRINT(WARN,
                         "Ignoring ACLNN temporary pool reset for unrelated "
                         "stream %p (active=%p)",
                         stream, active_stream_);
    return;
  }
  ResetLocked();
  active_stream_ = nullptr;
}

void NpuStreamOrderedDeviceMemoryPool::ResetLocked() {
  DestroyRetiredAclResourcesLocked();
  for (Chunk& chunk : chunks_) chunk.offset = 0;
  current_chunk_ = 0;
  current_bytes_ = 0;
}

void NpuStreamOrderedDeviceMemoryPool::DestroyRetiredAclResourcesLocked() {
  std::unordered_set<aclTensor*> destroyed_tensors;
  std::unordered_set<aclScalar*> destroyed_scalars;
  std::unordered_set<aclIntArray*> destroyed_int_arrays;
  for (RetiredAclResources& resources : retired_acl_resource_groups_) {
    for (aclTensor* tensor : resources.tensors) {
      if (tensor != nullptr && destroyed_tensors.insert(tensor).second) {
        dep::aclDestroyTensor(tensor);
      }
    }
    for (aclScalar* scalar : resources.scalars) {
      if (scalar != nullptr && destroyed_scalars.insert(scalar).second) {
        dep::aclDestroyScalar(scalar);
      }
    }
    for (aclIntArray* array : resources.int_arrays) {
      if (array != nullptr && destroyed_int_arrays.insert(array).second) {
        dep::aclDestroyIntArray(array);
      }
    }
  }
  retired_acl_resource_groups_.clear();
  metadata_groups_by_device_ptr_.clear();
}

}  // namespace xla
