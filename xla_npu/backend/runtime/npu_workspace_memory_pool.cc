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

#include "xla/xla_npu/backend/runtime/npu_workspace_memory_pool.h"

#include <algorithm>
#include <cstring>

#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {
NpuWorkspaceMemoryPool* g_instance = nullptr;
std::mutex g_instance_mutex;
}  // namespace

NpuWorkspaceMemoryPool* NpuWorkspaceMemoryPool::GetInstance() {
  std::lock_guard<std::mutex> lock(g_instance_mutex);
  if (g_instance == nullptr) {
    g_instance = new NpuWorkspaceMemoryPool();
    absl::Status status = g_instance->Initialize();
    if (!status.ok()) {
      NPU_LOG_FORMAT_PRINT(ERROR,
                           "Failed to initialize NpuWorkspaceMemoryPool: %s",
                           status.ToString().c_str());
      delete g_instance;
      g_instance = nullptr;
    }
  }
  return g_instance;
}

void NpuWorkspaceMemoryPool::DestroyInstance() {
  std::lock_guard<std::mutex> lock(g_instance_mutex);
  if (g_instance != nullptr) {
    delete g_instance;
    g_instance = nullptr;
  }
}

void NpuWorkspaceMemoryPool::ResetForStreamIfInitialized(aclrtStream stream) {
  std::lock_guard<std::mutex> lock(g_instance_mutex);
  if (g_instance != nullptr) {
    g_instance->ResetForStream(stream);
  }
}

NpuWorkspaceMemoryPool::~NpuWorkspaceMemoryPool() {
  if (device_memory_base_ != nullptr) {
    aclError ret = dep::aclrtFree(device_memory_base_);
    if (ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Failed to free workspace memory pool: %d",
                           ret);
    } else {
      NPU_LOG_FORMAT_PRINT(INFO, "Workspace memory pool freed successfully");
    }
    device_memory_base_ = nullptr;
  }
  initialized_ = false;
}

absl::Status NpuWorkspaceMemoryPool::Initialize() {
  if (initialized_) {
    return absl::OkStatus();
  }

  for (size_t i = 0; i < kMaxStreamCount; ++i) {
    stream_mappings_[i].stream_id = 0;
    stream_mappings_[i].block_index = i;
    stream_mappings_[i].next_offset = 0;
    stream_mappings_[i].active = false;
  }
  stream_mapping_count_ = 0;

  aclError ret =
      dep::aclrtMalloc(&device_memory_base_, kWorkspaceMemoryPoolTotalSize,
                       ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate workspace memory pool of size %zu bytes: %d",
        kWorkspaceMemoryPoolTotalSize, ret));
  }

  initialized_ = true;
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Workspace memory pool initialized: total=%zu MB, per_stream=%zu MB, "
      "max_streams=%zu",
      kWorkspaceMemoryPoolTotalSize / (1024 * 1024),
      kWorkspaceMemoryPerStream / (1024 * 1024), kMaxStreamCount);

  return absl::OkStatus();
}

uint64_t NpuWorkspaceMemoryPool::GetStreamId(aclrtStream stream) const {
  return reinterpret_cast<uint64_t>(stream);
}

absl::StatusOr<size_t> NpuWorkspaceMemoryPool::GetOrAssignStreamIndex(
    aclrtStream stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t stream_id = GetStreamId(stream);

  for (size_t i = 0; i < stream_mapping_count_; ++i) {
    if (stream_mappings_[i].stream_id == stream_id &&
        stream_mappings_[i].active) {
      return stream_mappings_[i].block_index;
    }
  }

  if (stream_mapping_count_ >= kMaxStreamCount) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Maximum number of streams (%zu) exceeded", kMaxStreamCount));
  }

  size_t block_index = stream_mapping_count_;
  stream_mappings_[stream_mapping_count_].stream_id = stream_id;
  stream_mappings_[stream_mapping_count_].block_index = block_index;
  stream_mappings_[stream_mapping_count_].next_offset = 0;
  stream_mappings_[stream_mapping_count_].active = true;
  ++stream_mapping_count_;

  NPU_LOG_FORMAT_PRINT(DEBUG, "Assigned memory block %zu to stream %p",
                       block_index, stream);

  return block_index;
}

absl::StatusOr<void*> NpuWorkspaceMemoryPool::AllocateForStream(
    aclrtStream stream, size_t size) {
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Workspace memory pool not initialized");
  }

  if (size == 0) {
    return nullptr;
  }

  size_t aligned_size =
      (size + kWorkspaceMemoryAlignment - 1) & ~(kWorkspaceMemoryAlignment - 1);

  if (aligned_size > kWorkspaceMemoryPerStream) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Requested workspace size %zu exceeds per-stream limit %zu",
        aligned_size, kWorkspaceMemoryPerStream));
  }

  TF_ASSIGN_OR_RETURN(auto block_index, GetOrAssignStreamIndex(stream));

  size_t offset = 0;
  bool must_drain = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    StreamMapping& mapping = stream_mappings_[block_index];
    if (aligned_size > kWorkspaceMemoryPerStream - mapping.next_offset) {
      must_drain = true;
    } else {
      offset = mapping.next_offset;
      mapping.next_offset += aligned_size;
    }
  }

  // ACLNN's workspace setup can perform a host-side synchronous memcpy that
  // is not ordered against an already queued kernel. Never reuse a slice
  // until the stream is drained. Hitting the per-stream bump limit is the one
  // safe point where we reclaim all earlier slices.
  if (must_drain) {
    NPU_LOG_FORMAT_PRINT(
        WARN,
        "Workspace bump block exhausted for stream %p; draining before reuse",
        stream);
    aclError ret = dep::aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      return absl::InternalError(absl::StrFormat(
          "Failed to synchronize stream before workspace reuse: %d", ret));
    }
    ResetForStream(stream);
    return AllocateForStream(stream, size);
  }

  void* stream_memory = static_cast<char*>(device_memory_base_) +
                        block_index * kWorkspaceMemoryPerStream + offset;

  NPU_LOG_FORMAT_PRINT(
      DEBUG,
      "Allocated workspace for stream %p: block=%zu, size=%zu, "
      "aligned_size=%zu, offset=%zu, addr=%p",
      stream, block_index, size, aligned_size, offset, stream_memory);

  return stream_memory;
}

absl::StatusOr<void*> NpuWorkspaceMemoryPool::GetStreamMemoryBase(
    aclrtStream stream) {
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Workspace memory pool not initialized");
  }

  TF_ASSIGN_OR_RETURN(auto block_index, GetOrAssignStreamIndex(stream));

  void* stream_memory = static_cast<char*>(device_memory_base_) +
                        block_index * kWorkspaceMemoryPerStream;

  return stream_memory;
}

size_t NpuWorkspaceMemoryPool::GetStreamMemorySize(aclrtStream stream) const {
  (void)stream;
  return kWorkspaceMemoryPerStream;
}

void NpuWorkspaceMemoryPool::ResetForStream(aclrtStream stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t stream_id = GetStreamId(stream);
  for (size_t i = 0; i < stream_mapping_count_; ++i) {
    if (stream_mappings_[i].stream_id == stream_id &&
        stream_mappings_[i].active) {
      stream_mappings_[i].next_offset = 0;
      NPU_LOG_FORMAT_PRINT(DEBUG, "Reset workspace bump offset for stream %p",
                           stream);
      return;
    }
  }
}

void NpuWorkspaceMemoryPool::ReleaseForStream(aclrtStream stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t stream_id = GetStreamId(stream);

  for (size_t i = 0; i < stream_mapping_count_; ++i) {
    if (stream_mappings_[i].stream_id == stream_id &&
        stream_mappings_[i].active) {
      stream_mappings_[i].active = false;
      stream_mappings_[i].next_offset = 0;
      NPU_LOG_FORMAT_PRINT(DEBUG, "Released memory block %zu for stream %p",
                           stream_mappings_[i].block_index, stream);
      return;
    }
  }
}

}  // namespace xla
