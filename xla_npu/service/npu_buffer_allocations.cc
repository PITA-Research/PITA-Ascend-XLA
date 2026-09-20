/* Copyright 2019 The OpenXLA Authors.
Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file is modified from OpenXLA source code.
Modification: Custom adaptation for NPU/XLA-NPU scenarios.
==============================================================================*/

#include "xla/xla_npu/service/npu_buffer_allocations.h"

#include <cstdint>
#include <set>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::Status NpuBufferAllocations::TearDown(
    const std::set<se::DeviceMemoryBase>& live_addresses,
    absl::Span<const BufferAllocation* const> allocations) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  absl::Status status;
  const int64_t num_buffers = allocations.size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    se::DeviceMemoryBase buffer_address = GetDeviceAddress(allocation.index());

    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "TearDown: allocation %d, is_entry_param=%d, maybe_live_out=%d, "
        "in_live_addresses=%d, is_temp=%d, addr=%p",
        i, allocation.is_entry_computation_parameter(),
        allocation.maybe_live_out(),
        static_cast<int>(live_addresses.count(buffer_address)),
        allocation.IsPreallocatedTempBuffer(), buffer_address.opaque());

    // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
    // and temp buffers.
    //
    // Only deallocate if:
    // 1. It is NOT an entry computation parameter (entry parameters are owned by caller)
    // 2. AND either
    //    a. It's maybe_live_out but not actually live out (not in live_addresses), OR
    //    b. It's a preallocated temporary buffer
    //
    // This is more conservative: if the buffer is an entry computation parameter
    // (i.e., input to current computation), we NEVER deallocate it even if
    // it's also marked maybe_live_out (this can happen when the output from a
    // previous computation is reused as input to the current computation).
    if (!allocation.is_entry_computation_parameter() &&
        allocation.IsPreallocatedTempBuffer()) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "TearDown: deallocating allocation %d, addr=%p", i,
                           buffer_address.opaque());
      auto dealloc_result =
          memory_allocator_->Deallocate(device_ordinal_, buffer_address);
      if (!dealloc_result.ok() && status.ok()) {
        status = dealloc_result;
      }
    } else {
      NPU_LOG_FORMAT_PRINT(DEBUG, "TearDown: skipping allocation %d, addr=%p",
                           i, buffer_address.opaque());
    }
  }
  return status;
}

se::DeviceMemoryBase NpuBufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase& NpuBufferAllocations::GetMutableDeviceAddress(
    BufferAllocation::Index buffer_index) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase NpuBufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  int64_t index = buffer_slice.index();
  se::DeviceMemoryBase base = GetDeviceAddress(index);

  int64_t offset = buffer_slice.offset();
  CHECK_LE(buffer_slice.offset(), base.size())
      << "slice offset " << offset << " must be smaller than buffer #" << index
      << " size " << base.size();

  int64_t extent = offset + buffer_slice.size();
  CHECK_LE(extent, base.size())
      << "slice extent " << extent << " must be smaller than buffer #" << index
      << " size " << base.size();

  return base.GetByteSlice(buffer_slice.offset(), buffer_slice.size());
}

}  // namespace xla
