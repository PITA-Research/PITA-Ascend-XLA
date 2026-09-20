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

#include "xla/service/buffer_assignment.h"

#include "xla/xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h"

namespace xla {

const BufferAllocation& NpuBufferAssignment::GetAllocation(
    BufferAllocation::Index index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, allocations_.size());
  return allocations_[index];
}

BufferAllocation* NpuBufferAssignment::NewEmptyAllocation(
    int64_t size, LogicalBuffer::Color color) {
  BufferAllocation::Index index = allocations_.size();
  allocations_.emplace_back(index, size, color);
  BufferAllocation* allocation = &allocations_.back();
  return allocation;
}

void NpuBufferAssignment::Finalize() {}

}  // namespace xla