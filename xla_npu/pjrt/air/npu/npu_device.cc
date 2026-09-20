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

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"

#include "xla/xla_npu/pjrt/air/npu/npu_device.h"

namespace xla {

PjRtNpuDevice::PjRtNpuDevice(int id, int process_id)
    : description_(id, process_id) {}

void PjRtNpuDevice::AttachMemorySpace(PjRtMemorySpace* memory_space) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a PjRtNpuDevice to a PjRtMemorySpace owned by a "
      "different client, the device's client: %s, the memory space's client: "
      "%s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_id_.emplace(memory_space->kind_id(), memory_space);
}

absl::Span<PjRtMemorySpace* const> PjRtNpuDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> PjRtNpuDevice::default_memory_space() const {
  if (memory_spaces_.empty()) {
    return absl::FailedPreconditionError(
        "PjRtNpuDevice::default_memory_space(): No memory space found.");
  }
  return memory_spaces_.front();
}

absl::StatusOr<PjRtMemorySpace*> PjRtNpuDevice::memory_space_by_kind(
    absl::string_view memory_space_kind) const {
  auto it =
      absl::c_find_if(memory_spaces_, [memory_space_kind](PjRtMemorySpace* ms) {
        return ms->kind() == memory_space_kind;
      });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", memory_space_kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*> PjRtNpuDevice::memory_space_by_kind_id(
    int id) const {
  auto it = memory_spaces_by_id_.find(id);
  if (it == memory_spaces_by_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<bool> PjRtNpuDevice::PoisonExecution(int32_t launch_id,
                                                    absl::Status error) {
  return false;
}

}  // namespace xla
