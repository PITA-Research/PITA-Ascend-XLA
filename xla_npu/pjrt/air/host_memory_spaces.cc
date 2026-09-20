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

#include "tsl/platform/fingerprint.h"

#include "host_memory_spaces.h"

namespace xla {

NpuDeviceMemorySpace::NpuDeviceMemorySpace(int id, PjRtDevice* device)
    : id_(id), device_(device) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  auto* client = device_->client();
  debug_string_ =
      absl::StrFormat("NpuDeviceMemory(id=%i, process_index=%i, client=%s)",
                      id_, client->process_index(), client->platform_name());
  to_string_ = absl::StrFormat("NPU_DEVICE_%i", id_);
}

const int NpuDeviceMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(NpuDeviceMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

}  // namespace xla