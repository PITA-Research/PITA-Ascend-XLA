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

#ifndef XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_
#define XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"

namespace xla {
class NpuTopology {
 public:
  struct NpuDevice {
    int process_id;
    int local_device_id;

    bool operator==(const NpuDevice& other) const {
      return process_id == other.process_id &&
             local_device_id == other.local_device_id;
    }
  };

  explicit NpuTopology(std::vector<NpuDevice> npu_devices,
                       std::vector<std::string> machine_attributes)
      : npu_devices_(std::move(npu_devices)),
        machine_attributes_(std::move(machine_attributes)) {}

  int number_of_devices() const { return npu_devices_.size(); }
  absl::Span<const NpuDevice> devices() const { return npu_devices_; }
  absl::Span<const std::string> machine_attributes() const {
    return machine_attributes_;
  }

 private:
  const std::vector<NpuDevice> npu_devices_;
  const std::vector<std::string> machine_attributes_;
};

static const int kMaxNpuDevicesPerProcess = 1 << 11;

inline PjRtGlobalDeviceId PackCpuDeviceId(int process_index, int device_id) {
  return PjRtGlobalDeviceId(kMaxNpuDevicesPerProcess * process_index +
                            device_id);
}

inline int UnpackCpuProcessIndex(PjRtGlobalDeviceId global_device_id) {
  return global_device_id.value() / kMaxNpuDevicesPerProcess;
}

inline int UnpackCpuLocalDeviceId(PjRtGlobalDeviceId global_device_id) {
  return global_device_id.value() % kMaxNpuDevicesPerProcess;
}

}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_H_
