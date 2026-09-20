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

#ifndef XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_DESCRIPTION_H_
#define XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_DESCRIPTION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"

#include "xla/xla_npu/pjrt/plugin/npu_topology.h"

namespace xla {

class NpuTopologyDescription : public PjRtTopologyDescription {
 public:
  // `npu_device_ids` is the list of logical device ids for the NPU devices and
  // will be used to initialize the NPU topology.
  NpuTopologyDescription(const PjRtPlatformId platform_id,
                         const absl::string_view platform_name,
                         const absl::string_view platform_version,
                         std::vector<NpuTopology::NpuDevice> npu_devices,
                         absl::Span<const std::string> machine_attributes)
      : platform_id_(platform_id),
        platform_name_(platform_name),
        platform_version_(platform_version),
        npu_topology_(std::move(npu_devices),
                      std::vector<std::string>(machine_attributes.begin(),
                                               machine_attributes.end())) {}

  bool operator==(const NpuTopologyDescription& other) const {
    return this->platform_id() == other.platform_id() &&
           this->platform_name() == other.platform_name() &&
           this->platform_version() == other.platform_version() &&
           this->npu_topology().devices() == other.npu_topology().devices();
  }

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override {
    return platform_version_;
  }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  const NpuTopology& npu_topology() const { return npu_topology_; }
  const NpuTopology* npu_topology_ptr() const { return &npu_topology_; }

  // No subslice is supported.
  bool is_subslice_topology() const override { return false; }

  absl::StatusOr<int> ProcessCount() const override { return 1; }

  absl::StatusOr<int> ChipsPerProcess() const override {
    return npu_topology_.number_of_devices();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<std::string> Serialize() const override;

  absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
  LogicalDeviceOfDefaultTypeForId(
      xla::PjRtGlobalDeviceId device_id) const override;

  // Returns vendor specific attributes about the topology.
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

  absl::StatusOr<xla::PjRtTopologyDescriptionProto> ToProto() const override;

 private:
  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  const std::string platform_version_;
  const NpuTopology npu_topology_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_PLUGIN_NPU_TOPOLOGY_DESCRIPTION_H_