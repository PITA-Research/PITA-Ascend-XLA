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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_COPY_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_COPY_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// CopyFusion-like pass for NPU.
// CopyFusion checks if a fusion is followed by multiple copies and if so, adds
// those copies to the fusion, replacing the copies with get_tuple_elements.
class NpuCopyFusion : public HloModulePass {
 public:
  explicit NpuCopyFusion(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  absl::string_view name() const override { return "npu-copy-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::Status FuseCopyInstructionsIntoComputation(
      HloInstruction* hlo, HloComputation* fused_computation,
      const std::vector<HloInstruction*>& copies,
      const std::vector<HloInstruction*>& other_users);

  std::pair<std::vector<HloInstruction*>, std::vector<HloInstruction*>>
  ClassifyUsers(HloInstruction* hlo);

  absl::StatusOr<bool> DoCopyFusion(HloComputation* computation);

  const se::DeviceDescription& device_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_COPY_FUSION_H_
