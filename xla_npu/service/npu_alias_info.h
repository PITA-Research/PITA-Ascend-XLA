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

#ifndef XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_
#define XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_

#include <optional>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

class NpuAliasInfo : public AliasInfo {
 public:
  explicit NpuAliasInfo(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  // Backend-specific may-alias hint. If an empty optional is returned, the
  // default rules in HloDataflowAnalysis are used. `operand` should be an
  // operand of `user`. `operand_index` should be the output index of `operand`,
  // `user_index` should be the output index of `user`.
  std::optional<bool> MayAlias(const HloInstruction* operand,
                               const ShapeIndex& operand_index,
                               const HloInstruction* user,
                               const ShapeIndex& user_index) const override;

 protected:
  // aclnnScatterNd reads its input and writes its output in separate passes
  // (it is NOT an in-place op). The default AliasInfo rules mark a standalone
  // (non-fusion) kScatter as in-place (output shares operand #0's buffer).
  // That lets BufferAssigner fold a scatter output into its operand's buffer,
  // which is then also liveness-reused by sibling scatters; when the scheduler
  // runs a later scatter (and its consumers) before this scatter's consumers,
  // the shared buffer gets overwritten before read -- the kinematics cliff
  // (see fzx_tmp/repro_subtract/HANDOFF.md §5.11). Returning an empty vector
  // (not nullopt) forces a dedicated output buffer for kScatter.
  // Default ON (scatter not in-place). NPU_SCATTER_LEGACY_INPLACE=1 restores
  // the old in-place behavior as a debug/safety valve.
  std::optional<std::vector<std::pair<HloOperandIndex, ShapeIndex>>>
  GetNonDefaultInPlaceInputOutputPairs(const HloInstruction* user) const override;

  se::DeviceDescription device_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_ALIAS_INFO_H_
