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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// Convert existing loop/input fusions to NPU block-level fusions
// This PASS converts the loop/input fusion of the NPU backend to a block-level fusion, using the "__npu_block" kind.
// Block-level fusion enables better utilization of the NPU's unified buffer (UB) memory hierarchy and block-level parallel computing.
// Conversion example: Before: fusion with kind="kLoop" that operates elementwise
//          After: fusion with kind="__npu_block" with block-level tiling config
// Block-level configuration includes: - Output tile sizes (output_tile_sizes)
//             - Number of blocks (num_blocks)
//             - Number of stages (num_stages) - used for pipeline optimization
class NpuFusionBlockLevelRewriter : public HloModulePass {
 public:
  explicit NpuFusionBlockLevelRewriter(
      const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  ~NpuFusionBlockLevelRewriter() override = default;

  absl::string_view name() const override {
    return "npu-fusion-block-level-rewriter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Check if a fusion can be converted to block-level
  bool CanConvertToBlockLevel(const HloInstruction* fusion);

  // Run on a single computation
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  // Convert a loop/input fusion to block-level fusion
  absl::StatusOr<bool> ConvertFusionToBlockLevel(HloFusionInstruction* fusion);

  // Compute optimal tile sizes for the fusion
  absl::StatusOr<std::vector<int64_t>> ComputeTileSizes(
      const HloInstruction* fusion);

  // Check if the fusion contains only elementwise operations
  bool ContainsOnlyElementwiseOps(HloComputation* fusion);

  // Check if a fusion is safe for the current AFIR blocking autofuse path.
  bool ContainsOnlyAutofuseSafeOps(HloComputation* fusion);

  // Merge single-user elementwise producer fusions into this fusion.
  bool MergeElementwiseProducerFusions(HloFusionInstruction* fusion);

  // Sink scalar-constant broadcasts that are already operands of the fusion.
  bool SinkScalarBroadcastOperands(HloFusionInstruction* fusion);

  // Set up block-level configuration for the fusion
  absl::Status SetBlockLevelConfig(HloFusionInstruction* fusion,
                                   const std::vector<int64_t>& tile_sizes,
                                   int64_t num_blocks);

  const se::DeviceDescription device_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_BLOCK_LEVEL_REWRITER_H_
