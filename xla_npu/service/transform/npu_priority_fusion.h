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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// PriorityFusion-like pass for NPU.
class NpuPriorityFusion : public HloModulePass {
 public:
  NpuPriorityFusion(tsl::thread::ThreadPool* thread_pool,
                    const se::DeviceDescription& device,
                    HloCostAnalysis::Options cost_analysis_options,
                    SymbolicExprContext* symbolic_expr_context)
      : thread_pool_(thread_pool),
        device_info_(device),
        cost_analysis_options_(std::move(cost_analysis_options)),
        symbolic_expr_context_(symbolic_expr_context) {}

  absl::string_view name() const override { return "npu-priority-fusion"; }

 protected:
  HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                        const HloInstruction* consumer);

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       bool use_multi_output_fusion = false);

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Consumes a unit of compiler fuel and returns true if we should
  // continue with the transformation.
  bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);

  // Returns the decision if the constant can be fused into the user.
  bool CanFuseConstant(const HloInstruction* constant,
                       const HloInstruction* user);

  // Collect transitive fusible producers of `consumer` into `to_fuse`
  // (read-only DFS over the producer DAG). `visited` prevents revisiting, so
  // traversal terminates. Collection is pre-order (a producer is appended before
  // its own producers), which is the order Fuse() needs: each producer must be a
  // current operand of the growing fusion when it is absorbed.
  void CollectFusibleProducers(
      HloInstruction* consumer,
      absl::flat_hash_set<HloInstruction*>& visited,
      std::vector<HloInstruction*>& to_fuse);

  tsl::thread::ThreadPool* thread_pool_;
  const se::DeviceDescription& device_info_;

  // Cost model options that defines priorities in the queue.
  // Using the base HloCostAnalysis::Options (not GPU-specific).
  HloCostAnalysis::Options cost_analysis_options_;

  SymbolicExprContext* symbolic_expr_context_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PRIORITY_FUSION_H_
