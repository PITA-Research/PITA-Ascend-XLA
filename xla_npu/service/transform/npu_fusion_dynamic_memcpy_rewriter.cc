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

#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/backend_config.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

namespace {

// Check if a fusion computation only contains a single dynamic-slice or similar
// operation followed by get-tuple-elements. This pattern is suitable for
// dynamic memcpy optimization on NPU.
bool IsDynamicMemcpyPattern(HloInstruction* fusion) {
  if (fusion->opcode() != HloOpcode::kFusion) {
    return false;
  }

  HloComputation* comp = fusion->fused_instructions_computation();

  // Count operations in the fusion
  int64_t instruction_count = 0;
  bool has_dynamic_slice = false;
  bool has_slice = false;
  bool has_copy = false;
  bool has_only_simple_ops = true;

  for (HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    instruction_count++;

    // Check for slice operations
    if (HloPredicateIsOp<HloOpcode::kDynamicSlice,
                         HloOpcode::kDynamicUpdateSlice>(instr)) {
      has_dynamic_slice = true;
    } else if (HloPredicateIsOp<HloOpcode::kSlice>(instr)) {
      has_slice = true;
    } else if (HloPredicateIsOp<HloOpcode::kCopy>(instr)) {
      has_copy = true;
    } else if (!HloPredicateIsOp<HloOpcode::kBitcast,
                                 HloOpcode::kGetTupleElement,
                                 HloOpcode::kTuple>(instr)) {
      // If there are other operations, this is not a simple memcpy pattern
      has_only_simple_ops = false;
    }
  }

  // Must have at least one slice-related operation
  if (!has_dynamic_slice && !has_slice && !has_copy) {
    return false;
  }

  // All non-parameter operations should be slice/copy or simple data movement ops
  if (!has_only_simple_ops) {
    return false;
  }

  // Should be a relatively small fusion (simple memcpy pattern)
  if (instruction_count > 10) {
    return false;
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> NpuFusionDynamicMemcpyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Process each computation
  auto all_computations = module->computations(execution_threads);
  for (HloComputation* computation : all_computations) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    // Collect all fusion instructions first
    std::vector<HloInstruction*> fusion_instructions;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        fusion_instructions.push_back(instr);
      }
    }

    // Check if any fusion matches the dynamic memcpy pattern
    for (HloInstruction* fusion_instr : fusion_instructions) {
      if (IsDynamicMemcpyPattern(fusion_instr)) {
        // Mark the fusion as a dynamic memcpy fusion using NPU backend config
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "NpuFusionDynamicMemcpyRewriter: detected dynamic "
                             "memcpy pattern in fusion %s",
                             fusion_instr->name().data());

        // Set NPU backend config to mark this as a dynamic memcpy fusion
        npu_backend_config::BackendConfig backend_config;
        auto* fusion_config = backend_config.mutable_fusion_backend_config();
        fusion_config->set_kind("__dynamic_memcpy");

        TF_RETURN_IF_ERROR(fusion_instr->set_backend_config(backend_config));

        NPU_LOG_FORMAT_PRINT(INFO,
                             "NpuFusionDynamicMemcpyRewriter: set "
                             "backend_config for fusion %s as __dynamic_memcpy",
                             fusion_instr->name().data());

        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
