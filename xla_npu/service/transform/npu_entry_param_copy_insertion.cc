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

#include "xla/xla_npu/service/transform/npu_entry_param_copy_insertion.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::StatusOr<bool> NpuEntryParamCopyInsertion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (entry == nullptr) return false;

  const HloInstruction* root = entry->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kTuple) return false;

  // Step 1: Identify passthrough parameters — those whose value appears
  // directly as a root tuple element (no computation). These are
  // intentionally aliased and should NOT be copied.
  absl::flat_hash_set<HloInstruction*> passthrough_params;
  bool has_passthrough_output = false;
  bool has_computed_output = false;
  for (int i = 0; i < root->operand_count(); ++i) {
    const HloInstruction* operand = root->operand(i);
    // Walk through GTE, copy, bitcast to find the source parameter
    while (operand != nullptr) {
      if (operand->opcode() == HloOpcode::kParameter) {
        has_passthrough_output = true;
        passthrough_params.insert(
            entry->parameter_instruction(operand->parameter_number()));
        break;
      }
      if (operand->opcode() == HloOpcode::kGetTupleElement ||
          operand->opcode() == HloOpcode::kCopy ||
          operand->opcode() == HloOpcode::kBitcast) {
        operand = operand->operand(0);
        continue;
      }
      has_computed_output = true;
      break;  // not a passthrough chain
    }
  }

  // Mixed tuple results are a conservative correctness hazard: passthrough
  // entry params may still share allocations with computed outputs after buffer
  // assignment, which can corrupt rebuilt pytrees such as mjx Data.replace().
  // In that case, materialize every array parameter with an explicit copy and
  // only keep the aliasing fast path for pure passthrough tuples.
  const bool disable_passthrough_optimization =
      has_passthrough_output && has_computed_output;

  // Step 2: For each entry parameter that is NOT a passthrough, insert
  // a kCopy to give it its own buffer. This prevents the buffer
  // assignment from aliasing the parameter's buffer with intermediate
  // computation temporaries.
  bool changed = false;
  for (HloInstruction* param : entry->parameter_instructions()) {
    if (!disable_passthrough_optimization &&
        passthrough_params.contains(param)) {
      NPU_LOG_FORMAT_PRINT(
          INFO,
          "EntryParamCopyInsertion: param %d (%s) is passthrough, skipping",
          param->parameter_number(),
          param->shape().ToString().c_str());
      continue;
    }

    // Skip zero-element parameters (no data to copy)
    if (ShapeUtil::IsZeroElementArray(param->shape())) {
      continue;
    }

    // Insert copy after the parameter
    HloInstruction* copy = entry->AddInstruction(
        HloInstruction::CreateUnary(param->shape(), HloOpcode::kCopy, param));

    // Replace all uses of the parameter with the copy, EXCEPT the copy
    // instruction itself (which uses the parameter as its operand).
    // Do this manually to avoid creating a self-loop (copy → copy).
    std::vector<HloInstruction*> users = param->users();
    for (HloInstruction* user : users) {
      if (user == copy) continue;  // skip the copy itself
      TF_RETURN_IF_ERROR(
          param->ReplaceUseWithDifferentShape(user, copy));
    }

    NPU_LOG_FORMAT_PRINT(
        INFO,
        "EntryParamCopyInsertion: inserted copy for param %d, shape=%s",
        param->parameter_number(),
        param->shape().ToString().c_str());

    changed = true;
  }

  return changed;
}

}  // namespace xla
