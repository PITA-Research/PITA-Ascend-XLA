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

#include "xla/xla_npu/service/transform/npu_copy_fusion.h"

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/str_format.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

namespace {

// Returns the maximum number of operands and outputs per fusion.
constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 96; }

// Returns whether the fusion fits in the budget. This is a simplified version
// for NPU that doesn't check device-specific properties.
bool FusionFitsInBudget(const HloInstruction& fusion,
                        const HloInstruction& user) {
  int64_t total_operands = fusion.operand_count();
  if (fusion.IsMultiOutputFusion()) {
    total_operands += fusion.fused_instruction_count();
  }
  return total_operands + fusion.user_count() <=
         MaxOperandsAndOutputsPerFusion();
}

// Returns true if only elementwise operations (from parameters) are reachable
// in the fusion computation.
bool OnlyElementwiseOpsReachableFromParams(HloComputation* fused_computation) {
  std::queue<const HloInstruction*> q;
  absl::flat_hash_set<const HloInstruction*> visited;
  for (auto param : fused_computation->parameter_instructions()) {
    q.push(param);
    visited.insert(param);
  }
  while (!q.empty()) {
    const HloInstruction* hlo = q.front();
    q.pop();
    for (auto user : hlo->users()) {
      if ((!user->IsElementwiseOnOperand(user->operand_index(hlo)) ||
           HloPredicateIsOp<HloOpcode::kCopy>(user)) &&
          HloPredicateIsNotOp<HloOpcode::kBitcast, HloOpcode::kTuple>(user)) {
        return false;
      }
      if (visited.insert(user).second) {
        q.push(user);
      }
    }
  }
  return true;
}

// Returns true if the instruction is a scatter operation.
bool IsScatter(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kScatter;
}

}  // namespace

absl::StatusOr<bool> NpuCopyFusion::DoCopyFusion(HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();
  for (HloInstruction* hlo : defs_before_uses) {
    if (HloPredicateIsNotOp<HloOpcode::kFusion>(hlo) ||
        hlo->IsMultiOutputFusion()) {
      continue;
    }
    HloComputation* fused_computation = hlo->fused_instructions_computation();
    if (!OnlyElementwiseOpsReachableFromParams(fused_computation)) {
      continue;
    }
    HloInstruction* root = fused_computation->root_instruction();
    // Skip scatter operations and multi-output fusions with slice roots
    if (IsScatter(root)) {
      continue;
    }
    auto [copies, other_users] = ClassifyUsers(hlo);
    if (copies.empty()) {
      continue;
    }
    auto fusion_adaptor = HloFusionAdaptor::ForComputation(fused_computation);
    auto dynamic_update_slices =
        GetOutputDefiningDynamicUpdateSlices(fusion_adaptor->GetRoots());
    // Skip dynamic update slice fusions which might be emitted in-place.
    if (!dynamic_update_slices.empty() &&
        (HloPredicateIsNotOp<HloOpcode::kTuple>(root) ||
         dynamic_update_slices.size() == root->shape().tuple_shapes().size())) {
      continue;
    }
    int64_t num_outputs =
        hlo->IsMultiOutputFusion() ? root->operand_count() : int64_t{1};
    int64_t total_outputs = num_outputs + copies.size();
    if (total_outputs > MaxOperandsAndOutputsPerFusion()) {
      VLOG(1) << "Skipping fusion as it would exceed "
                 "MaxOperandsAndOutputsPerFusion(): "
              << total_outputs << " > " << MaxOperandsAndOutputsPerFusion();
      continue;
    }
    changed = true;
    TF_RETURN_IF_ERROR(FuseCopyInstructionsIntoComputation(
        hlo, fused_computation, copies, other_users));
  }
  return changed;
}

std::pair<std::vector<HloInstruction*>, std::vector<HloInstruction*>>
NpuCopyFusion::ClassifyUsers(HloInstruction* hlo) {
  std::vector<HloInstruction*> copies;
  std::vector<HloInstruction*> other_users;
  for (auto user : hlo->users()) {
    HloInstruction* copy_user = user;
    if (HloPredicateIsOp<HloOpcode::kGetTupleElement>(copy_user) &&
        copy_user->user_count() == 1) {
      copy_user = copy_user->users()[0];
    }
    if (HloPredicateIsOp<HloOpcode::kBitcast>(copy_user) &&
        copy_user->user_count() == 1) {
      copy_user = copy_user->users()[0];
    }
    if (HloPredicateIsOp<HloOpcode::kCopy>(copy_user) &&
        copy_user->shape() == copy_user->operand(0)->shape() &&
        !copy_user->shape().IsTuple() && !copy_user->HasControlDependencies() &&
        FusionFitsInBudget(*hlo, *copy_user)) {
      copies.push_back(copy_user);
    } else {
      other_users.push_back(user);
    }
  }

  return {copies, other_users};
}

absl::Status NpuCopyFusion::FuseCopyInstructionsIntoComputation(
    HloInstruction* hlo, HloComputation* fused_computation,
    const std::vector<HloInstruction*>& copies,
    const std::vector<HloInstruction*>& other_users) {
  HloInstruction* root = fused_computation->root_instruction();
  int64_t num_outputs =
      hlo->IsMultiOutputFusion() ? root->operand_count() : int64_t{1};
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(copies.size() + num_outputs);
  if (hlo->IsMultiOutputFusion()) {
    for (HloInstruction* operand : root->operands()) {
      tuple_elements.push_back(operand);
    }
  } else {
    tuple_elements.push_back(root);
  }
  for (auto copy : copies) {
    HloInstruction* user = copy;
    std::vector<HloInstruction*> operand_chain;
    while (user->operand(0) != hlo) {
      operand_chain.push_back(user);
      user = user->mutable_operand(0);
    }
    HloInstruction* clone = root;
    if (hlo->IsMultiOutputFusion()) {
      clone = root->mutable_operand(user->tuple_index());
      operand_chain.pop_back();
    }
    for (auto it = operand_chain.rbegin(); it != operand_chain.rend(); ++it) {
      clone = fused_computation->AddInstruction(
          (*it)->CloneWithNewOperands((*it)->shape(), {clone}));
    }
    tuple_elements.push_back(clone);
  }

  HloInstruction* new_root = fused_computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
  fused_computation->set_root_instruction(new_root, true);
  *hlo->mutable_shape() = new_root->shape();

  if (HloPredicateIsOp<HloOpcode::kTuple>(root)) {
    TF_RETURN_IF_ERROR(fused_computation->RemoveInstruction(root));
  } else {
    auto gte = hlo->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(hlo, 0));
    TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWithDifferentShape(other_users, gte));
  }
  for (int64_t i = 0; i < copies.size(); ++i) {
    auto gte = hlo->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(hlo, num_outputs + i));
    TF_RETURN_IF_ERROR(hlo->parent()->ReplaceInstruction(copies[i], gte));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> NpuCopyFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Only for the entry computation we can be sure that the copies do not share
  // a buffer with a parameter of the fusion that it will be fused with.
  return DoCopyFusion(module->entry_computation());
}

}  // namespace xla
