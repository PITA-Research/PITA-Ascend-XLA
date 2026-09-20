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

#include "xla/xla_npu/service/transform/npu_rename_fusions.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

constexpr absl::string_view FusionKindToString(
    HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kCustom:
      return "custom";
    case HloInstruction::FusionKind::kLoop:
      return "loop";
    case HloInstruction::FusionKind::kInput:
      return "input";
    case HloInstruction::FusionKind::kOutput:
      return "output";
  }
}

// Check if an instruction is "trivial" and should be skipped when finding the hero.
bool IsTrivialInstruction(const HloInstruction* instr) {
  // Skip data movement operations
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCopy:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kParameter:
      return true;
    default:
      return false;
  }
}

// Compute-intensive operations that should be prioritized as heroes.
bool IsComputeIntensiveInstruction(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kGather:
    case HloOpcode::kSort:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
    case HloOpcode::kFft:
      return true;
    default:
      return false;
  }
}

// Simplified version of FindNonTrivialHero for NPU.
// Returns the first non-trivial instruction reachable from the given root.
HloInstructionAdaptor FindNonTrivialHero(const HloInstructionAdaptor& root) {
  // Find the first non-trivial instruction reachable from root
  // using a simple BFS traversal
  absl::flat_hash_set<HloInstructionAdaptor> visited;
  std::vector<HloInstructionAdaptor> queue = {root};

  // First, try to find a compute-intensive instruction
  while (!queue.empty()) {
    HloInstructionAdaptor current = queue.back();
    queue.pop_back();

    if (visited.insert(current).second) {
      const HloInstruction* instr = &current.instruction();
      if (!IsTrivialInstruction(instr)) {
        if (IsComputeIntensiveInstruction(instr)) {
          return current;
        }
      }

      // Add operands to queue
      for (const auto& operand : current.GetOperands()) {
        if (!visited.contains(operand)) {
          queue.push_back(operand);
        }
      }
    }
  }

  // If no compute-intensive instruction, find the first non-trivial one
  visited.clear();
  queue = {root};
  while (!queue.empty()) {
    HloInstructionAdaptor current = queue.back();
    queue.pop_back();

    if (visited.insert(current).second) {
      const HloInstruction* instr = &current.instruction();
      if (!IsTrivialInstruction(instr)) {
        return current;
      }

      for (const auto& operand : current.GetOperands()) {
        if (!visited.contains(operand)) {
          queue.push_back(operand);
        }
      }
    }
  }

  // As a fallback, return the root
  return root;
}

// Generate hero names from fusion instructions.
std::string MakeFusionHeroNames(const HloInstruction* instruction) {
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(
          const_cast<HloInstruction*>(instruction));
  absl::btree_set<absl::string_view> heroes;

  for (auto root : fusion_adaptor->GetRoots()) {
    HloInstructionAdaptor hero = FindNonTrivialHero(root);
    heroes.insert(HloOpcodeString(hero.opcode()));
  }

  // If no hero found, use "unknown"
  if (heroes.empty()) {
    return "unknown";
  }

  return absl::StrReplaceAll(absl::StrJoin(heroes, "_"), {{"-", "_"}});
}

// Rename a fusion instruction and its computation.
void RenameFusion(HloModule* module, HloInstruction* instruction) {
  std::string hero_names = MakeFusionHeroNames(instruction);
  module->SetAndUniquifyInstrName(
      instruction, absl::StrCat(FusionKindToString(instruction->fusion_kind()),
                                "_", hero_names, "_fusion"));
  module->SetAndUniquifyComputationName(
      instruction->fused_instructions_computation(),
      absl::StrCat("fused_", hero_names));

  NPU_LOG_FORMAT_PRINT(DEBUG, "NpuRenameFusions: renamed fusion to %s",
                       instruction->name().data());
}

}  // namespace

absl::StatusOr<bool> NpuRenameFusions::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (HloPredicateIsNotOp<HloOpcode::kFusion>(instruction) ||
          instruction->fusion_kind() == HloInstruction::FusionKind::kCustom) {
        continue;
      }
      RenameFusion(module, instruction);
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
