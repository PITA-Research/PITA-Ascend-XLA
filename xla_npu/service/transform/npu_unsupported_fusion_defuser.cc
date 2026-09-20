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

#include "xla/xla_npu/service/transform/npu_unsupported_fusion_defuser.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_fusion_whitelist.h"

namespace xla {

namespace {

// Determine whether it is a harmless wrapper (layout/shape adaptation instructions that do not affect semantics).
bool IsWrapperOp(HloOpcode opcode) {
  return opcode == HloOpcode::kBitcast ||
         opcode == HloOpcode::kReshape ||
         opcode == HloOpcode::kCopy ||
         opcode == HloOpcode::kConvert;
}

// Start at the root and peel off the wrapper to find the real hero op.
const HloInstruction* PeelWrappers(const HloInstruction* root) {
  const HloInstruction* current = root;
  while (IsWrapperOp(current->opcode())) {
    if (current->operand_count() == 0) break;
    current = current->operand(0);
  }
  return current;
}

bool IsAfirUnsupportedOp(const HloInstruction* hlo) {
  return NpuFusionWhitelist::ShouldSkipFusion(hlo);
}

// Determine whether hero is a trivial operator that Autofuse is not suitable for processing.
bool IsUnsupportedHero(const HloInstruction* hero) {
  HloOpcode opcode = hero->opcode();
  return opcode == HloOpcode::kTranspose ||
         opcode == HloOpcode::kBroadcast ||
         opcode == HloOpcode::kConstant ||
         IsAfirUnsupportedOp(hero);
}

bool ContainsAfirUnsupportedOp(const HloComputation* fused_comp) {
  for (const HloInstruction* hlo : fused_comp->instructions()) {
    if (IsAfirUnsupportedOp(hlo)) {
      return true;
    }
  }
  return false;
}

// aclnnFmaAddcmul is emitted as a direct custom-call and lowered by
// NpuIrEmitterUnnested to one ACLNN thunk.  AFIR cannot lower custom-call
// instructions, so any fusion containing it must be defused before codegen.
bool ContainsFmaAddcmulCustomCall(const HloComputation* fused_comp) {
  for (const HloInstruction* hlo : fused_comp->instructions()) {
    if (hlo->opcode() == HloOpcode::kCustomCall &&
        hlo->custom_call_target() == "__npu_fma_addcmul") {
      return true;
    }
  }
  return false;
}

// A reshape is "trivial" for AFIR iff it only adds/removes dimensions of size 1
// (e.g. [1,128]<->[128], [32,3,1]<->[32,3]). Such reshapes are lowered
// correctly by ConvertReshapeOp's one-dim path. A reshape that merges or splits
// real dimensions (e.g. [73,2,2,18] -> [73,4,18], or [4,3] -> [2,6]) is
// non-trivial: AFIR aliasing produces wrong element addressing for it, so the
// fusion must be defused and the reshape run as a standalone aclrtMemcpy.
bool IsNonTrivialReshape(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kReshape || hlo->operand_count() != 1) {
    return false;
  }
  const Shape& in = hlo->operand(0)->shape();
  const Shape& out = hlo->shape();
  int64_t i = 0;
  int64_t j = 0;
  while (i < in.dimensions_size() || j < out.dimensions_size()) {
    while (i < in.dimensions_size() && in.dimensions(i) == 1) ++i;
    while (j < out.dimensions_size() && out.dimensions(j) == 1) ++j;
    if (i < in.dimensions_size() && j < out.dimensions_size()) {
      if (in.dimensions(i) != out.dimensions(j)) {
        return true;  // non-1 dims differ -> merge/split of real dims
      }
      ++i;
      ++j;
    } else if (i < in.dimensions_size() || j < out.dimensions_size()) {
      return true;  // one side still has a non-1 dim left
    }
  }
  return false;
}

bool ContainsNonTrivialReshape(const HloComputation* fused_comp) {
  for (const HloInstruction* hlo : fused_comp->instructions()) {
    if (IsNonTrivialReshape(hlo)) {
      return true;
    }
  }
  return false;
}

bool IsSimpleAfirArithmeticOrBroadcastOp(HloOpcode opcode) {
  return NpuFusionWhitelist::IsAfirArithmeticOp(opcode) ||
         opcode == HloOpcode::kBroadcast;
}

bool ContainsOnlySimpleAfirArithmeticFusion(
    const HloComputation* fused_comp) {
  bool has_nontrivial_compute = false;

  for (const HloInstruction* hlo : fused_comp->instructions()) {
    switch (hlo->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
        continue;
      default:
        break;
    }

    if (IsWrapperOp(hlo->opcode()) ||
        IsSimpleAfirArithmeticOrBroadcastOp(hlo->opcode())) {
      has_nontrivial_compute |=
          IsSimpleAfirArithmeticOrBroadcastOp(hlo->opcode());
      continue;
    }
    return false;
  }

  return has_nontrivial_compute;
}

bool IsAclnnHeroNamedFusion(const HloComputation* fused_comp) {
  return absl::StartsWith(fused_comp->name(), "fused_dot") ||
         absl::StartsWith(fused_comp->name(), "fused_scatter") ||
         absl::StartsWith(fused_comp->name(), "fused_reduce") ||
         absl::StartsWith(fused_comp->name(), "fused_cholesky") ||
         absl::StartsWith(fused_comp->name(), "fused_while");
}

// Verify that there are no other substantive calculation instructions except wrapper + parameter + constant.
bool OnlyWrappersConstantsAndHero(const HloComputation* fused_comp,
                                  const HloInstruction* hero) {
  for (const HloInstruction* hlo : fused_comp->instructions()) {
    if (hlo->opcode() == HloOpcode::kParameter) continue;
    if (hlo == hero) continue;
    if (hlo->opcode() == HloOpcode::kConstant) continue;
    if (IsWrapperOp(hlo->opcode())) continue;
    return false;
  }
  return true;
}

// Validates that all input to hero ultimately comes from kParameter/kConstant (either directly or via a wrapper).
bool HeroInputsFromParametersOrConstants(const HloInstruction* hero) {
  for (int64_t i = 0; i < hero->operand_count(); ++i) {
    const HloInstruction* operand = hero->operand(i);
    while (IsWrapperOp(operand->opcode())) {
      if (operand->operand_count() == 0) break;
      operand = operand->operand(0);
    }
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> NpuUnsupportedFusionDefuser::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    if (computation->IsFusionComputation() ||
        computation->IsAsyncComputation()) {
      continue;
    }

    // The vector returned by MakeInstructionPostOrder will change because fusion is removed.
    // So first collect the fusion instructions that need to be processed
    std::vector<HloFusionInstruction*> fusions_to_defuse;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kFusion) continue;
      if (instr->user_count() == 0 &&
          instr != computation->root_instruction()) {
        continue;
      }
      HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instr);
      const HloComputation* fused_comp =
          fusion->fused_instructions_computation();

      const HloInstruction* hero =
          PeelWrappers(fused_comp->root_instruction());

      if (IsAclnnHeroNamedFusion(fused_comp) ||
          ContainsOnlySimpleAfirArithmeticFusion(fused_comp) ||
          ContainsAfirUnsupportedOp(fused_comp) ||
          ContainsFmaAddcmulCustomCall(fused_comp) ||
          ContainsNonTrivialReshape(fused_comp)) {
        fusions_to_defuse.push_back(fusion);
        continue;
      }

      if (!IsUnsupportedHero(hero)) continue;
      if (!HeroInputsFromParametersOrConstants(hero)) continue;
      if (!OnlyWrappersConstantsAndHero(fused_comp, hero)) continue;

      fusions_to_defuse.push_back(fusion);
    }

    // defuse one by one
    for (HloFusionInstruction* fusion : fusions_to_defuse) {
      const HloComputation* fused_comp =
          fusion->fused_instructions_computation();
      const HloInstruction* hero =
          PeelWrappers(fused_comp->root_instruction());

      NPU_LOG_FORMAT_PRINT(DEBUG, "Defusing unsupported fusion: %s with hero op: %s",
                       std::string(fusion->name()).c_str(),
                       HloOpcodeString(hero->opcode()));

      // Construct a clone map: instruction in fused computation -> equivalent instruction in parent
      absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;

      // Mapping parameters to external operands of fusion
      for (const HloInstruction* param :
           fused_comp->parameter_instructions()) {
        clone_map[param] =
            fusion->mutable_operand(param->parameter_number());
      }

      // Clone all non-parameter instructions into parent computation in post-order
      for (const HloInstruction* hlo :
           fused_comp->MakeInstructionPostOrder()) {
        if (hlo->opcode() == HloOpcode::kParameter) continue;

        std::vector<HloInstruction*> new_operands;
        new_operands.reserve(hlo->operand_count());
        for (const HloInstruction* operand : hlo->operands()) {
          new_operands.push_back(clone_map.at(operand));
        }

        HloInstruction* clone =
            computation->AddInstruction(
                hlo->CloneWithNewOperands(hlo->shape(), new_operands));
        clone_map[hlo] = clone;
      }

      // Replace all uses of fusion with cloned root
      HloInstruction* cloned_root =
          clone_map.at(fused_comp->root_instruction());
      TF_RETURN_IF_ERROR(fusion->ReplaceAllUsesWith(cloned_root));

      // Remove fusion instruction
      TF_RETURN_IF_ERROR(
          computation->RemoveInstructionAndUnusedOperands(fusion));

      NPU_LOG_FORMAT_PRINT(
          INFO,
          "Defusing unsupported fusion '%s' hero=%s",
          std::string(fusion->name()).c_str(),
          HloOpcodeString(hero->opcode()));

      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
