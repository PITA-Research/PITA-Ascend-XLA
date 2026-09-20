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

#include "xla/xla_npu/service/transform/npu_priority_fusion.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_fusion_whitelist.h"

namespace xla {

namespace {

// Helper function to check if two instructions have compatible sharding.
// Simplified version that assumes most instructions are sharding-compatible.
bool IsFusibleOnSharding(const HloInstruction* consumer,
                         const HloInstruction* producer) {
  // Simplified check: only return false if sharding is explicitly set and differs
  if (consumer->has_sharding() && producer->has_sharding()) {
    return consumer->sharding() == producer->sharding();
  }
  return true;
}

bool IsNonScalarBroadcast(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBroadcast &&
         instr->operand_count() == 1 &&
         !ShapeUtil::IsScalar(instr->operand(0)->shape());
}

bool IsArithmeticLoopFusionCandidate(const HloInstruction* consumer,
                                     const HloInstruction* producer) {
  if (consumer == nullptr || producer == nullptr) {
    return false;
  }

  // Keep the tiny-fusion heuristic narrowly scoped to the arithmetic
  // elementwise chains that currently explode compile count in eval.
  if (!consumer->IsElementwise() || !producer->IsElementwise()) {
    return false;
  }

  return NpuFusionWhitelist::IsAfirArithmeticOp(consumer->opcode()) ||
         NpuFusionWhitelist::IsAfirArithmeticOp(producer->opcode());
}

// Helper function to check basic fusibility constraints.
bool CanFuseNpu(const HloInstruction* consumer,
                const HloInstruction* producer) {
  // Check if consumer can be fused (XLA-level check)
  // This is required because CreateFusion() has a CHECK for fused_root->IsFusible()
  // and will crash if called with a non-fusible instruction.
  if (!consumer->IsFusible()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Consumer %s is not fusible",
                         consumer->name().data());
    return false;
  }

  // Check if producer can be fused (XLA-level check)
  // This is required because FuseInstruction() has a CHECK for IsFusible()
  // and will crash if called with a non-fusible instruction.
  if (!producer->IsFusible()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Producer %s is not fusible",
                         producer->name().data());
    return false;
  }

  if (NpuFusionWhitelist::ShouldSkipFusion(consumer)) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Consumer %s is whitelisted to skip fusion",
                         consumer->name().data());
    return false;
  }
  if (NpuFusionWhitelist::ShouldSkipFusion(producer)) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Producer %s is whitelisted to skip fusion",
                         producer->name().data());
    return false;
  }

  // Don't fuse if producer is a fusion instruction.
  if (producer->opcode() == HloOpcode::kFusion) {
    return false;
  }

  // AFIR/AscendC autofuse currently mishandles non-scalar broadcasts inside a
  // fused computation: it allocates the operand tile using the pre-broadcast
  // shape, but downstream elementwise ops read it with the output tile size.
  // Keep these broadcasts outside fusion so they are materialized before being
  // consumed by the fused kernel.
  if (IsNonScalarBroadcast(producer) || IsNonScalarBroadcast(consumer)) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Producer %s is a non-scalar broadcast; keep it outside fusion",
        producer->name().data());
    return false;
  }

  // AFIR/AscendC autofuse emits incorrect kernels when a `divide` is the hero
  // of a fusion that contains a shape-reinterpreting operand (reshape/squeeze).
  // The codegen allocates the operand buffer by its declared shape but tiles by
  // the output shape, reading garbage — the same bug class as the fused_add
  // broadcast issue above. This shows up as linalg.solve's LU normalization
  // divide `divide(reshape(p), q)` silently corrupting the factorization.
  // Keep `divide` out of fusion so it runs as a correct aclnn op; add/sub/mul
  // and all other elementwise ops still fuse.
  auto is_divide = [](const HloInstruction* instr) -> bool {
    return instr->opcode() == HloOpcode::kDivide;
  };
  if (is_divide(producer) || is_divide(consumer)) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Producer/consumer %s/%s is a divide; keep out of fusion",
        producer->name().data(), consumer->name().data());
    return false;
  }

  // AFIR/AscendC autofuse cannot schedule a `transpose` inside a fused
  // computation. The graph attribute declares a single axis table with one
  // fixed size per logical axis, but a transpose remaps those axes so the same
  // axis carries different sizes before and after the node (e.g. mul output
  // repeats {8,3,2,3} -> transpose output repeats {3,8,2,3} share axis z0 which
  // is 8 on one side and 3 on the other). The AxesReorderScheduler then finds
  // no legal tiling and codegen fails with "Optimize fail" / "Autofuse
  // schedule/codegen failed" (fused_gather_89 = mul + transpose{1,0,2,3}).
  // Keep transpose out of fusion whether it is the producer or the consumer;
  // it materializes and runs as a correct aclnn op instead. add/sub/mul and the
  // rest of the elementwise set still fuse.
  auto is_transpose = [](const HloInstruction* instr) -> bool {
    return instr->opcode() == HloOpcode::kTranspose;
  };
  if (is_transpose(producer) || is_transpose(consumer)) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Producer/consumer %s/%s is a transpose; keep out of fusion",
        producer->name().data(), consumer->name().data());
    return false;
  }

  // Don't fuse instructions with different sharding
  if (!IsFusibleOnSharding(consumer, producer)) {
    return false;
  }

  // Only fuse if producer has exactly one user (the consumer)
  if (producer->user_count() != 1) {
    return false;
  }

  // Don't fuse if output shape is too large (simplified heuristic to avoid
  // register pressure issues)
  int64_t output_size = ShapeUtil::ByteSizeOfElements(producer->shape());
  constexpr int64_t kMaxFusionOutputSizeBytes = 128 * 1024;  // 128 KB
  if (output_size > kMaxFusionOutputSizeBytes) {
    return false;
  }

  // Tiny arithmetic loop fusions are profitable for graph reduction but not
  // for the current AFIR cold-compile path: eval produces many shape-distinct
  // add/sub/mul kernels only a few bytes to a few KB large, and compile time
  // dominates runtime. Let ACLNN handle these tiny arithmetic ops directly and
  // reserve fusion for larger tensors where compile cost can amortize.
  constexpr int64_t kMinArithmeticFusionOutputSizeBytes = 4 * 1024;  // 4 KB
  if (output_size < kMinArithmeticFusionOutputSizeBytes &&
      IsArithmeticLoopFusionCandidate(consumer, producer)) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Producer/consumer %s/%s is a tiny arithmetic fusion candidate "
        "(%ld bytes); keep out of fusion",
        producer->name().data(), consumer->name().data(), output_size);
    return false;
  }

  return true;
}

}  // namespace

HloInstruction::FusionKind NpuPriorityFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  // For NPU, we use kLoop fusion by default for most operations.
  // This can be refined later for specific patterns.
  return HloInstruction::FusionKind::kLoop;
}

HloInstruction* NpuPriorityFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer,
                                        bool use_multi_output_fusion) {
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "NpuPriorityFusion: fusing %s (%s) into %s (%s)",
      producer->name().data(), HloOpcodeString(producer->opcode()),
      consumer->name().data(), HloOpcodeString(consumer->opcode()));

  HloComputation* computation = consumer->parent();
  auto kind = ChooseKind(producer, consumer);
  HloInstruction* fusion_instruction = consumer;

  // Phase 1: Create or get fusion instruction
  if (HloPredicateIsNotOp<HloOpcode::kFusion>(consumer)) {
    fusion_instruction = computation->AddInstruction(
        HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
    absl::Status replace_status =
        consumer->ReplaceAllUsesWith(fusion_instruction);
    if (!replace_status.ok()) {
      NPU_LOG_FORMAT_PRINT(
          ERROR, "Failed to replace uses of %s with fusion %s: %s",
          consumer->name().data(), fusion_instruction->name().data(),
          replace_status.ToString().data());
      return nullptr;
    }
  } else if (kind != fusion_instruction->fusion_kind()) {
    fusion_instruction->set_fusion_kind(kind);
  }

  fusion_instruction->set_called_computations_execution_thread(
      computation->execution_thread());

  // Phase 2: Fuse producer into the fusion instruction
  if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
    if (use_multi_output_fusion) {
      // Merge producer fusion using multi-output approach
      fusion_instruction->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      fusion_instruction->MergeFusionInstruction(producer);
    }
  } else {
    if (use_multi_output_fusion) {
      fusion_instruction->FuseInstructionIntoMultiOutput(producer);
      if (producer->user_count() != 0) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Producer %s still has users after multi-output "
                             "fusion, user_count=%d",
                             producer->name().data(), producer->user_count());
      }
      absl::Status remove_status =
          producer->parent()->RemoveInstruction(producer);
      if (!remove_status.ok()) {
        NPU_LOG_FORMAT_PRINT(ERROR, "Failed to remove instruction %s: %s",
                             producer->name().data(),
                             remove_status.ToString().data());
      }
    } else {
      fusion_instruction->FuseInstruction(producer);
    }
  }

  if (fusion_instruction != consumer) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "NpuPriorityFusion: created new fusion: %s",
                         fusion_instruction->name().data());
  }

  return fusion_instruction;
}

bool NpuPriorityFusion::ConsumeFuel(HloInstruction* producer,
                                    HloInstruction* consumer) {
  // Always allow fusion for now. Can be enhanced with fuel mechanism later.
  return true;
}

bool NpuPriorityFusion::CanFuseConstant(const HloInstruction* constant,
                                        const HloInstruction* user) {
  // Only fuse scalar constants (dimensions().size() == 0)
  // Non-scalar constants are not supported by AFIR converter
  if (constant->shape().dimensions().size() != 0) {
    return false;
  }

  // Check if constant is small enough to fuse
  constexpr int64_t kMaxConstantSizeBytes = 256;  // 256 bytes threshold
  if (constant->literal().size_bytes() > kMaxConstantSizeBytes) {
    return false;  // Constant too large
  }

  // Only fuse Broadcast and constant scalar/element-wise operations
  if (user->opcode() == HloOpcode::kBroadcast || user->IsElementwise() ||
      user->opcode() == HloOpcode::kSlice ||
      user->opcode() == HloOpcode::kReshape ||
      user->opcode() == HloOpcode::kTranspose ||
      user->opcode() == HloOpcode::kBitcast) {
    return true;
  }

  return false;  // Constant can only be fused into elementwise-like ops
}

void NpuPriorityFusion::CollectFusibleProducers(
    HloInstruction* consumer,
    absl::flat_hash_set<HloInstruction*>& visited,
    std::vector<HloInstruction*>& to_fuse) {
  // Read-only transitive walk over the producer DAG: for each operand, if it
  // can be fused into `consumer` (same gate as the legacy inner loop), append
  // it and recurse into its own operands. The recursion is what lets a
  // producer-of-producer (e.g. `add` behind `mul` behind `sub`) enter the
  // fusion — the old "fuse one operand then break" driver stopped one level
  // short and never revisited the newly created fusion to pull it in.
  for (int i = 0; i < consumer->operand_count(); ++i) {
    HloInstruction* producer = consumer->mutable_operand(i);
    if (!visited.insert(producer).second) {
      continue;  // already considered; also bounds recursion on the DAG
    }
    if (producer->opcode() == HloOpcode::kConstant &&
        !CanFuseConstant(producer, consumer)) {
      continue;
    }
    if (!CanFuseNpu(consumer, producer)) {
      continue;  // div / transpose / non-scalar broadcast / multi-user / >128KB
    }
    to_fuse.push_back(producer);
    CollectFusibleProducers(producer, visited, to_fuse);
  }
}

absl::StatusOr<bool> NpuPriorityFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Copy computations to a vector to avoid iterator invalidation during fusion
  std::vector<HloComputation*> all_computations_list;
  for (HloComputation* computation : module->computations(execution_threads)) {
    all_computations_list.push_back(computation);
  }

  // Process each computation from the copied list
  for (HloComputation* computation : all_computations_list) {
    if (computation == nullptr) {
      NPU_LOG_FORMAT_PRINT(WARN,
                           "NpuPriorityFusion: skipping null computation");
      continue;
    }

    if (computation->IsFusionComputation()) {
      continue;
    }

    // Get all instructions in post-order
    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();

    // Transitive producer-consumer fusion.
    //
    // The legacy "fuse one operand then break" driver only pulled a consumer's
    // *direct* operand into the fusion, and (because the snapshot below is
    // fixed and kFusion consumers are skipped) never revisited a newly created
    // fusion to pull in that operand's own producers. So in `add -> mul -> sub`
    // only `mul` fused into `sub` while `add` stayed outside. Instead we now
    // *collect* all transitive fusible producers of the consumer first (read-
    // only DFS, CollectFusibleProducers), then fuse them into one fusion.
    //
    // `absorbed` tracks instructions already pulled into a fusion during this
    // pass, so we skip their stale entries in the post-order snapshot below
    // (they remain in `instructions` but are now dead / owned by a fusion).
    absl::flat_hash_set<HloInstruction*> absorbed;

    // Iterate in reverse post-order (from roots to leaves)
    for (auto consumer_it = instructions.rbegin();
         consumer_it != instructions.rend(); ++consumer_it) {
      HloInstruction* consumer = *consumer_it;

      // Skip fusion instructions and instructions already absorbed into a
      // fusion during this pass (stale entries in the snapshot).
      if (consumer->opcode() == HloOpcode::kFusion || absorbed.count(consumer)) {
        continue;
      }

      // Skip instructions with multiple users (simplification)
      if (consumer->user_count() != 0 &&
          consumer->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }

      // Collect all transitive fusible producers of this consumer (read-only).
      std::vector<HloInstruction*> to_fuse;
      absl::flat_hash_set<HloInstruction*> visited;
      CollectFusibleProducers(consumer, visited, to_fuse);
      if (to_fuse.empty()) {
        continue;
      }

      // Fuse every collected producer into one fusion rooted at `consumer`.
      // Fuse() reuses the fusion instruction when the consumer is already a
      // fusion (Phase 1 below), so sequential calls keep absorbing into the same
      // fusion. Pre-order collection guarantees each producer is a current
      // operand of the growing fusion when it is absorbed (mul first -> add
      // becomes a fusion operand -> add absorbed next).
      HloInstruction* fusion = consumer;
      for (HloInstruction* producer : to_fuse) {
        HloInstruction* new_fusion =
            Fuse(producer, fusion, /*use_multi_output_fusion=*/false);
        if (new_fusion == nullptr) {
          break;  // safety; should not happen after collection
        }
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "NpuPriorityFusion: successfully fused %s into %s",
            producer->name().data(), new_fusion->name().data());
        fusion = new_fusion;
        absorbed.insert(producer);
        changed = true;
      }
    }
  }

  // Run DCE once at the module level to clean up unused instructions
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(bool dce_changed, dce.Run(module, execution_threads));
  changed = changed || dce_changed;

  return changed;
}

}  // namespace xla
