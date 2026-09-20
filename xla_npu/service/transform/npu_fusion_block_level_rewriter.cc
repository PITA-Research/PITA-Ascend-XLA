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

#include "xla/xla_npu/service/transform/npu_fusion_block_level_rewriter.h"

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// NPU Unified Buffer (UB) size in bytes (typical for Ascend 910)
constexpr int64_t kNpuUBSize = 256 * 1024;  // 256KB

// Default tile sizes for block-level fusion
constexpr int64_t kDefaultTileSize1D = 1024;
constexpr int64_t kDefaultTileSize2D = 64;
constexpr int64_t kDefaultTileSize3D = 32;
constexpr int64_t kDefaultTileSize4D = 16;

// Default number of blocks for block-level fusion
constexpr int64_t kDefaultNumBlocks = 1;

// Default number of pipeline stages
constexpr int64_t kDefaultNumStages = 2;

// Minimum number of elements to enable block-level fusion
constexpr int64_t kMinElementsForBlockLevel = 1024;

// NPU fusion kind constants
constexpr char kNpuBlockingFusionKind[] = "__npu_blocking";

bool IsScalarConstantBroadcastFusion(const HloFusionInstruction* fusion) {
  if (fusion->IsMultiOutputFusion()) {
    return false;
  }

  const HloInstruction* root =
      fusion->fused_instructions_computation()->root_instruction();
  if (root->opcode() != HloOpcode::kBroadcast || root->operand_count() != 1) {
    return false;
  }

  const HloInstruction* scalar = root->operand(0);
  if (scalar->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(scalar->shape())) {
    return false;
  }

  for (const HloInstruction* instr :
       fusion->fused_instructions_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter || instr == root ||
        instr == scalar) {
      continue;
    }
    return false;
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;

  // Get all instructions in post-order
  std::vector<HloInstruction*> instrs = computation->MakeInstructionPostOrder();

  for (auto instr_it = instrs.rbegin(); instr_it != instrs.rend();
       ++instr_it) {
    HloInstruction* instr = *instr_it;
    // Only process loop or input fusion instructions
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }

    if (instr->user_count() == 0 && instr != computation->root_instruction()) {
      continue;
    }

    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instr);

    // Skip if already a custom fusion
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
      continue;
    }

    // Skip multi-output fusions
    if (fusion->IsMultiOutputFusion()) {
      continue;
    }

    // Check if this fusion can be converted to block-level
    if (!CanConvertToBlockLevel(fusion)) {
      VLOG(4) << "Fusion cannot be converted to block-level: "
              << fusion->ToString();
      continue;
    }

    changed |= MergeElementwiseProducerFusions(fusion);
    changed |= SinkScalarBroadcastOperands(fusion);

    VLOG(3) << "Converting fusion to block-level: " << fusion->ToString();

    TF_ASSIGN_OR_RETURN(bool converted, ConvertFusionToBlockLevel(fusion));
    changed |= converted;
  }

  return changed;
}

bool NpuFusionBlockLevelRewriter::CanConvertToBlockLevel(
    const HloInstruction* fusion) {
  // Must be a loop or input fusion
  if (fusion->opcode() != HloOpcode::kFusion) {
    return false;
  }

  const auto* fusion_instr = Cast<HloFusionInstruction>(fusion);
  HloInstruction::FusionKind kind = fusion_instr->fusion_kind();
  if (kind != HloInstruction::FusionKind::kLoop &&
      kind != HloInstruction::FusionKind::kInput) {
    return false;
  }

  // The current AFIR blocking autofuse scheduler assumes a fixed axis space
  // across the whole graph.  Only admit shape-preserving arithmetic chains and
  // scalar broadcasts; layout/indexing-changing ops such as transpose, reshape,
  // bitcast, slice, concatenate, and non-scalar broadcast must stay outside.
  if (!ContainsOnlyAutofuseSafeOps(fusion_instr->called_computation())) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Fusion %s contains ops outside blocking-autofuse safe arithmetic set",
        fusion->name().data());
    return false;
  }

  // Check if the output is large enough to benefit from block-level fusion
  int64_t num_elements = ShapeUtil::ElementsIn(fusion->shape());
  if (num_elements < kMinElementsForBlockLevel) {
    return false;
  }

  return true;
}

absl::StatusOr<bool> NpuFusionBlockLevelRewriter::ConvertFusionToBlockLevel(
    HloFusionInstruction* fusion) {
  // Compute optimal tile sizes
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tile_sizes,
                      ComputeTileSizes(fusion));

  // Change the fusion kind to custom with block kind
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());

  // Get backend config or create if not present
  npu_backend_config::FusionBackendConfig backend_config;
  if (npu_config.has_fusion_backend_config()) {
    backend_config = npu_config.fusion_backend_config();
  }

  // Set the fusion kind to NPU block
  backend_config.set_kind(std::string(kNpuBlockingFusionKind));

  // Set up block-level configuration
  TF_RETURN_IF_ERROR(
      SetBlockLevelConfig(fusion, tile_sizes, kDefaultNumBlocks));

  // Save the backend config
  *npu_config.mutable_fusion_backend_config() = backend_config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));

  VLOG(3) << "Converted fusion to block-level: " << fusion->ToString();
  return true;
}

absl::StatusOr<std::vector<int64_t>>
NpuFusionBlockLevelRewriter::ComputeTileSizes(const HloInstruction* fusion) {
  const Shape& shape = fusion->shape();
  int64_t rank = shape.dimensions_size();

  std::vector<int64_t> tile_sizes(rank);

  // Compute tile sizes based on rank
  switch (rank) {
    case 1:
      tile_sizes[0] = kDefaultTileSize1D;
      break;
    case 2:
      tile_sizes[0] = kDefaultTileSize2D;
      tile_sizes[1] = kDefaultTileSize2D;
      break;
    case 3:
      tile_sizes[0] = kDefaultTileSize3D;
      tile_sizes[1] = kDefaultTileSize3D;
      tile_sizes[2] = kDefaultTileSize3D;
      break;
    default:
      // For higher dimensional tensors, use smaller tiles
      for (int i = 0; i < rank; ++i) {
        tile_sizes[i] = kDefaultTileSize4D;
      }
      break;
  }

  // Clamp tile sizes to the actual shape dimensions
  const auto& dimensions = shape.dimensions();
  for (int i = 0; i < rank; ++i) {
    tile_sizes[i] = std::min(tile_sizes[i], dimensions[i]);
  }

  return tile_sizes;
}

bool NpuFusionBlockLevelRewriter::ContainsOnlyElementwiseOps(
    HloComputation* fusion) {
  for (HloInstruction* instr : fusion->MakeInstructionPostOrder()) {
    // Skip parameters
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Check if it's an elementwise operation
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kClamp:
      case HloOpcode::kAbs:
      case HloOpcode::kNegate:
      case HloOpcode::kExp:
      case HloOpcode::kLog:
      case HloOpcode::kSqrt:
      case HloOpcode::kRsqrt:
      case HloOpcode::kPower:
      case HloOpcode::kTanh:
      case HloOpcode::kLogistic:
      case HloOpcode::kBitcast:
      case HloOpcode::kReshape:
      case HloOpcode::kBroadcast:
        continue;
      default:
        return false;
    }
  }
  return true;
}

bool NpuFusionBlockLevelRewriter::ContainsOnlyAutofuseSafeOps(
    HloComputation* fusion) {
  for (HloInstruction* instr : fusion->MakeInstructionPostOrder()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
        continue;
      case HloOpcode::kBroadcast:
        // Only scalar broadcasts are safe for the generated kernel.  Non-scalar
        // broadcasts (e.g. f32[64] -> f32[64,64]) are mishandled by AFIR/AscendC
        // autofuse: the operand tile is sized by the pre-broadcast shape while
        // the elementwise op reads it with the output tile size.  Keep them out
        // of merged producer fusions so they stay materialized outside.
        if (instr->operand_count() == 1 &&
            ShapeUtil::IsScalar(instr->operand(0)->shape())) {
          continue;
        }
        return false;
      default:
        return false;
    }
  }
  return true;
}

bool NpuFusionBlockLevelRewriter::MergeElementwiseProducerFusions(
    HloFusionInstruction* fusion) {
  bool changed = false;

  for (int64_t i = 0; i < fusion->operand_count();) {
    HloInstruction* operand = fusion->mutable_operand(i);
    if (operand->opcode() != HloOpcode::kFusion) {
      ++i;
      continue;
    }

    HloFusionInstruction* producer = Cast<HloFusionInstruction>(operand);
    HloInstruction::FusionKind kind = producer->fusion_kind();
    bool producer_allowed =
        ContainsOnlyAutofuseSafeOps(producer->called_computation());
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "NpuFusionBlockLevelRewriter: considering producer fusion %s into %s, "
        "user_count=%d, multi_output=%d, kind=%d, allowed_ops=%d",
        producer->name().data(), fusion->name().data(),
        producer->user_count(), producer->IsMultiOutputFusion(),
        static_cast<int>(kind), producer_allowed);

    if (producer->user_count() != 1) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "NpuFusionBlockLevelRewriter: skip merging %s into %s: "
          "user_count=%d",
          producer->name().data(), fusion->name().data(),
          producer->user_count());
      ++i;
      continue;
    }

    if (producer->IsMultiOutputFusion()) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "NpuFusionBlockLevelRewriter: skip merging %s into %s: "
          "multi-output fusion",
          producer->name().data(), fusion->name().data());
      ++i;
      continue;
    }

    if (kind != HloInstruction::FusionKind::kLoop &&
        kind != HloInstruction::FusionKind::kInput) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "NpuFusionBlockLevelRewriter: skip merging %s into %s: kind=%d",
          producer->name().data(), fusion->name().data(),
          static_cast<int>(kind));
      ++i;
      continue;
    }

    if (!producer_allowed) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "NpuFusionBlockLevelRewriter: skip merging %s into %s: "
          "producer contains ops outside add/subtract/multiply/divide/broadcast",
          producer->name().data(), fusion->name().data());
      ++i;
      continue;
    }

    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "NpuFusionBlockLevelRewriter: merging elementwise producer fusion "
        "%s into %s",
        producer->name().data(), fusion->name().data());
    fusion->MergeFusionInstruction(producer);
    if (producer->user_count() == 0 &&
        producer != producer->parent()->root_instruction()) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG,
          "NpuFusionBlockLevelRewriter: removing merged producer fusion %s",
          producer->name().data());
      TF_CHECK_OK(producer->parent()->RemoveInstruction(producer));
    }
    changed = true;
  }

  return changed;
}

bool NpuFusionBlockLevelRewriter::SinkScalarBroadcastOperands(
    HloFusionInstruction* fusion) {
  bool changed = false;

  for (int64_t i = 0; i < fusion->operand_count();) {
    HloInstruction* broadcast = fusion->mutable_operand(i);
    if (broadcast->opcode() != HloOpcode::kBroadcast ||
        broadcast->operand_count() != 1) {
      if (broadcast->opcode() == HloOpcode::kFusion) {
        HloFusionInstruction* producer_fusion =
            Cast<HloFusionInstruction>(broadcast);
        if (IsScalarConstantBroadcastFusion(producer_fusion)) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG,
              "NpuFusionBlockLevelRewriter: merging scalar broadcast fusion "
              "%s into %s",
              producer_fusion->name().data(), fusion->name().data());
          fusion->MergeFusionInstruction(producer_fusion);
          if (producer_fusion->user_count() == 0 &&
              producer_fusion !=
                  producer_fusion->parent()->root_instruction()) {
            NPU_LOG_FORMAT_PRINT(
                DEBUG,
                "NpuFusionBlockLevelRewriter: removing merged scalar "
                "broadcast fusion %s",
                producer_fusion->name().data());
            TF_CHECK_OK(
                producer_fusion->parent()->RemoveInstruction(producer_fusion));
          }
          changed = true;
          continue;
        }
      }
      ++i;
      continue;
    }

    HloInstruction* scalar = broadcast->mutable_operand(0);
    if (scalar->opcode() != HloOpcode::kConstant ||
        !ShapeUtil::IsScalar(scalar->shape())) {
      ++i;
      continue;
    }

    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "NpuFusionBlockLevelRewriter: sinking scalar broadcast %s into %s",
        broadcast->name().data(), fusion->name().data());
    fusion->FuseInstruction(broadcast);
    changed = true;

    if (absl::c_linear_search(fusion->operands(), scalar)) {
      fusion->FuseInstruction(scalar);
    }
  }

  return changed;
}

absl::Status NpuFusionBlockLevelRewriter::SetBlockLevelConfig(
    HloFusionInstruction* fusion, const std::vector<int64_t>& tile_sizes,
    int64_t num_blocks) {
  TF_ASSIGN_OR_RETURN(
      auto npu_config,
      fusion->backend_config<npu_backend_config::NpuBackendConfig>());

  // Get or create backend config
  npu_backend_config::FusionBackendConfig* backend_config =
      npu_config.mutable_fusion_backend_config();

  npu_backend_config::NpuBlockLevelConfig block_config;
  *block_config.mutable_output_tile_sizes() = {tile_sizes.begin(),
                                               tile_sizes.end()};
  block_config.set_num_blocks(num_blocks);
  block_config.set_num_stages(kDefaultNumStages);
  *backend_config->mutable_block_config() = block_config;

  TF_RETURN_IF_ERROR(fusion->set_backend_config(npu_config));
  return absl::OkStatus();
}

}  // namespace xla
