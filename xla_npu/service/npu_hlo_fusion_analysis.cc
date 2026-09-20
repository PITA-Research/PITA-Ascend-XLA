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

#include "xla/xla_npu/service/npu_hlo_fusion_analysis.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"

namespace xla {

namespace {
// Returns a description of a transpose hero, that is compatible with all roots.
// A root is compatible with the transpose hero if
//   * Either the root has a traspose hero with the same normalized dimensions
//   * Or the root output shape is equal to the the transpose input shape
std::optional<TransposeDescription> FindConsistentTransposeHero(
    const absl::InlinedVector<HloInstructionAdaptor, 2>& hlo_roots,
    const absl::InlinedVector<HloInstructionAdaptor, 2>& heroes) {
  std::optional<TransposeDescription> tiled_transpose_hero;
  std::vector<const HloInstruction*> non_transpose_roots;

  for (auto [root, hero] : llvm::zip(hlo_roots, heroes)) {
    if (auto tr =
            NpuGetDescriptionForTiledTransposeEmitter(hero.instruction())) {
      if (!tiled_transpose_hero) {
        // First transpose hero found.
        tiled_transpose_hero = tr;
      } else if (!tiled_transpose_hero->IsEquivalent(*tr)) {
        // Transpose heroes have different shape.
        return std::nullopt;
      }
    } else {
      non_transpose_roots.push_back(&root.instruction());
    }
  }

  if (!tiled_transpose_hero) return std::nullopt;

  for (auto* root : non_transpose_roots) {
    // Roots that don't have a transpose hero, should have a shape compatible with the transpose input.
    if (!ShapeUtil::IsReshapeOrTransposeBitcast(
            root->shape(), tiled_transpose_hero->input_shape(),
            /*ignore_element_type=*/true)) {
      return std::nullopt;
    }
  }

  return tiled_transpose_hero;
}

bool UseConcatenateFusion(absl::Span<const HloInstructionAdaptor> roots,
                          absl::Span<const HloInstructionAdaptor> heroes) {
  if (heroes.size() != 1) return false;
  if (heroes.front().opcode() != HloOpcode::kConcatenate) return false;
  // The concat emitter does not support multiple outputs yet.
  if (roots.front().shape().IsTuple()) return false;
  // Limit the number of operands because the concat emitter produces code for
  // each operand, hurting occupancy.
  if (heroes.front().instruction().operand_count() > 4) return false;
  // The loop emitter is faster when warp divergence and occupancy are both low.
  return true;
}

NpuHloFusionAnalysis::EmitterFusionKind GetEmitterFusionKind(
    // const FusionBackendConfig& fusion_backend_config,
    absl::Span<const HloInstructionAdaptor> fusion_roots,
    absl::Span<const HloInstructionAdaptor> fusion_heroes,
    const std::optional<TransposeDescription>& tiled_transpose,
    const se::DeviceDescription& device_info) {
  // We expect that the last dimension is swapped with a different dimension.
  if (tiled_transpose.has_value()) {
    return NpuHloFusionAnalysis::EmitterFusionKind::kTranspose;
  }

  if (fusion_roots.size() > 1) {
    return NpuHloFusionAnalysis::EmitterFusionKind::kLoop;
  }

  if (fusion_roots[0].opcode() == HloOpcode::kScatter) {
    return NpuHloFusionAnalysis::EmitterFusionKind::kScatter;
  }

  if (UseConcatenateFusion(fusion_roots, fusion_heroes)) {
    return NpuHloFusionAnalysis::EmitterFusionKind::kConcatenate;
  }

  return NpuHloFusionAnalysis::EmitterFusionKind::kLoop;
}

const Shape& GetShape(const HloInstructionAdaptor& adaptor) {
  return adaptor.shape();
}

const Shape& GetShape(const HloInstruction* instruction) {
  return instruction->shape();
}

template <typename Container>
int SmallestBitWidth(const Container& args) {
  int bits = std::numeric_limits<int>::max();
  for (const auto& operand : args) {
    const Shape& shape = GetShape(operand);
    if (!shape.IsArray()) continue;
    bits = std::min(bits, shape.element_type() == PRED
                              ? 8
                              : primitive_util::BitWidth(shape.element_type()));
  }
  return bits;
}

}  // namespace

bool IsNpuFusionKind(const HloInstruction& hlo, absl::string_view kind) {
  return false;
}

NpuHloFusionAnalysis::NpuHloFusionAnalysis(
    HloFusionSpec fusion_spec, EmitterFusionKind emitter_fusion_kind,
    const se::DeviceDescription* device_info,
    std::optional<TransposeDescription> tiled_transpose,
    NpuHloFusionAnalysis::InputOutputInfo input_output_info)
    : fusion_spec_(std::move(fusion_spec)),
      emitter_fusion_kind_(emitter_fusion_kind),
      device_info_(device_info),
      tiled_transpose_(tiled_transpose),
      input_output_info_(std::move(input_output_info)) {}

NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(
    std::unique_ptr<HloFusionAdaptor> fusion,
    const se::DeviceDescription* device_info) {
  absl::InlinedVector<HloInstructionAdaptor, 2> roots = fusion->GetRoots();
  absl::InlinedVector<HloInstructionAdaptor, 2> heroes;
  for (auto root : roots) {
    heroes.push_back(NpuFindNonTrivialHero(root));
  }

  InputOutputInfo input_output_info{
      /*smallest_input_dtype_bits=*/SmallestBitWidth(fusion->GetParameters()),
      /*smallest_output_dtype_bits=*/SmallestBitWidth(roots),
  };

  std::optional<TransposeDescription> tiled_transpose_hero =
      FindConsistentTransposeHero(roots, heroes);

  EmitterFusionKind emitter_fusion_kind =
      GetEmitterFusionKind(roots, heroes, tiled_transpose_hero, *device_info);
  // The loop emitter always uses the roots as heroes. Even if we found some
  // non-trivial heroes in the fusion, some logic in GetEmitterFusionKind might
  // still decide to use the loop emitter and in that case we need to reset the
  // heroes to the roots.
  if (emitter_fusion_kind == EmitterFusionKind::kLoop) {
    heroes = roots;
    tiled_transpose_hero = std::nullopt;
  }

  HloFusionSpec fusion_spec(std::move(fusion), std::move(roots),
                            std::move(heroes));

  return NpuHloFusionAnalysis(std::move(fusion_spec), emitter_fusion_kind,
                              device_info, tiled_transpose_hero,
                              std::move(input_output_info));
}

// static
NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(
    const HloInstruction& instruction,
    const se::DeviceDescription& device_info) {
  return Create(HloFusionAdaptor::ForInstruction(&instruction), &device_info);
}

// static
NpuHloFusionAnalysis NpuHloFusionAnalysis::Create(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info) {
  return NpuHloFusionAnalysis::Create(
      // std::move(fusion_backend_config),
      HloFusionAdaptor::ForProducerConsumer(&producer, &consumer),
      &device_info);
}

const HloInstruction* NpuHloFusionAnalysis::FindHeroReduction() const {
  return nullptr;
}

}  // namespace xla
