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

#include "xla/xla_npu/service/transform/npu_dot_contracting_dims_normalizer.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct DimClassification {
  std::vector<int64_t> batch_dims;
  std::vector<int64_t> keep_dims;
  std::vector<int64_t> contract_dims;
};

static DimClassification ClassifyDims(
    int rank,
    const std::vector<int64_t>& batch,
    const std::vector<int64_t>& contract) {
  absl::flat_hash_set<int64_t> batch_set(batch.begin(), batch.end());
  absl::flat_hash_set<int64_t> contract_set(contract.begin(), contract.end());

  DimClassification dc;
  for (auto d : batch) dc.batch_dims.push_back(d);
  for (int i = 0; i < rank; ++i) {
    if (!batch_set.contains(i) && !contract_set.contains(i))
      dc.keep_dims.push_back(i);
  }
  for (auto d : contract) dc.contract_dims.push_back(d);
  return dc;
}

static std::vector<int64_t> BuildCanonicalPermutation(
    const DimClassification& dc, bool is_lhs) {
  std::vector<int64_t> perm;
  perm.reserve(dc.batch_dims.size() + dc.keep_dims.size() +
               dc.contract_dims.size());

  for (auto d : dc.batch_dims) perm.push_back(d);
  if (is_lhs) {
    for (auto d : dc.keep_dims) perm.push_back(d);
    for (auto d : dc.contract_dims) perm.push_back(d);
  } else {
    for (auto d : dc.contract_dims) perm.push_back(d);
    for (auto d : dc.keep_dims) perm.push_back(d);
  }
  return perm;
}

static bool IsIdentityPermutation(const std::vector<int64_t>& perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) return false;
  }
  return true;
}

static std::vector<int64_t> InversePermutation(
    const std::vector<int64_t>& perm) {
  std::vector<int64_t> inv(perm.size());
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    inv[perm[i]] = i;
  }
  return inv;
}

static std::vector<int64_t> RemapDims(
    const std::vector<int64_t>& inv_perm,
    const std::vector<int64_t>& old_dims) {
  std::vector<int64_t> new_dims;
  new_dims.reserve(old_dims.size());
  for (auto d : old_dims) {
    new_dims.push_back(inv_perm[d]);
  }
  return new_dims;
}

static Shape MakeMaterializedTransposeShape(
    const Shape& operand_shape, const std::vector<int64_t>& permutation) {
  Shape transpose_shape =
      ShapeUtil::PermuteDimensions(permutation, operand_shape);

  // PermuteDimensions deliberately preserves the physical order and makes the
  // transpose a bitcast. ACLNN thunks consume contiguous C-order buffers, so
  // the inserted transpose must materialize the logical permutation instead.
  LayoutUtil::SetToDefaultLayout(&transpose_shape);
  return transpose_shape;
}

static bool NormalizeMaterializedLayouts(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // Tuple and GTE are buffer aliases, not layout conversions. Visit producers
    // before users so nested tuples and extracted values describe the physical
    // layout of normalized producers before CopyInsertion sees the graph.
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTuple) {
        for (int64_t i = 0; i < instruction->operand_count(); ++i) {
          Shape* element =
              instruction->mutable_shape()->mutable_tuple_shapes(i);
          const Shape& operand_shape = instruction->operand(i)->shape();
          if (!ShapeUtil::Equal(*element, operand_shape)) {
            *element = operand_shape;
            changed = true;
          }
        }
        continue;
      }
      if (instruction->opcode() == HloOpcode::kGetTupleElement) {
        const Shape& element = instruction->operand(0)->shape().tuple_shapes(
            instruction->tuple_index());
        if (!ShapeUtil::Equal(instruction->shape(), element)) {
          *instruction->mutable_shape() = element;
          changed = true;
        }
        continue;
      }
      if ((instruction->opcode() != HloOpcode::kTranspose &&
           instruction->opcode() != HloOpcode::kDot) ||
          !instruction->shape().IsArray() ||
          LayoutUtil::IsMonotonicWithDim0Major(
              instruction->shape().layout())) {
        continue;
      }

      // ACLNN transpose and dot materialize contiguous C-order output buffers.
      // Keeping a bitcast/non-C-order HLO layout makes a later executable or
      // host transfer reinterpret that physical buffer with the wrong strides.
      LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
      changed = true;
    }
  }
  return changed;
}

template <typename RepeatedField>
static std::vector<int64_t> ToVector(const RepeatedField& rf) {
  return std::vector<int64_t>(rf.begin(), rf.end());
}

// ---------------------------------------------------------------------------
// NpuDotContractingDimsNormalizer
// ---------------------------------------------------------------------------

absl::StatusOr<bool> NpuDotContractingDimsNormalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int total_dots = 0;
  int normalized_dots = 0;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> dots;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kDot) {
        dots.push_back(instr);
      }
    }
    total_dots += dots.size();
    for (HloInstruction* instr : dots) {
      TF_ASSIGN_OR_RETURN(bool normalized,
                          NormalizeDot(Cast<HloDotInstruction>(instr)));
      if (normalized) normalized_dots++;
      changed |= normalized;
    }
  }
  LOG(INFO) << "[DotCanon] Total dots: " << total_dots
            << ", normalized: " << normalized_dots;
  changed |= NormalizeMaterializedLayouts(module, execution_threads);
  return changed;
}

absl::StatusOr<bool> NpuDotContractingDimsNormalizer::NormalizeDot(
    HloDotInstruction* dot) {
  const DotDimensionNumbers& dn = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  auto lhs_batch = ToVector(dn.lhs_batch_dimensions());
  auto rhs_batch = ToVector(dn.rhs_batch_dimensions());
  auto lhs_contract = ToVector(dn.lhs_contracting_dimensions());
  auto rhs_contract = ToVector(dn.rhs_contracting_dimensions());

  DimClassification lhs_dc = ClassifyDims(
      lhs->shape().dimensions_size(), lhs_batch, lhs_contract);
  DimClassification rhs_dc = ClassifyDims(
      rhs->shape().dimensions_size(), rhs_batch, rhs_contract);

  auto lhs_perm = BuildCanonicalPermutation(lhs_dc, /*is_lhs=*/true);
  auto rhs_perm = BuildCanonicalPermutation(rhs_dc, /*is_lhs=*/false);

  bool lhs_changed = !IsIdentityPermutation(lhs_perm);
  bool rhs_changed = !IsIdentityPermutation(rhs_perm);

  // Constant folding can leave a column-major literal even when dot dimension
  // numbers are already canonical. Unlike ACLNN-produced intermediate buffers,
  // constants retain their literal's physical byte order. Relayout the bytes,
  // not just the shape, before handing them to the C-order runtime. Clone only
  // for this dot so shared constants (including bitcast users) keep their layout.
  auto materialize_constant = [&](HloInstruction* operand) {
    if (operand->opcode() != HloOpcode::kConstant ||
        LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout())) {
      return operand;
    }
    Shape shape = operand->shape();
    LayoutUtil::SetToDefaultLayout(&shape);
    HloInstruction* constant = dot->parent()->AddInstruction(
        HloInstruction::CreateConstant(operand->literal().Relayout(shape)));
    constant->set_metadata(operand->metadata());
    return constant;
  };
  HloInstruction* materialized_lhs = materialize_constant(lhs);
  HloInstruction* materialized_rhs = materialize_constant(rhs);
  if (!lhs_changed && !rhs_changed && materialized_lhs == lhs &&
      materialized_rhs == rhs) {
    return false;
  }
  lhs = materialized_lhs;
  rhs = materialized_rhs;

  LOG(INFO) << "[DotCanon] Normalizing dot: " << dot->ToString();
  LOG(INFO) << "[DotCanon]   lhs_perm=[" << absl::StrJoin(lhs_perm, ",")
            << "] lhs_changed=" << lhs_changed;
  LOG(INFO) << "[DotCanon]   rhs_perm=[" << absl::StrJoin(rhs_perm, ",")
            << "] rhs_changed=" << rhs_changed;

  DotDimensionNumbers new_dn = dn;
  HloInstruction* new_lhs = lhs;
  HloInstruction* new_rhs = rhs;

  if (lhs_changed) {
    Shape new_lhs_shape =
        MakeMaterializedTransposeShape(lhs->shape(), lhs_perm);
    new_lhs = dot->parent()->AddInstruction(
        HloInstruction::CreateTranspose(new_lhs_shape, lhs, lhs_perm));

    auto lhs_inv = InversePermutation(lhs_perm);
    auto new_lb = RemapDims(lhs_inv, lhs_batch);
    auto new_lc = RemapDims(lhs_inv, lhs_contract);
    new_dn.mutable_lhs_batch_dimensions()->Assign(
        new_lb.begin(), new_lb.end());
    new_dn.mutable_lhs_contracting_dimensions()->Assign(
        new_lc.begin(), new_lc.end());
  }

  if (rhs_changed) {
    Shape new_rhs_shape =
        MakeMaterializedTransposeShape(rhs->shape(), rhs_perm);
    new_rhs = dot->parent()->AddInstruction(
        HloInstruction::CreateTranspose(new_rhs_shape, rhs, rhs_perm));

    auto rhs_inv = InversePermutation(rhs_perm);
    auto new_rb = RemapDims(rhs_inv, rhs_batch);
    auto new_rc = RemapDims(rhs_inv, rhs_contract);
    new_dn.mutable_rhs_batch_dimensions()->Assign(
        new_rb.begin(), new_rb.end());
    new_dn.mutable_rhs_contracting_dimensions()->Assign(
        new_rc.begin(), new_rc.end());
  }

  HloInstruction* new_dot = dot->parent()->AddInstruction(
      HloInstruction::CreateDot(dot->shape(), new_lhs, new_rhs,
                                new_dn, dot->precision_config()));
  new_dot->set_metadata(dot->metadata());
  if (dot->has_sharding()) {
    new_dot->set_sharding(dot->sharding());
  }

  TF_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, new_dot));

  LOG(INFO) << "[DotCanon]  => " << new_dot->ToString();
  return true;
}

}  // namespace xla
