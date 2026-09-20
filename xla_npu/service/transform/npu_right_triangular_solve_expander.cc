/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#include "xla/xla_npu/service/transform/npu_right_triangular_solve_expander.h"

#include <numeric>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
absl::StatusOr<bool> NpuRightTriangularSolveExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kTriangularSolve ||
          instr->triangular_solve_options().left_side()) continue;
      // The runtime represents transpose as a boolean and has no conjugation
      // contract. Do not silently interpret a complex adjoint as a transpose.
      if (primitive_util::IsComplexType(instr->shape().element_type())) {
        return absl::UnimplementedError(
            "NPU right triangular solve currently supports real element types");
      }
      auto options = instr->triangular_solve_options();
      switch (options.transpose_a()) {
        case TriangularSolveOptions::NO_TRANSPOSE:
          options.set_transpose_a(TriangularSolveOptions::TRANSPOSE);
          break;
        case TriangularSolveOptions::TRANSPOSE:
        case TriangularSolveOptions::ADJOINT:  // A^H == A^T for real A.
          options.set_transpose_a(TriangularSolveOptions::NO_TRANSPOSE);
          break;
        default:
          return absl::InvalidArgumentError("Invalid triangular solve transpose option");
      }
      options.set_left_side(true);
      // lower and unit_diagonal describe the stored A, which is unchanged.
      auto* b = instr->mutable_operand(1);
      std::vector<int64_t> permutation(b->shape().dimensions_size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::swap(permutation[permutation.size() - 1],
                permutation[permutation.size() - 2]);
      Shape transposed_shape = ShapeUtil::PermuteDimensions(permutation, b->shape());
      LayoutUtil::SetToDefaultLayout(&transposed_shape);
      auto* bt = comp->AddInstruction(HloInstruction::CreateTranspose(
          transposed_shape, b, permutation));
      auto* solve = comp->AddInstruction(HloInstruction::CreateTriangularSolve(
          transposed_shape, instr->mutable_operand(0), bt, options));
      auto* result = comp->AddInstruction(HloInstruction::CreateTranspose(
          instr->shape(), solve, permutation));
      bt->set_metadata(instr->metadata());
      solve->set_metadata(instr->metadata());
      result->set_metadata(instr->metadata());
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, result));
      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
