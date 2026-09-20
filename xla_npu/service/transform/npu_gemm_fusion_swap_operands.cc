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

#include "xla/xla_npu/service/transform/npu_gemm_fusion_swap_operands.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Prefer FRACTAL_NZ layout for matrix operations
constexpr NpuLayoutFormat kPreferredMatrixModeLayout =
    NpuLayoutFormat::kFRACTAL_NZ;

// Prefer NC1HWC0 layout for convolution operations
constexpr NpuLayoutFormat kPreferredConvModeLayout = NpuLayoutFormat::kNC1HWC0;

// Minimum matrix size to consider operand swapping
constexpr int64_t kMinMatrixSizeForSwapping = 1024;

// Aspect ratio threshold for considering swapping
constexpr double kAspectRatioThreshold = 4.0;

// Data sizes in bytes for different types
int64_t GetDataSize(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::F16:
    case PrimitiveType::BF16:
      return 2;
    case PrimitiveType::F32:
    case PrimitiveType::S32:
      return 4;
    case PrimitiveType::S8:
    case PrimitiveType::U8:
      return 1;
    default:
      return 4;
  }
}

}  // namespace

absl::StatusOr<bool> NpuGemmFusionSwapOperands::RunImpl(
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

absl::StatusOr<bool> NpuGemmFusionSwapOperands::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;
  // Get all instructions in post-order
  std::vector<HloInstruction*> instrs = computation->MakeInstructionPostOrder();
  for (HloInstruction* instr : instrs) {
    // Only process dot instructions
    if (instr->opcode() != HloOpcode::kDot) {
      continue;
    }
    auto* dot = Cast<HloDotInstruction>(instr);
    // Skip if operands should not be swapped
    if (!ShouldSwapOperands(dot)) {
      VLOG(4) << "GEMM operands will not be swapped: " << dot->ToString();
      continue;
    }
    VLOG(3) << "Attempting to swap GEMM operands: " << dot->ToString();
    TF_ASSIGN_OR_RETURN(bool swapped, SwapGemmOperands(dot));
    changed |= swapped;
  }
  return changed;
}

bool NpuGemmFusionSwapOperands::ShouldSwapOperands(
    const HloDotInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();

  // Skip batched matrix multiplication for now
  if (lhs_shape.dimensions_size() > 2 || rhs_shape.dimensions_size() > 2) {
    return false;
  }

  // Check minimum size
  int64_t lhs_elements = ShapeUtil::ElementsIn(lhs_shape);
  int64_t rhs_elements = ShapeUtil::ElementsIn(rhs_shape);
  if (lhs_elements * rhs_elements < kMinMatrixSizeForSwapping) {
    return false;
  }

  // Get preferred layouts
  NpuLayoutFormat lhs_layout = GetPreferredLayoutFormat(lhs_shape, true);
  NpuLayoutFormat rhs_layout = GetPreferredLayoutFormat(rhs_shape, false);

  // Check if swapping would improve layout alignment
  bool lhs_preferred = (lhs_layout == kPreferredMatrixModeLayout);
  bool rhs_preferred = (rhs_layout == kPreferredMatrixModeLayout);

  // Consider swapping if layouts don't match the preferred format
  if (!lhs_preferred && !rhs_preferred) {
    // Both suboptimal, check aspect ratios
    int64_t lhs_rows = lhs_shape.dimensions(0);
    int64_t lhs_cols = lhs_shape.dimensions(1);
    int64_t rhs_rows = rhs_shape.dimensions(0);
    int64_t rhs_cols = rhs_shape.dimensions(1);

    double lhs_aspect = static_cast<double>(std::max(lhs_rows, lhs_cols)) /
                        static_cast<double>(std::min(lhs_rows, lhs_cols));
    double rhs_aspect = static_cast<double>(std::max(rhs_rows, rhs_cols)) /
                        static_cast<double>(std::min(rhs_rows, rhs_cols));

    // Swap if RHS has better aspect ratio for NPU layout
    return rhs_aspect > lhs_aspect * kAspectRatioThreshold;
  }

  return !lhs_preferred && rhs_preferred;
}

NpuLayoutFormat NpuGemmFusionSwapOperands::GetPreferredLayoutFormat(
    const Shape& shape, bool is_lhs) {
  // For simple 2D matrices in GEMM, prefer FRACTAL_NZ
  if (shape.dimensions_size() == 2) {
    return NpuLayoutFormat::kFRACTAL_NZ;
  }

  // For higher dimensional tensors (like in convolutions), use NC1HWC0
  if (shape.dimensions_size() >= 4) {
    return NpuLayoutFormat::kNC1HWC0;
  }

  return NpuLayoutFormat::kDefault;
}

bool NpuGemmFusionSwapOperands::IsLayoutCompatible(const Shape& shape,
                                                   NpuLayoutFormat format) {
  switch (format) {
    case NpuLayoutFormat::kNC1HWC0:
      // NC1HWC0 requires 4D or 5D tensors
      return shape.dimensions_size() >= 4;

    case NpuLayoutFormat::kFRACTAL_NZ:
      // FRACTAL_NZ works best with 2D tensors
      return shape.dimensions_size() == 2;

    default:
      return true;
  }
}

absl::StatusOr<bool> NpuGemmFusionSwapOperands::SwapGemmOperands(
    HloDotInstruction* dot) {
  // Get the operands
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  // Transpose the operands
  std::vector<int64_t> transpose_permutation;

  // Create transpose for LHS
  if (lhs->shape().dimensions_size() == 2) {
    transpose_permutation = {1, 0};
  } else {
    // For higher dimensional tensors, transpose last two dimensions
    int64_t rank = lhs->shape().dimensions_size();
    for (int64_t i = 0; i < rank; ++i) {
      if (i < rank - 2) {
        transpose_permutation.push_back(i);
      } else if (i == rank - 2) {
        transpose_permutation.push_back(rank - 1);
      } else {
        transpose_permutation.push_back(rank - 2);
      }
    }
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * lhs_transpose,
                      CreateLayoutTranspose(lhs, transpose_permutation));

  TF_ASSIGN_OR_RETURN(HloInstruction * rhs_transpose,
                      CreateLayoutTranspose(rhs, transpose_permutation));

  // Swap dimension numbers
  DotDimensionNumbers new_dim_numbers = dot->dot_dimension_numbers();

  // Swap LHS and RHS contracting dimensions
  std::vector<int64_t> new_lhs_contracting(
      new_dim_numbers.rhs_contracting_dimensions().begin(),
      new_dim_numbers.rhs_contracting_dimensions().end());
  std::vector<int64_t> new_rhs_contracting(
      new_dim_numbers.lhs_contracting_dimensions().begin(),
      new_dim_numbers.lhs_contracting_dimensions().end());

  new_dim_numbers.mutable_lhs_contracting_dimensions()->Assign(
      new_lhs_contracting.begin(), new_lhs_contracting.end());
  new_dim_numbers.mutable_rhs_contracting_dimensions()->Assign(
      new_rhs_contracting.begin(), new_rhs_contracting.end());

  // Swap batch dimensions if present
  if (new_dim_numbers.lhs_batch_dimensions_size() > 0) {
    std::vector<int64_t> new_lhs_batch(
        new_dim_numbers.rhs_batch_dimensions().begin(),
        new_dim_numbers.rhs_batch_dimensions().end());
    std::vector<int64_t> new_rhs_batch(
        new_dim_numbers.lhs_batch_dimensions().begin(),
        new_dim_numbers.lhs_batch_dimensions().end());
    new_dim_numbers.mutable_lhs_batch_dimensions()->Assign(
        new_lhs_batch.begin(), new_lhs_batch.end());
    new_dim_numbers.mutable_rhs_batch_dimensions()->Assign(
        new_rhs_batch.begin(), new_rhs_batch.end());
  }

  // Create new dot with swapped operands
  HloInstruction* new_dot = dot->parent()->AddInstruction(
      HloInstruction::CreateDot(dot->shape(), lhs_transpose, rhs_transpose,
                                new_dim_numbers, dot->precision_config()));

  // Replace the old dot
  TF_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, new_dot));

  VLOG(3) << "Swapped GEMM operands: " << new_dot->ToString();
  return true;
}

absl::StatusOr<HloInstruction*>
NpuGemmFusionSwapOperands::CreateLayoutTranspose(
    HloInstruction* operand, const std::vector<int64_t>& permutation) {
  Shape new_shape = ShapeUtil::PermuteDimensions(permutation, operand->shape());

  // Create transpose instruction
  HloInstruction* transpose = operand->parent()->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, operand, permutation));

  return transpose;
}

}  // namespace xla
