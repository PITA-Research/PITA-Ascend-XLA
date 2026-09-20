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

#include "xla/xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.h"

#include <cstdint>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace {

constexpr int64_t kMaxLeftOrderDotK = 64;

bool IsF32Array(const HloInstruction* instr) {
  return instr != nullptr && instr->shape().IsArray() &&
         instr->shape().element_type() == F32;
}

bool IsZeroF32ScalarConstant(const HloInstruction* instr) {
  if (instr == nullptr || instr->opcode() != HloOpcode::kConstant ||
      instr->shape().element_type() != F32 ||
      !ShapeUtil::IsScalar(instr->shape())) {
    return false;
  }
  return instr->literal().Get<float>({}) == 0.0f;
}

bool IsAddReducer(const HloInstruction* reduce) {
  if (reduce->called_computations().empty()) {
    return false;
  }
  const HloComputation* reducer = reduce->called_computations()[0];
  if (reducer == nullptr || reducer->root_instruction() == nullptr) {
    return false;
  }
  return reducer->root_instruction()->opcode() == HloOpcode::kAdd;
}

bool SameDimensions(const Shape& lhs, const Shape& rhs) {
  if (!lhs.IsArray() || !rhs.IsArray() ||
      lhs.dimensions_size() != rhs.dimensions_size()) {
    return false;
  }
  for (int64_t i = 0; i < lhs.dimensions_size(); ++i) {
    if (lhs.dimensions(i) != rhs.dimensions(i)) {
      return false;
    }
  }
  return true;
}

bool OutputIsInputPrefix(const Shape& input_shape, const Shape& output_shape) {
  const int64_t input_rank = input_shape.dimensions_size();
  if (!output_shape.IsArray() ||
      output_shape.dimensions_size() != input_rank - 1) {
    return false;
  }
  for (int64_t i = 0; i + 1 < input_rank; ++i) {
    if (output_shape.dimensions(i) != input_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

bool IsSingleLastDimReduce(const HloInstruction* reduce,
                           const Shape& operand_shape) {
  if (reduce->dimensions().size() != 1) {
    return false;
  }
  return reduce->dimensions()[0] == operand_shape.dimensions_size() - 1;
}

absl::StatusOr<HloInstruction*> RewriteReduceToDot(HloInstruction* reduce) {
  if (reduce->opcode() != HloOpcode::kReduce || reduce->operand_count() != 2 ||
      !IsF32Array(reduce) || !IsZeroF32ScalarConstant(reduce->operand(1)) ||
      !IsAddReducer(reduce)) {
    return nullptr;
  }

  HloInstruction* multiply = reduce->mutable_operand(0);
  if (multiply->opcode() != HloOpcode::kMultiply ||
      multiply->operand_count() != 2 || !IsF32Array(multiply)) {
    return nullptr;
  }

  HloInstruction* lhs = multiply->mutable_operand(0);
  HloInstruction* rhs = multiply->mutable_operand(1);
  if (!IsF32Array(lhs) || !IsF32Array(rhs) ||
      !SameDimensions(lhs->shape(), rhs->shape())) {
    return nullptr;
  }

  const Shape& input_shape = multiply->shape();
  const int64_t rank = input_shape.dimensions_size();
  if (rank < 1 || !IsSingleLastDimReduce(reduce, input_shape) ||
      !OutputIsInputPrefix(input_shape, reduce->shape())) {
    return nullptr;
  }

  const int64_t dot_len = input_shape.dimensions(rank - 1);
  if (dot_len <= 0 || dot_len > kMaxLeftOrderDotK) {
    return nullptr;
  }

  DotDimensionNumbers dot_dims;
  for (int64_t dim = 0; dim + 1 < rank; ++dim) {
    dot_dims.add_lhs_batch_dimensions(dim);
    dot_dims.add_rhs_batch_dimensions(dim);
  }
  dot_dims.add_lhs_contracting_dimensions(rank - 1);
  dot_dims.add_rhs_contracting_dimensions(rank - 1);

  PrecisionConfig precision_config;
  HloInstruction* dot = reduce->parent()->AddInstruction(
      HloInstruction::CreateDot(reduce->shape(), lhs, rhs, dot_dims,
                                precision_config));
  dot->set_metadata(reduce->metadata());
  if (reduce->has_sharding()) {
    dot->set_sharding(reduce->sharding());
  }

  TF_RETURN_IF_ERROR(reduce->parent()->ReplaceInstruction(reduce, dot));
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Rewrite f32 reduce(multiply) to left-order dot candidate: %s",
      dot->ToString().c_str());
  return dot;
}

}  // namespace

absl::StatusOr<bool> NpuDotFp32LeftOrderRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> reductions;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kReduce) {
        reductions.push_back(instr);
      }
    }

    for (HloInstruction* reduce : reductions) {
      TF_ASSIGN_OR_RETURN(HloInstruction* dot, RewriteReduceToDot(reduce));
      changed |= dot != nullptr;
    }
  }

  return changed;
}

}  // namespace xla
