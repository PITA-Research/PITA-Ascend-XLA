/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */

#include "xla/xla_npu/service/transform/npu_variadic_reduce_expander.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

using Values = std::vector<HloInstruction*>;

absl::Status ValidateReducer(const HloInstruction* reduce) {
  const HloComputation* reducer = reduce->to_apply();
  const HloInstruction* root = reducer->root_instruction();
  const int64_t count = reduce->operand_count() / 2;
  if (root->opcode() != HloOpcode::kTuple ||
      root->operand_count() != count || reducer->num_parameters() != 2 * count) {
    return absl::UnimplementedError(
        "NPU variadic reduce requires a flat tuple reducer");
  }
  for (const HloInstruction* instr : reducer->MakeInstructionPostOrder()) {
    if (instr == root) continue;
    if (!ShapeUtil::IsScalar(instr->shape()) || instr->HasSideEffect() ||
        !instr->control_predecessors().empty() ||
        (instr->opcode() != HloOpcode::kParameter &&
         instr->opcode() != HloOpcode::kConstant &&
         (!instr->IsElementwise() || !instr->called_computations().empty()))) {
      return absl::UnimplementedError(absl::StrCat(
          "NPU cannot expand non-elementwise variadic reducer: ",
          instr->ToString()));
    }
  }
  for (const Shape& shape : reduce->shape().tuple_shapes()) {
    if (!shape.IsArray() || shape.is_dynamic() ||
        (shape.has_layout() &&
         !LayoutUtil::IsMonotonicWithDim0Major(shape.layout()))) {
      return absl::UnimplementedError(
          "NPU variadic reduce requires static C-order array results");
    }
  }
  for (int64_t i = 0; i < count; ++i) {
    if (reduce->operand(i)->shape().is_dynamic()) {
      return absl::UnimplementedError("NPU variadic reduce requires static inputs");
    }
  }
  return absl::OkStatus();
}

// Lift the original scalar computation to arrays with the same dimensions.
// In particular, preserve comparison types, NaN predicates and index tie rules.
Values ApplyReducer(HloComputation* computation, const HloComputation* reducer,
                    const Values& lhs, const Values& rhs) {
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> lifted;
  const int64_t count = lhs.size();
  for (int64_t i = 0; i < count; ++i) {
    lifted[reducer->parameter_instruction(i)] = lhs[i];
    lifted[reducer->parameter_instruction(count + i)] = rhs[i];
  }
  const HloInstruction* root = reducer->root_instruction();
  for (const HloInstruction* instr : reducer->MakeInstructionPostOrder()) {
    if (instr == root || instr->opcode() == HloOpcode::kParameter) continue;
    Shape shape = ShapeUtil::MakeShape(instr->shape().element_type(),
                                      lhs[0]->shape().dimensions());
    if (instr->opcode() == HloOpcode::kConstant) {
      HloInstruction* scalar = computation->AddInstruction(instr->Clone());
      lifted[instr] = computation->AddInstruction(
          HloInstruction::CreateBroadcast(shape, scalar, {}));
    } else {
      Values operands;
      for (const HloInstruction* operand : instr->operands()) {
        operands.push_back(lifted.at(operand));
      }
      lifted[instr] = computation->AddInstruction(
          instr->CloneWithNewOperands(shape, operands));
    }
  }
  Values result;
  for (const HloInstruction* operand : root->operands()) {
    result.push_back(lifted.at(operand));
  }
  return result;
}

HloInstruction* Slice(HloComputation* computation, HloInstruction* value,
                      int64_t axis, int64_t start, int64_t limit) {
  std::vector<int64_t> sizes(value->shape().dimensions().begin(),
                            value->shape().dimensions().end());
  std::vector<int64_t> starts(sizes.size(), 0);
  std::vector<int64_t> limits = sizes;
  std::vector<int64_t> strides(sizes.size(), 1);
  starts[axis] = start;
  limits[axis] = limit;
  sizes[axis] = limit - start;
  return computation->AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(value->shape().element_type(), sizes), value,
      starts, limits, strides));
}

absl::Status Expand(HloInstruction* reduce) {
  TF_RETURN_IF_ERROR(ValidateReducer(reduce));
  HloComputation* computation = reduce->parent();
  const int64_t count = reduce->operand_count() / 2;
  Values values;
  Values init;
  for (int64_t i = 0; i < count; ++i) {
    values.push_back(reduce->mutable_operand(i));
    init.push_back(computation->AddInstruction(HloInstruction::CreateBroadcast(
        reduce->shape().tuple_shapes(i), reduce->mutable_operand(count + i), {})));
  }

  // Empty reductions return init; do not issue ACLNN work on empty tensors.
  if (ShapeUtil::ElementsIn(values[0]->shape()) == 0) {
    return computation->ReplaceInstruction(
        reduce, computation->AddInstruction(HloInstruction::CreateTuple(init)));
  }

  std::vector<int64_t> axes(reduce->dimensions().begin(),
                            reduce->dimensions().end());
  std::sort(axes.rbegin(), axes.rend());
  for (int64_t axis : axes) {
    int64_t size = values[0]->shape().dimensions(axis);
    while (size > 1) {
      Values lhs, rhs;
      const int64_t pairs = size / 2;
      for (HloInstruction* value : values) {
        HloInstruction* even_prefix =
            size % 2 ? Slice(computation, value, axis, 0, 2 * pairs) : value;
        std::vector<int64_t> dims(even_prefix->shape().dimensions().begin(),
                                 even_prefix->shape().dimensions().end());
        dims[axis] = pairs;
        dims.insert(dims.begin() + axis + 1, 2);
        HloInstruction* paired =
            computation->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(value->shape().element_type(), dims),
                even_prefix));
        dims.erase(dims.begin() + axis + 1);
        Shape pair_shape =
            ShapeUtil::MakeShape(value->shape().element_type(), dims);
        // Split adjacent pairs using unit-stride slices. The NPU slice path
        // does not need to interpret a strided slice for this lowering.
        lhs.push_back(computation->AddInstruction(HloInstruction::CreateReshape(
            pair_shape, Slice(computation, paired, axis + 1, 0, 1))));
        rhs.push_back(computation->AddInstruction(HloInstruction::CreateReshape(
            pair_shape, Slice(computation, paired, axis + 1, 1, 2))));
      }
      Values combined = ApplyReducer(computation, reduce->to_apply(), lhs, rhs);
      if (size % 2) {
        for (int64_t i = 0; i < count; ++i) {
          HloInstruction* tail = Slice(computation, values[i], axis, size - 1, size);
          Shape shape = combined[i]->shape();
          shape.set_dimensions(axis, pairs + 1);
          combined[i] =
              computation->AddInstruction(HloInstruction::CreateConcatenate(
                  shape, {combined[i], tail}, axis));
        }
      }
      values = std::move(combined);
      size = (size + 1) / 2;
    }
  }
  // Remove size-one reduced dimensions, then apply init exactly once. Do not
  // pad the tree with init: it need not be an identity for every tuple field.
  for (int64_t i = 0; i < count; ++i) {
    values[i] = computation->AddInstruction(HloInstruction::CreateReshape(
        reduce->shape().tuple_shapes(i), values[i]));
  }
  Values result = ApplyReducer(computation, reduce->to_apply(), init, values);
  HloInstruction* replacement =
      computation->AddInstruction(HloInstruction::CreateTuple(result));
  replacement->set_metadata(reduce->metadata());
  return computation->ReplaceInstruction(reduce, replacement);
}

}  // namespace

absl::StatusOr<bool> NpuVariadicReduceExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kReduce && instr->shape().IsTuple()) {
        TF_RETURN_IF_ERROR(Expand(instr));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
