/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#include "xla/xla_npu/service/transform/npu_scatter_window_expander.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <utility>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {
absl::StatusOr<bool> Expand(HloScatterInstruction* scatter) {
  const auto& dn = scatter->scatter_dimension_numbers();
  auto* reducer = scatter->called_computations()[0];
  const auto* root = reducer->root_instruction();
  if (root->opcode() != HloOpcode::kAdd) return false;
  if (scatter->operand_count() != 3 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0)->parameter_number() == root->operand(1)->parameter_number()) {
    return absl::UnimplementedError(
        "NPU scatter-add requires a single-output add(old, update) reducer");
  }
  // Only scatters emitted by this pass may bypass normalization. Shape alone
  // is insufficient: even a point scatter may contain out-of-bounds indices.
  if (scatter->get_frontend_attribute("npu.scatter_add_normalized") == "1") {
    return false;
  }
  auto* data = scatter->mutable_operand(0);
  auto* indices = scatter->mutable_operand(1);
  auto* updates = scatter->mutable_operand(2);
  const Shape& ds = data->shape();
  const Shape& is = indices->shape();
  const Shape& us = updates->shape();
  if (ds.is_dynamic() || is.is_dynamic() || us.is_dynamic() ||
      dn.input_batching_dims_size() != 0 ||
      dn.scatter_indices_batching_dims_size() != 0 ||
      (is.element_type() != S32 && is.element_type() != S64)) {
    return absl::UnimplementedError(
        "NPU scatter-add normalization requires static shapes, signed "
        "indices and normalized batching dimensions");
  }
  const int64_t size = ShapeUtil::ElementsIn(ds);
  const int64_t count = ShapeUtil::ElementsIn(us);
  if (size >= std::numeric_limits<int32_t>::max() ||
      count > std::numeric_limits<int32_t>::max()) {
    return absl::UnimplementedError("NPU scatter-add window exceeds int32 indexing");
  }
  auto* comp = scatter->parent();
  if (count == 0) {
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(scatter, data));
    return true;
  }
  const std::vector<int64_t> udims(us.dimensions().begin(), us.dimensions().end());
  auto shape = [&](PrimitiveType type) { return ShapeUtil::MakeShape(type, udims); };
  auto scalar = [&](int64_t value, PrimitiveType type) {
    return comp->AddInstruction(HloInstruction::CreateConstant(
        type == S64 ? LiteralUtil::CreateR0<int64_t>(value)
                    : LiteralUtil::CreateR0<int32_t>(value)));
  };
  auto constant = [&](int64_t value, PrimitiveType type) {
    return comp->AddInstruction(HloInstruction::CreateBroadcast(
        shape(type), scalar(value, type), {}));
  };
  auto binary = [&](HloOpcode op, HloInstruction* a, HloInstruction* b) {
    return comp->AddInstruction(HloInstruction::CreateBinary(a->shape(), op, a, b));
  };
  auto* zero = constant(0, S32);
  auto* flat_index = zero;
  HloInstruction* valid = comp->AddInstruction(HloInstruction::CreateCompare(
      shape(PRED), zero, zero, ComparisonDirection::kEq));

  // Complement of update_window_dims maps indices' non-vector dimensions to
  // updates. The remaining operand dimensions map window offsets in order.
  std::vector<int64_t> scatter_update_dims;
  for (int64_t d = 0; d < us.dimensions_size(); ++d) {
    if (std::find(dn.update_window_dims().begin(), dn.update_window_dims().end(), d)
        == dn.update_window_dims().end()) scatter_update_dims.push_back(d);
  }
  std::vector<int64_t> operand_window_dim(ds.dimensions_size(), -1);
  int64_t w = 0;
  for (int64_t d = 0; d < ds.dimensions_size(); ++d) {
    if (std::find(dn.inserted_window_dims().begin(), dn.inserted_window_dims().end(), d)
        == dn.inserted_window_dims().end()) {
      operand_window_dim[d] = dn.update_window_dims(w++);
    }
  }
  std::vector<int64_t> strides(ds.dimensions_size(), 1);
  for (int64_t d = ds.dimensions_size() - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * ds.dimensions(d + 1);
  }
  for (int64_t k = 0; k < dn.scatter_dims_to_operand_dims_size(); ++k) {
    const int64_t od = dn.scatter_dims_to_operand_dims(k);
    const int64_t iv = dn.index_vector_dim();
    HloInstruction* component = indices;
    if (iv < is.dimensions_size()) {
      std::vector<int64_t> starts(is.dimensions_size(), 0);
      std::vector<int64_t> limits(is.dimensions().begin(), is.dimensions().end());
      std::vector<int64_t> steps(is.dimensions_size(), 1);
      starts[iv] = k;
      limits[iv] = k + 1;
      auto slice_dims = limits;
      slice_dims[iv] = 1;
      component = comp->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(is.element_type(), slice_dims), indices, starts, limits, steps));
      slice_dims.erase(slice_dims.begin() + iv);
      component = comp->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(is.element_type(), slice_dims), component));
    }
    // index_vector_dim == index rank denotes an implicit singleton vector.
    component = comp->AddInstruction(HloInstruction::CreateBroadcast(
        shape(is.element_type()), component, scatter_update_dims));
    const int64_t width = operand_window_dim[od] < 0 ? 1 :
        us.dimensions(operand_window_dim[od]);
    auto* lower = comp->AddInstruction(HloInstruction::CreateCompare(
        shape(PRED), component, constant(0, is.element_type()), ComparisonDirection::kGe));
    auto* upper = comp->AddInstruction(HloInstruction::CreateCompare(
        shape(PRED), component, constant(ds.dimensions(od) - width, is.element_type()),
        ComparisonDirection::kLe));
    auto* in_bounds = binary(HloOpcode::kAnd, lower, upper);
    valid = binary(HloOpcode::kAnd, valid, in_bounds);
    // Sanitize before narrowing or multiplying: huge s64 / negative indices
    // must not overflow into an apparently valid flattened index.
    component = comp->AddInstruction(HloInstruction::CreateTernary(
        shape(is.element_type()), HloOpcode::kSelect, in_bounds, component,
        constant(0, is.element_type())));
    if (is.element_type() != S32) {
      component = comp->AddInstruction(HloInstruction::CreateConvert(shape(S32), component));
    }
    flat_index = binary(HloOpcode::kAdd, flat_index,
        binary(HloOpcode::kMultiply, component, constant(strides[od], S32)));
  }
  for (int64_t od = 0; od < ds.dimensions_size(); ++od) {
    if (operand_window_dim[od] < 0) continue;
    auto* offset = comp->AddInstruction(HloInstruction::CreateIota(
        shape(S32), operand_window_dim[od]));
    flat_index = binary(HloOpcode::kAdd, flat_index,
        binary(HloOpcode::kMultiply, offset, constant(strides[od], S32)));
  }
  flat_index = comp->AddInstruction(HloInstruction::CreateTernary(
      shape(S32), HloOpcode::kSelect, valid, flat_index, constant(size, S32)));
  flat_index = comp->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {count, 1}), flat_index));
  // HloConstantFolding deliberately skips some broadcast/iota expressions.
  // Evaluate bounded constant index graphs here so static VJP indices do not
  // launch dozens of arithmetic kernels on each loop iteration.
  if (indices->opcode() == HloOpcode::kConstant && count <= 65536) {
    HloEvaluator evaluator;
    Literal literal;
    if (evaluator.TryEvaluate(flat_index, &literal,
                              /*recursively_evaluate_nonconstant_operands=*/true)) {
      flat_index = comp->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
    }
  }
  const auto flat_shape = ShapeUtil::MakeShape(ds.element_type(), {size});
  const auto padded_shape = ShapeUtil::MakeShape(ds.element_type(), {size + 1});
  auto* flat_data = comp->AddInstruction(HloInstruction::CreateReshape(flat_shape, data));
  auto* sink = comp->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(ds.element_type())));
  sink = comp->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(ds.element_type(), {1}), sink, {}));
  flat_data = comp->AddInstruction(HloInstruction::CreateConcatenate(
      padded_shape, {flat_data, sink}, 0));
  auto* flat_updates = comp->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(us.element_type(), {count}), updates));
  ScatterDimensionNumbers point_dims;
  point_dims.add_inserted_window_dims(0);
  point_dims.add_scatter_dims_to_operand_dims(0);
  point_dims.set_index_vector_dim(1);
  auto* result = comp->AddInstruction(HloInstruction::CreateScatter(
      padded_shape, flat_data, flat_index, flat_updates, reducer, point_dims,
      /*indices_are_sorted=*/false, /*unique_indices=*/false));
  result->set_metadata(scatter->metadata());
  result->set_frontend_attribute("npu.scatter_add_normalized", "1");
  result = comp->AddInstruction(HloInstruction::CreateSlice(
      flat_shape, result, {0}, {size}, {1}));
  result = comp->AddInstruction(HloInstruction::CreateReshape(scatter->shape(), result));
  result->set_metadata(scatter->metadata());
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(scatter, result));
  return true;
}
}  // namespace

absl::StatusOr<bool> NpuScatterWindowExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kScatter) continue;
      TF_ASSIGN_OR_RETURN(bool expanded, Expand(static_cast<HloScatterInstruction*>(instr)));
      changed |= expanded;
    }
  }
  return changed;
}
}  // namespace xla
