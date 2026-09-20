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

#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"

#include <algorithm>
#include <functional>
#include <set>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"

// Ascend-MLIR AFIR Dialect
#include "Dialect/AFIR/AFIR.h"

namespace xla {

namespace {

// Helper class to manage the conversion state
class HloToAfirConverter {
 public:
  HloToAfirConverter(mlir::ModuleOp module, mlir::MLIRContext* context,
                     const HloToAfirOptions& options)
      : module_op_ptr_(module.getOperation()),
        context_(context),
        builder_(context),
        options_(options) {}

  mlir::ModuleOp getModule() const {
    return mlir::cast<mlir::ModuleOp>(module_op_ptr_);
  }

  // Convert the entire HLO module
  absl::Status Convert(const HloModule& hlo_module);

  // Convert a single HLO computation
  absl::StatusOr<mlir::func::FuncOp> ConvertComputation(
      const HloComputation& computation, bool is_entry);

  // Set module-level graph attributes
  void SetModuleGraphAttr(const HloModule& hlo_module);
  int64_t SetModuleGraphAttr(const HloComputation& computation);

  // Fix tensor shapes to match max_rank from asc_graph_attr
  // This handles cases like [2,2] -> [2,1,2] reshape by inserting dimensions
  void FixTensorShapes(const HloComputation& computation, int64_t max_rank);

 private:
  // Convert a single HLO instruction
  absl::StatusOr<mlir::Value> ConvertInstruction(
      const HloInstruction* instruction);

  // Instruction-specific converters
  absl::StatusOr<mlir::Value> ConvertParameterOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertConstantOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertAddOp(const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertSubtractOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertMultiplyOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertDivideOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertMaximumOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertMinimumOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertBroadcastOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertTanhOp(const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertNegateOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertExponentialOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertConvertOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertReshapeOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertReduceOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertConcatenateOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertTransposeOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertSliceOp(const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertFusionOp(
      const HloInstruction* instruction);
  absl::StatusOr<mlir::Value> ConvertCallOp(const HloInstruction* instruction);

  std::pair<mlir::ArrayAttr, mlir::ArrayAttr> CreateBinaryOpCommonAttrs(
      mlir::Type result_type, int64_t index_maps_cnt);

  absl::StatusOr<mlir::Value> CreateScalarBroadcastOp(mlir::Location loc,
                                                      mlir::Type result_type,
                                                      mlir::Value operand,
                                                      int64_t rank);

  absl::StatusOr<mlir::Value> CreateRankExtendingBroadcastOp(
      mlir::Location loc, mlir::Type result_type, mlir::Value operand);

  void ProcessHloComputationParameterShapes(
      const HloComputation& computation, int64_t& max_rank,
      const Shape*& shape_for_axes, std::unique_ptr<Shape>& merged_shape);

  // Type conversion
  mlir::Type ConvertShape(const Shape& shape);
  mlir::Type ConvertPrimitiveType(PrimitiveType prim_type);

  // Location conversion
  mlir::Location ConvertLocation(const HloInstruction* instr);
  mlir::Location ConvertLocation(const HloComputation* comp);

  // AFIR-specific helpers
  mlir::AffineMap GetIdentityAffineMap(int64_t rank);
  mlir::ArrayAttr CreateIndexingMaps(llvm::ArrayRef<mlir::AffineMap> maps);
  mlir::afir::AscTensorGroupsAttr CreateDefaultOutputAttr(
      mlir::Type tensor_type);

  // Helper: Check if reshape only adds/removes dimensions of size 1
  // and returns the MLIR type that should be visible to consumers.
  struct ReshapeTypeInfo {
    bool is_one_dim_reshape;
    mlir::Type effective_type;
  };
  ReshapeTypeInfo GetReshapeEffectiveType(const HloInstruction* reshape_instr);
  absl::StatusOr<llvm::SmallVector<mlir::ReassociationIndices>>
  BuildOneDimReassociation(llvm::ArrayRef<int64_t> source_shape,
                           llvm::ArrayRef<int64_t> target_shape);

  static std::string ShapeToString(const llvm::SmallVector<int64_t>& shape);
  llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>
  CollectTargetShapes(const HloComputation& computation);
  const HloInstruction* FindTargetOperation(
      const HloInstruction* instruction,
      const llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>&
          target_shapes);
  std::pair<size_t, int64_t> ComputeBroadcastAlignment(
      const llvm::SmallVector<int64_t>& old_shape,
      const llvm::SmallVector<int64_t>& target_shape);
  llvm::SmallVector<int64_t> ComputeNewShapeForBroadcast(
      const llvm::SmallVector<int64_t>& old_shape,
      const llvm::SmallVector<int64_t>& target_shape, size_t alignment_start);
  bool UpdateFunctionSignature(mlir::BlockArgument blockArg,
                               mlir::Type new_type);
  void UpdateLoadOperation(mlir::Value value, mlir::Type new_type,
                           int64_t max_rank);
  bool ProcessInstruction(
      const HloInstruction* instruction,
      const llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>&
          target_shapes,
      int64_t max_rank);

  bool ValidateAndPrepareInstruction(const HloInstruction* instruction,
                                     mlir::Value& value,
                                     mlir::Value& value_to_update,
                                     mlir::BlockArgument& blockArg,
                                     mlir::RankedTensorType& tensor_type);
  bool IsBroadcastTarget(const HloInstruction* target_op);
  llvm::SmallVector<int64_t> ComputeNewShape(
      const llvm::SmallVector<int64_t>& old_shape,
      const llvm::SmallVector<int64_t>& target_shape,
      const HloInstruction* target_op, int64_t current_rank);
  void ApplyShapeUpdate(mlir::BlockArgument blockArg, mlir::Value value,
                        mlir::Value value_to_update, mlir::Type new_type,
                        int64_t max_rank);

  void FixBroadcastIndexingMaps(const HloComputation& computation);

  bool GetValidBroadcastOp(const HloInstruction* instr,
                           mlir::afir::BroadcastOp& bcast_op);
  mlir::AffineMap BuildResultIndexingMap(int64_t result_rank);
  llvm::SmallVector<bool> IdentifyBroadcastDimensions(
      mlir::RankedTensorType operand_type, mlir::RankedTensorType result_type,
      absl::Span<const int64_t> broadcast_dims);
  mlir::AffineMap BuildOperandIndexingMap(
      mlir::RankedTensorType operand_type, mlir::RankedTensorType result_type,
      const llvm::SmallVector<bool>& is_broadcast_dim,
      absl::Span<const int64_t> broadcast_dims);
  void ProcessSingleBroadcast(const HloInstruction* instr);

  // Module state
  mlir::Operation* module_op_ptr_;
  mlir::MLIRContext* context_;
  mlir::OpBuilder builder_;

  const HloToAfirOptions& options_;

  // Value mapping for SSA conversion
  absl::flat_hash_map<const HloInstruction*, mlir::Value> value_map_;

  // Function mapping
  absl::flat_hash_map<const HloComputation*, mlir::func::FuncOp> function_map_;
};

std::pair<mlir::ArrayAttr, mlir::ArrayAttr>
HloToAfirConverter::CreateBinaryOpCommonAttrs(mlir::Type result_type,
                                              int64_t index_maps_cnt) {
  int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
  auto identity_map = GetIdentityAffineMap(rank);

  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  llvm::SmallVector<mlir::AffineMap> maps;
  for (int64_t i = 0; i < index_maps_cnt; i++) {
    maps.push_back(identity_map);
  }
  auto indexing_maps_attr = CreateIndexingMaps(maps);
  return {indexing_maps_attr, outputs_array};
}

absl::StatusOr<mlir::Value> HloToAfirConverter::CreateScalarBroadcastOp(
    mlir::Location loc, mlir::Type result_type, mlir::Value operand,
    int64_t rank) {
  auto scalar_map = GetIdentityAffineMap(0);
  auto identity_map = GetIdentityAffineMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps;
  maps.push_back(scalar_map);
  maps.push_back(identity_map);

  auto indexing_maps_attr = CreateIndexingMaps(maps);

  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  return builder_
      .create<mlir::afir::BroadcastOp>(
          loc, result_type, operand, indexing_maps_attr, nullptr,
          builder_.getDictionaryAttr({}), nullptr, outputs_array)
      .getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::CreateRankExtendingBroadcastOp(
    mlir::Location loc, mlir::Type result_type, mlir::Value operand) {
  auto operand_type = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
  auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);
  if (!operand_type || !result_tensor_type) {
    return absl::InternalError("Rank-extending broadcast requires ranked tensors");
  }

  int64_t operand_rank = operand_type.getRank();
  int64_t result_rank = result_tensor_type.getRank();
  if (operand_rank > result_rank) {
    NPU_LOG_FORMAT_PRINT(
        WARN,
        "[RankReduceBlock] operand_rank=%d, result_rank=%d",
        operand_rank, result_rank);
    return absl::UnimplementedError(
        "Rank-reducing reshape before concat is not supported");
  }

  auto operand_shape = operand_type.getShape();
  auto result_shape = result_tensor_type.getShape();
  llvm::SmallVector<mlir::AffineExpr> operand_exprs;
  int64_t result_dim = 0;
  for (int64_t operand_dim = 0; operand_dim < operand_rank; ++operand_dim) {
    while (result_dim < result_rank && result_shape[result_dim] == 1 &&
           operand_shape[operand_dim] != 1) {
      result_dim++;
    }
    if (result_dim >= result_rank) {
      return absl::InternalError(
          "Could not align operand shape with concat input shape");
    }
    operand_exprs.push_back(builder_.getAffineDimExpr(result_dim));
    result_dim++;
  }

  auto operand_map =
      mlir::AffineMap::get(result_rank, 0, operand_exprs, context_);
  auto result_map = GetIdentityAffineMap(result_rank);
  auto indexing_maps_attr = CreateIndexingMaps({operand_map, result_map});

  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  return builder_
      .create<mlir::afir::BroadcastOp>(
          loc, result_type, operand, indexing_maps_attr, nullptr,
          builder_.getDictionaryAttr({}), nullptr, outputs_array)
      .getResult();
}

absl::Status HloToAfirConverter::Convert(const HloModule& hlo_module) {
  const HloComputation* entry = hlo_module.entry_computation();
  if (!entry) {
    return absl::InvalidArgumentError("Module has no entry computation");
  }

  // Set module-level asc_graph_attr if emit_afir_dialect is enabled
  if (options_.emit_afir_dialect) {
    SetModuleGraphAttr(hlo_module);
  }

  // Convert all embedded computations FIRST (before entry computation)
  // This ensures that fused/called computations are available when
  // the entry computation references them via fusion or call instructions
  for (const HloComputation* computation : hlo_module.computations()) {
    if (computation == entry) continue;

    TF_ASSIGN_OR_RETURN(auto func, ConvertComputation(*computation, false));
  }

  // Convert entry computation LAST (after all embedded computations)
  TF_ASSIGN_OR_RETURN(auto entry_func, ConvertComputation(*entry, true));

  return absl::OkStatus();
}

absl::StatusOr<mlir::func::FuncOp> HloToAfirConverter::ConvertComputation(
    const HloComputation& computation, bool is_entry) {
  // Collect parameter types
  llvm::SmallVector<mlir::Type> param_types;
  for (const HloInstruction* param : computation.parameter_instructions()) {
    param_types.push_back(ConvertShape(param->shape()));
  }

  // Determine return type
  mlir::Type result_type;
  const HloInstruction* root = computation.root_instruction();

  result_type = ConvertShape(root->shape());

  // Create function type
  mlir::TypeRange param_types_range(param_types);
  auto func_type = builder_.getFunctionType(param_types_range, result_type);

  // Create function location
  mlir::Location loc = ConvertLocation(&computation);

  // Set insertion point to module body before creating function
  builder_.setInsertionPointToEnd(getModule().getBody());

  // Replace all '.' with '_' in computation name for funcOp
  std::string func_name = std::string(computation.name());
  size_t pos = 0;
  while ((pos = func_name.find('.', pos)) != std::string::npos) {
    func_name.replace(pos, 1, "_");
    pos++;
  }

  // Create function
  mlir::func::FuncOp func =
      builder_.create<mlir::func::FuncOp>(loc, func_name, func_type);

  function_map_[&computation] = func;

  // Create function body
  mlir::Block* block = func.addEntryBlock();
  builder_.setInsertionPointToEnd(block);

  // Map parameters and create load nodes when emit_afir_dialect is true
  for (size_t i = 0; i < computation.num_parameters(); ++i) {
    const HloInstruction* param = computation.parameter_instruction(i);
    mlir::Value arg = block->getArgument(i);

    if (options_.preserve_names && !param->name().empty()) {
      arg.setLoc(mlir::NameLoc::get(builder_.getStringAttr(param->name())));
    }

    // Create load node for AFIR dialect
    if (!options_.emit_afir_dialect) {
      value_map_[param] = arg;
      continue;
    }
    mlir::Type param_type = ConvertShape(param->shape());
    int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(param_type).getRank();
    auto identity_map = GetIdentityAffineMap(rank);

    // Create load output attribute
    auto output_attr = CreateDefaultOutputAttr(param_type);
    llvm::SmallVector<mlir::Attribute> output_attrs;
    output_attrs.push_back(output_attr);
    auto outputs_array = builder_.getArrayAttr(output_attrs);

    // Create indexing_maps for load (single map for input)
    llvm::SmallVector<mlir::AffineMap> load_maps;
    load_maps.push_back(identity_map);
    auto indexing_maps_attr = CreateIndexingMaps(load_maps);

    // Create load operation
    auto load_op = builder_.create<mlir::afir::LoadOp>(
        ConvertLocation(param), param_type, arg, indexing_maps_attr,
        nullptr,                         // loop_axis
        builder_.getDictionaryAttr({}),  // ir_attr_def
        nullptr,                         // tmp_buffers
        outputs_array);

    value_map_[param] = load_op.getResult();
  }

  // Convert instructions in post-order (topological sort)
  std::vector<HloInstruction*> instructions =
      computation.MakeInstructionPostOrder();

  for (const HloInstruction* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) continue;

    TF_ASSIGN_OR_RETURN(mlir::Value value, ConvertInstruction(instr));
    value_map_[instr] = value;
  }

  // Set return value with store node when emit_afir_dialect is true
  mlir::Value result = value_map_[computation.root_instruction()];

  if (!options_.emit_afir_dialect) {
    builder_.create<mlir::func::ReturnOp>(
        ConvertLocation(computation.root_instruction()), result);
    return func;
  }

  int64_t rank = llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
  auto identity_map = GetIdentityAffineMap(rank);

  // Create store output attribute
  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  // Create indexing_maps for store (single map for output)
  llvm::SmallVector<mlir::AffineMap> store_maps;
  store_maps.push_back(identity_map);
  auto indexing_maps_attr = CreateIndexingMaps(store_maps);

  // Create store operation
  auto store_op = builder_.create<mlir::afir::StoreOp>(
      ConvertLocation(computation.root_instruction()), result_type, result,
      indexing_maps_attr,
      nullptr,                         // loop_axis
      builder_.getDictionaryAttr({}),  // ir_attr_def
      nullptr,                         // tmp_buffers
      outputs_array);

  // Return the stored value
  builder_.create<mlir::func::ReturnOp>(
      ConvertLocation(computation.root_instruction()), store_op.getResult());

  return func;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertInstruction(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kParameter:
      return ConvertParameterOp(instruction);

    case HloOpcode::kConstant:
      return ConvertConstantOp(instruction);

    case HloOpcode::kAdd:
      return ConvertAddOp(instruction);

    case HloOpcode::kSubtract:
      return ConvertSubtractOp(instruction);

    case HloOpcode::kMultiply:
      return ConvertMultiplyOp(instruction);

    case HloOpcode::kDivide:
      return ConvertDivideOp(instruction);

    case HloOpcode::kBroadcast:
      return ConvertBroadcastOp(instruction);

    case HloOpcode::kMaximum:
      return ConvertMaximumOp(instruction);

    case HloOpcode::kMinimum:
      return ConvertMinimumOp(instruction);

    case HloOpcode::kTanh:
      return ConvertTanhOp(instruction);

    case HloOpcode::kNegate:
      return ConvertNegateOp(instruction);

    case HloOpcode::kExp:
      return ConvertExponentialOp(instruction);

    case HloOpcode::kConvert:
      return ConvertConvertOp(instruction);

    case HloOpcode::kReshape:
      return ConvertReshapeOp(instruction);

    case HloOpcode::kReduce:
      return ConvertReduceOp(instruction);

    case HloOpcode::kConcatenate:
      return ConvertConcatenateOp(instruction);

    case HloOpcode::kTranspose:
      return ConvertTransposeOp(instruction);

    case HloOpcode::kSlice:
      return ConvertSliceOp(instruction);

    case HloOpcode::kFusion:
      return ConvertFusionOp(instruction);

    case HloOpcode::kCall:
      return ConvertCallOp(instruction);

    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported opcode: ", HloOpcodeString(instruction->opcode()),
          " (instruction: ", instruction->name(), ")"));
  }
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertParameterOp(
    const HloInstruction* instruction) {
  // Parameters are handled in computation conversion
  auto it = value_map_.find(instruction);
  if (it == value_map_.end()) {
    return absl::InternalError(absl::StrCat(
        "Parameter not found in value map: ", instruction->name()));
  }
  return it->second;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConstantOp(
    const HloInstruction* instruction) {
  mlir::Location loc = ConvertLocation(instruction);
  const Literal& literal = instruction->literal();

  // Handle different element types
  mlir::Type result_type = ConvertShape(instruction->shape());
  bool is_scalar = literal.shape().dimensions().size() == 0;

  // Create constant based on element type
  switch (literal.shape().element_type()) {
    case F32: {
      float value = literal.GetFirstElement<float>();
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

      if (options_.emit_afir_dialect) {
        if (!is_scalar) {
          return absl::UnimplementedError(
              "AFIR dialect only supports scalar constants");
        }
        int64_t rank = tensor_type.getRank();
        auto indexing_maps_attr =
            CreateIndexingMaps({GetIdentityAffineMap(rank)});

        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(CreateDefaultOutputAttr(result_type));
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        auto value_attr =
            mlir::FloatAttr::get(tensor_type.getElementType(), value);
        auto scalar_op = builder_.create<mlir::afir::ScalarOp>(
            loc, result_type, value_attr, indexing_maps_attr, nullptr,
            builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return scalar_op.getResult();
      } else {
        auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
        auto op =
            builder_.create<mlir::arith::ConstantOp>(loc, result_type, attr);
        return op.getResult();
      }
    }
    case F16: {
      // F16 literal 存储为 Eigen::half；转 float 后用 FloatAttr 发射
      // （test_multiply_by_zero 等 f16 标量常量融合此前在此报
      // "Unsupported constant element type: F16"）
      half hval = literal.GetFirstElement<half>();
      float value = static_cast<float>(hval);
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

      if (options_.emit_afir_dialect) {
        if (!is_scalar) {
          return absl::UnimplementedError(
              "AFIR dialect only supports scalar constants");
        }
        int64_t rank = tensor_type.getRank();
        auto indexing_maps_attr =
            CreateIndexingMaps({GetIdentityAffineMap(rank)});

        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(CreateDefaultOutputAttr(result_type));
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        auto value_attr =
            mlir::FloatAttr::get(tensor_type.getElementType(), value);
        auto scalar_op = builder_.create<mlir::afir::ScalarOp>(
            loc, result_type, value_attr, indexing_maps_attr, nullptr,
            builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return scalar_op.getResult();
      } else {
        auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
        auto op =
            builder_.create<mlir::arith::ConstantOp>(loc, result_type, attr);
        return op.getResult();
      }
    }
    case BF16: {
      bfloat16 bval = literal.GetFirstElement<bfloat16>();
      float value = static_cast<float>(bval);
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

      if (options_.emit_afir_dialect) {
        if (!is_scalar) {
          return absl::UnimplementedError(
              "AFIR dialect only supports scalar constants");
        }
        int64_t rank = tensor_type.getRank();
        auto indexing_maps_attr =
            CreateIndexingMaps({GetIdentityAffineMap(rank)});

        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(CreateDefaultOutputAttr(result_type));
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        auto value_attr =
            mlir::FloatAttr::get(tensor_type.getElementType(), value);
        auto scalar_op = builder_.create<mlir::afir::ScalarOp>(
            loc, result_type, value_attr, indexing_maps_attr, nullptr,
            builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return scalar_op.getResult();
      } else {
        auto attr = mlir::DenseElementsAttr::get(tensor_type, value);
        auto op =
            builder_.create<mlir::arith::ConstantOp>(loc, result_type, attr);
        return op.getResult();
      }
    }
    case PRED: {
      bool value = literal.GetFirstElement<bool>();
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

      if (options_.emit_afir_dialect) {
        if (!is_scalar) {
          return absl::UnimplementedError(
              "AFIR dialect only supports scalar constants");
        }
        int64_t rank = tensor_type.getRank();
        auto indexing_maps_attr =
            CreateIndexingMaps({GetIdentityAffineMap(rank)});

        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(CreateDefaultOutputAttr(result_type));
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        auto value_attr =
            mlir::IntegerAttr::get(tensor_type.getElementType(), value);
        auto scalar_op = builder_.create<mlir::afir::ScalarOp>(
            loc, result_type, value_attr, indexing_maps_attr, nullptr,
            builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return scalar_op.getResult();
      } else {
        auto ele_attr = mlir::DenseElementsAttr::get(tensor_type, value);
        auto op = builder_.create<mlir::arith::ConstantOp>(loc, result_type,
                                                           ele_attr);
        return op.getResult();
      }
    }
    case S32: {
      int32_t value = literal.GetFirstElement<int32_t>();
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);

      if (options_.emit_afir_dialect) {
        if (!is_scalar) {
          return absl::UnimplementedError(
              "AFIR dialect only supports scalar constants");
        }
        int64_t rank = tensor_type.getRank();
        auto indexing_maps_attr =
            CreateIndexingMaps({GetIdentityAffineMap(rank)});

        llvm::SmallVector<mlir::Attribute> output_attrs;
        output_attrs.push_back(CreateDefaultOutputAttr(result_type));
        auto outputs_array = builder_.getArrayAttr(output_attrs);

        auto value_attr =
            mlir::IntegerAttr::get(tensor_type.getElementType(), value);
        auto scalar_op = builder_.create<mlir::afir::ScalarOp>(
            loc, result_type, value_attr, indexing_maps_attr, nullptr,
            builder_.getDictionaryAttr({}), nullptr, outputs_array);
        return scalar_op.getResult();
      } else {
        auto ele_attr = mlir::DenseElementsAttr::get(tensor_type, value);
        auto op = builder_.create<mlir::arith::ConstantOp>(loc, result_type,
                                                           ele_attr);
        return op.getResult();
      }
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported constant element type: ",
                       PrimitiveType_Name(literal.shape().element_type())));
  }
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertAddOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Add operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.add when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    // Create AddOp with all required attributes
    auto add_op = builder_.create<mlir::afir::AddOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return add_op.getResult();
  }
  // Use arith.add as fallback
  auto add_op = builder_.create<mlir::arith::AddIOp>(loc, lhs, rhs);
  return add_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertSubtractOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Subtract operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.sub when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    auto sub_op = builder_.create<mlir::afir::SubOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return sub_op.getResult();
  }

  // Use arith.sub as fallback
  auto sub_op = builder_.create<mlir::arith::SubIOp>(loc, lhs, rhs);
  return sub_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertMultiplyOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Multiply operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.mul when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    auto mul_op = builder_.create<mlir::afir::MulOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return mul_op.getResult();
  }

  // Use arith.mul as fallback
  auto mul_op = builder_.create<mlir::arith::MulIOp>(loc, lhs, rhs);
  return mul_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertDivideOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Divide operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.div when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    auto div_op = builder_.create<mlir::afir::DivOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return div_op.getResult();
  }

  // Use arith.div as fallback
  auto div_op = builder_.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
  return div_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertMaximumOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Maximum operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.MaximumOp when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    auto max_op = builder_.create<mlir::afir::MaximumOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return max_op.getResult();
  }

  return absl::UnimplementedError(
      "Maximum operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertMinimumOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Minimum operation must have 2 operands, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value lhs = value_map_[instruction->operand(0)];
  mlir::Value rhs = value_map_[instruction->operand(1)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.MinimumOp when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 3);
    auto min_op = builder_.create<mlir::afir::MinimumOp>(
        loc, result_type, lhs, rhs, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return min_op.getResult();
  }

  return absl::UnimplementedError(
      "Minimum operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertTanhOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tanh operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.tanh when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 2);
    auto tanh_op = builder_.create<mlir::afir::TanhOp>(
        loc, result_type, operand, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return tanh_op.getResult();
  }

  // Use math.tanh as fallback
  auto tanh_op = builder_.create<mlir::math::TanhOp>(loc, operand);
  return tanh_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertNegateOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Negate operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.NegOp when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 2);
    auto neg_op = builder_.create<mlir::afir::NegOp>(
        loc, result_type, operand, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return neg_op.getResult();
  }

  // Use arith.negf as fallback
  auto neg_op = builder_.create<mlir::arith::NegFOp>(loc, operand);
  return neg_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertExponentialOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Exponential operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.ExpOp when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 2);
    auto exp_op = builder_.create<mlir::afir::ExpOp>(
        loc, result_type, operand, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return exp_op.getResult();
  }

  // Use math.exp as fallback
  auto exp_op = builder_.create<mlir::math::ExpOp>(loc, operand);
  return exp_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConvertOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Convert operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.CastOp when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(result_type, 2);
    auto cast_op = builder_.create<mlir::afir::CastOp>(
        loc, result_type, operand, indexing_maps_attr, nullptr,
        builder_.getDictionaryAttr({}), nullptr, outputs_array);
    return cast_op.getResult();
  }

  // Use arith.ExtFOp, arith.TruncFOp, arith.IndexCastOp, etc. as fallback
  // For simplicity, we'll use the standard arith operations based on the types
  auto operand_type = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
  auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);

  if (!operand_type || !result_tensor_type) {
    return absl::InternalError("Convert operands must be ranked tensors");
  }

  auto operand_elem_type = operand_type.getElementType();
  auto result_elem_type = result_tensor_type.getElementType();
  // Handle float to float conversions
  if (llvm::isa<mlir::FloatType>(operand_elem_type) &&
      llvm::isa<mlir::FloatType>(result_elem_type)) {
    auto operand_width =
        llvm::cast<mlir::FloatType>(operand_elem_type).getWidth();
    auto result_width =
        llvm::cast<mlir::FloatType>(result_elem_type).getWidth();
    if (operand_width < result_width) {
      // Extend floating point
      auto extf_op =
          builder_.create<mlir::arith::ExtFOp>(loc, result_type, operand);
      return extf_op.getResult();
    }
    if (operand_width > result_width) {
      // Truncate floating point
      auto truncf_op =
          builder_.create<mlir::arith::TruncFOp>(loc, result_type, operand);
      return truncf_op.getResult();
    }
    // Same width, just bitcast
    auto bitcast_op =
        builder_.create<mlir::arith::BitcastOp>(loc, result_type, operand);
    return bitcast_op.getResult();
  }

  // Handle integer to integer conversions
  if (llvm::isa<mlir::IntegerType>(operand_elem_type) &&
      llvm::isa<mlir::IntegerType>(result_elem_type)) {
    auto operand_width =
        llvm::cast<mlir::IntegerType>(operand_elem_type).getWidth();
    auto result_width =
        llvm::cast<mlir::IntegerType>(result_elem_type).getWidth();
    if (operand_width < result_width) {
      // Extend integer (signed or unsigned)
      auto operand_signed =
          llvm::cast<mlir::IntegerType>(operand_elem_type).isSigned();
      if (operand_signed) {
        auto exti_op =
            builder_.create<mlir::arith::ExtSIOp>(loc, result_type, operand);
        return exti_op.getResult();
      }
      auto exti_op =
          builder_.create<mlir::arith::ExtUIOp>(loc, result_type, operand);
      return exti_op.getResult();
    }
    if (operand_width > result_width) {
      // Truncate integer
      auto trunci_op =
          builder_.create<mlir::arith::TruncIOp>(loc, result_type, operand);
      return trunci_op.getResult();
    }
    // Same width, just bitcast
    auto bitcast_op =
        builder_.create<mlir::arith::BitcastOp>(loc, result_type, operand);
    return bitcast_op.getResult();
  }

  // Handle float to integer conversions
  if (llvm::isa<mlir::FloatType>(operand_elem_type) &&
      llvm::isa<mlir::IntegerType>(result_elem_type)) {
    auto result_signed =
        llvm::cast<mlir::IntegerType>(result_elem_type).isSigned();
    if (result_signed) {
      auto fptosi_op =
          builder_.create<mlir::arith::FPToSIOp>(loc, result_type, operand);
      return fptosi_op.getResult();
    }
    auto fptoui_op =
        builder_.create<mlir::arith::FPToUIOp>(loc, result_type, operand);
    return fptoui_op.getResult();
  }

  // Handle integer to float conversions
  if (llvm::isa<mlir::IntegerType>(operand_elem_type) &&
      llvm::isa<mlir::FloatType>(result_elem_type)) {
    auto operand_signed =
        llvm::cast<mlir::IntegerType>(operand_elem_type).isSigned();
    if (operand_signed) {
      auto sitofp_op =
          builder_.create<mlir::arith::SIToFPOp>(loc, result_type, operand);
      return sitofp_op.getResult();
    }
    auto uitofp_op =
        builder_.create<mlir::arith::UIToFPOp>(loc, result_type, operand);
    return uitofp_op.getResult();
  }

  return absl::UnimplementedError(absl::StrCat("Unsupported type conversion"));
}

absl::StatusOr<llvm::SmallVector<mlir::ReassociationIndices>>
HloToAfirConverter::BuildOneDimReassociation(
    llvm::ArrayRef<int64_t> source_shape,
    llvm::ArrayRef<int64_t> target_shape) {
  llvm::SmallVector<mlir::ReassociationIndices> reassociation;
  if (target_shape.empty()) {
    if (source_shape.empty()) {
      return reassociation;
    }
    return absl::InvalidArgumentError(
        "Cannot build one-dim reassociation to scalar target");
  }

  int64_t source_idx = 0;
  for (int64_t target_idx = 0; target_idx < target_shape.size();
       ++target_idx) {
    mlir::ReassociationIndices group;

    while (source_idx < source_shape.size() && source_shape[source_idx] == 1) {
      group.push_back(source_idx++);
    }

    if (source_idx < source_shape.size() &&
        source_shape[source_idx] == target_shape[target_idx]) {
      group.push_back(source_idx++);
    } else if (target_shape[target_idx] != 1 || group.empty()) {
      llvm::SmallVector<int64_t> source_shape_vec(source_shape.begin(),
                                                  source_shape.end());
      llvm::SmallVector<int64_t> target_shape_vec(target_shape.begin(),
                                                  target_shape.end());
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot build one-dim reassociation from ",
          ShapeToString(source_shape_vec), " to ",
          ShapeToString(target_shape_vec)));
    }

    while (source_idx < source_shape.size() && source_shape[source_idx] == 1) {
      group.push_back(source_idx++);
    }

    reassociation.push_back(group);
  }

  if (source_idx != source_shape.size()) {
    llvm::SmallVector<int64_t> source_shape_vec(source_shape.begin(),
                                                source_shape.end());
    return absl::InvalidArgumentError(absl::StrCat(
        "One-dim reassociation did not consume source shape ",
        ShapeToString(source_shape_vec)));
  }

  return reassociation;
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertReshapeOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Reshape operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  const Shape& input_shape = instruction->operand(0)->shape();
  const Shape& output_shape = instruction->shape();

  // For non-afir mode, just return the operand (reshape is no-op in MLIR)
  if (!options_.emit_afir_dialect) {
    return operand;
  }
  // For emit_afir_dialect mode, use AFIR operations to implement reshape

  auto all_dims_are_one = [](const Shape& shape) {
    for (int64_t dim : shape.dimensions()) {
      if (dim != 1) return false;
    }
    return true;
  };

  // Special case: [1, ...] -> scalar reshape.
  // This is a true rank-reducing tensor reshape, not a broadcast. Emitting an
  // AFIR broadcast here leaves a rank-1 operand paired with scalar indexing
  // maps, which fails MLIR verification when the pass manager runs.
  if (input_shape.dimensions().size() > 0 &&
      output_shape.dimensions().size() == 0 && all_dims_are_one(input_shape)) {
    auto result_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type);
    auto operand_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!result_tensor_type || !operand_tensor_type) {
      return absl::InternalError(
          "Scalar rank-reducing reshape requires ranked tensors");
    }
    llvm::SmallVector<mlir::ReassociationIndices> reassociation;
    return builder_
        .create<mlir::tensor::CollapseShapeOp>(loc, result_type, operand,
                                               reassociation)
        .getResult();
  }

  // Special case: scalar -> [1] reshape
  // Implement this as a broadcast operation
  // Because some ops e.g. concat, do not support input tensor with empty axis.
  if (input_shape.dimensions().size() == 0 &&
      output_shape.dimensions().size() == 1 &&
      output_shape.dimensions(0) == 1) {
    int64_t rank =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type).getRank();
    return CreateScalarBroadcastOp(loc, result_type, operand, rank);
  }

  // Check if reshape only adds or removes dimensions of size 1
  // This covers cases like:
  // - [1,128] -> [128] or [128] -> [1,128]
  // - [32,3,1] -> [32,3] or [32,3] -> [32,3,1]
  // - [1,128,1] -> [128] or [128] -> [1,128,1]
  // - [1] -> [1,1] or [1,1] -> [1]
  // - [1] -> [1,1,1] or [1,1,1] -> [1]
  bool is_one_dim_reshape = true;
  int64_t input_idx = 0;
  int64_t output_idx = 0;
  while (input_idx < input_shape.dimensions().size() ||
         output_idx < output_shape.dimensions().size()) {
    // Skip dimensions of size 1 in input
    while (input_idx < input_shape.dimensions().size() &&
           input_shape.dimensions(input_idx) == 1) {
      input_idx++;
    }
    // Skip dimensions of size 1 in output
    while (output_idx < output_shape.dimensions().size() &&
           output_shape.dimensions(output_idx) == 1) {
      output_idx++;
    }

    // If both indices are still valid, check if dimensions match
    if (input_idx < input_shape.dimensions().size() &&
        output_idx < output_shape.dimensions().size()) {
      if (input_shape.dimensions(input_idx) !=
          output_shape.dimensions(output_idx)) {
        is_one_dim_reshape = false;
        break;
      }
      input_idx++;
      output_idx++;
    } else if (input_idx < input_shape.dimensions().size() ||
               output_idx < output_shape.dimensions().size()) {
      // One shape still has non-1 dimensions left
      is_one_dim_reshape = false;
      break;
    }
  }

  if (is_one_dim_reshape) {
    auto result_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type);
    auto operand_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!result_tensor_type || !operand_tensor_type) {
      return absl::InternalError(
          "One-dim reshape requires ranked tensor operand and result");
    }
    if (operand_tensor_type == result_tensor_type) {
      return operand;
    }

    llvm::SmallVector<int64_t> input_dims(input_shape.dimensions().begin(),
                                          input_shape.dimensions().end());
    llvm::SmallVector<int64_t> output_dims(output_shape.dimensions().begin(),
                                           output_shape.dimensions().end());
    llvm::SmallVector<int64_t> core_dims;
    for (int64_t dim : input_dims) {
      if (dim != 1) {
        core_dims.push_back(dim);
      }
    }
    if (core_dims.empty()) {
      core_dims.push_back(1);
    }

    mlir::Value reshaped = operand;
    mlir::Type element_type = result_tensor_type.getElementType();
    auto core_type = mlir::RankedTensorType::get(core_dims, element_type);

    if (input_dims != core_dims) {
      TF_ASSIGN_OR_RETURN(auto collapse_reassociation,
                          BuildOneDimReassociation(input_dims, core_dims));
      reshaped = builder_
                     .create<mlir::tensor::CollapseShapeOp>(
                         loc, core_type, reshaped, collapse_reassociation)
                     .getResult();
    }

    if (output_dims != core_dims) {
      TF_ASSIGN_OR_RETURN(auto expand_reassociation,
                          BuildOneDimReassociation(output_dims, core_dims));
      reshaped = builder_
                     .create<mlir::tensor::ExpandShapeOp>(
                         loc, result_type, reshaped, expand_reassociation)
                     .getResult();
    }

    return reshaped;
  }

  // Non-trivial reshapes (merge/split of non size-1 dimensions, e.g.
  // f32[73,2,2,18] -> f32[73,4,18]) cannot be lowered here: the AFIR kernel
  // computes element addresses from the aliased buffer's shape, and aliasing
  // a reshape that merges real dimensions produces wrong strides/indexing.
  // Such reshapes are defused out of fusion computations by
  // NpuUnsupportedFusionDefuser (ContainsNonTrivialReshape) so they run as a
  // standalone kReshape aclrtMemcpy thunk, which is correct because HLO
  // reshape never reorders row-major data. If one still reaches AFIR here,
  // fail loudly rather than emit a silently-wrong kernel.
  return absl::UnimplementedError(
      absl::StrCat("Unsupported reshape: ", ShapeUtil::HumanString(input_shape),
                   " -> ", ShapeUtil::HumanString(output_shape),
                   ". Only reshapes that add/remove dimensions of size 1 are "
                   "currently supported in AFIR; non-trivial reshapes must be "
                   "defused (see NpuUnsupportedFusionDefuser)."));
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertReduceOp(
    const HloInstruction* instruction) {
  // HLO reduce instruction has N data operands followed by N init operands
  // For example, reduce(arg0, arg1, ..., init0, init1, ...)
  // The actual reduction is applied to (arg0, init0), (arg1, init1), etc.

  // Get the number of data operands (total operands / 2)
  int64_t num_data_operands = instruction->operand_count() / 2;
  if (num_data_operands < 1) {
    return absl::InvalidArgumentError(
        "Reduce operation must have at least one data operand");
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Get the reduce computation
  if (!instruction->has_to_apply()) {
    return absl::InvalidArgumentError(
        "Reduce operation must have a to_apply computation");
  }
  const HloComputation* reduce_computation = instruction->to_apply();

  // For simplicity, we'll handle single-operand reduce here
  // Multi-operand reduce would need more complex handling
  if (num_data_operands != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Reduce with ", num_data_operands, " operands is not yet supported"));
  }

  mlir::Value data_operand = value_map_[instruction->operand(0)];

  // Get dimensions to reduce - AFIR reduce ops only support reducing along a single axis
  const auto& dimensions = instruction->dimensions();
  if (dimensions.size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("AFIR reduce operations only support reducing along a "
                     "single dimension, got ",
                     dimensions.size(), " dimensions"));
  }

  // For AFIR dialect, use the dedicated reduce ops
  if (options_.emit_afir_dialect) {
    auto data_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(data_operand.getType());
    auto result_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type);

    if (!data_tensor_type || !result_tensor_type) {
      return absl::InternalError("Reduce operands must be ranked tensors");
    }
    auto [indexing_maps_attr, outputs_array] =
        CreateBinaryOpCommonAttrs(data_operand.getType(), 1);

    // Create axis attribute for AFIR reduce op
    auto axis_attr = builder_.getI32IntegerAttr(dimensions[0]);

    // Check if the reduce computation corresponds to a known operation
    const HloInstruction* reduce_root = reduce_computation->root_instruction();

    // For common reductions like sum, max, min, we can use dedicated AFIR ops
    if (reduce_root->opcode() == HloOpcode::kAdd) {
      // Sum reduction
      return builder_
          .create<mlir::afir::SumOp>(
              loc, result_type, data_operand, axis_attr, indexing_maps_attr,
              nullptr, builder_.getDictionaryAttr({}), nullptr, outputs_array)
          .getResult();
    } else if (reduce_root->opcode() == HloOpcode::kMaximum) {
      // Max reduction
      return builder_
          .create<mlir::afir::MaxOp>(
              loc, result_type, data_operand, axis_attr, indexing_maps_attr,
              nullptr, builder_.getDictionaryAttr({}), nullptr, outputs_array)
          .getResult();
    } else if (reduce_root->opcode() == HloOpcode::kMinimum) {
      // Min reduction
      return builder_
          .create<mlir::afir::MinOp>(
              loc, result_type, data_operand, axis_attr, indexing_maps_attr,
              nullptr, builder_.getDictionaryAttr({}), nullptr, outputs_array)
          .getResult();
    } else {
      // Generic reduce - not yet supported
      return absl::UnimplementedError(absl::StrCat(
          "Reduce computation type '", HloOpcodeString(reduce_root->opcode()),
          "' is not yet supported for AFIR dialect"));
    }
  }

  // Non-AFIR mode: use MLIR standard reduce operations
  // For now, return unimplemented
  return absl::UnimplementedError(
      "Reduce operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertConcatenateOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Concatenate operation must have at least 2 operands, got ",
        instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Get the concatenate dimension
  const auto& concat_dimensions = instruction->dimensions();
  if (concat_dimensions.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Concatenate must have exactly 1 dimension, got ",
                     concat_dimensions.size()));
  }
  int64_t concat_dim = concat_dimensions[0];

  // Collect all operands
  llvm::SmallVector<mlir::Value> operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = value_map_.find(operand);
    if (it == value_map_.end()) {
      return absl::InternalError(absl::StrCat(
          "Concatenate operand not found in value map: ", operand->name()));
    }
    // operands.push_back(it->second);
    mlir::Value operand_value = it->second;
    if (options_.emit_afir_dialect) {
      mlir::Type expected_operand_type = ConvertShape(operand->shape());
      auto expected_tensor_type =
          llvm::dyn_cast<mlir::RankedTensorType>(expected_operand_type);
      auto actual_tensor_type =
          llvm::dyn_cast<mlir::RankedTensorType>(operand_value.getType());
      if (expected_tensor_type && actual_tensor_type &&
          expected_tensor_type.getRank() != actual_tensor_type.getRank()) {
        NPU_LOG_FORMAT_PRINT(
            WARN,
            "[RankMismatch] Concat=%s, operand=%s, opcode=%s, "
            "expected_rank=%d (shape=%s), actual_rank=%d%s",
            instruction->name(),
            operand->name(),
            HloOpcodeString(operand->opcode()),
            expected_tensor_type.getRank(),
            ShapeUtil::HumanString(operand->shape()).c_str(),
            actual_tensor_type.getRank(),
            operand->opcode() == HloOpcode::kReshape
                ? absl::StrCat(", reshape_input=",
                               ShapeUtil::HumanString(
                                   operand->operand(0)->shape()),
                               ", reshape_output=",
                               ShapeUtil::HumanString(operand->shape()))
                      .c_str()
                : "");
        TF_ASSIGN_OR_RETURN(
            operand_value,
            CreateRankExtendingBroadcastOp(ConvertLocation(operand),
                                           expected_operand_type,
                                           operand_value));
      }
    }
    operands.push_back(operand_value);
  }

  // Use AFIR dialect
  if (options_.emit_afir_dialect) {
    auto result_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type);
    if (!result_tensor_type) {
      return absl::InternalError("Concatenate result must be a ranked tensor");
    }

    int64_t result_rank = result_tensor_type.getRank();

    // Build identity indexing maps for all inputs (AFIR ConcatOp uses identity maps)
    auto identity_map = GetIdentityAffineMap(result_rank);
    llvm::SmallVector<mlir::AffineMap> maps;
    // Each input gets an identity map
    for (size_t i = 0; i < operands.size(); ++i) {
      maps.push_back(identity_map);
    }
    auto indexing_maps_attr = CreateIndexingMaps(maps);

    // Build outputs attribute
    auto output_attr = CreateDefaultOutputAttr(result_type);
    llvm::SmallVector<mlir::Attribute> output_attrs;
    output_attrs.push_back(output_attr);
    auto outputs_array = builder_.getArrayAttr(output_attrs);

    // Create concat_axis attribute for AFIR concat op
    auto concat_axis_attr = builder_.getI32IntegerAttr(concat_dim);

    // Create ConcatOp with all operands
    auto concat_op = builder_.create<mlir::afir::ConcatOp>(
        loc, result_type, operands, concat_axis_attr, indexing_maps_attr,
        nullptr,                         // loop_axis
        builder_.getDictionaryAttr({}),  // ir_attr_def
        nullptr,                         // tmp_buffers
        outputs_array);

    return concat_op.getResult();
  }

  // Non-AFIR mode: use MLIR standard operations
  // For now, return unimplemented
  return absl::UnimplementedError(
      "Concatenate operation requires emit_afir_dialect to be enabled");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertTransposeOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Transpose operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  if (!options_.emit_afir_dialect) {
    return absl::UnimplementedError(
        "Transpose operation requires emit_afir_dialect to be enabled");
  }

  auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);
  if (!result_tensor_type) {
    return absl::InternalError("Transpose result must be a ranked tensor");
  }

  int64_t rank = result_tensor_type.getRank();

  // Build identity indexing maps for input and result
  auto identity_map = GetIdentityAffineMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps;
  maps.push_back(identity_map);
  maps.push_back(identity_map);
  auto indexing_maps_attr = CreateIndexingMaps(maps);

  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  // Build permutation attribute from HLO instruction dimensions
  const auto& dimensions = instruction->dimensions();
  llvm::SmallVector<mlir::Attribute> perm_attrs;
  for (int64_t dim : dimensions) {
    perm_attrs.push_back(builder_.getI64IntegerAttr(dim));
  }
  auto permutation_attr = builder_.getArrayAttr(perm_attrs);

  auto transpose_op = builder_.create<mlir::afir::TransposeOp>(
      loc, result_type, operand, permutation_attr, indexing_maps_attr,
      nullptr,                         // loop_axis
      builder_.getDictionaryAttr({}),  // ir_attr_def
      nullptr,                         // tmp_buffers
      outputs_array);

  return transpose_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertSliceOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Slice operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  if (!options_.emit_afir_dialect) {
    return absl::UnimplementedError(
        "Slice operation requires emit_afir_dialect to be enabled");
  }

  auto source_tensor_type =
      llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
  if (!source_tensor_type) {
    return absl::InternalError("Slice source must be a ranked tensor");
  }
  auto result_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(result_type);
  if (!result_tensor_type) {
    return absl::InternalError("Slice result must be a ranked tensor");
  }

  int64_t rank = source_tensor_type.getRank();
  auto source_shape = source_tensor_type.getShape();

  // Compute base_offset = sum(start[i] * product(source_shape[i+1..rank-1]))
  int64_t offset = 0;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t phys = 1;
    for (int64_t j = i + 1; j < rank; ++j) {
      phys *= source_shape[j];
    }
    offset += instruction->slice_starts(i) * phys;
  }

  // ir_attr_def carries:
  //   "offset": linear base offset into the source buffer
  //   "source_shape": original source shape for strides computation in AFIRToASCIRText
  std::string source_shape_str;
  for (int64_t i = 0; i < rank; ++i) {
    if (i > 0) source_shape_str += ",";
    source_shape_str += std::to_string(source_shape[i]);
  }
  auto ir_attr_def = builder_.getDictionaryAttr(
      {builder_.getNamedAttr("offset",
                             builder_.getStringAttr(std::to_string(offset))),
       builder_.getNamedAttr("source_shape",
                             builder_.getStringAttr(source_shape_str))});

  // The slice-load operand must share the same type as the result (slice output shape).
  // operand is the result of the parameter LoadOp; we:
  //   1. Take the parameter LoadOp's input (block argument) as the real operand
  //   2. Update the block argument's type to result_type (slice output shape)
  //   3. Erase the now-unused parameter LoadOp
  mlir::Value raw_arg = operand;
  mlir::Operation* param_load_op = operand.getDefiningOp();
  if (param_load_op && llvm::isa<mlir::afir::LoadOp>(param_load_op) &&
      param_load_op->getNumOperands() == 1) {
    raw_arg = param_load_op->getOperand(0);
    param_load_op->erase();
  }
  // Update the block argument type to match result_type so LoadOp verify passes,
  // and sync the enclosing func's signature so func type stays consistent.
  raw_arg.setType(result_type);
  if (auto block_arg = llvm::dyn_cast<mlir::BlockArgument>(raw_arg)) {
    mlir::Block* block = block_arg.getOwner();
    if (auto func_op =
            llvm::dyn_cast<mlir::func::FuncOp>(block->getParentOp())) {
      unsigned arg_idx = block_arg.getArgNumber();
      llvm::SmallVector<mlir::Type> new_input_types(
          func_op.getFunctionType().getInputs().begin(),
          func_op.getFunctionType().getInputs().end());
      new_input_types[arg_idx] = result_type;
      func_op.setType(builder_.getFunctionType(
          new_input_types, func_op.getFunctionType().getResults()));
    }
  }

  // LoadOp takes exactly ONE identity indexing map.
  int64_t result_rank = result_tensor_type.getRank();
  llvm::SmallVector<mlir::AffineMap> maps;
  maps.push_back(GetIdentityAffineMap(result_rank));
  auto indexing_maps_attr = CreateIndexingMaps(maps);

  auto output_attr = CreateDefaultOutputAttr(result_type);
  llvm::SmallVector<mlir::Attribute> output_attrs;
  output_attrs.push_back(output_attr);
  auto outputs_array = builder_.getArrayAttr(output_attrs);

  auto load_op = builder_.create<mlir::afir::LoadOp>(loc, result_type, raw_arg,
                                                     indexing_maps_attr,
                                                     nullptr,  // loop_axis
                                                     ir_attr_def,
                                                     nullptr,  // tmp_buffers
                                                     outputs_array);

  return load_op.getResult();
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertBroadcastOp(
    const HloInstruction* instruction) {
  if (instruction->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Broadcast operation must have 1 operand, got ",
                     instruction->operand_count()));
  }

  mlir::Location loc = ConvertLocation(instruction);
  mlir::Value operand = value_map_[instruction->operand(0)];
  mlir::Type result_type = ConvertShape(instruction->shape());

  // Use afir.broadcast when emit_afir_dialect is true
  if (options_.emit_afir_dialect) {
    auto result_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(result_type);
    auto operand_tensor_type =
        llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());

    int64_t result_rank = result_tensor_type.getRank();
    int64_t operand_rank = operand_tensor_type.getRank();

    // Get broadcast dimensions from HLO instruction
    const auto& broadcast_dims = instruction->dimensions();

    // Build indexing maps for broadcast
    // For broadcast, we need to map from result dimensions to operand dimensions
    llvm::SmallVector<mlir::AffineExpr> operand_exprs;
    llvm::SmallVector<mlir::AffineExpr> result_exprs;

    // Result map: identity (use all dimensions)
    for (int64_t i = 0; i < result_rank; ++i) {
      result_exprs.push_back(builder_.getAffineDimExpr(i));
    }
    auto result_map =
        mlir::AffineMap::get(result_rank, 0, result_exprs, context_);

    // Operand map: map result dims to operand dims based on broadcast_dims
    // broadcast_dims specifies which dimensions in the result correspond to the operand's dimensions
    // For example: if broadcasting from (2,) to (3, 2) with broadcast_dims={1},
    // then operand map should be (d0, d1) -> (d1)
    llvm::SmallVector<mlir::AffineExpr> exprs;
    for (int64_t i = 0; i < operand_rank; ++i) {
      if (i < broadcast_dims.size()) {
        // This operand dimension maps to a result dimension
        int64_t result_dim = broadcast_dims[i];
        exprs.push_back(builder_.getAffineDimExpr(result_dim));
      } else {
        // This operand dimension is broadcast (size 1), use constant 0
        exprs.push_back(builder_.getAffineConstantExpr(0));
      }
    }
    auto operand_map = mlir::AffineMap::get(result_rank, 0, exprs, context_);

    // Build outputs attribute
    auto output_attr = CreateDefaultOutputAttr(result_type);
    llvm::SmallVector<mlir::Attribute> output_attrs;
    output_attrs.push_back(output_attr);
    auto outputs_array = builder_.getArrayAttr(output_attrs);

    // Build indexing_maps array (2 maps: operand, result)
    llvm::SmallVector<mlir::AffineMap> maps;
    maps.push_back(operand_map);  // operand (broadcasted)
    maps.push_back(result_map);   // result
    auto indexing_maps_attr = CreateIndexingMaps(maps);

    // Create BroadcastOp with all required attributes
    auto broadcast_op = builder_.create<mlir::afir::BroadcastOp>(
        loc, result_type, operand, indexing_maps_attr,
        nullptr,                         // loop_axis (use default)
        builder_.getDictionaryAttr({}),  // ir_attr_def (use default)
        nullptr,                         // tmp_buffers (use default)
        outputs_array);
    return broadcast_op.getResult();
  }

  // Fallback: Broadcast is only supported with AFIR dialect
  return absl::UnimplementedError(
      "Broadcast operation requires emit_afir_dialect=true");
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertFusionOp(
    const HloInstruction* instruction) {
  // Fusion instruction is similar to call - it invokes a fused computation
  // The fused computation should already be converted to a function
  HloComputation* fused_computation = instruction->called_computations()[0];

  // Look up the converted function
  auto it = function_map_.find(fused_computation);
  if (it == function_map_.end()) {
    return absl::InternalError(
        absl::StrCat("Fused computation not found in function map: ",
                     fused_computation->name()));
  }

  mlir::func::FuncOp callee_func = it->second;

  // Collect operands
  llvm::SmallVector<mlir::Value> operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto operand_it = value_map_.find(operand);
    if (operand_it == value_map_.end()) {
      return absl::InternalError(absl::StrCat(
          "Fusion operand not found in value map: ", operand->name()));
    }
    operands.push_back(operand_it->second);
  }

  mlir::Location loc = ConvertLocation(instruction);

  // Create call operation
  auto call_op =
      builder_.create<mlir::func::CallOp>(loc, callee_func, operands);

  // Fusion produces a single result
  return call_op.getResult(0);
}

absl::StatusOr<mlir::Value> HloToAfirConverter::ConvertCallOp(
    const HloInstruction* instruction) {
  // Call instruction invokes a nested computation
  HloComputation* called_computation = instruction->called_computations()[0];

  // Look up the converted function
  auto it = function_map_.find(called_computation);
  if (it == function_map_.end()) {
    return absl::InternalError(
        absl::StrCat("Called computation not found in function map: ",
                     called_computation->name()));
  }

  mlir::func::FuncOp callee_func = it->second;

  // Collect operands
  llvm::SmallVector<mlir::Value> operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto operand_it = value_map_.find(operand);
    if (operand_it == value_map_.end()) {
      return absl::InternalError(absl::StrCat(
          "Call operand not found in value map: ", operand->name()));
    }
    operands.push_back(operand_it->second);
  }

  mlir::Location loc = ConvertLocation(instruction);

  // Create call operation
  auto call_op =
      builder_.create<mlir::func::CallOp>(loc, callee_func, operands);

  // Call produces a single result (for non-tuple results)
  return call_op.getResult(0);
}

HloToAfirConverter::ReshapeTypeInfo HloToAfirConverter::GetReshapeEffectiveType(
    const HloInstruction* reshape_instr) {
  ReshapeTypeInfo result;
  result.is_one_dim_reshape = false;

  if (reshape_instr->opcode() != HloOpcode::kReshape ||
      reshape_instr->operand_count() != 1) {
    result.effective_type = ConvertShape(reshape_instr->shape());
    return result;
  }

  const Shape& input_shape = reshape_instr->operand(0)->shape();
  const Shape& output_shape = reshape_instr->shape();

  // Check if reshape only adds/removes dimensions of size 1
  bool is_one_dim = true;
  int64_t input_idx = 0;
  int64_t output_idx = 0;

  while (input_idx < input_shape.dimensions().size() ||
         output_idx < output_shape.dimensions().size()) {
    // Skip dimensions of size 1 in input
    while (input_idx < input_shape.dimensions().size() &&
           input_shape.dimensions(input_idx) == 1) {
      input_idx++;
    }
    // Skip dimensions of size 1 in output
    while (output_idx < output_shape.dimensions().size() &&
           output_shape.dimensions(output_idx) == 1) {
      output_idx++;
    }

    // If both indices are still valid, check if dimensions match
    if (input_idx < input_shape.dimensions().size() &&
        output_idx < output_shape.dimensions().size()) {
      if (input_shape.dimensions(input_idx) !=
          output_shape.dimensions(output_idx)) {
        is_one_dim = false;
        break;
      }
      input_idx++;
      output_idx++;
    } else if (input_idx < input_shape.dimensions().size() ||
               output_idx < output_shape.dimensions().size()) {
      // One shape still has non-1 dimensions left
      is_one_dim = false;
      break;
    }
  }

  result.is_one_dim_reshape = is_one_dim;
  result.effective_type = ConvertShape(output_shape);
  return result;
}

mlir::Type HloToAfirConverter::ConvertShape(const Shape& shape) {
  if (shape.IsTuple()) {
    // Handle tuple type
    llvm::SmallVector<mlir::Type> element_types;
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      element_types.push_back(ConvertShape(elem_shape));
    }
    return builder_.getTupleType(element_types);
  }

  // Handle tensor type
  mlir::Type element_type = ConvertPrimitiveType(shape.element_type());

  // Create ranked tensor type (even for scalars, use 0-dimensional tensor)
  llvm::ArrayRef<int64_t> shape_dims(shape.dimensions().begin(),
                                     shape.dimensions().end());
  return mlir::RankedTensorType::get(shape_dims, element_type);
}

mlir::Type HloToAfirConverter::ConvertPrimitiveType(PrimitiveType prim_type) {
  switch (prim_type) {
    case PRED:
      if (options_.emit_afir_dialect) {
        return builder_.getIntegerType(8, /*isSigned=*/false);
      }
      return builder_.getI1Type();
    case S8:
      return builder_.getI8Type();
    case S16:
      return builder_.getI16Type();
    case S32:
      return builder_.getI32Type();
    case S64:
      return builder_.getI64Type();
    case U8:
      return builder_.getIntegerType(8, /*isSigned=*/false);
    case U16:
      return builder_.getIntegerType(16, /*isSigned=*/false);
    case U32:
      return builder_.getIntegerType(32, /*isSigned=*/false);
    case U64:
      return builder_.getIntegerType(64, /*isSigned=*/false);
    case F16:
      return builder_.getF16Type();
    case F32:
      return builder_.getF32Type();
    case F64:
      return builder_.getF64Type();
    case BF16:
      return builder_.getBF16Type();
    default:
      // Return none type for unsupported types
      return builder_.getNoneType();
  }
}

mlir::Location HloToAfirConverter::ConvertLocation(
    const HloInstruction* instr) {
  const auto& metadata = instr->metadata();
  if (!metadata.source_file().empty()) {
    return mlir::FileLineColLoc::get(context_, metadata.source_file(),
                                     metadata.source_line(),
                                     metadata.source_column());
  }

  if (options_.preserve_names && !instr->name().empty()) {
    return mlir::NameLoc::get(builder_.getStringAttr(instr->name()));
  }

  return builder_.getUnknownLoc();
}

mlir::Location HloToAfirConverter::ConvertLocation(const HloComputation* comp) {
  if (options_.preserve_names && !comp->name().empty()) {
    return mlir::NameLoc::get(builder_.getStringAttr(comp->name()));
  }
  return builder_.getUnknownLoc();
}

mlir::AffineMap HloToAfirConverter::GetIdentityAffineMap(int64_t rank) {
  if (rank == 0) {
    return builder_.getEmptyAffineMap();
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int64_t i = 0; i < rank; ++i) {
    exprs.push_back(builder_.getAffineDimExpr(i));
  }
  return mlir::AffineMap::get(rank, 0, exprs, context_);
}

mlir::ArrayAttr HloToAfirConverter::CreateIndexingMaps(
    llvm::ArrayRef<mlir::AffineMap> maps) {
  llvm::SmallVector<mlir::Attribute> attrs;
  for (auto map : maps) {
    attrs.push_back(mlir::AffineMapAttr::get(map));
  }
  return builder_.getArrayAttr(attrs);
}

mlir::afir::AscTensorGroupsAttr HloToAfirConverter::CreateDefaultOutputAttr(
    mlir::Type tensor_type) {
  auto ranked_tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(tensor_type);
  if (!ranked_tensor_type) {
    // Return a minimal valid attribute for non-ranked tensors
    return mlir::afir::AscTensorGroupsAttr::get(
        context_, {},  // vectorized_axis
        {},            // vectorized_strides
        -1,            // tensor_id
        -1,            // reuse_id
        mlir::afir::PositionConfigAttr::get(context_, mlir::afir::Position::GM,
                                            0, false),
        -1  // position_id
    );
  }

  int64_t rank = ranked_tensor_type.getRank();

  // Create vectorized_axis (all dimensions)
  llvm::SmallVector<int64_t> vectorized_axis;
  for (int64_t i = 0; i < rank; ++i) {
    vectorized_axis.push_back(i);
  }

  // Create vectorized_strides (column-major strides)
  llvm::SmallVector<mlir::Attribute> vectorized_strides;
  int64_t stride = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    vectorized_strides.push_back(builder_.getI64IntegerAttr(stride));
    stride *= ranked_tensor_type.getDimSize(i);
  }
  std::reverse(vectorized_strides.begin(), vectorized_strides.end());

  // Create position attribute (GM by default)
  auto pos_attr =
      mlir::afir::PositionConfigAttr::get(context_, mlir::afir::Position::GM,
                                          0,     // depth
                                          false  // is_double_buffer
      );

  return mlir::afir::AscTensorGroupsAttr::get(context_, vectorized_axis,
                                              vectorized_strides,
                                              -1,  // tensor_id
                                              -1,  // reuse_id
                                              pos_attr,
                                              -1  // position_id
  );
}

void HloToAfirConverter::SetModuleGraphAttr(const HloModule& hlo_module) {
  // Create axes based on the shape of the entry computation parameters
  const HloComputation* entry = hlo_module.entry_computation();
  if (!entry || entry->num_parameters() == 0) {
    return;
  }

  // Find the maximum rank across all parameters
  int64_t max_rank = 0;
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const Shape& param_shape = entry->parameter_instruction(i)->shape();
    int64_t param_rank = param_shape.dimensions_size();
    if (param_rank > max_rank) {
      max_rank = param_rank;
    }
  }

  // Use the first parameter's shape for axis sizes, but ensure we have axes for max_rank
  const Shape& first_param_shape = entry->parameter_instruction(0)->shape();
  int64_t first_rank = first_param_shape.dimensions_size();

  // AFIR asc_graph_attr cannot be printed and parsed back with an empty axes
  // list.  Treat an all-scalar computation as a one-element graph on a dummy
  // axis so rank-0 fusions such as f32[] add/subtract can still be compiled.
  if (max_rank == 0) {
    max_rank = 1;
  }

  // Create axis attributes
  llvm::SmallVector<mlir::afir::AxisAttr> axes;
  for (int64_t i = 0; i < max_rank; ++i) {
    std::string axis_name = "z" + std::to_string(i);
    std::string size_str = (i < first_rank)
                               ? std::to_string(first_param_shape.dimensions(i))
                               : "1";
    auto axis_attr = mlir::afir::AxisAttr::get(
        context_, i, builder_.getStringAttr(axis_name),
        mlir::afir::AxisType::Original, false, builder_.getStringAttr(size_str),
        nullptr, llvm::ArrayRef<int64_t>());
    axes.push_back(axis_attr);
  }
  // Create asc_graph_attr
  auto graph_attr = mlir::afir::AscGraphAttrGroupsAttr::get(
      context_, -1, axes, mlir::afir::AscGraphType::Compute,
      llvm::ArrayRef<mlir::StringAttr>());

  // Set module attribute
  getModule()->setAttr("afir.asc_graph_attr", graph_attr);
}

void HloToAfirConverter::ProcessHloComputationParameterShapes(
    const HloComputation& computation, int64_t& max_rank,
    const Shape*& shape_for_axes, std::unique_ptr<Shape>& merged_shape) {
  // Iterate through all parameters, find the maximum rank and merge dimensions with the same rank
  for (int64_t i = 0; i < computation.num_parameters(); ++i) {
    const Shape& param_shape = computation.parameter_instruction(i)->shape();
    int64_t param_rank = param_shape.dimensions_size();
    // Find parameters with greater rank, update maximum rank and reset merge shape
    if (param_rank > max_rank) {
      max_rank = param_rank;
      shape_for_axes = &param_shape;
      merged_shape = nullptr;
    }
    // The ranks are the same, and the dimensions are merged (taking the maximum value of each axis)
    else if (param_rank == max_rank && shape_for_axes != nullptr) {
      // Initializing merged shapes for the first time
      if (!merged_shape) {
        std::vector<int64_t> dims;
        for (int64_t dim = 0; dim < max_rank; ++dim) {
          dims.push_back(shape_for_axes->dimensions(dim));
        }
        merged_shape = std::make_unique<Shape>(
            ShapeUtil::MakeShape(shape_for_axes->element_type(), dims));
      }

      // Take the maximum value dimension by dimension and generate a new merged dimension.
      std::vector<int64_t> merged_dims;
      for (int64_t dim = 0; dim < max_rank; ++dim) {
        int64_t existing_size = merged_shape->dimensions(dim);
        int64_t new_size = param_shape.dimensions(dim);
        int64_t max_size = std::max(existing_size, new_size);
        merged_dims.push_back(max_size);
      }

      // Update merged shapes
      merged_shape = std::make_unique<Shape>(
          ShapeUtil::MakeShape(shape_for_axes->element_type(), merged_dims));
    }
  }

  // If merged shapes exist, use the merged shape
  if (merged_shape) {
    shape_for_axes = merged_shape.get();
  }
}

int64_t HloToAfirConverter::SetModuleGraphAttr(
    const HloComputation& computation) {
  // For computations without parameters, derive axis information from the root instruction's output shape
  const Shape* shape_for_axes = nullptr;
  std::unique_ptr<Shape>
      merged_shape;  // For storing merged shape when comparing shapes
  int64_t max_rank = 0;

  // For computations without parameters, use the root instruction's output shape
  // For computations with parameters, use the maximum rank across all parameters
  const HloInstruction* root = computation.root_instruction();
  if (root != nullptr && root->shape().IsArray()) {
    shape_for_axes = &root->shape();
    max_rank = shape_for_axes->dimensions_size();
  }
  if (computation.num_parameters() > 0) {
    // Find the maximum rank across all parameters
    ProcessHloComputationParameterShapes(computation, max_rank, shape_for_axes,
                                         merged_shape);
  }

  // Use the derived shape for axis sizes
  int64_t first_rank = shape_for_axes->dimensions_size();

  // AFIR asc_graph_attr cannot be printed and parsed back with an empty axes
  // list.  Treat an all-scalar computation as a one-element graph on a dummy
  // axis so rank-0 fusions such as f32[] add/subtract can still be compiled.
  if (max_rank == 0) {
    max_rank = 1;
  }

  // Create axis attributes
  llvm::SmallVector<mlir::afir::AxisAttr> axes;
  for (int64_t i = 0; i < max_rank; ++i) {
    std::string axis_name = "z" + std::to_string(i);
    std::string size_str =
        (i < first_rank) ? std::to_string(shape_for_axes->dimensions(i)) : "1";
    auto axis_attr = mlir::afir::AxisAttr::get(
        context_, i, builder_.getStringAttr(axis_name),
        mlir::afir::AxisType::Original, false, builder_.getStringAttr(size_str),
        nullptr, llvm::ArrayRef<int64_t>());
    axes.push_back(axis_attr);
  }
  // Create asc_graph_attr
  auto asc_graph_attr = mlir::afir::AscGraphAttrGroupsAttr::get(
      context_, -1, axes, mlir::afir::AscGraphType::Compute,
      llvm::ArrayRef<mlir::StringAttr>());

  // Set module attribute
  getModule()->setAttr("afir.asc_graph_attr", asc_graph_attr);

  return max_rank;
}

std::string HloToAfirConverter::ShapeToString(
    const llvm::SmallVector<int64_t>& shape) {
  std::string shape_str;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      shape_str += ", ";
    }
    shape_str += std::to_string(shape[i]);
  }
  return shape_str;
}

llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>
HloToAfirConverter::CollectTargetShapes(const HloComputation& computation) {
  llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>
      target_shapes;

  for (const HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() == HloOpcode::kReshape) {
      // Check if this reshape is used by a concat or broadcast
      for (const HloInstruction* user : instr->users()) {
        if (user->opcode() == HloOpcode::kConcatenate ||
            user->opcode() == HloOpcode::kBroadcast) {
          const Shape& shape = instr->shape();
          llvm::SmallVector<int64_t> shape_vec(shape.dimensions().begin(),
                                               shape.dimensions().end());
          target_shapes[user] = shape_vec;

          NPU_LOG_FORMAT_PRINT(
              DEBUG, "Found target shape from reshape %s (used by %s): [%s]",
              std::string(instr->name()).c_str(),
              std::string(user->name()).c_str(),
              ShapeToString(shape_vec).c_str());
          break;
        }
      }
    } else if (instr->opcode() == HloOpcode::kBroadcast) {
      // For direct broadcast operation, use its output shape as target shape
      const Shape& shape = instr->shape();
      llvm::SmallVector<int64_t> shape_vec(shape.dimensions().begin(),
                                           shape.dimensions().end());
      target_shapes[instr] = shape_vec;

      NPU_LOG_FORMAT_PRINT(DEBUG, "Found target shape from broadcast %s: [%s]",
                           std::string(instr->name()).c_str(),
                           ShapeToString(shape_vec).c_str());
    }
  }

  return target_shapes;
}

const HloInstruction* HloToAfirConverter::FindTargetOperation(
    const HloInstruction* instruction,
    const llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>&
        target_shapes) {
  for (const HloInstruction* user : instruction->users()) {
    if (target_shapes.count(user)) {
      return user;
    }
  }
  return nullptr;
}

std::pair<size_t, int64_t> HloToAfirConverter::ComputeBroadcastAlignment(
    const llvm::SmallVector<int64_t>& old_shape,
    const llvm::SmallVector<int64_t>& target_shape) {
  size_t old_rank = old_shape.size();
  size_t target_rank = target_shape.size();
  size_t best_alignment_start = 0;
  int64_t best_match_score = -1;

  // Try aligning old_shape[0] to target_shape[0], target_shape[1], ..., target_shape[target_rank - old_rank]
  for (size_t align_start = 0; align_start + old_rank <= target_rank;
       ++align_start) {
    int64_t match_score = 0;

    // Check if old_shape matches target_shape at this alignment
    for (size_t i = 0; i < old_rank; ++i) {
      size_t target_idx = align_start + i;
      if (old_shape[i] == target_shape[target_idx]) {
        match_score += 10;  // Exact match gets high score
      } else if (target_shape[target_idx] == 1) {
        match_score += 1;  // Can broadcast from 1 gets low score
      } else if (old_shape[i] == 1) {
        match_score += 1;  // Can broadcast to 1 gets low score
      } else {
        match_score = -1;  // Mismatch
        break;
      }
    }

    // Prefer alignments that maximize match_score
    if (match_score > best_match_score) {
      best_match_score = match_score;
      best_alignment_start = align_start;
    }
  }

  return {best_alignment_start, best_match_score};
}

llvm::SmallVector<int64_t> HloToAfirConverter::ComputeNewShapeForBroadcast(
    const llvm::SmallVector<int64_t>& old_shape,
    const llvm::SmallVector<int64_t>& target_shape, size_t alignment_start) {
  size_t old_rank = old_shape.size();
  size_t target_rank = target_shape.size();
  llvm::SmallVector<int64_t> new_shape(target_shape.begin(),
                                       target_shape.end());

  // Fill new_shape based on best alignment
  for (size_t i = 0; i < target_rank; ++i) {
    if (i >= alignment_start && i < alignment_start + old_rank) {
      // This position has a corresponding old_shape dimension
      size_t old_idx = i - alignment_start;
      new_shape[i] = old_shape[old_idx];
    } else {
      // This position doesn't have corresponding old_shape dimension, fill with 1
      new_shape[i] = 1;
    }
  }

  return new_shape;
}

bool HloToAfirConverter::UpdateFunctionSignature(mlir::BlockArgument blockArg,
                                                 mlir::Type new_type) {
  if (auto func =
          dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp())) {
    auto func_type = func.getFunctionType();
    llvm::SmallVector<mlir::Type> inputs(func_type.getInputs().begin(),
                                         func_type.getInputs().end());
    inputs[blockArg.getArgNumber()] = new_type;

    auto new_func_type = mlir::FunctionType::get(func.getContext(), inputs,
                                                 func.getResultTypes());

    auto old_ranked_type = dyn_cast<mlir::RankedTensorType>(
        func_type.getInputs()[blockArg.getArgNumber()]);
    auto new_ranked_type = dyn_cast<mlir::RankedTensorType>(new_type);
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "Updating function %s arg %d from rank %d to rank %d",
                         func.getName().str().c_str(), blockArg.getArgNumber(),
                         old_ranked_type ? old_ranked_type.getRank() : -1,
                         new_ranked_type ? new_ranked_type.getRank() : -1);

    func.setType(new_func_type);
    blockArg.setType(new_type);

    auto updated_ranked_type = dyn_cast<mlir::RankedTensorType>(
        func.getFunctionType().getInputs()[blockArg.getArgNumber()]);
    NPU_LOG_FORMAT_PRINT(
        DEBUG, "After update: rank %d",
        updated_ranked_type ? updated_ranked_type.getRank() : -1);
    return true;
  }
  return false;
}

void HloToAfirConverter::UpdateLoadOperation(mlir::Value value,
                                             mlir::Type new_type,
                                             int64_t max_rank) {
  if (mlir::Operation* op = value.getDefiningOp()) {
    if (auto loadOp = dyn_cast<mlir::afir::LoadOp>(op)) {
      // Cast new_type to mlir::TensorType if it's a RankedTensorType
      if (auto tensor_type = dyn_cast<mlir::TensorType>(new_type)) {
        loadOp.getResult().setType(tensor_type);
      } else {
        // If new_type is not a TensorType, this is an error
        // Log and return without updating
        NPU_LOG_FORMAT_PRINT(
            DEBUG,
            "  Warning: new_type is not a TensorType, skipping LoadOp update");
        return;
      }
      loadOp.setIndexingMapsAttr(
          CreateIndexingMaps({GetIdentityAffineMap(max_rank)}));
    }
  }
}

bool HloToAfirConverter::ValidateAndPrepareInstruction(
    const HloInstruction* instruction, mlir::Value& value,
    mlir::Value& value_to_update, mlir::BlockArgument& blockArg,
    mlir::RankedTensorType& tensor_type) {
  auto it = value_map_.find(instruction);
  if (it == value_map_.end()) {
    return false;
  }

  value = it->second;
  NPU_LOG_FORMAT_PRINT(DEBUG, "Processing instruction: %s",
                       std::string(instruction->name()).c_str());

  // Check if this value is produced by a load operation
  value_to_update = value;
  if (mlir::Operation* op = value.getDefiningOp()) {
    if (auto loadOp = dyn_cast<mlir::afir::LoadOp>(op)) {
      value_to_update = loadOp.getInput();
      NPU_LOG_FORMAT_PRINT(
          DEBUG, "Found load op, will update its input (block argument)");
    }
  }

  // Check if the value to update is a block argument
  blockArg = dyn_cast<mlir::BlockArgument>(value_to_update);
  if (!blockArg) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "This is NOT a BlockArgument, skipping");
    return false;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "This is a BlockArgument, arg_num = %d",
                       blockArg.getArgNumber());

  tensor_type = dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensor_type) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Not a tensor type");
    return false;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Current rank: %d", tensor_type.getRank());
  return true;
}

bool HloToAfirConverter::IsBroadcastTarget(const HloInstruction* target_op) {
  if (target_op->opcode() == HloOpcode::kBroadcast) {
    return true;
  }
  if (target_op->opcode() == HloOpcode::kConcatenate) {
    // Check if any of the concat's users is a broadcast
    for (const HloInstruction* concat_user : target_op->users()) {
      if (concat_user->opcode() == HloOpcode::kBroadcast) {
        return true;
      }
    }
  }
  return false;
}

llvm::SmallVector<int64_t> HloToAfirConverter::ComputeNewShape(
    const llvm::SmallVector<int64_t>& old_shape,
    const llvm::SmallVector<int64_t>& target_shape,
    const HloInstruction* target_op, int64_t current_rank) {
  // Handle scalar case (rank 0)
  if (current_rank == 0) {
    llvm::SmallVector<int64_t> new_shape(target_shape.size(), 1);
    return new_shape;
  }

  // Check if the target operation is a broadcast
  if (IsBroadcastTarget(target_op)) {
    auto [alignment_start, match_score] =
        ComputeBroadcastAlignment(old_shape, target_shape);

    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Broadcast best alignment: start_pos=%d, match_score=%d",
        alignment_start, match_score);

    auto new_shape =
        ComputeNewShapeForBroadcast(old_shape, target_shape, alignment_start);

    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Broadcast target: old_shape=[%s], computed new_shape=[%s]",
        ShapeToString(old_shape).c_str(), ShapeToString(new_shape).c_str());
    return new_shape;
  }
  // For reshape: use target_shape directly
  NPU_LOG_FORMAT_PRINT(DEBUG, "Reshape target: using target_shape directly");
  return llvm::SmallVector<int64_t>(target_shape.begin(), target_shape.end());
}

void HloToAfirConverter::ApplyShapeUpdate(mlir::BlockArgument blockArg,
                                          mlir::Value value,
                                          mlir::Value value_to_update,
                                          mlir::Type new_type,
                                          int64_t max_rank) {
  // Extract shape for logging
  if (auto ranked_type = dyn_cast<mlir::RankedTensorType>(new_type)) {
    llvm::SmallVector<int64_t> shape(ranked_type.getShape().begin(),
                                     ranked_type.getShape().end());
    NPU_LOG_FORMAT_PRINT(DEBUG, "New shape: [%s]",
                         ShapeToString(shape).c_str());
  }

  // Update the block argument
  value_to_update.setType(new_type);

  // Update function signature
  UpdateFunctionSignature(blockArg, new_type);

  // Update load operation's output type and indexing maps
  UpdateLoadOperation(value, new_type, max_rank);
}

bool HloToAfirConverter::ProcessInstruction(
    const HloInstruction* instruction,
    const llvm::DenseMap<const HloInstruction*, llvm::SmallVector<int64_t>>&
        target_shapes,
    int64_t max_rank) {
  mlir::Value value;
  mlir::Value value_to_update;
  mlir::BlockArgument blockArg;
  mlir::RankedTensorType tensor_type;

  // Validate and prepare instruction for processing
  if (!ValidateAndPrepareInstruction(instruction, value, value_to_update,
                                     blockArg, tensor_type)) {
    return false;
  }

  int64_t current_rank = tensor_type.getRank();
  // If rank already matches max_rank, no need to fix
  if (current_rank == max_rank) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Rank already matches max_rank, skipping");
    return false;
  }

  // Find the target shape for this instruction
  const HloInstruction* target_op =
      FindTargetOperation(instruction, target_shapes);
  if (!target_op) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "No target concat/broadcast found, skipping");
    return false;
  }

  // Skip if this instruction has any user that is not the target broadcast/concat.
  // Mutating the shape of a parameter that is also used by elementwise ops
  // (e.g. multiply, add) will cause those ops to infer a different result type
  // than what was declared during conversion, leading to verification failures.
  for (const HloInstruction* user : instruction->users()) {
    if (user == target_op) continue;
    if (user->opcode() == HloOpcode::kBroadcast ||
        user->opcode() == HloOpcode::kConcatenate) {
      continue;
    }
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Instruction %s has non-broadcast/concat user %s, skipping shape fix",
        std::string(instruction->name()).c_str(),
        std::string(user->name()).c_str());
    return false;
  }

  const llvm::SmallVector<int64_t>& target_shape =
      target_shapes.find(target_op)->second;
  NPU_LOG_FORMAT_PRINT(DEBUG, "Target shape from %s: [%s]",
                       std::string(target_op->name()).c_str(),
                       ShapeToString(target_shape).c_str());

  // Get the old shape (current tensor shape)
  llvm::SmallVector<int64_t> old_shape(tensor_type.getShape().begin(),
                                       tensor_type.getShape().end());

  // Compute new shape based on target operation type
  llvm::SmallVector<int64_t> new_shape =
      ComputeNewShape(old_shape, target_shape, target_op, current_rank);

  // Create new tensor type
  auto new_type =
      mlir::RankedTensorType::get(new_shape, tensor_type.getElementType());

  // Apply the shape update
  ApplyShapeUpdate(blockArg, value, value_to_update, new_type, max_rank);

  return true;
}

bool HloToAfirConverter::GetValidBroadcastOp(
    const HloInstruction* instr, mlir::afir::BroadcastOp& bcast_op) {
  if (instr->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  auto it = value_map_.find(instr);
  if (it == value_map_.end()) {
    return false;
  }

  mlir::Value broadcast_result = it->second;
  mlir::Operation* broadcast_op = broadcast_result.getDefiningOp();
  if (!broadcast_op) {
    return false;
  }

  bcast_op = dyn_cast<mlir::afir::BroadcastOp>(broadcast_op);
  return static_cast<bool>(bcast_op);
}

mlir::AffineMap HloToAfirConverter::BuildResultIndexingMap(
    int64_t result_rank) {
  // Result map: identity
  llvm::SmallVector<mlir::AffineExpr> result_exprs;
  for (int64_t i = 0; i < result_rank; ++i) {
    result_exprs.push_back(builder_.getAffineDimExpr(i));
  }
  return mlir::AffineMap::get(result_rank, 0, result_exprs, context_);
}

llvm::SmallVector<bool> HloToAfirConverter::IdentifyBroadcastDimensions(
    mlir::RankedTensorType operand_type, mlir::RankedTensorType result_type,
    absl::Span<const int64_t> broadcast_dims) {
  int64_t operand_rank = operand_type.getRank();
  int64_t result_rank = result_type.getRank();

  auto operand_shape = operand_type.getShape();
  auto result_shape = result_type.getShape();

  // Identify broadcast vs data dimensions.
  // Use broadcast_dims to map operand dim i to the correct result dim,
  // rather than comparing by same-index position (which is wrong when
  // broadcast_dims reorders dimensions, e.g. broadcast [1,4]->[64,1,4]
  // with broadcast_dims={1,2} where operand dim 0 maps to result dim 1).
  llvm::SmallVector<bool> is_broadcast_dim(operand_rank, false);
  for (int64_t i = 0; i < operand_rank; ++i) {
    if (operand_shape[i] == 1) {
      int64_t result_dim = -1;
      if (i < (int64_t)broadcast_dims.size()) {
        result_dim = broadcast_dims[i];
      }
      if (result_dim >= 0 && result_dim < result_rank &&
          result_shape[result_dim] == 1) {
        is_broadcast_dim[i] = false;
      } else {
        is_broadcast_dim[i] = true;
      }
    }
  }

  return is_broadcast_dim;
}

mlir::AffineMap HloToAfirConverter::BuildOperandIndexingMap(
    mlir::RankedTensorType operand_type, mlir::RankedTensorType result_type,
    const llvm::SmallVector<bool>& is_broadcast_dim,
    absl::Span<const int64_t> broadcast_dims) {
  int64_t operand_rank = operand_type.getRank();
  int64_t result_rank = result_type.getRank();

  auto operand_shape = operand_type.getShape();
  auto result_shape = result_type.getShape();

  // Build operand map
  llvm::SmallVector<mlir::AffineExpr> operand_exprs;
  int64_t data_dim_idx = 0;
  for (int64_t i = 0; i < operand_rank; ++i) {
    if (is_broadcast_dim[i]) {
      operand_exprs.push_back(builder_.getAffineConstantExpr(0));
    } else {
      if (data_dim_idx < broadcast_dims.size()) {
        int64_t result_dim = broadcast_dims[data_dim_idx];
        operand_exprs.push_back(builder_.getAffineDimExpr(result_dim));
        data_dim_idx++;
      } else {
        // Fallback: find matching dimension
        bool found = false;
        for (int64_t j = 0; j < result_rank; ++j) {
          if (operand_shape[i] == result_shape[j]) {
            operand_exprs.push_back(builder_.getAffineDimExpr(j));
            found = true;
            break;
          }
        }
        if (!found) {
          operand_exprs.push_back(builder_.getAffineConstantExpr(0));
        }
      }
    }
  }

  return mlir::AffineMap::get(result_rank, 0, operand_exprs, context_);
}

void HloToAfirConverter::ProcessSingleBroadcast(const HloInstruction* instr) {
  mlir::afir::BroadcastOp bcast_op;
  if (!GetValidBroadcastOp(instr, bcast_op)) {
    return;
  }

  // Get the operand
  mlir::Value operand = bcast_op.getOperand();
  auto operand_tensor_type =
      dyn_cast<mlir::RankedTensorType>(operand.getType());
  auto broadcast_result = bcast_op.getResult();
  auto result_tensor_type =
      dyn_cast<mlir::RankedTensorType>(broadcast_result.getType());
  if (!operand_tensor_type || !result_tensor_type) {
    return;
  }

  // Get broadcast_dims from HLO instruction
  const auto& broadcast_dims = instr->dimensions();

  // Build result indexing map
  auto result_map = BuildResultIndexingMap(result_tensor_type.getRank());

  // Identify broadcast dimensions
  auto is_broadcast_dim =
      IdentifyBroadcastDimensions(operand_tensor_type, result_tensor_type,
                                  broadcast_dims);

  // Build operand indexing map
  auto operand_map =
      BuildOperandIndexingMap(operand_tensor_type, result_tensor_type,
                              is_broadcast_dim, broadcast_dims);

  // Update the broadcast operation's indexing_maps
  llvm::SmallVector<mlir::AffineMap> new_maps;
  new_maps.push_back(operand_map);
  new_maps.push_back(result_map);
  bcast_op.setIndexingMapsAttr(CreateIndexingMaps(new_maps));
}

void HloToAfirConverter::FixBroadcastIndexingMaps(
    const HloComputation& computation) {
  for (const HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() == HloOpcode::kBroadcast) {
      ProcessSingleBroadcast(instr);
    }
  }
}

void HloToAfirConverter::FixTensorShapes(const HloComputation& computation,
                                         int64_t max_rank) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "FixTensorShapes called with max_rank = %d",
                       max_rank);
  NPU_LOG_FORMAT_PRINT(DEBUG, "value_map_ has %d entries", value_map_.size());

  // Reshape is now represented explicitly with tensor.collapse_shape /
  // tensor.expand_shape. Do not mutate block argument, function signature, or
  // load result types here: doing so can make concat operands disagree with the
  // original HLO result type, e.g. 64x7 + 64x7 + 64x7 + 64x7 -> 64x19.
  // Keep only the broadcast indexing-map repair, which does not change value
  // types.
  FixBroadcastIndexingMaps(computation);
}

}  // namespace

// ============================================================================
// Public API Implementation
// ============================================================================

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(
    const HloModule& module, mlir::MLIRContext* context,
    const HloToAfirOptions& options) {
  context->loadDialect<mlir::tensor::TensorDialect>();

  // Create MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

  HloToAfirConverter converter(*mlir_module, context, options);
  TF_RETURN_IF_ERROR(converter.Convert(module));

  if (options.verify_module) {
    if (failed(mlir_module->verify())) {
      return absl::InternalError("Generated AFIR module failed verification");
    }
  }

  return mlir_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloComputationToAfir(
    const HloComputation& computation, mlir::MLIRContext* context,
    const HloToAfirOptions& options) {
  context->loadDialect<mlir::tensor::TensorDialect>();

  // Create MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

  HloToAfirConverter converter(*mlir_module, context, options);

  // Collect all computations that this computation depends on
  std::set<const HloComputation*> needed_computations;
  std::function<void(const HloComputation*)> collect_dependencies =
      [&](const HloComputation* comp) {
        if (needed_computations.count(comp)) return;
        needed_computations.insert(comp);

        // Recursively collect fused and called computations
        for (const HloInstruction* instruction : comp->instructions()) {
          if (instruction->opcode() == HloOpcode::kFusion) {
            collect_dependencies(
                instruction->fused_expression_root()->parent());
          } else if (instruction->opcode() == HloOpcode::kCall ||
                     instruction->opcode() == HloOpcode::kConditional) {
            for (const HloComputation* called_comp :
                 instruction->called_computations()) {
              collect_dependencies(called_comp);
            }
          }
        }
      };

  collect_dependencies(&computation);

  // Convert all needed computations (non-entry first, then entry)
  for (const HloComputation* comp : needed_computations) {
    if (comp == &computation) continue;  // Skip entry computation for now
    TF_ASSIGN_OR_RETURN(
        auto func, converter.ConvertComputation(*comp, /*is_entry=*/false));
  }

  // Convert entry computation last
  TF_ASSIGN_OR_RETURN(
      auto func, converter.ConvertComputation(computation, /*is_entry=*/true));

  // Set module-level asc_graph_attr and fix tensor shapes if emit_afir_dialect is enabled
  NPU_LOG_FORMAT_PRINT(DEBUG, "emit_afir_dialect = %d",
                       options.emit_afir_dialect);
  if (options.emit_afir_dialect) {
    int64_t max_rank = converter.SetModuleGraphAttr(computation);

    // Check if the computation contains any concatenate/broadcast operations
    for (const HloInstruction* instr : computation.instructions()) {
      if (instr->opcode() == HloOpcode::kConcatenate ||
          instr->opcode() == HloOpcode::kBroadcast) {
        converter.FixTensorShapes(computation, max_rank);
        break;
      }
    }
  }

  // Run DCE pass to eliminate dead code (unused operations)
  if (options.emit_afir_dialect) {
    mlir::PassManager pm(context);
    // Use MLIR's built-in RemoveDeadValues pass
    pm.addPass(mlir::createRemoveDeadValuesPass());
    std::string diagnostic_message;
    mlir::ScopedDiagnosticHandler diagnostic_handler(
        context, [&](mlir::Diagnostic& diagnostic) {
          llvm::raw_string_ostream os(diagnostic_message);
          diagnostic.print(os);
          os << "\n";
          os.flush();
          return mlir::success();
        });
    if (failed(pm.run(*mlir_module))) {
      return absl::InternalError(absl::StrCat(
          "AFIR DCE pass failed",
          diagnostic_message.empty()
              ? ""
              : absl::StrCat(":\n", diagnostic_message)));
    }
  }

  if (options.verify_module) {
    if (failed(mlir_module->verify())) {
      return absl::InternalError("Generated AFIR module failed verification");
    }
  }

  return mlir_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToAfir(
    const HloModule& module, const HloToAfirOptions& options) {
  // Create context on heap to ensure it lives long enough
  // Note: The caller must ensure the module is used before the context is destroyed
  // For safety, we recommend using the API that takes an explicit context pointer
  static thread_local std::unique_ptr<mlir::MLIRContext> thread_context;
  thread_context = std::make_unique<mlir::MLIRContext>();
  // Load required dialects
  thread_context->loadDialect<mlir::func::FuncDialect>();
  thread_context->loadDialect<mlir::arith::ArithDialect>();
  thread_context->loadDialect<mlir::tensor::TensorDialect>();
  thread_context->loadDialect<mlir::afir::AFIRDialect>();
  TF_ASSIGN_OR_RETURN(auto mlir_module,
                      ConvertHloToAfir(module, thread_context.get(), options));
  return mlir_module;
}

}  // namespace xla
