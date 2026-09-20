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

#include "xla/xla_npu/service/npu_ir_emitter_unnested.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/human_readable_json.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/backend/codegen/npu_custom.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/backend/runtime/npu_aclnn_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_call_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_conditional_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_while_thunk.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_execution_stream_assignment.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"
#include "xla/xla_npu/service/transform/npu_tril_utils.h"

namespace xla {

NpuIrEmitterUnnested::NpuIrEmitterUnnested(
    NpuIrEmitterContext* ir_emitter_context)
    : NpuIrEmitter(ir_emitter_context, /*is_nested=*/false),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())) {}

std::unique_ptr<NpuIrEmitterUnnested> NpuIrEmitterUnnested::Create(
    NpuIrEmitterContext* ir_emitter_context) {
  return std::unique_ptr<NpuIrEmitterUnnested>(
      new NpuIrEmitterUnnested(ir_emitter_context));
}

absl::Status NpuIrEmitterUnnested::EmitConstant(
    const HloConstantInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Emit const thunk");

  TF_ASSIGN_OR_RETURN(NpuDenseDataIntermediate content,
                      NpuLiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, {}));

  ir_emitter_context_->emit_constant(num_elements, element_bytes, global_name,
                                     slice.index(), std::move(content), &b_);
  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::HandleGather(const HloInstruction* instr) {
  const auto* gather_instr = Cast<HloGatherInstruction>(instr);
  const auto& dnums = gather_instr->gather_dimension_numbers();
  auto slice_sizes = gather_instr->gather_slice_sizes();
  const auto& operand_shape = instr->operand(0)->shape();
  const auto& indices_shape = instr->operand(1)->shape();
  const auto& output_shape = instr->shape();

  // Collect dnums signatures for error messages
  auto dnums_signature = [&]() -> std::string {
    std::string sig = "offset_dims={";
    for (int64_t i = 0; i < dnums.offset_dims().size(); ++i) {
      if (i > 0) sig += ",";
      sig += std::to_string(dnums.offset_dims(i));
    }
    sig += "}, collapsed_slice_dims={";
    for (int64_t i = 0; i < dnums.collapsed_slice_dims().size(); ++i) {
      if (i > 0) sig += ",";
      sig += std::to_string(dnums.collapsed_slice_dims(i));
    }
    sig += "}, start_index_map={";
    for (int64_t i = 0; i < dnums.start_index_map().size(); ++i) {
      if (i > 0) sig += ",";
      sig += std::to_string(dnums.start_index_map(i));
    }
    sig += "}, index_vector_dim=" + std::to_string(dnums.index_vector_dim());
    sig += ", slice_sizes={";
    for (int64_t i = 0; i < slice_sizes.size(); ++i) {
      if (i > 0) sig += ",";
      sig += std::to_string(slice_sizes[i]);
    }
    sig += "}";
    return sig;
  };

  // Basic condition: index_vector_dim must be the last dimension of indices
  if (dnums.index_vector_dim() != (int64_t)indices_shape.dimensions().size() - 1) {
    return Internal(
        "Unsupported gather: index_vector_dim=%lld != indices last dim (%lld). "
        "operand=%s, indices=%s, output=%s, %s",
        dnums.index_vector_dim(), (int64_t)indices_shape.dimensions().size() - 1,
        NpuTool::GetXlaShapeString(operand_shape).c_str(),
        NpuTool::GetXlaShapeString(indices_shape).c_str(),
        NpuTool::GetXlaShapeString(output_shape).c_str(),
        dnums_signature().c_str());
  }

  // The size of index_vector_dim must be equal to start_index_map.size()
  if (indices_shape.dimensions(dnums.index_vector_dim()) !=
      (int64_t)dnums.start_index_map().size()) {
    return Internal(
        "Unsupported gather: index_vector_dim size=%lld != "
        "start_index_map.size=%lld. "
        "operand=%s, indices=%s, output=%s, %s",
        indices_shape.dimensions(dnums.index_vector_dim()),
        dnums.start_index_map().size(),
        NpuTool::GetXlaShapeString(operand_shape).c_str(),
        NpuTool::GetXlaShapeString(indices_shape).c_str(),
        NpuTool::GetXlaShapeString(output_shape).c_str(),
        dnums_signature().c_str());
  }

  // slice_size range for each gathered dim: 1 ≤ slice_size ≤ dim_size
  // - slice_size == dim_size: trivial dim (pseudo-gather, no index required)
  // - slice_size == 1: unit gather (take a single element)
  // - 1 < slice_size < dim_size: partial slice gather (take continuous range)
  // At most 1 gathered dim is allowed to have partial slices (the rest must be 1 or dim_size)
  int64_t non_trivial_count = 0;
  int64_t partial_slice_count = 0;
  for (int64_t i = 0; i < (int64_t)dnums.start_index_map().size(); ++i) {
    int64_t gd = dnums.start_index_map(i);
    if (slice_sizes[gd] < 1 || slice_sizes[gd] > operand_shape.dimensions(gd)) {
      return Internal(
          "Unsupported gather: slice_sizes[%lld]=%lld out of range [1, %lld]. "
          "operand=%s, indices=%s, output=%s, %s",
          gd, slice_sizes[gd], operand_shape.dimensions(gd),
          NpuTool::GetXlaShapeString(operand_shape).c_str(),
          NpuTool::GetXlaShapeString(indices_shape).c_str(),
          NpuTool::GetXlaShapeString(output_shape).c_str(),
          dnums_signature().c_str());
    }
    if (slice_sizes[gd] != operand_shape.dimensions(gd)) {
      non_trivial_count++;
    }
    if (slice_sizes[gd] > 1 && slice_sizes[gd] < operand_shape.dimensions(gd)) {
      partial_slice_count++;
    }
  }
  if (partial_slice_count > 1) {
    return Internal(
        "Unsupported gather: multiple gathered dims with partial slices "
        "(1 < slice_size < dim_size). operand=%s, indices=%s, output=%s, %s",
        NpuTool::GetXlaShapeString(operand_shape).c_str(),
        NpuTool::GetXlaShapeString(indices_shape).c_str(),
        NpuTool::GetXlaShapeString(output_shape).c_str(),
        dnums_signature().c_str());
  }

  // The slice_sizes of non-gathered dim must be equal to the corresponding dimensions of operand
  {
    std::set<int64_t> gather_set(dnums.start_index_map().begin(),
                                 dnums.start_index_map().end());
    for (int64_t i = 0; i < (int64_t)operand_shape.dimensions().size(); ++i) {
      if (gather_set.count(i) == 0 &&
          slice_sizes[i] != operand_shape.dimensions(i)) {
        return Internal(
            "Unsupported gather: partial slice not supported "
            "(slice_sizes[%lld]=%lld != operand[%lld]=%lld). "
            "operand=%s, indices=%s, output=%s, %s",
            i, slice_sizes[i], i, operand_shape.dimensions(i),
            NpuTool::GetXlaShapeString(operand_shape).c_str(),
            NpuTool::GetXlaShapeString(indices_shape).c_str(),
            NpuTool::GetXlaShapeString(output_shape).c_str(),
            dnums_signature().c_str());
      }
    }
  }

  // Multidimensional gather also requires gathered dims to be continuous
  // if (dnums.start_index_map().size() > 1) {
  //   std::vector<int64_t> sorted_dims(dnums.start_index_map().begin(),
  //                                    dnums.start_index_map().end());
  //   std::sort(sorted_dims.begin(), sorted_dims.end());
  //   for (size_t i = 1; i < sorted_dims.size(); ++i) {
  //     if (sorted_dims[i] != sorted_dims[i - 1] + 1) {
  //       return Internal(
  //           "Unsupported gather: non-contiguous gathered dims. "
  //           "operand=%s, indices=%s, output=%s, %s",
  //           NpuTool::GetXlaShapeString(operand_shape).c_str(),
  //           NpuTool::GetXlaShapeString(indices_shape).c_str(),
  //           NpuTool::GetXlaShapeString(output_shape).c_str(),
  //           dnums_signature().c_str());
  //     }
  //   }
  // }

  // Pass all checks and emit ACLNN thunk (processed according to gather_lowering_kind at runtime)
  return EmitNpuAclnnThunk(instr);
}

absl::Status NpuIrEmitterUnnested::EmitNpuAclnnThunk(
    const HloInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Emit aclnn thunk");

  auto thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  if (instr->opcode() == HloOpcode::kScatter &&
      thunk_info.scatter_reduction_kind ==
          NpuThunk::ThunkInfo::ScatterReductionKind::kAdd &&
      !thunk_info.scatter_add_normalized) {
    return absl::UnimplementedError(
        "NPU scatter-add must pass semantic normalization before emission");
  }
  if (instr->opcode() == HloOpcode::kReduce) {
    using ReducerKind = NpuThunk::ThunkInfo::ReduceReducerKind;
    const auto kind = thunk_info.reduce_reducer_kind;
    if (instr->shape().IsTuple() || kind == ReducerKind::kUnsupported ||
        ((kind == ReducerKind::kAnd || kind == ReducerKind::kOr) &&
         instr->operand(0)->shape().element_type() != PRED)) {
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported NPU reducer; coupled reductions must be expanded: ",
          instr->ToString()));
    }
  }

  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context_->buffer_assignment(),
                          NpuConstant::GetDefaultBufferAlignment(), instr));
  const std::string fingerprint = "";
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;

  AddThunkToThunkSequence(std::make_unique<NpuAclnnThunk>(
      fingerprint,
      std::move(thunk_info),
      kernel_arguments.GetArgumentBufferSlices(),
      kernel_arguments.GetArgumentOutputFlags(), dropout_seed));

  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitFmaAddcmulThunk(
    const HloInstruction* instr, float value) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context_->buffer_assignment(),
                          NpuConstant::GetDefaultBufferAlignment(), instr));
  NpuThunk::ThunkInfo thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  thunk_info.is_fma_addcmul = true;
  thunk_info.fma_addcmul_value = value;
  AddThunkToThunkSequence(std::make_unique<NpuAclnnThunk>(
      /*fingerprint=*/"", std::move(thunk_info),
      kernel_arguments.GetArgumentBufferSlices(),
      kernel_arguments.GetArgumentOutputFlags(),
      /*sdpa_dropout_seed=*/std::nullopt));
  return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice>
NpuIrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                               const ShapeIndex& index) const {
  return NpuGetAllocationSlice(ir_emitter_context_->buffer_assignment(), instr,
                               index);
}

absl::Status NpuIrEmitterUnnested::EmitTrilThunk(const HloInstruction* instr,
                                                  int64_t diagonal) {
  NPU_LOG_FORMAT_PRINT(INFO,
                       "Emitting aclnnTril thunk for '%s' with diagonal=%lld",
                       std::string(instr->name()).c_str(), (long long)diagonal);

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      emitters::KernelArguments::Create(
          ir_emitter_context_->buffer_assignment(),
          NpuConstant::GetDefaultBufferAlignment(), instr));

  NpuThunk::ThunkInfo thunk_info =
      NpuThunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId());
  thunk_info.is_tril = true;
  thunk_info.tril_diagonal = diagonal;

  AddThunkToThunkSequence(std::make_unique<NpuAclnnThunk>(
      /*fingerprint=*/"", std::move(thunk_info),
      kernel_arguments.GetArgumentBufferSlices(),
      kernel_arguments.GetArgumentOutputFlags(),
      /*sdpa_dropout_seed=*/std::nullopt));
  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitFusion(
    const HloFusionInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Emit fusion thunk");

  const HloComputation* fused_computation =
      instr->fused_instructions_computation();

  // Check if it is a single-operator fusion that Autofuse cannot handle.
  // Strip the kBitcast/kReshape/kCopy/kConvert wrapper starting from fused_expression_root,
  // Find the real hero op. If all non-parameter directives are hero + wrappers,
  // Skip the Autofuse compilation pipeline and go directly to the ACLNN thunk path.
  const HloInstruction* root = fused_computation->root_instruction();

  // Peel off the wrapper and find the hero
  const HloInstruction* hero = root;
  while (hero->opcode() == HloOpcode::kBitcast ||
         hero->opcode() == HloOpcode::kReshape ||
         hero->opcode() == HloOpcode::kCopy ||
         hero->opcode() == HloOpcode::kConvert) {
    if (hero->operand_count() == 0) break;
    hero = hero->operand(0);
  }

  // Verify that all non-parameter instructions in fused computation are wrapper or hero
  bool is_trivial_fusion = true;
  for (const HloInstruction* hlo : fused_computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kParameter) continue;
    if (hlo == hero || hlo->opcode() == HloOpcode::kBitcast ||
        hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kCopy ||
        hlo->opcode() == HloOpcode::kConvert) {
      continue;
    }
    is_trivial_fusion = false;
    break;
  }

  if (is_trivial_fusion &&
      hero->opcode() != HloOpcode::kFusion &&
      hero->opcode() != HloOpcode::kConstant &&
      hero->opcode() != HloOpcode::kCall &&
      hero->opcode() != HloOpcode::kCustomCall) {
    NPU_LOG_FORMAT_PRINT(
        INFO,
        "Trivial fusion '%s' (root=%s hero=%s), emitting via ACLNN thunk "
        "instead of autofuse",
        std::string(instr->name()).c_str(),
        HloOpcodeString(root->opcode()),
        HloOpcodeString(hero->opcode()));

    // Buffer slice query uses outer fusion instruction
    TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                        emitters::KernelArguments::Create(
                            ir_emitter_context_->buffer_assignment(),
                            NpuConstant::GetDefaultBufferAlignment(), instr));

    std::vector<BufferAllocation::Slice> fusion_slices =
        kernel_arguments.GetArgumentBufferSlices();
    std::vector<bool> fusion_flags = kernel_arguments.GetArgumentOutputFlags();

    // ThunkInfo 使用 hero（内部指令）构建，但 hero 的操作数顺序可能与 fusion 的
    // 外部 operand 顺序不一致：hero 的 operand(0) 可能经过 wrapper（reshape/
    // bitcast/copy/convert）追溯到 parameter_number != 0 的参数。KernelArguments
    // 返回的 slice 是按 fusion operand（= parameter_number）顺序排列的，因此这里
    // 必须按 hero 的操作数顺序重排，否则非交换算子（如 subtract）的两个输入会被
    // 交换。这正是 jax.random.uniform 把 range = maxval - minval 算成
    // minval - maxval（=-1）、导致 uniform 全塌到下界的根因。
    std::vector<BufferAllocation::Slice> hero_slices;
    std::vector<bool> hero_flags;
    hero_slices.reserve(hero->operand_count() + 1);
    hero_flags.reserve(hero->operand_count() + 1);
    bool reorder_ok =
        fusion_slices.size() == instr->operand_count() + 1;
    for (int i = 0; i < hero->operand_count() && reorder_ok; ++i) {
      const HloInstruction* op = hero->operand(i);
      while (op->opcode() == HloOpcode::kBitcast ||
             op->opcode() == HloOpcode::kReshape ||
             op->opcode() == HloOpcode::kCopy ||
             op->opcode() == HloOpcode::kConvert) {
        if (op->operand_count() == 0) break;
        op = op->operand(0);
      }
      if (op->opcode() == HloOpcode::kParameter &&
          op->parameter_number() < instr->operand_count()) {
        hero_slices.push_back(fusion_slices[op->parameter_number()]);
        hero_flags.push_back(false);
      } else {
        // hero 操作数不追溯到参数（如常量），回退到 fusion operand 顺序。
        reorder_ok = false;
      }
    }
    if (reorder_ok && !fusion_flags.empty()) {
      // 追加 fusion 输出 slice（written=true 的那个），保持最后。
      int64_t out_idx = fusion_flags.size() - 1;
      for (int64_t i = 0; i < static_cast<int64_t>(fusion_flags.size()); ++i) {
        if (fusion_flags[i]) {
          out_idx = i;
          break;
        }
      }
      hero_slices.push_back(fusion_slices[out_idx]);
      hero_flags.push_back(true);
    }
    bool use_reordered =
        reorder_ok &&
        static_cast<int>(hero_slices.size()) == hero->operand_count() + 1;

    std::vector<BufferAllocation::Slice> out_slices;
    std::vector<bool> out_flags;
    if (use_reordered) {
      out_slices = std::move(hero_slices);
      out_flags = std::move(hero_flags);
    } else {
      out_slices = std::move(fusion_slices);
      out_flags = std::move(fusion_flags);
    }

    // ThunkInfo is built using hero so that ExecuteOnStream presses the correct opcode
    //(such as kTranspose) is distributed to the corresponding ACLNN function
    NpuThunk::ThunkInfo thunk_info =
        NpuThunk::ThunkInfo::WithProfileAnnotation(
            hero, ir_emitter_context_->GetNextThunkId());

    AddThunkToThunkSequence(std::make_unique<NpuAclnnThunk>(
        /*fingerprint=*/"", std::move(thunk_info), std::move(out_slices),
        std::move(out_flags),
        /*sdpa_dropout_seed=*/std::nullopt));
    return absl::OkStatus();
  }

  // Detect tril mode: map directly to aclnnTril, bypass AFIR fusion, avoid loss of negative k constants
  {
    int64_t tril_k = 0;
    if (tril::MatchTrilFusion(fused_computation, &tril_k)) {
      return EmitTrilThunk(instr, tril_k);
    }
  }

  std::unique_ptr<NpuFusionInterface> emitter =
      std::make_unique<NpuCustomFusion>();
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  const NpuExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<NpuThunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(NpuExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }

  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitCall(const HloInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Emit call thunk");

  const HloComputation* called_computation =
      instr->called_computations().front();
  auto ir_emitter = NpuIrEmitterUnnested::Create(ir_emitter_context_);
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "====Start generate sub hlo instruction thunk====");
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(called_computation));
  NPU_LOG_FORMAT_PRINT(DEBUG, "====End generate sub hlo instruction thunk====");
  std::unique_ptr<NpuSequentialThunk> called_thunk_sequence =
      ir_emitter->ConsumeThunkSequence(
          NpuThunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()));
  AddThunkToThunkSequence(std::make_unique<NpuCallThunk>(
      NpuThunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(called_thunk_sequence->thunks())));
  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitWhile(const HloInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Emit while thunk: name=%s, shape=%s",
                       std::string(instr->name()).c_str(),
                       instr->shape().ToString().c_str());

  // Read compile-time trip_count (if while_loop_trip_count_annotator is annotated)
  std::optional<int64_t> trip_count = std::nullopt;
  auto config = instr->backend_config<WhileLoopBackendConfig>();
  if (config.ok() && config->has_known_trip_count()) {
    trip_count = config->known_trip_count().n();
    NPU_LOG_FORMAT_PRINT(DEBUG, "While loop trip count: %d", *trip_count);
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Generate thunk sequence for condition subcomputation
  auto ir_emitter_condition = NpuIrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitHloComputation(condition));

  // Generate thunk sequence for body subcomputation
  auto ir_emitter_body = NpuIrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_body->EmitHloComputation(body));

  // Get the buffer slice corresponding to the condition result (used to read back pred at runtime)
  TF_ASSIGN_OR_RETURN(
      auto pred, GetAllocationSliceForHlo(condition->root_instruction(), {}));

  NpuThunk::ThunkInfo cond_thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  cond_thunk_info.profile_annotation += "_condition";
  NpuThunk::ThunkInfo body_thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  body_thunk_info.profile_annotation += "_body";

  auto cond_thunk_seq =
      ir_emitter_condition->ConsumeThunkSequence(cond_thunk_info);
  auto body_thunk_seq =
      ir_emitter_body->ConsumeThunkSequence(body_thunk_info);

  NPU_LOG_FORMAT_PRINT(DEBUG, "EmitWhile: cond_thunks=%d, body_thunks=%d",
                       (int)cond_thunk_seq->thunks().size(),
                       (int)body_thunk_seq->thunks().size());

  // Get the top-level buffer slice (tuple pointer table or scalar data) of while operand
  TF_ASSIGN_OR_RETURN(auto while_operand_slice,
                      GetAllocationSliceForHlo(instr->operand(0), {}));

  // Compute element-level carry copy.
  // XLA's while loop carry relies on buffer aliasing: the new value written in body should be
  // The next read location of condition/body is the same memory. But buffer assignment is possible
  // Allocate body's calculation instructions (add, etc.) to a different location from the carry state buffer.
  //
  // Here the buffer slice is obtained directly from the specific instructions inside the body:
  //   src = i-th operand of body root tuple (instruction that generates new carry value, such as add)
  //   dst = i-th GTE user of body parameter (where carry state is read)
  // This avoids passing the while directive to the phi node, which is in the dataflow analysis
  // Will merge init and body_root, causing GetUniqueSlice to return the same slice
  // (The aliasing is considered correct at compile time), but the actual writing location is different at runtime.
  std::vector<NpuWhileElementCopy> element_copies;
  const HloInstruction* body_root = body->root_instruction();
  const HloInstruction* body_param = body->parameter_instruction(0);

  if (body_root->opcode() == HloOpcode::kTuple) {
    // Tuple carry: Compares src and dst for each tuple element.
    //
    // Improvement: Use the ShapeIndex {i} of body_param to directly obtain the carry state buffer,
    // Instead of searching for GTE(body_param, i). GTE user of body_param after CopyInsertion
    // becomes param_copy internal operand, the GTE search may find the wrong GTE or not find it.
    // The ShapeIndex path directly obtains the carry state buffer through dataflow analysis.
    for (int64_t i = 0; i < body_root->operand_count(); ++i) {
      const HloInstruction* body_root_elem = body_root->operand(i);
      if (!body_root_elem->shape().IsArray()) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
            "While carry element %d: skipped (non-array: %s)", (int)i,
            body_root_elem->shape().ToString().c_str());
        continue;
      }

      // src: 使用 body root 元素本身的 buffer。
      // 对 while carry，root 上的 Copy/Bitcast 可能正是为了物化稳定的结果槽位；
      // 继续 unwrap 到底层 producer 可能会重新落回被复用/覆盖的临时 buffer。
      const HloInstruction* src_instr = body_root_elem;

      auto src_status = GetAllocationSliceForHlo(src_instr, {});
      if (!src_status.ok()) {
        NPU_LOG_FORMAT_PRINT(WARN,
            "While carry element %d: src FAILED for '%s' (unwrapped from "
            "'%s'): %s",
            (int)i, std::string(src_instr->name()).c_str(),
            std::string(body_root_elem->name()).c_str(),
            src_status.status().ToString().c_str());
        continue;
      }

      // dst: Use the ShapeIndex {i} of body_param to obtain the carry state buffer.
      // The i-th element of body_param is the buffer at the i-th position of the while loop carry.
      // This is where condition and next body iteration read carry.
      auto dst_status = GetAllocationSliceForHlo(body_param, {i});
      if (!dst_status.ok()) {
        // Fallback: Try while operand of ShapeIndex {i}
        NPU_LOG_FORMAT_PRINT(WARN,
            "While carry element %d: body_param{%d} FAILED (%s), "
            "trying while_operand{%d} fallback",
            (int)i, (int)i,
            dst_status.status().ToString().c_str(), (int)i);
        dst_status = GetAllocationSliceForHlo(instr->operand(0), {i});
      }
      if (!dst_status.ok()) {
        NPU_LOG_FORMAT_PRINT(ERROR,
            "While carry element %d: CANNOT resolve carry state buffer. "
            "Carry propagation may be broken for this element.",
            (int)i);
        continue;
      }

      const auto& src = *src_status;
      const auto& dst = *dst_status;

      NPU_LOG_FORMAT_PRINT(INFO,
          "While carry element %d: src=(idx=%d,off=%lld,sz=%lld) '%s' "
          "dst=(idx=%d,off=%lld,sz=%lld) body_param{%d} %s",
          (int)i,
          src.index(), (long long)src.offset(), (long long)src.size(),
          std::string(src_instr->name()).c_str(),
          dst.index(), (long long)dst.offset(), (long long)dst.size(),
          (int)i,
          (src == dst) ? "ALIASED" : "NEEDS-COPY");

      if (src != dst) {
        element_copies.push_back({src, dst});
      }
    }
  } else if (body_root->shape().IsArray()) {
    // Scalar carry (not tuple)
    auto src_status = GetAllocationSliceForHlo(body_root, {});
    auto dst_status = GetAllocationSliceForHlo(body_param, {});

    if (src_status.ok() && dst_status.ok()) {
      const auto& src = *src_status;
      const auto& dst = *dst_status;

      if (src != dst) {
        element_copies.push_back({src, dst});
      }
    }
  }

  if (!element_copies.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "EmitWhile: %d element carry copies needed",
                         (int)element_copies.size());
    for (int64_t ci = 0; ci < (int64_t)element_copies.size(); ++ci) {
      const auto& c = element_copies[ci];
      NPU_LOG_FORMAT_PRINT(DEBUG,
          "  carry_copy[%d]: src=(idx=%d,off=%lld,sz=%lld) -> "
          "dst=(idx=%d,off=%lld,sz=%lld)",
          (int)ci, c.src.index(), (long long)c.src.offset(),
          (long long)c.src.size(),
          c.dst.index(), (long long)c.dst.offset(),
          (long long)c.dst.size());
    }
  } else {
    NPU_LOG_FORMAT_PRINT(DEBUG,
        "EmitWhile: NO element carry copies (relying on buffer aliasing)");
  }


  AddThunkToThunkSequence(std::make_unique<NpuWhileThunk>(
      NpuThunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      pred, while_operand_slice,
      std::move(cond_thunk_seq->thunks()),
      std::move(body_thunk_seq->thunks()),
      std::move(element_copies), trip_count));
  return absl::OkStatus();
}

absl::Status NpuIrEmitterUnnested::EmitConditional(
    const HloInstruction* instr) {
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "Emit conditional thunk: name=%s, branch_count=%d",
                       std::string(instr->name()).c_str(),
                       instr->branch_count());

  bool branch_index_is_bool =
      instr->operand(0)->shape().element_type() == PRED;

  // Get the buffer slice (branch index or predicate) of operand(0)
  TF_ASSIGN_OR_RETURN(
      auto branch_index_slice,
      GetAllocationSliceForHlo(instr->operand(0), {}));

  // Generate thunk sequence for each branch
  int32_t branch_count = instr->branch_count();
  std::vector<NpuThunkSequence> branch_thunks;
  branch_thunks.reserve(branch_count);

  for (int32_t b = 0; b < branch_count; ++b) {
    HloComputation* branch_computation = instr->branch_computation(b);
    auto ir_emitter = NpuIrEmitterUnnested::Create(ir_emitter_context_);
    TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(branch_computation));

    NpuThunk::ThunkInfo branch_thunk_info =
        NpuThunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId());
    branch_thunk_info.profile_annotation +=
        absl::StrCat("_branch_", b);

    auto branch_thunk_seq = ir_emitter->ConsumeThunkSequence(branch_thunk_info);
    branch_thunks.push_back(std::move(branch_thunk_seq->thunks()));

    NPU_LOG_FORMAT_PRINT(DEBUG, "  branch[%d] '%s': %d thunks",
                         b, std::string(branch_computation->name()).c_str(),
                         (int)branch_thunks.back().size());
  }

  AddThunkToThunkSequence(std::make_unique<NpuConditionalThunk>(
      NpuThunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      branch_index_slice, std::move(branch_thunks), branch_index_is_bool));
  return absl::OkStatus();
}

static const HloInstruction* FindCanonicalSendRecvStartOp(
    const HloInstruction* inst) {
  CHECK(inst->opcode() == HloOpcode::kSend ||
        inst->opcode() == HloOpcode::kRecv ||
        inst->opcode() == HloOpcode::kSendDone ||
        inst->opcode() == HloOpcode::kRecvDone);
  // If the instruction is wrapped in an async computation, return the instruction itself.
  if (inst->parent()->IsAsyncComputation()) {
    return inst;
  }

  // Find container while loop and index for the send/recv case or
  // return canonical start op directly.
  const HloInstruction* while_op = nullptr;
  int64_t i = -1;
  if (inst->opcode() == HloOpcode::kSend ||
      inst->opcode() == HloOpcode::kRecv) {
    CHECK_EQ(inst->users().size(), 1);
    const HloInstruction* unique_user = inst->users().front();

    // Return send/recv inst directly if this is a simple send/recv pair.
    if (unique_user->opcode() == HloOpcode::kSendDone ||
        unique_user->opcode() == HloOpcode::kRecvDone) {
      return inst;
    }

    // Find while loop and index, otherwise.
    CHECK(unique_user->opcode() == HloOpcode::kTuple ||
          unique_user->opcode() == HloOpcode::kWhile);
    if (unique_user->IsRoot()) {
      // send/recv op in the loop body.
      auto maybe_while_op =
          unique_user->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while_op);
      while_op = *maybe_while_op;
      i = unique_user->operand_index(inst);
    } else {
      // send/recv leading into the loop.
      CHECK_EQ(unique_user->users().size(), 1);
      CHECK(unique_user->users().front()->opcode() == HloOpcode::kWhile);
      while_op = unique_user->users().front();
      i = unique_user->operand_index(inst);
    }
  }

  // Find container while loop and index for the send-done/recv-done
  // case or return canonical start op directly.
  if (inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone) {
    const HloInstruction* operand = inst->operand(0);

    // Return send/recv inst directly if this is a simple send/recv  pair.
    if (operand->opcode() == HloOpcode::kSend ||
        operand->opcode() == HloOpcode::kRecv) {
      return operand;
    }

    // Find while loop and index, otherwise.
    CHECK(operand->opcode() == HloOpcode::kGetTupleElement);
    const auto* gte = Cast<HloGetTupleElementInstruction>(operand);
    const HloInstruction* iter_tuple = operand->operand(0);
    if (iter_tuple->opcode() == HloOpcode::kParameter) {
      // send-done/recv-done in the loop body.
      CHECK(Cast<HloParameterInstruction>(iter_tuple)->parameter_number() == 0);
      auto maybe_while =
          iter_tuple->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while);
      while_op = *maybe_while;
      i = gte->tuple_index();
    } else {
      // send-done/recv-done proceeding the loop.
      CHECK(iter_tuple->opcode() == HloOpcode::kWhile);
      while_op = iter_tuple;
      i = gte->tuple_index();
    }
  }

  // Extract canonical start op from while loop's init.
  CHECK(while_op != nullptr);
  CHECK(0 <= i && i < while_op->shape().tuple_shapes().size());
  const HloInstruction* init = while_op->operand(0);
  const HloInstruction* canonical_start_op = init->operand(i);
  CHECK(canonical_start_op->opcode() == HloOpcode::kSend ||
        canonical_start_op->opcode() == HloOpcode::kRecv);
  return canonical_start_op;
}

std::vector<const HloInstruction*> GetRealDependencyInstructions(
    const HloInstruction* instr) {
  std::vector<const HloInstruction*> real_deps;
  switch (instr->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
      return {FindCanonicalSendRecvStartOp(instr)};
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopyDone:
      return {instr->operand(0)};
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kCustomCall:
    case HloOpcode::kFusion:
    case HloOpcode::kCopy:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPartitionId:
    case HloOpcode::kFft:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kWhile:
    case HloOpcode::kCopyStart:
      return {instr};
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kTuple:
      for (const HloInstruction* operand : instr->operands()) {
        auto deps = GetRealDependencyInstructions(operand);
        real_deps.insert(real_deps.end(), deps.begin(), deps.end());
      }
      return real_deps;
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement: {
      auto deps = GetRealDependencyInstructions(instr->operand(0));
      real_deps.insert(real_deps.end(), deps.begin(), deps.end());
    }
      return real_deps;
    default:
      return {};
  }
}

absl::Status NpuIrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter) {
  return Internal("This should be unreachable");
}

absl::Status NpuIrEmitterUnnested::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAcos:
    case HloOpcode::kAcosh:
    case HloOpcode::kAsin:
    case HloOpcode::kAsinh:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kAtanh:
    case HloOpcode::kBroadcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kCholesky:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kCosh:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog1p:
    case HloOpcode::kLog:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSinh:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kXor:
    case HloOpcode::kDot:  // Automatic fusion does not currently support matmul, and is processed by ACLNN.
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:        // reduce requires special handling
    case HloOpcode::kReduceWindow:  // reduce-window is handled by aclnnMaxPool
    case HloOpcode::kSlice:   // Static slice operations
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:  // dynamic-update-slice via aclnnInplaceCopy
    case HloOpcode::kScatter:  // scatter replace/add lowering via aclnn
    case HloOpcode::kCopy:     // copy inserted by scatter-copy-insertion pass
    case HloOpcode::kSort:     // sort handled via aclnnSort
    case HloOpcode::kTopK:     // topk handled via aclnnTopk
    case HloOpcode::kPad:      // pad handled via aclnnConstantPadNd
      return EmitNpuAclnnThunk(instr);

    case HloOpcode::kGather:
      return HandleGather(instr);

    case HloOpcode::kCall:
      return EmitCall(Cast<HloCallInstruction>(instr));
    case HloOpcode::kWhile:
      return EmitWhile(instr);
    case HloOpcode::kConditional:
      return EmitConditional(instr);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(instr));
    case HloOpcode::kCustomCall:
      if (instr->custom_call_target() == "tril") {
        // Parse diagonal offset from opaque string "k=<value>"
        int64_t k = 0;
        const auto* custom_call = Cast<const HloCustomCallInstruction>(instr);
        std::string opaque(custom_call->opaque());
        if (absl::StartsWith(opaque, "k=")) {
          if (!absl::SimpleAtoi(opaque.substr(2), &k)) {
            NPU_LOG_FORMAT_PRINT(ERROR,
                                 "Failed to parse tril k value from '%s'",
                                 opaque.c_str());
          }
        }
        return EmitTrilThunk(instr, k);
      }
      if (instr->custom_call_target() == "__npu_fma_addcmul") {
        const auto* custom_call = Cast<const HloCustomCallInstruction>(instr);
        std::string opaque(custom_call->opaque());
        constexpr absl::string_view kPrefix = "value_bits=";
        uint32_t bits = 0;
        if (!absl::StartsWith(opaque, kPrefix)) {
          return Internal("Invalid aclnnFmaAddcmul opaque: %s",
                          std::string(custom_call->opaque()));
        }
        opaque.erase(0, kPrefix.size());
        if (!absl::SimpleAtoi(opaque, &bits)) {
          return Internal("Invalid aclnnFmaAddcmul opaque: %s",
                          std::string(custom_call->opaque()));
        }
        float value;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return EmitFmaAddcmulThunk(instr, value);
      }
      return absl::OkStatus();
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(instr));
    // HLO module is already scheduled, so instructions for ordering
    // are noops.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    // We don't need to emit thunks for these operations because
    // their semantics are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      NPU_LOG_FORMAT_PRINT(DEBUG, "Emit no thunk");
      return absl::OkStatus();
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(instr->opcode()));
  }

  return Internal("Unhandled HLO instruction");
}

absl::Status NpuIrEmitterUnnested::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation)) {
    return Internal("Sequence not found for computation: %s",
                    computation->name());
  }

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  absl::flat_hash_map<const HloInstruction*, NpuThunk*> instr_to_thunk;
  int id = 0;
  for (const HloInstruction* instr : sequence.instructions()) {
    // ========== Print the complete information of instr ==========
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    NPU_LOG_FORMAT_PRINT(DEBUG, "[FULL Info for HloInstruction %d] ", id++);
    HloPrintOptions options = NpuTool::GetDetailedPrintOptions();
    std::string detailed_str = instr->ToString(options);
    NPU_LOG_FORMAT_PRINT(DEBUG, detailed_str);
    int64_t previous_thunk_size = thunk_sequence_.size();
    TF_RETURN_IF_ERROR(EmitHloInstruction(instr));

    NPU_LOG_FORMAT_PRINT(DEBUG, "  ----------------------------------------");
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Target thunk Info: ");

    if (thunk_sequence_.size() > previous_thunk_size) {
      instr_to_thunk[instr] = thunk_sequence_.back().get();

      const auto& thunk = thunk_sequence_.back();
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Code: %s",
                           HloOpcodeString(thunk->thunk_info().op_code));
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Name: %s",
                           thunk->thunk_info().op_name);
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Op Shape size: %d",
                           thunk->thunk_info().param_shapes.size());
      for (int i = 0; i < thunk->thunk_info().param_shapes.size(); ++i) {
        NpuTool::PrintXlaShape(thunk->thunk_info().param_shapes[i],
                               absl::StrFormat("    Op param %d shape: ", i));
      }
    } else {
      // Some op types do not allocate thunks, such as constant types
      NPU_LOG_FORMAT_PRINT(DEBUG, "    No thunk info.");
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
