/* Copyright 2019 The OpenXLA Authors.
Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file is modified from OpenXLA source code.
Modification: Custom adaptation for NPU/XLA-NPU scenarios.
==============================================================================*/

#include "xla/xla_npu/backend/runtime/npu_thunk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/literal.h"
#include "xla/service/global_device_id.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_id.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/config_loader.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

//===----------------------------------------------------------------------===//
// NpuThunk::ExecuteParams
//===----------------------------------------------------------------------===//

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options,
    const NpuBufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    ExecutionStreamIdMap additional_compute_streams) {
  return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream,
                       run_options.run_options().device_to_host_stream(),
                       run_options.run_options().host_to_device_stream(),
                       run_options.run_options().send_device_memory_function(),
                       run_options.run_options().recv_device_memory_function(),
                       run_options.run_options().ffi_execution_context(),
                       additional_compute_streams, false,
                       run_options.run_options().run_id().ToInt());
}

NpuThunk::ExecuteParams NpuThunk::ExecuteParams::CloneWithNewAllocations(
    const NpuThunk::ExecuteParams& params,
    const NpuBufferAllocations& buffer_allocations) {
  return ExecuteParams(
      &buffer_allocations, params.stream, params.command_buffer_trace_stream,
      params.device_to_host_stream, params.host_to_device_stream,
      params.send_device_memory_function, params.recv_device_memory_function,
      params.ffi_execution_context, params.additional_compute_streams);
}

NpuThunk::ExecuteParams::ExecuteParams(
    const NpuBufferAllocations* buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream, se::Stream* device_to_host_stream,
    se::Stream* host_to_device_stream,
    SendDeviceMemoryFunction* send_device_memory_function,
    RecvDeviceMemoryFunction* recv_device_memory_function,
    const ffi::ExecutionContext* ffi_execution_context,
    ExecutionStreamIdMap additional_compute_streams, bool mock_collectives,
    int64_t execution_id)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
      device_to_host_stream(device_to_host_stream),
      host_to_device_stream(host_to_device_stream),
      send_device_memory_function(send_device_memory_function),
      recv_device_memory_function(recv_device_memory_function),
      ffi_execution_context(ffi_execution_context),
      additional_compute_streams(additional_compute_streams),
      mock_collectives(mock_collectives),
      execution_id(execution_id) {}

//===----------------------------------------------------------------------===//

/*static*/ absl::string_view NpuThunk::KindToString(NpuThunk::Kind kind) {
#define CASE(x)     \
  case NpuThunk::x: \
    return #x
  switch (kind) {
    // # go/keep-sorted start
    CASE(kAllGather);
    CASE(kAllGatherDone);
    CASE(kAllGatherStart);
    CASE(kAllReduce);
    CASE(kAllReduceDone);
    CASE(kAllReduceStart);
    CASE(kAllToAll);
    CASE(kAllToAllDone);
    CASE(kAllToAllStart);
    CASE(kBuffersDebugChecksum);
    CASE(kBuffersDebugFloatCheck);
    CASE(kCollectiveBroadcast);
    CASE(kCollectiveBroadcastDone);
    CASE(kCollectiveBroadcastStart);
    CASE(kCollectiveKernel);
    CASE(kCollectivePermute);
    CASE(kCollectivePermuteDone);
    CASE(kCollectivePermuteStart);
    CASE(kCall);
    CASE(kCommandBuffer);
    CASE(kConditional);
    CASE(kConvolution);
    CASE(kConvolutionReorder);
    CASE(kCopy);
    CASE(kCopyDone);
    CASE(kCuDnn);
    CASE(kCubSort);
    CASE(kCublasLtMatmul);
    CASE(kCustomCall);
    CASE(kCustomKernel);
    CASE(kDynamicSlice);
    CASE(kFft);
    CASE(kGemm);
    CASE(kGroupDone);
    CASE(kGroupStart);
    CASE(kHostExecuteDone);
    CASE(kHostExecuteStart);
    CASE(kHostRecv);
    CASE(kHostRecvDone);
    CASE(kHostSend);
    CASE(kHostSendDone);
    CASE(kInfeed);
    CASE(kKernel);
    CASE(kMemset32BitValue);
    CASE(kMemzero);
    CASE(kNorm);
    CASE(kNvshmemAllReduceDone);
    CASE(kNvshmemAllReduceStart);
    CASE(kNvshmemCollectivePermute);
    CASE(kNvshmemCollectivePermuteDone);
    CASE(kNvshmemCollectivePermuteStart);
    CASE(kNvshmemRecv);
    CASE(kNvshmemRecvDone);
    CASE(kNvshmemSend);
    CASE(kNvshmemSendDone);
    CASE(kOutfeed);
    CASE(kPartitionId);
    CASE(kRaggedAllToAll);
    CASE(kRaggedAllToAllDone);
    CASE(kRaggedAllToAllStart);
    CASE(kRecv);
    CASE(kRecvDone);
    CASE(kReduceScatter);
    CASE(kReduceScatterDone);
    CASE(kReduceScatterStart);
    CASE(kReplicaId);
    CASE(kSelectK);
    CASE(kSend);
    CASE(kSendDone);
    CASE(kSequential);
    CASE(kTriangularSolve);
    CASE(kWaitForStreams);
    CASE(kWhile);
    // # go/keep-sorted end
  }
}

/*static*/
absl::StatusOr<se::Stream*> NpuThunk::GetStreamForExecution(
    NpuExecutionStreamId stream_id, const ExecuteParams& params) {
  if (stream_id == kDefaultExecutionStreamId) {
    return params.stream;
  }
  auto iter = params.additional_compute_streams.find(stream_id);
  if (iter == params.additional_compute_streams.end()) {
    return absl::InvalidArgumentError("Invalid execution stream id.");
  }
  return iter->second;
}

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind) {
  return os << NpuThunk::KindToString(kind);
}

bool IsReductionCollective(NpuThunk::Kind kind) {
  return kind == NpuThunk::kAllReduce || kind == NpuThunk::kAllReduceStart ||
         kind == NpuThunk::kReduceScatter ||
         kind == NpuThunk::kReduceScatterStart ||
         kind == NpuThunk::kNvshmemAllReduceStart;
  ;
}

NpuThunk::ThunkInfo NpuThunk::ThunkInfo::WithProfileAnnotation(
    const HloInstruction* instr, NpuThunkId thunk_id) {
  ThunkInfo thunk_info;
  thunk_info.profile_annotation = instr->name();
  thunk_info.thunk_id = thunk_id;

  HloPrintOptions options = NpuTool::GetDetailedPrintOptions();
  thunk_info.hloInstrDesc = instr->ToString(options);
  thunk_info.op_name = std::string(instr->name());
  thunk_info.op_code = instr->opcode();
  for (int i = 0; i < instr->operand_count(); ++i) {
    thunk_info.param_shapes.push_back(instr->operand(i)->shape());
  }
  // Flatten output shape: For tuple output (such as kSort), add elements shape one by one,
  // Make sure param_shapes corresponds one-to-one with args_ created by KernelArguments
  if (instr->shape().IsTuple()) {
    for (const Shape& elem_shape : instr->shape().tuple_shapes()) {
      thunk_info.param_shapes.push_back(elem_shape);
    }
  } else {
    thunk_info.param_shapes.push_back(instr->shape());
  }

  // Save dimensions information for operators such as concatenate
  // Only instructions of type HloDimensionsInstruction have the dimensions attribute
  if (HloDimensionsInstruction::ClassOf(instr)) {
    auto dimensions_span = instr->dimensions();
    thunk_info.dimensions =
        std::vector<int64_t>(dimensions_span.begin(), dimensions_span.end());
  }

  // DynamicSlice uses dynamic_slice_sizes instead of dimensions
  if (instr->opcode() == HloOpcode::kDynamicSlice) {
    const auto* dynamic_slice_instr =
        dynamic_cast<const HloDynamicSliceInstruction*>(instr);
    if (dynamic_slice_instr) {
      thunk_info.dimensions = dynamic_slice_instr->dynamic_slice_sizes();
    }
  }

  // Recognize only a scalar binary operator applied directly to both reducer
  // parameters. A tuple or a compound expression must never become ReduceSum
  // just because its name contains "add" or its root happens to be kAdd.
  if (instr->opcode() == HloOpcode::kReduce &&
      !instr->called_computations().empty()) {
    const HloComputation* reducer = instr->called_computations()[0];
    if (reducer != nullptr) {
      const HloInstruction* root = reducer->root_instruction();
      if (root != nullptr && instr->operand_count() == 2 &&
          !instr->shape().IsTuple() && root->operand_count() == 2 &&
          root->operand(0)->opcode() == HloOpcode::kParameter &&
          root->operand(1)->opcode() == HloOpcode::kParameter &&
          ((root->operand(0)->parameter_number() == 0 &&
            root->operand(1)->parameter_number() == 1) ||
           (root->operand(0)->parameter_number() == 1 &&
            root->operand(1)->parameter_number() == 0))) {
        switch (root->opcode()) {
          case HloOpcode::kAdd:
            thunk_info.reduce_reducer_kind =
                ThunkInfo::ReduceReducerKind::kSum;
            break;
          case HloOpcode::kMaximum:
            thunk_info.reduce_reducer_kind =
                ThunkInfo::ReduceReducerKind::kMax;
            break;
          case HloOpcode::kMinimum:
            thunk_info.reduce_reducer_kind =
                ThunkInfo::ReduceReducerKind::kMin;
            break;
          case HloOpcode::kAnd:
            thunk_info.reduce_reducer_kind =
                ThunkInfo::ReduceReducerKind::kAnd;
            break;
          case HloOpcode::kOr:
            thunk_info.reduce_reducer_kind =
                ThunkInfo::ReduceReducerKind::kOr;
            break;
          default:
            break;
        }
      }
    }
  }

  // Gather: Extract complete dnums and determine the lowering type
  if (instr->opcode() == HloOpcode::kGather) {
    const auto* gather_instr = dynamic_cast<const HloGatherInstruction*>(instr);
    if (gather_instr) {
      const auto& dnums = gather_instr->gather_dimension_numbers();
      thunk_info.dimensions.clear();
      for (int64_t v : dnums.start_index_map()) {
        thunk_info.dimensions.push_back(v);
      }

      // Save complete dnums
      for (int64_t v : dnums.offset_dims()) {
        thunk_info.gather_offset_dims.push_back(v);
      }
      for (int64_t v : dnums.collapsed_slice_dims()) {
        thunk_info.gather_collapsed_slice_dims.push_back(v);
      }
      for (int64_t v : dnums.start_index_map()) {
        thunk_info.gather_start_index_map.push_back(v);
      }
      thunk_info.gather_index_vector_dim = dnums.index_vector_dim();
      auto slice_sizes = gather_instr->gather_slice_sizes();
      thunk_info.gather_slice_sizes.assign(slice_sizes.begin(),
                                           slice_sizes.end());
      for (int64_t v : dnums.operand_batching_dims()) {
        thunk_info.gather_operand_batching_dims.push_back(v);
      }
      for (int64_t v : dnums.start_indices_batching_dims()) {
        thunk_info.gather_start_indices_batching_dims.push_back(v);
      }

      // Determine lowering type
      const auto& operand_shape = instr->operand(0)->shape();
      int64_t operand_rank = operand_shape.dimensions().size();
      int64_t index_rank = instr->operand(1)->shape().dimensions().size();

      auto is_single_effective_dim_gather = [&]() -> bool {
        // Single valid dimension gather: start_index_map can have multiple dimensions, but exactly 1
        // slice_size for dim!= operand_size (non-trivial), the rest is "pseudo-gather"
        // Also overrides classic single-dimensional gather with start_index_map.size() == 1
        // Also covers the case of partial slice (1 < slice_size < dim_size)
        if (dnums.index_vector_dim() != index_rank - 1) return false;
        if (instr->operand(1)->shape().dimensions(dnums.index_vector_dim()) !=
            (int64_t)dnums.start_index_map().size())
          return false;

        // Statistically non-trivial gathered dim
        int64_t non_trivial = 0;
        int64_t effective_dim = -1;
        int64_t effective_pos = -1;
        int64_t partial_slice_size = 0;
        for (int64_t i = 0; i < (int64_t)dnums.start_index_map().size(); ++i) {
          int64_t gd = dnums.start_index_map(i);
          if (slice_sizes[gd] == operand_shape.dimensions(gd)) {
            // Trivial dim: slice covers the entire operand dimension, no gather required
            continue;
          }
          // Allow slice_size == 1 (unit gather) and 1 < slice_size < dim_size (partial slice)
          if (slice_sizes[gd] < 1 ||
              slice_sizes[gd] > operand_shape.dimensions(gd))
            return false;  // Illegal slice_size
          if (slice_sizes[gd] > 1) {
            // Partial slicing: allow up to 1 dim with this condition
            partial_slice_size = slice_sizes[gd];
          }
          non_trivial++;
          effective_dim = gd;
          effective_pos = i;
        }
        if (non_trivial > 1) return false;
        // Totally trivial (non_trivial == 0): take the first gathered dim as effective
        if (non_trivial == 0) {
          effective_dim = dnums.start_index_map(0);
          effective_pos = 0;
        }

        // The slice_sizes of non-gathered dim are equal to the corresponding dimensions of operand
        std::set<int64_t> gather_set(dnums.start_index_map().begin(),
                                     dnums.start_index_map().end());
        for (int64_t i = 0; i < operand_rank; ++i) {
          if (gather_set.count(i) == 0 &&
              slice_sizes[i] != operand_shape.dimensions(i))
            return false;
        }

        thunk_info.gather_effective_dim = effective_dim;
        thunk_info.gather_effective_index_pos = effective_pos;
        thunk_info.gather_partial_slice_size = partial_slice_size;
        return true;
      };

      auto is_multi_dim_flatten_gather = [&]() -> bool {
        // Multidimensional gather can be reduced to single dimension by flatten gathered dims + flat indices:
        // 1. start_index_map has multiple dimensions
        // 2. All gathered dims have slice_size 1 (cell gather), or
        //    Exactly one dim has 1 < slice_size < dim_size (partial slice)
        // 3. All units gathered dim are in collapsed_slice_dims
        // 4. gathered dims must be consecutive
        // 5. The slice_sizes of non-gathered dim are equal to the corresponding dimensions of operand
        // 6. index_vector_dim is the last dimension of indices
        // 7. index_vector_dim size == start_index_map.size()
        if (dnums.start_index_map().size() < 2) return false;
        if (dnums.index_vector_dim() != index_rank - 1) return false;
        if (instr->operand(1)->shape().dimensions(dnums.index_vector_dim()) !=
            (int64_t)dnums.start_index_map().size())
          return false;

        // Count the number of partial slices dim and record its slice_size
        int64_t partial_count = 0;
        int64_t partial_slice_size = 0;
        for (int64_t i = 0; i < (int64_t)dnums.start_index_map().size(); ++i) {
          int64_t gd = dnums.start_index_map(i);
          int64_t ss = slice_sizes[gd];
          int64_t ds = operand_shape.dimensions(gd);
          if (ss == ds) continue;  // ordinary dim
          if (ss == 1) continue;   // unit gather
          if (ss > 1 && ss < ds) {
            partial_count++;
            partial_slice_size = ss;
            continue;
          }
          return false;  // Illegal slice_size
        }
        if (partial_count > 1) return false;

        // unit gathered dim (slice_size=1) does not need to be forced in collapsed_slice_dims:
        // After flattening, these dims are absorbed to flat dims, and the GatherV2 output will lack the corresponding
        // size-1 dimension, but with the same number of elements, adapted through the output view at runtime.

        // gathered dims must be consecutive
        std::vector<int64_t> sorted_gather_dims(
            dnums.start_index_map().begin(), dnums.start_index_map().end());
        std::sort(sorted_gather_dims.begin(), sorted_gather_dims.end());
        for (size_t i = 1; i < sorted_gather_dims.size(); ++i) {
          if (sorted_gather_dims[i] != sorted_gather_dims[i - 1] + 1)
            return false;
        }

        // The slice_sizes of non-gathered dim are equal to the corresponding dimensions of operand
        std::set<int64_t> gather_set(sorted_gather_dims.begin(),
                                     sorted_gather_dims.end());
        for (int64_t i = 0; i < operand_rank; ++i) {
          if (gather_set.count(i) == 0 &&
              slice_sizes[i] != operand_shape.dimensions(i))
            return false;
        }

        thunk_info.gather_partial_slice_size = partial_slice_size;
        return true;
      };

      if (is_single_effective_dim_gather()) {
        // Single valid dimension gather: distinguish simple vs with-view based on collapsed_slice_dims
        if (dnums.collapsed_slice_dims().size() == 1 &&
            dnums.collapsed_slice_dims(0) ==
                thunk_info.gather_effective_dim) {
          thunk_info.gather_lowering_kind =
              ThunkInfo::GatherLoweringKind::kSimpleGatherV2;
        } else if (dnums.start_index_map().size() >= 2 &&
                   is_multi_dim_flatten_gather()) {
          // collapsed_slice_dims contains dimensions other than effective_dim (such as trivial
          // gathered dim is also collapsed), at this time the single-dimensional GatherV2 output shape
          // These dimensions are preserved, causing the number of elements to not match the XLA output.
          // Downgrade to a multidimensional flatten path, flattening all gathered dims.
          thunk_info.gather_lowering_kind =
              ThunkInfo::GatherLoweringKind::kMultiDimFlattenGather;
        } else {
          thunk_info.gather_lowering_kind =
              ThunkInfo::GatherLoweringKind::kGatherV2WithView;
        }
      } else if (is_multi_dim_flatten_gather()) {
        thunk_info.gather_lowering_kind =
            ThunkInfo::GatherLoweringKind::kMultiDimFlattenGather;
      } else {
        thunk_info.gather_lowering_kind =
            ThunkInfo::GatherLoweringKind::kUnsupported;
      }
    }
  }

  // Scatter: inspect the reducer computation to determine reduction kind
  if (instr->opcode() == HloOpcode::kScatter) {
    const auto* scatter_instr =
        dynamic_cast<const HloScatterInstruction*>(instr);
    if (scatter_instr && !scatter_instr->called_computations().empty()) {
      thunk_info.scatter_add_normalized =
          instr->get_frontend_attribute("npu.scatter_add_normalized") == "1";
      const auto& scatter_dims =
          scatter_instr->scatter_dimension_numbers();
      // Always record scatter dimension for single-dim scatters so the
      // execution phase can use it even when no specific lowering kind matches.
      if (scatter_dims.scatter_dims_to_operand_dims_size() == 1) {
        thunk_info.scatter_dimension =
            scatter_dims.scatter_dims_to_operand_dims(0);
      }

      bool ordered_update_window_dims = true;
      for (int64_t i = 0; i < scatter_dims.update_window_dims_size(); ++i) {
        if (scatter_dims.update_window_dims(i) != i) {
          ordered_update_window_dims = false;
          break;
        }
      }

      // x.at[:, i].set(update) becomes a single-index replace scatter where
      // updates are the operand slice with the scatter dimension removed.
      // Shape contract: index rank <= 1, exactly 1 index element,
      // update rank + 1 == data rank (scatter dim removed from updates).
      //
      // When the shape contract is not met but update rank == data rank,
      // this is a same-rank single-dim scatter (operand [A,B,K],
      // indices [N,1], updates [A,B,N]) that goes through aclnnScatter.
      if (scatter_dims.scatter_dims_to_operand_dims_size() == 1 &&
          scatter_dims.inserted_window_dims_size() == 1 &&
          scatter_dims.inserted_window_dims(0) ==
              scatter_dims.scatter_dims_to_operand_dims(0) &&
          scatter_dims.input_batching_dims_size() == 0 &&
          scatter_dims.scatter_indices_batching_dims_size() == 0 &&
          ordered_update_window_dims) {
        const Shape& data_shape = instr->operand(0)->shape();
        const Shape& index_shape = instr->operand(1)->shape();
        const Shape& update_shape = instr->operand(2)->shape();
        const int64_t scatter_dim =
            scatter_dims.scatter_dims_to_operand_dims(0);
        if (data_shape.IsArray() && index_shape.IsArray() &&
            update_shape.IsArray() &&
            index_shape.dimensions_size() <= 1 &&
            ShapeUtil::ElementsIn(index_shape) == 1 &&
            update_shape.dimensions_size() + 1 ==
                data_shape.dimensions_size()) {
          thunk_info.scatter_lowering_kind =
              NpuThunk::ThunkInfo::ScatterLoweringKind::kSingleSliceIndexCopy;
          thunk_info.scatter_dimension = scatter_dim;
        } else if (data_shape.IsArray() && index_shape.IsArray() &&
                   update_shape.IsArray() &&
                   update_shape.dimensions_size() ==
                       data_shape.dimensions_size()) {
          thunk_info.scatter_lowering_kind =
              NpuThunk::ThunkInfo::ScatterLoweringKind::
                  kSameRankSingleDimScatter;
          thunk_info.scatter_dimension = scatter_dim;
        }
      }

      // A vmapped single-dimension replace scatter carries the vmap axis as
      // an HLO scatter batch dimension. aclnnScatter can express that directly
      // with an expanded index view along the scattered operand dimension.
      if (scatter_dims.scatter_dims_to_operand_dims_size() == 1 &&
          scatter_dims.inserted_window_dims_size() == 1 &&
          scatter_dims.inserted_window_dims(0) ==
              scatter_dims.scatter_dims_to_operand_dims(0) &&
          scatter_dims.input_batching_dims_size() == 1 &&
          scatter_dims.scatter_indices_batching_dims_size() == 1 &&
          ordered_update_window_dims) {
        thunk_info.scatter_lowering_kind =
            NpuThunk::ThunkInfo::ScatterLoweringKind::kBatchedSingleDimScatter;
        thunk_info.scatter_dimension =
            scatter_dims.scatter_dims_to_operand_dims(0);
        thunk_info.scatter_batch_dimension = scatter_dims.input_batching_dims(0);
        thunk_info.scatter_indices_batch_dimension =
            scatter_dims.scatter_indices_batching_dims(0);
      }

      // x.at[..., start:start + window].set(update) keeps the scatter
      // dimension in the update window. Lower a single static in-bounds window
      // as an output subview copy instead of treating it as ScatterNd points.
      if (instr->operand_count() == 3 &&
          scatter_dims.scatter_dims_to_operand_dims_size() == 1 &&
          scatter_dims.inserted_window_dims_size() == 0 &&
          scatter_dims.input_batching_dims_size() == 0 &&
          scatter_dims.scatter_indices_batching_dims_size() == 0 &&
          ordered_update_window_dims) {
        const auto* indices_constant =
            dynamic_cast<const HloConstantInstruction*>(instr->operand(1));
        const Shape& data_shape = instr->operand(0)->shape();
        const Shape& index_shape = instr->operand(1)->shape();
        const Shape& update_shape = instr->operand(2)->shape();
        const int64_t scatter_dimension =
            scatter_dims.scatter_dims_to_operand_dims(0);
        std::optional<int64_t> scatter_start;
        if (indices_constant != nullptr &&
            ShapeUtil::ElementsIn(index_shape) == 1) {
          switch (index_shape.element_type()) {
            case PrimitiveType::S32:
              scatter_start =
                  indices_constant->literal().GetFirstElement<int32_t>();
              break;
            case PrimitiveType::S64:
              scatter_start =
                  indices_constant->literal().GetFirstElement<int64_t>();
              break;
            default:
              break;
          }
        }

        bool full_non_scatter_window =
            data_shape.IsArray() && update_shape.IsArray() &&
            scatter_dimension >= 0 &&
            scatter_dimension < data_shape.dimensions_size() &&
            scatter_dims.update_window_dims_size() ==
                update_shape.dimensions_size() &&
            update_shape.dimensions_size() == data_shape.dimensions_size();
        for (int64_t dim = 0;
             full_non_scatter_window && dim < data_shape.dimensions_size();
             ++dim) {
          if (dim != scatter_dimension &&
              data_shape.dimensions(dim) != update_shape.dimensions(dim)) {
            full_non_scatter_window = false;
          }
        }

        if (scatter_start.has_value() && full_non_scatter_window &&
            *scatter_start >= 0 &&
            *scatter_start + update_shape.dimensions(scatter_dimension) <=
                data_shape.dimensions(scatter_dimension)) {
          thunk_info.scatter_lowering_kind =
              NpuThunk::ThunkInfo::ScatterLoweringKind::kStaticWindowCopy;
          thunk_info.scatter_dimension = scatter_dimension;
          thunk_info.scatter_window_start = *scatter_start;
        }
      }

      // Multi-dim scatter: when scatter_dims_to_operand_dims_size() > 1,
      // aclnnScatterNd cannot handle it directly. Linearize indices to
      // single-dim and use aclnnScatterNd on a flattened 1D view.
      if (scatter_dims.scatter_dims_to_operand_dims_size() > 1 &&
          thunk_info.scatter_lowering_kind ==
              NpuThunk::ThunkInfo::ScatterLoweringKind::kScatterNd &&
          ordered_update_window_dims) {
        thunk_info.scatter_lowering_kind =
            NpuThunk::ThunkInfo::ScatterLoweringKind::kMultiDimScatter;
        const Shape& data_shape = instr->operand(0)->shape();
        for (int i = 0; i < data_shape.dimensions_size(); ++i) {
          thunk_info.scatter_operand_dims.push_back(
              data_shape.dimensions(i));
        }
        for (int i = 0;
             i < scatter_dims.scatter_dims_to_operand_dims_size(); ++i) {
          thunk_info.scatter_dims_to_operand.push_back(
              scatter_dims.scatter_dims_to_operand_dims(i));
        }
        for (int i = 0; i < scatter_dims.update_window_dims_size(); ++i) {
          thunk_info.scatter_update_window_dims.push_back(
              scatter_dims.update_window_dims(i));
        }
        for (int i = 0; i < scatter_dims.inserted_window_dims_size(); ++i) {
          thunk_info.scatter_inserted_window_dims.push_back(
              scatter_dims.inserted_window_dims(i));
        }
        thunk_info.scatter_index_vector_dim =
            scatter_dims.index_vector_dim();
        if (scatter_dims.input_batching_dims_size() > 0) {
          thunk_info.scatter_operand_batch_dim =
              scatter_dims.input_batching_dims(0);
        }
        if (scatter_dims.scatter_indices_batching_dims_size() > 0) {
          thunk_info.scatter_indices_batch_dim =
              scatter_dims.scatter_indices_batching_dims(0);
        }
      }

      const HloComputation* reducer =
          scatter_instr->called_computations()[0];
      const HloInstruction* root = reducer->root_instruction();
      if (root->opcode() == HloOpcode::kParameter) {
        // reducer body just returns the update parameter → replace
        thunk_info.scatter_reduction_kind =
            NpuThunk::ThunkInfo::ScatterReductionKind::kReplace;
      } else if (root->opcode() == HloOpcode::kAdd &&
                 root->operand_count() == 2 &&
                 root->operand(0)->opcode() == HloOpcode::kParameter &&
                 root->operand(1)->opcode() == HloOpcode::kParameter &&
                 root->operand(0)->parameter_number() !=
                     root->operand(1)->parameter_number()) {
        // reducer body returns add(old, update) → add scatter
        thunk_info.scatter_reduction_kind =
            NpuThunk::ThunkInfo::ScatterReductionKind::kAdd;
      } else if (root->opcode() == HloOpcode::kMultiply) {
        // reducer body returns multiply(old, update) → mul scatter
        thunk_info.scatter_reduction_kind =
            NpuThunk::ThunkInfo::ScatterReductionKind::kMul;
      } else {
        // max/min/other → unsupported
        thunk_info.scatter_reduction_kind =
            NpuThunk::ThunkInfo::ScatterReductionKind::kUnsupported;
      }
    }
  }

  // Iota: extract iota_dimension from HloIotaInstruction
  if (instr->opcode() == HloOpcode::kIota) {
    const auto* iota_instr = dynamic_cast<const HloIotaInstruction*>(instr);
    if (iota_instr) {
      thunk_info.iota_dimension = iota_instr->iota_dimension();
    }
  }

  // ReduceWindow: extract window config and detect reducer type
  if (instr->opcode() == HloOpcode::kReduceWindow) {
    const auto* reduce_window_instr =
        dynamic_cast<const HloReduceWindowInstruction*>(instr);
    if (reduce_window_instr) {
      const Window& window = reduce_window_instr->window();
      for (const auto& dim : window.dimensions()) {
        thunk_info.window_dimensions.push_back(dim.size());
        thunk_info.window_strides.push_back(dim.stride());
        thunk_info.window_padding_low.push_back(dim.padding_low());
        thunk_info.window_padding_high.push_back(dim.padding_high());
        thunk_info.window_dilations.push_back(dim.window_dilation());
      }

      // Detect reducer type (check root instruction of computation body)
      const HloComputation* reducer = reduce_window_instr->to_apply();
      if (reducer) {
        const HloInstruction* root = reducer->root_instruction();
        if (root->opcode() == HloOpcode::kMaximum) {
          thunk_info.reduce_window_reducer_kind =
              ThunkInfo::ReduceWindowReducerKind::kMax;
        } else if (root->opcode() == HloOpcode::kAdd) {
          // Detect cumsum mode: add reducer + init=0 + single non-trivial window dimension
          const HloInstruction* init_value = reduce_window_instr->operand(1);
          bool init_is_zero = false;
          if (init_value->opcode() == HloOpcode::kConstant) {
            const Literal& lit = init_value->literal();
            switch (lit.shape().element_type()) {
              case xla::S32:
                init_is_zero = (lit.Get<int32_t>({}) == 0);
                break;
              case xla::F32:
                init_is_zero = (lit.Get<float>({}) == 0.0f);
                break;
              case xla::F64:
                init_is_zero = (lit.Get<double>({}) == 0.0);
                break;
              case xla::S64:
                init_is_zero = (lit.Get<int64_t>({}) == 0);
                break;
              default:
                break;
            }
          }

          if (init_is_zero) {
            const Shape& input_shape =
                reduce_window_instr->operand(0)->shape();
            const Window& window = reduce_window_instr->window();
            int64_t cumsum_dim = -1;
            bool is_cumsum = true;

            for (int i = 0; i < window.dimensions().size(); ++i) {
              const auto& wdim = window.dimensions(i);
              bool is_trivial =
                  (wdim.size() == 1 && wdim.stride() == 1 &&
                   wdim.padding_low() == 0 && wdim.padding_high() == 0 &&
                   wdim.window_dilation() == 1 && wdim.base_dilation() == 1 &&
                   !wdim.window_reversal());
              if (is_trivial) continue;

              if (cumsum_dim >= 0) {
                is_cumsum = false;
                break;
              }

              int64_t dim_size = input_shape.dimensions(i);
              if (wdim.size() == dim_size && wdim.stride() == 1 &&
                  wdim.padding_low() == dim_size - 1 &&
                  wdim.padding_high() == 0 && wdim.window_dilation() == 1 &&
                  wdim.base_dilation() == 1 && !wdim.window_reversal()) {
                cumsum_dim = i;
              } else {
                is_cumsum = false;
              }
            }

            if (is_cumsum && cumsum_dim >= 0) {
              thunk_info.reduce_window_reducer_kind =
                  ThunkInfo::ReduceWindowReducerKind::kCumsum;
              thunk_info.cumsum_dim = cumsum_dim;
            }
          }
        }
        // Other reducers remain kUnsupported
      }
    }
  }

  // Sort: extract stable/descending flags and determine comparator direction
  if (instr->opcode() == HloOpcode::kSort) {
    const auto* sort_instr =
        dynamic_cast<const HloSortInstruction*>(instr);
    if (sort_instr) {
      thunk_info.sort_stable = sort_instr->is_stable();
      // Determine descending from the comparator's comparison direction
      if (!sort_instr->called_computations().empty()) {
        const HloComputation* comparator =
            sort_instr->called_computations()[0];
        const HloInstruction* root = comparator->root_instruction();
        if (root->opcode() == HloOpcode::kCompare) {
          std::string root_str = root->ToString();
          // GT or GE → descending; LT or LE → ascending
          thunk_info.sort_descending =
              (root_str.find("direction=GT") != std::string::npos ||
               root_str.find("direction=GE") != std::string::npos);
        }
      }
    }
  }

  // TopK: extract k / largest and the effective sort dimension (last dim).
  if (instr->opcode() == HloOpcode::kTopK) {
    const auto* topk_instr =
        dynamic_cast<const HloTopKInstruction*>(instr);
    if (topk_instr) {
      thunk_info.topk_k = topk_instr->k();
      thunk_info.topk_largest = topk_instr->largest();
      thunk_info.topk_dimension =
          instr->operand(0)->shape().dimensions_size() - 1;
    }
  }

  // Pad: extract low/high/interior padding for each dimension.
  if (instr->opcode() == HloOpcode::kPad) {
    const auto* pad_instr =
        dynamic_cast<const HloPadInstruction*>(instr);
    if (pad_instr) {
      const PaddingConfig& padding_config = pad_instr->padding_config();
      for (const auto& dim : padding_config.dimensions()) {
        thunk_info.pad_edge_padding_low.push_back(dim.edge_padding_low());
        thunk_info.pad_edge_padding_high.push_back(dim.edge_padding_high());
        thunk_info.pad_interior_padding.push_back(dim.interior_padding());
      }
    }
  }

  // Cholesky: extract lower/upper triangular flag
  if (instr->opcode() == HloOpcode::kCholesky) {
    thunk_info.cholesky_is_lower = instr->cholesky_options().lower();
  }

  // TriangularSolve: extract options
  if (instr->opcode() == HloOpcode::kTriangularSolve) {
    const auto& opts = instr->triangular_solve_options();
    thunk_info.triangular_solve_upper = !opts.lower();
    thunk_info.triangular_solve_transpose =
        (opts.transpose_a() == TriangularSolveOptions::TRANSPOSE);
    thunk_info.triangular_solve_unitriangular = opts.unit_diagonal();
    thunk_info.triangular_solve_left_side = opts.left_side();
  }

  return thunk_info;
}

bool NpuThunk::IsCollective() const {
  switch (kind()) {
    // go/keep-sorted start
    case kAllGather:
    case kAllGatherDone:
    case kAllGatherStart:
    case kAllReduce:
    case kAllReduceDone:
    case kAllReduceStart:
    case kAllToAll:
    case kAllToAllDone:
    case kAllToAllStart:
    case kCollectiveBroadcast:
    case kCollectiveBroadcastDone:
    case kCollectiveBroadcastStart:
    case kCollectivePermute:
    case kCollectivePermuteDone:
    case kCollectivePermuteStart:
    case kGroupDone:
    case kGroupStart:
    case kRaggedAllToAll:
    case kRaggedAllToAllDone:
    case kRaggedAllToAllStart:
    case kRecv:
    case kRecvDone:
    case kReduceScatter:
    case kReduceScatterDone:
    case kReduceScatterStart:
    case kSend:
    case kSendDone:
      // go/keep-sorted end
      return true;
    default:
      return false;
  }
}

void NpuThunk::ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
}

void NpuThunk::ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
}

// Handle all special values in the constant buffer during the Initialize phase
// These special values (-inf, +inf, nan) are only written during graph allocation and are not generated during calculations.
absl::Status NpuThunk::InitializeSpecialValues(
    const InitializeParams& params,
    const std::vector<BufferAllocation::Slice>& args) {
  const auto* allocations = params.buffer_allocations;

  // Iterate over all args, checking and replacing special values
  // Note: We only deal with buffers of scalar floating point type
  for (const auto& slice : args) {
    // Skip entry computation parameter: parameter buffer is not a constant,
    // And dep::aclrtMemcpy will destroy the device memory of the parameter buffer when there is no aclrtSetDevice.
    if (slice.allocation()->is_entry_computation_parameter()) {
      continue;
    }

    // Get the shape corresponding to the buffer (from thunk_info)
    // Since we have no direct shape information, a simple strategy is used here:
    // Check all small buffers (< 16 bytes), these are most likely scalar constants
    auto addr = allocations->GetDeviceAddress(slice);
    void* device_addr = addr.opaque();
    size_t size = addr.size();
    // Only handle small buffers (possibly scalar)
    if (size > sizeof(float) * 4) {
      continue;
    }

    // Only handles 4-byte or 8-byte buffers (float or double)
    if (size != sizeof(float) && size != sizeof(double)) {
      continue;
    }

    // Only handle buffers of floating point type; skip integer/other types.
    // The bit pattern of integer types such as s32 may be NaN/Inf under float interpretation.
    // For example, s32(2147483647) = 0x7FFFFFFF is NaN in float,
    // Cleaning by mistake will cause constant values to be cleared.
    if (!primitive_util::IsFloatingPointType(slice.element_type())) {
      continue;
    }

    // Try handling as float
    if (size == sizeof(float)) {
      float value_host = 0.0f;
      aclError ret = dep::aclrtMemcpy(&value_host, sizeof(float), device_addr,
                                      sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_SUCCESS) {
        continue;  // Skip failed reads
      }

      // Check and replace special values
      bool need_replace = false;
      float replacement_value = 0.0f;

      if (std::isinf(value_host) && value_host < 0) {
        need_replace = true;
        replacement_value = -3.4028235e+38f;  // -FLT_MAX
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected -inf in buffer (slice "
                             "%d:%d), replacing with -FLT_MAX",
                             slice.index(), slice.offset());
      } else if (std::isinf(value_host) && value_host > 0) {
        need_replace = true;
        replacement_value = 3.4028235e+38f;  // FLT_MAX
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected +inf in buffer (slice "
                             "%d:%d), replacing with FLT_MAX",
                             slice.index(), slice.offset());
      } else if (std::isnan(value_host)) {
        need_replace = true;
        replacement_value = 0.0f;
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Initialize: Detected nan in buffer (slice "
                             "%d:%d), replacing with 0.0f",
                             slice.index(), slice.offset());
      }

      if (need_replace) {
        ret = dep::aclrtMemcpy(device_addr, sizeof(float), &replacement_value,
                               sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret == ACL_SUCCESS) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG,
              "Initialize: Successfully replaced special value in buffer");
        }
      }
    }
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===n// Print Functions for Debugging
//===----------------------------------------------------------------------===n
/*static*/ void NpuThunk::PrintKernelConfig(
    const NpuKernelConfig* kernel_config) {
  if (kernel_config == nullptr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Kernel config is null");
    return;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "---- NPU Kernel Config: ----");
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Kernel type: %d",
                       static_cast<int>(kernel_config->npu_kernel_type));
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Artifact name: %s",
                       kernel_config->npu_artifact_info.name.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Artifact dir: %s",
                       kernel_config->npu_artifact_info.dir.c_str());
}

/*static*/ void NpuThunk::PrintComputationGraphs(
    const std::unordered_map<std::string, std::string>& computation_hlo_map,
    const std::string& entry_computation_name) {
  if (computation_hlo_map.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "No computation graphs available");
    return;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Computation Graphs (%d total): ----",
                       computation_hlo_map.size());
  for (const auto& [name, hlo_text] : computation_hlo_map) {
    bool is_entry = (name == entry_computation_name);
    PrintComputationHlo(hlo_text, name, is_entry);
  }
}

/*static*/ void NpuThunk::PrintComputationHlo(const std::string& hlo_text,
                                              const std::string& name,
                                              bool is_entry) {
  if (hlo_text.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Computation %s HLO is empty", name.c_str());
    return;
  }

  const char* entry_marker = is_entry ? " [ENTRY]" : "";
  NPU_LOG_FORMAT_PRINT(DEBUG, "==== Computation: %s%s ====", name.c_str(),
                       entry_marker);

  // Print HLO text line by line
  size_t start = 0;
  size_t end = hlo_text.find('\n');
  while (end != std::string::npos) {
    std::string line = hlo_text.substr(start, end - start);
    if (!line.empty()) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "  %s", line.c_str());
    }
    start = end + 1;
    end = hlo_text.find('\n', start);
  }
  // Print last line if any
  if (start < hlo_text.length()) {
    std::string line = hlo_text.substr(start);
    if (!line.empty()) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "  %s", line.c_str());
    }
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "==== End of Computation: %s ====", name.c_str());
}

//===----------------------------------------------------------------------===//
// Unified Stream Sync and Print Control
//===----------------------------------------------------------------------===//

/*static*/ absl::Status NpuThunk::SyncStreamIfEnabled(se::Stream* stream) {
  if (NPU_CONFIG_GET_BOOL(NpuConfig::Key::ENABLE_SYNC_AFTER_EXECUTE)) {
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Sync after execute enabled");
  }
  return absl::OkStatus();
}

/*static*/ bool NpuThunk::IsPrintIOValuesEnabled() {
  return NPU_CONFIG_GET_BOOL(NpuConfig::Key::ENABLE_PRINT_IO_VALUES);
}

/*static*/ void NpuThunk::PrintParamInfo(int param_id,
                                         const std::string& param_type,
                                         int device_ordinal,
                                         const BufferAllocation::Slice& slice,
                                         const Shape& shape,
                                         void* device_addr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Param %d (%s) info: ----", param_id,
                       param_type.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Device id: %d, Arg: alloc #%d, offset: %d: %p",
                       device_ordinal, slice.index(), slice.offset(),
                       device_addr);
  NPU_LOG_FORMAT_PRINT(DEBUG, "  Shape: %s", shape.ToString(true).c_str());
  NpuTool::PrintNpuDeviceMemoryValueByShape("  Value on device:", shape,
                                            device_addr);
}

}  // namespace xla
