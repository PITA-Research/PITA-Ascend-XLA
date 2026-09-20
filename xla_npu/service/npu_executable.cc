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

#include "xla/xla_npu/service/npu_executable.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/rendezvous.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_pass_pipeline.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"
#include "xla/xla_npu/service/npu_stream_executor_util.h"

namespace xla {

namespace {

// Chooses the correct allocations to be used within the NpuExecutable code.
std::vector<const BufferAllocation*> GatherAllocationPtrs(
    const NpuExecutable::Params& params,
    const std::deque<BufferAllocation>& thunk_pass_allocations) {
  const std::vector<BufferAllocation>* allocation_vec = nullptr;
  if (params.mlir_allocations.has_value()) {
    allocation_vec = &params.mlir_allocations.value();
  } else if (params.buffer_assignment != nullptr) {
    allocation_vec = &params.buffer_assignment->Allocations();
  }

  std::vector<const BufferAllocation*> alloc_ptrs;
  if (allocation_vec != nullptr) {
    alloc_ptrs.reserve(allocation_vec->size());
    for (const BufferAllocation& alloc : *allocation_vec) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  if (!thunk_pass_allocations.empty()) {
    alloc_ptrs.reserve(alloc_ptrs.size() + thunk_pass_allocations.size());
    for (const BufferAllocation& alloc : thunk_pass_allocations) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  return alloc_ptrs;
}

class NpuExecutableThunkPassBufferAllocator
    : public NpuThunkPassBufferAllocator {
 public:
  ~NpuExecutableThunkPassBufferAllocator() override = default;

  explicit NpuExecutableThunkPassBufferAllocator(
      BufferAllocation::Index start_idx)
      : next_idx_(start_idx) {}

  absl::StatusOr<BufferAllocation* absl_nonnull> NewEmptyAllocation(
      int64_t size) override {
    allocations_.push_back(BufferAllocation(next_idx_++, size, /*color=*/0));
    return &allocations_.back();
  }

  std::deque<BufferAllocation>& MutableAllocations() { return allocations_; }

 private:
  BufferAllocation::Index next_idx_ = 0;
  // std::deque is used to ensure pointer stability.
  std::deque<BufferAllocation> allocations_;
};

}  // namespace

using ::tsl::profiler::ScopedAnnotation;

// Returns the set of `ExecutionStreamIds` requested by all `Thunks` in the
// `NpuExecutable`. At run time `Thunks` may use additional streams to launch
// compute operations in parallel.
static absl::flat_hash_set<NpuExecutionStreamId> GetExecutionStreamIds(
    const NpuSequentialThunk& thunks) {
  absl::flat_hash_set<NpuExecutionStreamId> stream_ids;
  thunks.ForAllThunks([&](const NpuThunk* thunk) {
    if (thunk->execution_stream_id() > 0) {
      stream_ids.insert(thunk->execution_stream_id());
    }
  });
  return stream_ids;
}

absl::StatusOr<std::unique_ptr<NpuExecutable>> NpuExecutable::Create(
    Params params) {
  int64_t next_idx = 0;
  if (params.mlir_allocations.has_value()) {
    next_idx = params.mlir_allocations->size();
  } else if (params.buffer_assignment != nullptr) {
    next_idx = params.buffer_assignment->Allocations().size();
  }

  NpuExecutableThunkPassBufferAllocator allocator(next_idx);
  return std::unique_ptr<NpuExecutable>(new NpuExecutable(
      std::move(params), std::move(allocator.MutableAllocations())));
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
NpuExecutable::NpuExecutable(
    NpuExecutable::Params params,
    std::deque<BufferAllocation> thunk_pass_allocations)
    : Executable(std::move(params.debug_module)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      thunks_(std::move(params.executable)),
      execution_stream_ids_(GetExecutionStreamIds(*thunks_)),
      module_name_(params.module_name),
      program_shape_(params.program_shape),
      allocation_ptrs_(GatherAllocationPtrs(params, thunk_pass_allocations)),
      allocations_(std::move(params.mlir_allocations)),
      buffer_assignment_(std::move(params.buffer_assignment)),
      thunk_pass_allocations_(std::move(thunk_pass_allocations)),
      debug_buffer_assignment_show_max_(
          params.debug_options.xla_debug_buffer_assignment_show_max()),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)),
      enable_debug_info_manager_(params.enable_debug_info_manager) {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               buffer_assignment_);
  }
}

NpuExecutable::~NpuExecutable() {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

namespace {

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync);

absl::Status ExecuteThunksImpl(
    const DebugOptions* debug_options, const std::string& module_name,
    ModuleIdentifier module_id, NpuSequentialThunk& thunk_sequence,
    NpuThunk::ExecutableSource executable_source,
    const ServiceExecutableRunOptions* run_options,
    const NpuBufferAllocations& buffer_allocations, bool block_host_until_done,
    const absl::flat_hash_set<NpuExecutionStreamId>& execution_stream_ids) {
  bool use_highest_priority_for_async_stream =
      debug_options
          ? debug_options->xla_gpu_enable_highest_priority_async_stream()
          : false;

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  stream_executor::StreamPriority stream_priority =
      stream_executor::StreamPriority::Default;
  if (executor->GetPlatform()->id() == stream_executor::sycl::kSyclPlatformId) {
    use_highest_priority_for_async_stream = false;
  }
  if (use_highest_priority_for_async_stream) {
    stream_priority = stream_executor::StreamPriority::Highest;
  }
  se::Stream* command_buffer_trace_stream = nullptr;

  // Borrow stream for additional compute streams
  NpuThunk::ExecutionStreamIdMap additional_execution_streams;
  std::vector<StreamPool::Ptr> additional_streams;
  if (!execution_stream_ids.empty()) {
    if (run_options->HasStreamBorrower()) {
      TF_ASSIGN_OR_RETURN(additional_streams, run_options->BorrowStreams(
                                                  executor->device_ordinal(),
                                                  execution_stream_ids.size()));
      int64_t i = 0;
      for (NpuExecutionStreamId stream_id : execution_stream_ids) {
        additional_execution_streams[stream_id] =
            additional_streams.at(i).get();
        i++;
      }
      VLOG(2) << "Using " << additional_execution_streams.size()
              << " additional compute streams.";
    } else {
      VLOG(2) << "No stream borrower created. Assigning the default stream to "
                 "all parallel computes.";
      for (NpuExecutionStreamId stream_id : execution_stream_ids) {
        additional_execution_streams[stream_id] = main_stream;
      }
    }
  }

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::unique_ptr<se::EventBasedTimer> execution_timer;

  {  // Initialize thunks using prepared resources before execution.
    NpuThunk::InitializeParams initialize_params{
        executor,
        executable_source,
        &buffer_allocations,
        main_stream,
        command_buffer_trace_stream,
        run_options->run_options().ffi_execution_context(),
        run_options->local_device_count()};

    tsl::profiler::TraceMe trace_initialize("Thunks::Initialize");
    TF_RETURN_IF_ERROR(thunk_sequence.Initialize(initialize_params));
  }

  // Prepare parameters for thunks execution.
  NpuThunk::ExecuteParams execute_params =
      NpuThunk::ExecuteParams::Create(*run_options, buffer_allocations,
                                      main_stream, command_buffer_trace_stream,
                                      // &collective_params,
                                      // &collective_cliques,
                                      std::move(additional_execution_streams));

  VLOG(1) << "[" << run_options->device_ordinal()
          << "] Start NpuExecutable::ExecuteOnStream module: " << module_name;
  TF_RETURN_IF_ERROR(thunk_sequence.ExecuteOnStream(execute_params));
  VLOG(1) << "[" << run_options->device_ordinal()
          << "] End NpuExecutable::ExecuteOnStream module: " << module_name;

  return MaybeSyncAndProfile(run_options, execution_timer.get(),
                             block_host_until_done ? main_stream : nullptr);
}

namespace {
// Wrap RunId into a unique struct to guarantee we do not accidentally try to
// run multiple unrelated rendezvous for a same key.
struct InitializationKey {
  RunId run_id;

  template <typename H>
  friend H AbslHashValue(H h, const InitializationKey& key) {
    return H::combine(std::move(h), key.run_id);
  }
};

bool operator==(const InitializationKey& a, const InitializationKey& b) {
  return a.run_id == b.run_id;
}
}  // namespace

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync = nullptr) {
  // If we're measuring the execution time then it's important to queue the
  // stop event before triggering any synchronization.
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed,
                        execution_timer->GetElapsedDuration());
    profile->set_compute_time_ns(
        std::max(absl::ToDoubleNanoseconds(elapsed), 1.0));
  }

  if (stream_to_sync) {
    absl::Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return Internal(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.message());
    }
  }

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<const NpuExecutable::BufferAllocToDeviceMemoryMap*>
NpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<stream_executor::DeviceMemoryBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceMemoryBase global;
    if (static_cast<bool>(module_handle) && global_status.ok()) {
      // The constant was defined in the PTX and has been allocated by the CUDA
      // driver.
      global = *global_status;
      VLOG(3) << "Resolved global " << info.symbol_name << " to "
              << global.opaque();

      if (!info.content.span().empty()) {
        // This means the constant did not have an initializer in the PTX and
        // therefore must be initialized by XLA here.
        TF_RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                          info.content.span().size()));
        submitted_mem_copies = true;
      }
    } else {
      // The constant was not defined in the PTX and therefore must be both
      // allocated and initialized by XLA here.
      // Zero-size constants (e.g. shape (0,)) have empty content and need
      // no device allocation; skip them to avoid dereferenceable(0) in LLVM
      // IR that llvm-link rejects.
      if (info.content.span().empty()) {
        VLOG(3) << "Skipping zero-size constant " << info.symbol_name;
        if (info.allocation_index != -1) {
          InsertOrDie(globals.get(), info.allocation_index,
                      se::DeviceMemoryBase());
        }
        continue;
      }

      TF_ASSIGN_OR_RETURN(auto shared, executor->CreateOrShareConstant(
                                           stream, info.content.span()));
      global = *shared;
      VLOG(3) << "Allocated (or shared) global " << info.symbol_name << " at "
              << global.opaque();
      // XLA will continue to own this global at least until this executable is
      // destroyed (longer if another, longer-lived executable shares the same
      // constant).
      shared_constants_.push_back(std::move(shared));
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    TF_CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

absl::StatusOr<se::DeviceMemoryBase> NpuExecutable::BufferForAllocation(
    VariantArguments arguments,
    const NpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx, se::Stream* stream) {
  if (allocation.is_thread_local()) {
    return se::DeviceMemoryBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64_t param_no = allocation.parameter_number();
    se::DeviceMemoryBase registered_buffer = [&] {
      if (auto unowned_shapedbuffers =
              std::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return std::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceMemoryBase();
      }
    }();
    if (registered_buffer.is_null() && registered_buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run NPU XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has zero "
          "elements.",
          allocation.param_shape_index().ToString(), param_no);
    }
    return registered_buffer;
  } else if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceMemoryBase();
    } else {
      return it->second;
    }
  } else {
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    const int64_t buffer_size = allocation.size();
    se::DeviceMemoryBase buffer_address;
    if (buffer_size > 0) {
      TF_ASSIGN_OR_RETURN(
          se::OwningDeviceMemory buffer,
          memory_allocator->Allocate(device_ordinal, buffer_size,
                                     /*retry_on_failure=*/true,
                                     /*memory_space=*/allocation.color()));
      buffer_address = buffer.Release();
    }
    return buffer_address;
  }
}

static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceMemoryBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return NpuConstant::kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return NpuConstant::kConstantBufferAlignBytes;
    } else {
      return NpuConstant::kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

absl::StatusOr<NpuBufferAllocations> NpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const NpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    se::Stream* stream) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  const int64_t num_buffers = allocations.size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    TF_ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(arguments, globals, allocation, memory_allocator,
                            device_ordinal, i, stream));
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

absl::StatusOr<ExecutionOutput> NpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

absl::StatusOr<ScopedShapedBuffer> NpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

absl::StatusOr<ExecutionOutput> NpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "NpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  NPU_LOG_FORMAT_PRINT(DEBUG, "=== ExecuteAsyncOnStreamImpl called for module: %s ===", module_name_.c_str());
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> npu_lock(
      std::in_place_index_t<0>{}, &GetNpuMutex(executor));
  const NpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/program_shape_.result(),
                         memory_allocator, device_ordinal,
                         executor->device_ordinal());

  TF_ASSIGN_OR_RETURN(
      NpuBufferAllocations buffer_allocations,
      GenerateBufferAllocations(arguments, globals, memory_allocator,
                                device_ordinal, run_options->stream()));
  VLOG(3) << buffer_allocations.ToString();
  absl::Span<const BufferAllocation* const> allocations = GetAllocations();

  const bool is_entire_tuple_contents_aliased = [&] {
    for (auto& p : result.MutableResult()->buffers().leaves()) {
      if (!output_info_.contains(p.first)) {
        continue;
      }
      // handle alias
      const OutputInfo& output_info = output_info_.at(p.first);
      if (!output_info.alias_config.has_value()) {
        return false;
      }
    }
    return true;
  }();

  std::set<se::DeviceMemoryBase> buffers_in_result;

  NPU_LOG_FORMAT_PRINT(DEBUG, "ExecuteAsyncOnStreamImpl: output_info_.size()=%d",
                       output_info_.size());

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceMemoryBase& result_buffer = p.second;

    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Processing output: index %s, allocation_index=%d, has_alias_config=%d",
        index.ToString().c_str(), output_info.allocation_index,
        output_info.alias_config.has_value());
    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    bool needs_passthrough =
        output_info.passthrough && !output_info.alias_config;
    if (output_info.alias_config || needs_passthrough) {
      int param_number = output_info.alias_config.has_value()
                             ? output_info.alias_config->parameter_number
                             : allocation->parameter_number();
      ShapeIndex param_shape_index =
          output_info.alias_config.has_value()
              ? output_info.alias_config->parameter_index
              : allocation->param_shape_index();
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Processing: index %s, allocation_index=%d, "
                           "passthrough=%d, has_alias=%d, param_number=%d",
                           index.ToString().c_str(),
                           output_info.allocation_index,
                           output_info.passthrough,
                           output_info.alias_config.has_value(), param_number);
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Allocation: is_entry_param=%d, param_number=%d, "
                           "param_shape_index={%s}",
                           allocation->is_entry_computation_parameter(),
                           allocation->parameter_number(),
                           allocation->param_shape_index().ToString().c_str());
      MaybeOwningDeviceMemory* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceMemory* {
        // ScopedBuffer is never an owned buffer.
        if (std::holds_alternative<absl::Span<const ShapedBuffer* const>>(
                arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              std::get<absl::Span<ExecutionInput>>(arguments);
          ExecutionInput& input = unowned_execution_input[param_number];
          return input.MutableBuffer(param_shape_index);
        }
      }();
      NPU_LOG_FORMAT_PRINT(
          DEBUG, "maybe_owning_memory=%p, has_ownership=%d",
          (void*)maybe_owning_memory,
          maybe_owning_memory ? maybe_owning_memory->HasOwnership() : 0);
      if (output_info.alias_config && output_info.alias_config->must_alias() &&
          maybe_owning_memory && !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (maybe_owning_memory && maybe_owning_memory->HasOwnership()) {
        std::optional<se::OwningDeviceMemory> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Passthrough: allocation_index=%d, "
                             "param_number=%d, transferred ownership",
                             output_info.allocation_index, param_number);
        // The caller is giving us the
        // input buffer, but in case of error from the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vector, which will be fed to the ExecutionOutput, which will use
        // the indices to drop the addresses from its own ScopedShapedBuffer
        // result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else if (output_info.passthrough) {
        // For passthrough computation, if the caller doesn't donate ownership,
        // we need to allocate a new buffer and copy the data.
        // We cannot use the external buffer directly because ExecutionOutput's
        // ScopedShapedBuffer will try to release it when ConsumeResult() is
        // called (which clears aliased_indices_).
        se::DeviceMemoryBase argument_buffer =
            maybe_owning_memory->AsDeviceMemoryBase();
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(program_shape_.result(), index));
        if (allocation_size == 0) {
          result_buffer = se::DeviceMemoryBase();
        } else {
          absl::StatusOr<se::OwningDeviceMemory> allocated_buffer =
              memory_allocator->Allocate(device_ordinal, allocation_size,
                                         /*retry_on_failure=*/true,
                                         /*memory_space=*/allocation->color());
          if (!allocated_buffer.ok()) {
            return allocated_buffer.status();
          }
          result_buffer = allocated_buffer->Release();
          // Copy data from argument buffer to result buffer
          TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
              &result_buffer, argument_buffer, allocation_size));
        }
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Passthrough: allocation_index=%d, "
                             "param_number=%d, copied to new buffer %p",
                             output_info.allocation_index, param_number,
                             (void*)result_buffer.opaque());
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(program_shape_.result(), index)
                      .IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(program_shape_.result(), index));
        absl::StatusOr<se::OwningDeviceMemory> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return allocated_buffer.status();
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  TF_RETURN_IF_ERROR(ExecuteThunks(buffer_allocations, run_options));

  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, GetAllocations()));

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    for (int i = 0; i < args->size(); ++i) {
      ExecutionInput& argument = (*args)[i];
      for (auto& index_buffer : *argument.MutableBuffers()) {
        NPU_LOG_FORMAT_PRINT(
            DEBUG,
            "MarkToBeReleasedArguments: arg %d, index %s, has_ownership=%d", i,
            index_buffer.first.ToString().c_str(),
            index_buffer.second.HasOwnership());
      }
    }
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

absl::Status NpuExecutable::ExecuteThunks(
    const NpuBufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("NpuExecutable::ExecuteThunks",
                                        {{"module_name", module_name_}});
  });

  if (VLOG_IS_ON(5)) {
    se::StreamExecutor* executor = run_options->stream()->parent();
    // Debug code to compare current allocation's address with previous run's
    // address, and report the allocation info if memory addressed changed.
    // Useful for identify in user's model if it is command buffer perf friendly
    // (no command buffer update cost).
    absl::MutexLock lock(module_handle_mutex_);
    if (module_allocations_.find(executor) == module_allocations_.end()) {
      std::vector<se::DeviceMemoryBase> allocs_addr;
      allocs_addr.reserve(buffer_allocations.size());
      for (int i = 0; i < buffer_allocations.size(); i++) {
        allocs_addr.push_back(buffer_allocations.GetDeviceAddress(i));
      }
      module_allocations_[executor] = std::move(allocs_addr);
    } else {
      for (int i = 0; i < buffer_allocations.size(); i++) {
        if (module_allocations_[executor][i].IsSameAs(
                buffer_allocations.GetDeviceAddress(i))) {
          continue;
        }
        module_allocations_[executor][i] =
            buffer_allocations.GetDeviceAddress(i);
        VLOG(5) << "Gpu address changed for module " << module_name_;
      }
    }
  }

  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  ModuleIdentifier unique_id = has_module() ? module().unique_id() : -1;
  NpuThunk::ExecutableSource executable_source = {
      text_,
      binary_,
  };

  TF_RETURN_IF_ERROR(ExecuteThunksImpl(
      has_module() ? &module_config().debug_options() : nullptr, module_name_,
      unique_id, *thunks_, executable_source, run_options, buffer_allocations,
      block_host_until_done, execution_stream_ids_));
  return absl::OkStatus();
}

int64_t NpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (const BufferAllocation* allocation : GetAllocations()) {
    if (allocation->is_constant()) {
      size += allocation->size();
    }
  }
  return size;
}

absl::StatusOr<absl::flat_hash_map<ShapeIndex, NpuExecutable::OutputInfo>>
NpuGetOutputInfo(const HloModule& hlo_module,
                 const BufferAssignment& assignment) {
  const HloInstruction* root =
      hlo_module.entry_computation()->root_instruction();

  InstructionValueSet root_value_set =
      assignment.dataflow_analysis().GetInstructionValueSet(root);
  if (root_value_set.IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  using OutputInfoMap =
      absl::flat_hash_map<ShapeIndex, NpuExecutable::OutputInfo>;
  OutputInfoMap output;

  int alias_count = 0;
  hlo_module.input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        alias_count++;
        NPU_LOG_FORMAT_PRINT(
            DEBUG,
            "NpuGetOutputInfo: module %s, alias: output {%s} -> param %d index "
            "{%s}",
            hlo_module.name().c_str(), output_index.ToString().c_str(),
            alias.parameter_number, alias.parameter_index.ToString().c_str());
      });
  NPU_LOG_FORMAT_PRINT(DEBUG, "NpuGetOutputInfo: module %s, total %d aliases",
                       hlo_module.name().c_str(), alias_count);

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> absl::Status {
        const auto& sources = root_value_set.element(index);
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        NpuExecutable::OutputInfo& info = output[index];
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();
        const BufferAllocation& allocation =
            assignment.GetAllocation(slice.index());
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter &&
                           allocation.is_entry_computation_parameter();
        NPU_LOG_FORMAT_PRINT(
            DEBUG,
            "NpuGetOutputInfo: output index %s, src_hlo opcode=%d (%s), "
            "is_entry_param=%d, passthrough=%d",
            index.ToString().c_str(), static_cast<int>(src_hlo->opcode()),
            HloOpcodeString(src_hlo->opcode()).data(),
            allocation.is_entry_computation_parameter(), info.passthrough);

        auto alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);
        if (alias_config.has_value()) {
          NPU_LOG_FORMAT_PRINT(DEBUG,
                               "NpuGetOutputInfo: output index %s -> param %d, "
                               "shape index {%s}, kind %d",
                               index.ToString().c_str(),
                               alias_config->parameter_number,
                               alias_config->parameter_index.ToString().c_str(),
                               static_cast<int>(alias_config->kind));
        } else {
          NPU_LOG_FORMAT_PRINT(
              DEBUG, "NpuGetOutputInfo: output index %s has no alias_config",
              index.ToString().c_str());
        }
        output[index].alias_config = alias_config;

        return absl::OkStatus();
      }));
  return output;
}
}  // namespace xla
