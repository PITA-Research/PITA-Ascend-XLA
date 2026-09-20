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
==============================================================================*/

#include <map>
#include <memory>
#include <numeric>
#include <functional>
#include <llvm/IR/DataLayout.h>
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"

#include "xla/xla_npu/pjrt/air/npu/npu_client.h"

#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo_module_util.h"
#include "xla/xla_npu/pjrt/air/npu_tool.h"
#include "xla/xla_npu/pjrt/air/npu/npu_device.h"
#include "xla/xla_npu/pjrt/air/npu/raw_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/abstract_npu_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/ge/graph_data.h"
#include "xla/xla_npu/pjrt/air/npu/service/npu_executable.h"
#include "xla/xla_npu/pjrt/air/host_memory_spaces.h"
#include "xla/xla_npu/dependency/header/metadef/graph/graph.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "xla/xla_npu/dependency/header/metadef/graph/ascend_string.h"
#include "xla/xla_npu/dependency/header/metadef/graph/operator_factory.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/pjrt/air/mair/test/test_mair_gen.h"
#include "xla/xla_npu/pjrt/air/mair/stablehlo2mair_converter.h"

namespace xla {

struct BufferInfo {
  tsl::AsyncValueRef<NpuDeviceMemory> buffer;
  bool owns_buffer;
  size_t buffer_size;
};

struct BufferAlloc {
  // All data members should have the same size.
  absl::InlinedVector<tsl::AsyncValueRef<NpuDeviceMemory>, 4> buffers;
  absl::InlinedVector<size_t, 4> allocation_sizes;

  void Allocate(const NpuDeviceMemory::Allocator& allocator) {
    for (int i = 0; i < buffers.size(); ++i) {
      auto status = NpuDeviceMemory::AllocateInto(
          allocation_sizes[i], buffers[i].AsPtr(), allocator);
      if (!status.ok()) {
        buffers[i].SetError(status);
        return;
      }
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(buffers[i]->untyped_data(),
                                          allocation_sizes[i]);
    }
  }
};

// Returns the CPU devices from the given PjRtCpuDevices.
// Precondition: `devices` doesn't contain nullptr.
static std::vector<NpuTopology::NpuDevice> GetDevices(
    absl::Span<const std::unique_ptr<PjRtNpuDevice>> devices) {
  std::vector<NpuTopology::NpuDevice> npu_devices;
  npu_devices.reserve(devices.size());
  for (const auto& device : devices) {
    npu_devices.push_back(NpuTopology::NpuDevice{
        device->process_index(), device->local_hardware_id().value()});
  }
  return npu_devices;
}

static tsl::ThreadOptions GetThreadOptions() {
  tsl::ThreadOptions thread_options;
  // On Mac OS the default stack size is 512KiB, which is too small for some
  // BLAS and LAPACK functions (https://github.com/google/jax/issues/20428).
  // On Linux we also observed that 2MB wasn't enough to run some OpenBLAS
  // functions.
  thread_options.stack_size = 8 * 1024 * 1024;
  return thread_options;
}

// The following few helpers are adapted from XLA:NPU to create a buffer table
// and assemble the buffer pointers in order to call into NpuExecutable.
static absl::StatusOr<BufferInfo> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc) {
  BufferInfo buffer_info;
  if (allocation.is_entry_computation_parameter()) {
    bool can_donate = false;
    TrackedNpuDeviceBuffer* arg = nullptr;
    size_t buffer_size;
    tsl::AsyncValuePtr<NpuDeviceMemory> out;
    std::tie(can_donate, arg) = arguments[allocation.parameter_number()];
    out = arg->buffer().AsPtr();
    buffer_size = arg->BufferSize();
    CHECK_EQ(allocation.size(), buffer_size)
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    buffer_info.buffer = out.CopyRef();
    buffer_info.owns_buffer = true;
    buffer_info.buffer_size = buffer_size;
    return buffer_info;
  }

  // Output and temporary buffer.
  auto out = NpuDeviceMemory::CreateDelayedMemory();

  buffer_alloc.buffers.push_back(out);
  buffer_alloc.allocation_sizes.push_back(allocation.size());

  buffer_info.buffer = std::move(out);
  buffer_info.owns_buffer = true;
  buffer_info.buffer_size = allocation.size();
  return buffer_info;
}

static absl::StatusOr<std::vector<BufferInfo>> CreateBufferTable(
    const NpuBufferAssignment& assignment,
    absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc) {
  std::vector<BufferInfo> buffer_table(assignment.Allocations().size());
  for (BufferAllocation::Index i = 0; i < buffer_table.size(); ++i) {
    const BufferAllocation& allocation = assignment.GetAllocation(i);
    TF_ASSIGN_OR_RETURN(
        buffer_table[i],
        MemoryForAllocation(allocation, arguments, buffer_alloc));
  }
  return std::move(buffer_table);
}

static absl::InlinedVector<BufferInfo, 4> CreateResultBufferInfo(
    absl::Span<const BufferAllocation::Index> buffer_indices,
    absl::Span<const BufferInfo> buffer_table) {
  absl::InlinedVector<BufferInfo, 4> output_buffer_info;
  output_buffer_info.reserve(buffer_indices.size());
  for (int i = 0; i < buffer_indices.size(); ++i) {
    output_buffer_info.push_back(buffer_table[buffer_indices[i]]);
  }
  return output_buffer_info;
}

PjRtNpuClient::PjRtNpuClient(
    int process_index, std::vector<std::unique_ptr<PjRtNpuDevice>> devices)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      transpose_cache_(1024),
      topology_(platform_id(), platform_name(), platform_version(),
                GetDevices(devices), {}),
      pjrt_client_thread_pool_(new tsl::thread::ThreadPool(
          tsl::Env::Default(), GetThreadOptions(), "XLAPjRtNpuClient", 1)),
      async_work_runner_(
          MakeThreadPoolAsyncWorkRunner(pjrt_client_thread_pool_.get())) {
  for (const std::unique_ptr<PjRtNpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.insert({device->global_device_id(), device.get()}).second)
        << "Duplicate device id: " << device->global_device_id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id().value();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }

  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    auto* const device = addressable_devices_[idx];
    CHECK(device != nullptr) << idx;
    auto* npu_device = tsl::down_cast<PjRtNpuDevice*>(device);

    // Use the device id to construct a globally unique memory space id.
    const int id = device->id();

    // The first attached memory space is returned as the default by
    // PjRtCpuDevice, so attach the device memory space first.
    auto npu_device_memory_space =
        std::make_unique<NpuDeviceMemorySpace>(id * 3, device);
    npu_device->AttachMemorySpace(npu_device_memory_space.get());
    memory_spaces_.push_back(npu_device_memory_space.get());
    owned_memory_spaces_.push_back(std::move(npu_device_memory_space));
  }
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> PjRtNpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_device_id %d",
                         local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> PjRtNpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<int64_t> PjRtNpuClient::GetOnDeviceBytesCount(
    PjRtMemorySpace* memory_space, const Shape& shape) const {
  auto pointer_size = llvm::DataLayout(DataLayout())
                          .getPointerSize(0 /* default address space */);
  if (shape.IsTuple() || shape.is_static()) {
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  }
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
  return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
PjRtNpuClient::AllocateRawBuffer(PjRtMemorySpace* memory_space,
                                 size_t on_device_bytes_count,
                                 bool retry_on_oom,
                                 tsl::AsyncValueRef<bool> allocate_after) {
  return xla::NpuRawBuffer::Allocate(memory_space, on_device_bytes_count);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtNpuClient::LinearizeHostBufferInto(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    const xla::Shape& device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
  return tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())
      ->CopyFromHostBuffer(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), device_shape,
          async_work_runner(), &transpose_mu_, &transpose_cache_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtNpuClient::DefineBuffer(
    const Shape& on_device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
        definition_device_events,
    bool raw_buffer_is_mutable) {
  absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events;
  for (auto& ev : definition_device_events) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "PjRtNpuClient::DefineBuffer event: %p",
                         ev.get());
    definition_events.push_back(
        tsl::down_cast<NpuTrackedDeviceEvent*>(ev.get())->event());
  }
  return std::unique_ptr<PjRtBuffer>(std::make_unique<CommonPjRtBufferImpl>(
      on_device_shape,
      std::make_unique<TrackedNpuDeviceBuffer>(
          /*owns_buffers=*/raw_buffer_is_mutable,
          tsl::down_cast<NpuRawBuffer*>(raw_buffer.get())->buffer(),
          ShapeUtil::ByteSizeOf(on_device_shape), std::move(definition_events)),
      raw_buffer->memory_space()));
}

absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                         tsl::RCReference<PjRtDeviceEvent>>>
PjRtNpuClient::CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                                        absl::string_view debug_info) {
  auto definition_event_promise = tsl::MakeIndirectAsyncValue();
  auto definition_event = tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::AsyncValueRef<NpuEvent>(definition_event_promise));
  return std::make_pair(tsl::MakeRef<NpuTrackedDeviceEventPromise>(
                            std::move(definition_event_promise)),
                        std::move(definition_event));
}

static void TestGraph(ge::GraphPtr& graph) {
  graph = std::make_shared<ge::Graph>("XLAGraph");
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options, GraphData& graph_data,
    const AotCompilationOptions* absl_nullable aot_options) {
  auto input_options = options;
  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }
  ExecutableBuildOptions& build_options = options.executable_build_options;

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layout_pointers,
                         &execution_options, execution_options.num_replicas(),
                         1, aot_options));

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));

  // Create device assignment(Do not delete!!! step into PjRtNpuExecutable::Execute requires it)
  int num_replicas = 1;
  int num_partitions = 1;
  TF_ASSIGN_OR_RETURN(auto default_device_assignment,
                      GetDefaultDeviceAssignment(num_replicas, num_partitions));
  auto device_assignment =
      std::make_shared<DeviceAssignment>(default_device_assignment);
  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        PjRtGlobalDeviceId device_id((*device_assignment)(replica, partition));
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (!addressable_devices.empty() && build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id().value());
    }
  }

  /* set input data */
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    Shape input_parameter = program_shape.parameters(i);
    ge::Shape input_data_shape =
        NpuTool::ConvertXlaShape2GeShape(input_parameter);
    ge::DataType input_data_type =
        NpuTool::ConvertXlaType2GeType(input_parameter.element_type());
    graph_data.inputs_shapes.emplace_back(input_data_shape);
    graph_data.input_dtypes.emplace_back(input_data_type);
  }

  /* create buffer assignment(Do not delete!!! alloc output buffer depends on it) */
  auto buffer_assignment = std::make_unique<NpuBufferAssignment>();

  TF_ASSIGN_OR_RETURN(
      auto npu_executable,
      NpuExecutable::Create(std::move(buffer_assignment), std::move(hlo_module),
                            nullptr, nullptr, graph_data));

  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices;
  result_buffer_indices.push_back(0);
  auto executable = std::make_unique<PjRtNpuExecutable>(
      num_replicas, num_partitions, device_assignment,
      options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(npu_executable), std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this, nullptr);

  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<DeviceAssignment> PjRtNpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions * num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, num_partitions);
    for (int i = 0; i < num_replicas; ++i) {
      for (int j = 0; j < num_partitions; ++j) {
        assignment(i, j) =
            addressable_devices().at(i * num_partitions + j)->id();
      }
    }
    return assignment;
  }
  NPU_LOG_FORMAT_PRINT(ERROR,
                       "Exception PjRtNpuClient::GetDefaultDeviceAssignment "
                       "num_replicas: %d, num_partitions: %d",
                       num_replicas, num_partitions);
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  tsl::profiler::TraceMe traceme(
      "PjRtNpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions "
      "options)");
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  NPU_LOG_FORMAT_PRINT(DEBUG, "stablehlo module: %s", os.str().c_str());

  mlir::ModuleOp mair_module = module.clone();
  xla::mair::Stablehlo2MairConverter(mair_module);

  std::string mair_result;
  llvm::raw_string_ostream mair_os(mair_result);
  mair_module.print(mair_os);
  NPU_LOG_FORMAT_PRINT(DEBUG, "mair module: %s", mair_os.str().c_str());

  std::map<ge::AscendString, ge::AscendString> ge_options = {
      {"ge.exec.deviceId", "0"},
  };

  if (!ge_initialized_) {
    if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
      return absl::InternalError("Failed to initialize GE.");
    } else {
      (void)ge_options.emplace(ge::AscendString("ge.session_device_id"),
                               ge::AscendString("0"));
      global_ge_session_ = std::make_unique<ge::Session>(ge_options);
      if (global_ge_session_ == nullptr) {
        return absl::InternalError("Failed to create GE session.");
      }
      ge_initialized_ = true;
      is_initialized = 1;
    }
  }

  auto graph_data = GraphData();
  graph_data.graph = std::make_shared<ge::Graph>("XLAGraph");
  mlir::air::GraphGenTester::Convert(mair_module, graph_data.graph);
  graph_data.id = global_graph_id++;

  if (global_ge_session_->AddGraph(graph_data.id, *graph_data.graph,
                                   ge_options) != ge::SUCCESS) {
    return absl::InternalError("Failed to AddGraph.");
  }

  //Build HLO modules and pass JAX verification
  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options));

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module));
  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [&arg_layout_modes, &out_layout_modes,
                          &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces, &LayoutUtil::GetWithDefaultLayout);
  };

  // This call will update result_layout in options.executable_build_options.
  TF_ASSIGN_OR_RETURN(
      auto arg_layouts_and_pointers,
      LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                       arg_memory_spaces, out_memory_spaces,
                       &LayoutUtil::GetWithDefaultLayout,
                       options.executable_build_options));
  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options, graph_data);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtNpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  return absl::UnimplementedError(
      "CompileAndLoad with XlaComputation is not supported.");
}

PjRtNpuExecutable::PjRtNpuExecutable(
    int num_replicas, int num_partitions,
    std::shared_ptr<DeviceAssignment> device_assignment,
    bool parameter_is_tupled_arguments, CompileOptions compile_options,
    std::unique_ptr<Executable> npu_executable,
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, PjRtNpuClient* client,
    std::unique_ptr<HloModule> unoptimized_hlo_module)
    : client_(client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      compile_options_(std::move(compile_options)),
      npu_executable_(std::move(npu_executable)),
      result_buffer_indices_(std::move(result_buffer_indices)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)),
      unoptimized_hlo_module_(std::move(unoptimized_hlo_module)) {}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtNpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  RunId run_id(options.launch_id);
  NPU_LOG_FORMAT_PRINT(DEBUG, "Executing computation %s",
                       std::string(name()).c_str());

  // only one device
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(1);
  if (returned_futures.has_value()) {
    returned_futures->resize(1);
  }

  // Fast-path if there is only one device — run the computation on the
  // current thread.
  auto statusor = ExecuteHelper(
      argument_handles[0], num_replicas_, num_partitions_, run_id, options,
      /*last_collective_launch_event=*/{}, returned_futures.has_value());
  if (!statusor.ok()) {
    return std::move(statusor).status();
  }

  wrapped_results[0] = std::move(statusor->buffers);
  if (returned_futures.has_value()) {
    (*returned_futures)[0] = std::move(*statusor->future);
  }

  return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return absl::UnimplementedError(
      "PjRtNpuExecutable::ExecuteSharded is not supported.");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtNpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return absl::UnimplementedError(
      "PjRtNpuExecutable::ExecutePortable is not supported.");
}

absl::StatusOr<PjRtLoadedExecutable::Result> PjRtNpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    PjRtNpuClient::CollectiveLaunchEvent last_collective_launch_event,
    bool fill_future, PjRtNpuDevice* device) const {
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    // only device 0
    const int64_t device_id = 0;
    PjRtGlobalDeviceId global_device_id(device_id);
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(global_device_id));
    device = tsl::down_cast<PjRtNpuDevice*>(pjrt_device);
  }

  CHECK_EQ(device->process_index(), client_->process_index());

  // Handle inputs.
  // `execute_event` indicates whether npu computation is complete and whether
  // there was an error.
  auto execute_event = tsl::MakeConstructedAsyncValueRef<NpuEvent>();
  // Asynchronously call needed
  MarkEventReadyOnExit ready_on_exit(execute_event);
  auto execute_usage_event = tsl::MakeRef<NpuTrackedDeviceEvent>(execute_event);

  absl::InlinedVector<std::pair<bool, TrackedNpuDeviceBuffer*>, 4>
      tracked_buffers;
  tracked_buffers.reserve(argument_handles.size());

  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as the
  // single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size());

  for (int i = 0; i < argument_handles.size(); ++i) {
    PjRtBuffer* handle = argument_handles[i];
    auto* npu_buffer = tsl::down_cast<CommonPjRtBuffer*>(handle);
    if (npu_buffer->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, npu_buffer->device()->DebugString(),
          device->DebugString());
    }

    TrackedNpuDeviceBuffer* tracked_buffer;
    auto get_buffer = [&](int i) -> absl::Status {
      CommonPjRtBuffer::ScopedHold usage_transaction =
          npu_buffer->GetBufferWithHold(CommonPjRtBuffer::ScopedHold::kUsage);
      if (!usage_transaction.ok()) {
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      }
      tracked_buffer = tensorflow::down_cast<TrackedNpuDeviceBuffer*>(
          usage_transaction.buffer());
      usage_transaction.ConvertUsageHold(execute_usage_event);
      tracked_buffers.emplace_back(/*can_donate=*/false, tracked_buffer);

     return absl::OkStatus();
    };

    TF_RETURN_IF_ERROR(get_buffer(i));

    // Definition events are never modified after buffer construction. If they
    // are available and have no error, they can be skipped in input deps.
    // In contrast, already known errors in the input are taken as deps so that
    // they can poison output buffers.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable() || definition_event.IsError()) {
      input_deps.push_back(definition_event.CopyRCRef());
    }
  }

  for (auto& tracked_buffer : tracked_buffers) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "argument handle buffer addr=%p, size=%zu",
                         tracked_buffer.second->buffer().get().untyped_data(),
                         tracked_buffer.second->buffer().get().size_bytes());
  }

  if (!input_deps.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "input is not ready");
  }

  auto npu_executable = tsl::down_pointer_cast<NpuExecutable>(npu_executable_);
  ge::Session* ge_session = client_->GetGlobalGeSession();
  const GraphData& graph_data = npu_executable->GetGraphData();

  // transfer to ge tensor
  std::vector<ge::Tensor> inputs;
  for (int i = 0; i < tracked_buffers.size(); ++i) {
    ge::TensorDesc desc;
    const ge::Shape& input_data_shape = graph_data.inputs_shapes[i];
    desc.SetOriginShape(input_data_shape);
    desc.SetOriginFormat(ge::FORMAT_ND);
    desc.SetPlacement(ge::Placement::kPlacementDevice);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetDataType(graph_data.input_dtypes[i]);
    ge::Tensor input;
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t* data) {};
    input.ResetData(
        (uint8_t*)tracked_buffers[i].second->buffer().get().untyped_data(),
        tracked_buffers[i].second->buffer().get().size_bytes(), kDoNothing);

    const auto& dimSize = input_data_shape.GetDims().size();
    input.SetShapeDimNum(dimSize);
    for (int j = 0; j < dimSize; ++j) {
      input.SetShapeDim(j, input_data_shape.GetDim(j));
    }
    input.SetTensorDesc(desc);
    NPU_LOG_FORMAT_PRINT(
        DEBUG, "input tensor %d data addr=%p, data size=%zu", i,
        tracked_buffers[i].second->buffer().get().untyped_data(),
        tracked_buffers[i].second->buffer().get().size_bytes());
    inputs.emplace_back(input);
  }

  /* Run the GE graph, use the output memory prepared above, refer to the usage of cpu and use result_buffers_info */
  std::vector<ge::Tensor> outputs;
  ge::Status ge_status = ge_session->RunGraph(graph_data.id, inputs, outputs);
  if (ge_status != ge::SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "run graph id ", graph_data.id, " failed, ge status is ", ge_status));
  }

  /* debug */
  auto output_data_ptr = outputs[0].GetData();
  auto output_data_size = outputs[0].GetSize();
  auto output_data_desc = outputs[0].GetTensorDesc();
  auto output_data_shape = output_data_desc.GetShape();
  auto output_data_type = output_data_desc.GetDataType();

  auto buffer_assignment = std::make_unique<NpuBufferAssignment>();
  int64_t size =
      output_data_shape.GetShapeSize() * GetSizeByDataType(output_data_type);
  auto allocation =
      buffer_assignment->NewEmptyAllocation(size, BufferValue::Color(0));
  allocation->set_entry_computation_parameter(0, ShapeIndex{0}, false);

  /* prepare output buffer */
  // `buffer_alloc` and `buffer_alloc_and_copy` are used to do real memory allocation and copy work.
  absl::InlinedVector<std::pair<bool, TrackedNpuDeviceBuffer*>, 4>
      output_tracked_buffers;
  output_tracked_buffers.reserve(outputs.size());
  void* output_memory_ptr = NULL;
  // Ascend aclrtMalloc/aclrtMemcpy do not support zero-size operations.
  if (output_data_size == 0) {
    output_memory_ptr = nullptr;
  } else {
    auto acl_ret = dep::aclrtMalloc(&output_memory_ptr, output_data_size,
                                    ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("malloc host memory failed, error code: ", acl_ret));
    }
    acl_ret =
        dep::aclrtMemcpy(output_memory_ptr, output_data_size, output_data_ptr,
                          output_data_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
      return absl::InternalError(absl::StrCat(
          "memcpy from host to device failed, error code: ", acl_ret));
    }
  }
  auto deleter = [output_memory_ptr]() { dep::aclrtFree(output_memory_ptr); };
  tsl::AsyncValueRef<NpuDeviceMemory> output_device_memory =
      NpuDeviceMemory::CreateForeignMemory(output_memory_ptr, output_data_size,
                                           deleter);
  auto output_tracked_buffer = TrackedNpuDeviceBuffer(
      /*owns_buffers=*/true, std::move(output_device_memory), output_data_size,
      absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4>{
          tsl::AsyncValueRef<NpuEvent>()});
  output_tracked_buffers.emplace_back(/*can_donate=*/false,
                                      &output_tracked_buffer);

  BufferAlloc buffer_alloc;  // not used
  // BufferAllocAndCopy buffer_alloc_and_copy;

  TF_ASSIGN_OR_RETURN(
      std::vector<BufferInfo> buffer_table,
      CreateBufferTable(
          *buffer_assignment,
          /* npu_executable->constants(), */ output_tracked_buffers,
          buffer_alloc));
  auto result_buffers_info =
      CreateResultBufferInfo(result_buffer_indices_, buffer_table);

  // Create output buffers.
  const Shape& result_shape =
      NpuTool::ConvertGeShape2XlaShape(output_data_shape, output_data_type);
  std::vector<std::unique_ptr<PjRtBuffer>> res;
  auto* memory_space = *device->default_memory_space();
  CHECK_EQ(result_buffers_info.size(), 1);

  // Program execution writes to output buffers so it's a definition event.
  auto tracked_device_buffer = std::make_unique<TrackedNpuDeviceBuffer>(
      result_buffers_info[0].owns_buffer,
      std::move(result_buffers_info[0].buffer),
      result_buffers_info[0].buffer_size,
      /*definition_event=*/execute_event);
  auto output_buffer = std::make_unique<CommonPjRtBufferImpl>(
      result_shape, std::move(tracked_device_buffer),
      *device->default_memory_space());
  res.push_back(std::move(output_buffer));

  if (fill_future) {
    auto [promise, future] = Future<>::MakePromise();
    promise.Set(absl::OkStatus());
    return Result({std::move(future), /*buffers=*/std::move(res)});
  }

  return Result({/*future=*/std::nullopt, std::move(res)});
}

void PjRtNpuExecutable::Delete() {}

bool PjRtNpuExecutable::IsDeleted() const { return false; }

absl::StatusOr<std::string> PjRtNpuExecutable::SerializeExecutable() const {
  return absl::UnimplementedError(
      "PjRtNpuExecutable::SerializeExecutable is not supported.");
}

absl::StatusOr<CompiledMemoryStats> PjRtNpuExecutable::GetCompiledMemoryStats()
    const {
  return absl::UnimplementedError(
      "PjRtNpuExecutable::GetCompiledMemoryStats is not supported.");
}

}  // namespace xla
