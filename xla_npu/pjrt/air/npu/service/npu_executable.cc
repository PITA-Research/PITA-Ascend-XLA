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

#include "xla/xla_npu/pjrt/air/npu/service/npu_executable.h"

#include <stdint.h>

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/setround.h"
#include "tsl/profiler/lib/traceme.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla {

absl::StatusOr<std::unique_ptr<NpuExecutable>> NpuExecutable::Create(
    std::unique_ptr<NpuBufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    const GraphData& graph_data) {
  std::unique_ptr<NpuExecutable> executable(new NpuExecutable(
      std::move(hlo_module), std::move(hlo_profile_printer_data),
      std::move(hlo_profile_index_map), std::move(assignment), graph_data));
  return executable;
}

NpuExecutable::NpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    std::unique_ptr<NpuBufferAssignment> assignment,
    const GraphData& graph_data)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      assignment_(std::move(assignment)),
      graph_data_(graph_data) {
  if (assignment_) {
    alloc_ptrs_.reserve(assignment_->Allocations().size());
    for (const BufferAllocation& alloc : assignment_->Allocations()) {
      alloc_ptrs_.push_back(&alloc);
    }
  }

  if (has_module()) {
    *shared_module()->metadata() = HloModuleMetadata(tsl::Env::Default());
  }
}

NpuExecutable::~NpuExecutable() {}

static absl::StatusOr<MaybeOwningDeviceMemory> MemoryForAllocation(
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal) {
  VLOG(3) << allocation.ToString();
  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory out,  // Ordinary memory allocation, whose life cycle is managed by XLA
      memory_allocator->Allocate(device_ordinal, buffer_size));
  VLOG(3) << "buffer allocated " << buffer_size << " bytes [" << out->opaque()
          << "]";

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->opaque(), buffer_size);
  return MaybeOwningDeviceMemory{std::move(out)};
}

absl::StatusOr<std::vector<MaybeOwningDeviceMemory>>
NpuExecutable::CreateBufferTable(se::DeviceMemoryAllocator* memory_allocator,
                                 int device_ordinal,
                                 absl::Span<ExecutionInput const> arguments) {
  std::vector<MaybeOwningDeviceMemory> buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  return std::move(buffers);
}

static int32_t GetDeviceOrdinal(const ExecutableRunOptions* run_options) {
  if (!run_options) {
    return 0;
  }
  if (run_options->device_ordinal() != -1) {
    return run_options->device_ordinal();
  }
  return run_options->stream()->parent()->device_ordinal();
}

absl::StatusOr<ExecutionOutput> NpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory> buffers,
    absl::Span<ExecutionInput> arguments) {}

absl::StatusOr<ExecutionOutput> NpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return absl::UnimplementedError(
      "NpuExecutable::ExecuteAsyncOnStream is not supported.");
}

int64_t NpuExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

int64_t NpuExecutable::SizeOfGeneratedCodeInBytes() const { return 0; }

void NpuExecutable::Finalize() {
  if (has_module()) {
    shared_module()->Finalize();
  }
  assignment_->Finalize();
}

}  // namespace xla
