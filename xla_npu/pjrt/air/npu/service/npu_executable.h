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

#ifndef XLA_SERVICE_NPU_NPU_EXECUTABLE_H_
#define XLA_SERVICE_NPU_NPU_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h"
#include "xla/xla_npu/pjrt/air/npu/service/npu_executable.h"
#include "xla/xla_npu/pjrt/air/npu/ge/graph_data.h"

namespace xla {

// CPU-targeting implementation of the XLA Executable interface.
// Wraps a JIT-ed object that can be executed "on device". We JIT for the host
// architecture, so JIT-ed code and host code share the same ABI.
class NpuExecutable : public Executable {
 public:
  // Creates a NpuExecutable from a thunk sequence.
  static absl::StatusOr<std::unique_ptr<NpuExecutable>> Create(
      std::unique_ptr<NpuBufferAssignment> assignment,
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
      const GraphData& graph_data);

  ~NpuExecutable() override;

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  GraphData GetGraphData() const { return graph_data_; }

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

  void set_ir_module_string(const std::string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  const std::string& module_name() const { return module_name_; }

  static int64_t ShapeSizeBytes(const Shape& shape);

  bool has_xnn_fusions() const { return has_xnn_fusions_; }
  bool has_ynn_fusions() const { return has_ynn_fusions_; }

  const NpuBufferAssignment& buffer_assignment() const { return *assignment_; }

  int64_t SizeOfGeneratedCodeInBytes() const override;

  absl::Span<const BufferAllocation * absl_nonnull const> GetAllocations()
      const override {
    return alloc_ptrs_;
  }

  // Finalize construction of the NpuExecutable and finalize all internal data
  // structures that might have been used at compile time.
  void Finalize();

 private:
  absl::StatusOr<std::vector<MaybeOwningDeviceMemory>> CreateBufferTable(
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
      absl::Span<ExecutionInput const> arguments);

  absl::StatusOr<ExecutionOutput> CreateResultShapedBuffer(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<MaybeOwningDeviceMemory> buffers,
      absl::Span<ExecutionInput> arguments);

  // Buffer assignment for the buffers we need to allocate.
  std::shared_ptr<NpuBufferAssignment> assignment_;
  std::vector<const BufferAllocation*> alloc_ptrs_;

  std::string ir_module_string_;

  // Unique identifier.
  std::string module_name_;

  bool has_xnn_fusions_ = false;

  // Whether the thunk executor contains any YNN fusion thunks.
  bool has_ynn_fusions_ = false;

  // TargetMachineOptionsProto target_machine_options_;

  // Entry function name for the computation.
  std::string entry_function_name_;

  GraphData graph_data_;

  NpuExecutable(std::unique_ptr<HloModule> hlo_module,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
                std::unique_ptr<NpuBufferAssignment> assignment,
                const GraphData& graph_data);
  NpuExecutable(const NpuExecutable&) = delete;
  NpuExecutable& operator=(const NpuExecutable&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_NPU_NPU_EXECUTABLE_H_