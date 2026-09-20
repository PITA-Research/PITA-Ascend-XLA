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

#ifndef XLA_XLA_NPU_SERVICE_NPU_EXECUTABLE_H_
#define XLA_XLA_NPU_SERVICE_NPU_EXECUTABLE_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"

namespace xla {
class NpuExecutable : public Executable {
 public:
  struct ConstantInfo {
    std::string symbol_name;
    NpuDenseDataIntermediate content;
    int allocation_index = -1;
  };

  struct OutputInfo {
    int allocation_index;

    bool passthrough = false;

    std::optional<HloInputOutputAliasConfig::Alias> alias_config;

    friend bool operator==(const OutputInfo& lhs, const OutputInfo& rhs) {
      return std::tie(lhs.allocation_index, lhs.passthrough,
                      lhs.alias_config) ==
             std::tie(rhs.allocation_index, rhs.passthrough, rhs.alias_config);
    }

    friend bool operator!=(const OutputInfo& lhs, const OutputInfo& rhs) {
      return !(lhs == rhs);
    }
  };

  struct Params {
    std::string asm_text;
    std::vector<uint8_t> binary;
    std::unique_ptr<NpuSequentialThunk> executable;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    ProgramShape program_shape;
    std::optional<std::vector<BufferAllocation>> mlir_allocations;
    std::unique_ptr<const BufferAssignment> buffer_assignment;
    std::unique_ptr<NpuAliasInfo> alias_info;
    DebugOptions debug_options;
    se::DeviceDescription device_description;
    std::unique_ptr<HloModule> debug_module = nullptr;
    bool enable_debug_info_manager = true;
  };

  static absl::StatusOr<std::unique_ptr<NpuExecutable>> Create(Params params);
  ~NpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

  absl::string_view name() const override { return module_name_; }

  xla::Shape result_shape() const override { return program_shape_.result(); }

  const absl::flat_hash_map<ShapeIndex, OutputInfo>& output_info() const {
    return output_info_;
  }

  ComputationLayout compute_computation_layout() const override {
    return ComputationLayout(program_shape_, /*ignore_layouts=*/false);
  }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const std::string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled code for the computation.
  // The compiled code is PTX in Cuda and unused empty string in ROCm.
  // This may be left empty for saving memory if we have a non-empty binary.
  // If both text() and binary() are empty, that means the HLO required no
  // custom kernels to be compiled.
  const std::string& text() const { return text_; }

  // Returns the binary stored in this NpuExecutable.
  // The binary is cubin in Cuda, and HSA code object in ROCm. It may be empty,
  // in which case compilation is left up to the GPU driver. If both text() and
  // binary() are empty, that means the HLO required no custom kernels to be
  // compiled.
  const std::vector<uint8_t>& binary() const { return binary_; }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override;

  using VariantArguments = std::variant<absl::Span<const ShapedBuffer* const>,
                                        absl::Span<ExecutionInput>>;
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation * absl_nonnull const> GetAllocations()
      const override {
    return allocation_ptrs_;
  }

  const std::vector<ConstantInfo>& constants() const { return constants_; }

  const BufferAssignment* buffer_assignment() const {
    return buffer_assignment_.get();
  }

  const NpuAliasInfo* alias_info() const { return alias_info_.get(); }

  const NpuSequentialThunk& GetThunk() { return *thunks_; }

  absl::Status ExecuteThunks(const NpuBufferAllocations& buffer_allocations,
                             const ServiceExecutableRunOptions* run_options);

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase>;

  absl::StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::Stream* stream);

 private:
  // Use NpuExecutable::Create() to create an instance.
  explicit NpuExecutable(Params params,
                         std::deque<BufferAllocation> thunk_pass_allocations);

  absl::StatusOr<NpuBufferAllocations> GenerateBufferAllocations(
      VariantArguments arguments,
      const NpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
      se::Stream* stream = nullptr);

  absl::StatusOr<se::DeviceMemoryBase> BufferForAllocation(
      VariantArguments arguments,
      const NpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
      int64_t arg_idx, se::Stream* stream = nullptr);
  // This string should be modified only before ExecuteOnStream.
  std::string ir_module_string_;

  // The compiled code for the computation.
  const std::string text_;

  // May be empty, in which case we leave compilation up to the GPU driver.
  std::vector<uint8_t> binary_;

  // The thunks to be invoked by this NpuExecutable. They are generated by the
  // IrEmitter (null if XLA:GPU runtime is enabled).
  std::unique_ptr<NpuSequentialThunk> thunks_;

  // Additional execution streams requested by `thunks_`.
  absl::flat_hash_set<NpuExecutionStreamId> execution_stream_ids_;

  std::string module_name_;

  ProgramShape program_shape_;

  // Provides information for allocating memory for every output/temp buffer.
  // Non-owning pointers - allocation objects reside either in allocations_
  // or buffer_assignment_.
  // A NpuExecutable can get its allocations in three ways:
  // 1 - From a regular compilation that uses allocations from MLIR.
  // 2 - From a regular compilation that uses the original allocations from
  //     the buffer assignment.
  // 3 - From loading the executable from an object file.
  // In cases 1 and 3, the allocations are stored in allocations_ and in
  // case 2, they are part of the buffer_assignment.
  const std::vector<const BufferAllocation*> allocation_ptrs_;

  // The allocations_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on allocation_ptrs_.
  std::optional<const std::vector<BufferAllocation>> allocations_;

  // The buffer_assignment_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on allocation_ptrs_.
  // This object is also used for dumping debug info.
  std::shared_ptr<const xla::BufferAssignment> buffer_assignment_;

  // Extra allocations added by thunk passes outside of the normal buffer
  // assignment process.
  // std::deque is used to ensure pointer stability.
  const std::deque<BufferAllocation> thunk_pass_allocations_;

  // Backend specific aliasing information whether operands can/should share the
  // buffer with the user.
  std::unique_ptr<NpuAliasInfo> alias_info_;

  int64_t debug_buffer_assignment_show_max_;

  absl::Mutex module_handle_mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // executable is destroyed.
  absl::flat_hash_map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  // Cache previous memory allocations for current module, this is used to help
  // identify if user's model have unstable pointers by turning on VLOG(5).
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::vector<se::DeviceMemoryBase>>
      module_allocations_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  // Retains shared ownership of on-device constants that are managed by XLA and
  // potentially shared with other executables.
  std::vector<std::shared_ptr<se::DeviceMemoryBase>> shared_constants_;
  bool enable_debug_info_manager_;

  NpuExecutable(const NpuExecutable&) = delete;
  NpuExecutable& operator=(const NpuExecutable&) = delete;
};

absl::StatusOr<absl::flat_hash_map<ShapeIndex, NpuExecutable::OutputInfo>>
NpuGetOutputInfo(const HloModule& hlo_module,
                 const BufferAssignment& assignment);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_EXECUTABLE_H_
