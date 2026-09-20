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

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_CONTEXT_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/name_uniquer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_id.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_execution_stream_assignment.h"
#include "xla/xla_npu/service/npu_ir_emission_utils.h"

namespace xla {
// NpuIrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class NpuIrEmitterContext {
 public:
  NpuIrEmitterContext(
      const std::string& output_dir, const HloModule* hlo_module,
      const BufferAssignment* buffer_assignment,
      const NpuExecutionStreamAssignment* execution_stream_assignment,
      std::string platform_name, const se::DeviceDescription& gpu_device_info,
      mlir::MLIRContext* mlir_context, llvm::Module* llvm_module,
      llvm::Module* llvm_module_constants, bool emit_kernels)
      : output_dir_(output_dir),
        hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        execution_stream_assignment_(execution_stream_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        mlir_context_(mlir_context),
        expr_context_(mlir_context_),
        llvm_module_(llvm_module),
        llvm_module_constants_(llvm_module_constants),
        emit_kernels_(emit_kernels) {}
  // Disallow copy and assign.
  NpuIrEmitterContext(const NpuIrEmitterContext&) = delete;
  NpuIrEmitterContext& operator=(const NpuIrEmitterContext&) = delete;

  std::string GetOutputDir() const { return output_dir_; }

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  const NpuExecutionStreamAssignment& execution_stream_assignment() const {
    return *execution_stream_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  const se::DeviceDescription& gpu_device_info() const {
    return gpu_device_info_;
  }

  mlir::MLIRContext* mlir_context() { return mlir_context_; }

  // pointer: go/totw/230. Alternatively, return by reference instead of pointer
  // (and require reference in ctor) to signal that it is always present.
  SymbolicExprContext* expr_context() { return &expr_context_; }

  llvm::Module* llvm_module() { return llvm_module_; }
  // A separate module can optionally be used to emit constants.
  llvm::Module* llvm_module_constants() {
    return (llvm_module_constants_ == nullptr) ? llvm_module_
                                               : llvm_module_constants_;
  }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

  absl::StatusOr<InlinedModule*> get_inlined_module() {
    if (inlined_module_ == nullptr) {
      TF_ASSIGN_OR_RETURN(InlinedModule inlined_module,
                          GetInlinedModule(hlo_module_));
      inlined_module_ =
          std::make_unique<InlinedModule>(std::move(inlined_module));
    }
    return inlined_module_.get();
  }

  std::vector<NpuExecutable::ConstantInfo>& constants() { return constants_; }

  // Emit a constant with a given number of element, given byte size of the
  // element, given symbol name and content.
  void emit_constant(int64_t num_elements, int64_t bytes_per_element,
                     absl::string_view symbol_name, int allocation_idx,
                     NpuDenseDataIntermediate content, llvm::IRBuilderBase* b);

  const DebugOptions& debug_options() const {
    return hlo_module_->config().debug_options();
  }

  bool emit_kernels() const { return emit_kernels_; }

  NpuThunkId GetNextThunkId() { return thunk_id_generator_.GetNextThunkId(); }

 private:
  const std::string output_dir_;

  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  const NpuExecutionStreamAssignment* execution_stream_assignment_;
  std::string platform_name_;
  const se::DeviceDescription& gpu_device_info_;
  mlir::MLIRContext* mlir_context_;
  SymbolicExprContext expr_context_;
  llvm::Module* llvm_module_;
  llvm::Module* llvm_module_constants_;
  NameUniquer name_uniquer_;
  std::vector<NpuExecutable::ConstantInfo> constants_;
  std::unique_ptr<InlinedModule> inlined_module_;

  // We should not emit kernels when loading thunks from a compilation result.
  const bool emit_kernels_;

  // Generates unique IDs for thunk creation.
  NpuThunkIdGenerator thunk_id_generator_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_CONTEXT_H_
