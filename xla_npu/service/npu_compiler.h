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

#ifndef XLA_XLA_NPU_SERVICE_NPU_COMPILER_H_
#define XLA_XLA_NPU_SERVICE_NPU_COMPILER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotune_results.pb.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_compile_module_to_llvm_ir.h"

namespace xla {

// The GPU compiler generates efficient GPU executables.
class NpuCompiler : public LLVMCompiler {
 public:
  NpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);

  using LLVMCompiler::Compile;

  // An attached device is passed in via stream_exec. We get GPU configuration
  // from the attached device OR from the `options` struct (in which case the
  // attached device is ignored during the compilation).
  // If you call this directly, follow it with RunBackend rather than Compile.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                     AotCompilationOptions const& options) override;

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  absl::Status RunPostSchedulingPipelines(
      HloModule* module, int64_t scheduler_mem_limit,
      const se::DeviceDescription& gpu_device_info,
      const NpuAliasInfo* alias_info);

  std::string target_triple() const { return target_triple_; }
  std::string data_layout() const { return data_layout_; }

  const char* GetDataLayout() const { return data_layout_; }

  const char* GetTargetTriple() const { return target_triple_; }

  int64_t GetPointerSize() const { return pointer_size_; }

  static absl::StatusOr<Compiler::GpuTargetConfig> GetTargetConfig(
      const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
      se::StreamExecutor* executor);

  mlir::MLIRContext* mlir_context() { return &mlir_context_; }
  SymbolicExprContext* symbolic_expr_context() {
    return &symbolic_expr_context_;
  }

  virtual std::unique_ptr<NpuAliasInfo> GetAliasInfo(
      const se::DeviceDescription& device_description) const {
    return std::make_unique<NpuAliasInfo>(device_description);
  }

  virtual absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& config,
      const stream_executor::DeviceDescription& device_description) {
    return false;
  }

  enum class AlgebraicSimplifierMode {
    kLayoutInsensitive,
    kPostFusionSimplification,
    kLayoutNormalization,
    kPostLayoutAssignment,
    kAfterSimplifyFPConversions,
    kGpuConvoluationCanonicalization,
  };

 protected:
  struct BackendCompileResult {
    std::string asm_text;
    std::vector<uint8_t> binary;
    // BinaryMap dnn_compiled_graphs;
  };

 private:
  struct CompileResultWithMetadata {
    BackendCompileResult backend_result;
    NpuCompileModuleResults compile_module_results;
  };

  // Schedule and compile the module.
  absl::StatusOr<CompileResultWithMetadata> CompileToBackendResult(
      HloModule* module, llvm::LLVMContext* llvm_context,
      const CompileOptions& options,
      const se::DeviceDescription& gpu_device_info);

  absl::StatusOr<BackendCompileResult> CompileSingleModule(
      const HloModuleConfig& module_config,
      const stream_executor::DeviceDescription& device_description,
      const HloModule* debug_module, llvm::Module* llvm_module,
      bool relocatable, const CompileOptions& options,
      std::optional<int> shard_number);

  virtual absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      const CompileOptions& options, std::optional<int> shard_number) {
    return BackendCompileResult{};  // The original pure virtual function is changed to an empty implementation
  }

  virtual absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const stream_executor::DeviceDescription& device_description,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) {
    return Unimplemented("LinkModules is not implemented.");
  }

  se::Platform::Id platform_id_;

  const char* target_triple_;

  const char* data_layout_;

  const int64_t pointer_size_;

  NpuCompiler(const NpuCompiler&) = delete;
  NpuCompiler& operator=(const NpuCompiler&) = delete;

  // Returns the LLVM command line options that we use for compilation.
  // THey need to be set globally whenever we call into LLVM.
  virtual std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) {
    return {};  // The original pure virtual function is changed to an empty implementation
  }

  // A MLIR context that can be used by pre-codegen passes. For codegen, we will
  // need to have a context with more dialects registered.
  mlir::MLIRContext mlir_context_;
  // A symbolic expression context that can be used by pre-codegen passes to
  // create symbolic expressions.
  SymbolicExprContext symbolic_expr_context_{&mlir_context_};
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_COMPILER_H_
