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

#include "xla/xla_npu/service/npu_compile_module_to_llvm_ir.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "tsl/platform/path.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"
#include "xla/xla_npu/service/npu_ir_emitter_unnested.h"

namespace xla {

namespace {

using tsl::profiler::ScopedAnnotation;

// Prints mlir diagnostic messages to VLOG level 2.
static mlir::LogicalResult DiagnosticHandler(mlir::Diagnostic& diag) {
  VLOG(2) << diag.str();
  return mlir::failure();
}

NpuCompileModuleResults InitializeResults(const HloModule* hlo_module,
                                          llvm::LLVMContext* llvm_context,
                                          const std::string& target_triple,
                                          const std::string& data_layout,
                                          const bool split_constants_module) {
  absl::string_view module_name = hlo_module->name();
  NpuCompileModuleResults results;
  results.module_name = module_name;
  results.llvm_module =
      std::make_unique<llvm::Module>(module_name, *llvm_context);
  results.llvm_module->setTargetTriple(llvm::Triple(target_triple));
  results.llvm_module->setDataLayout(data_layout);

  if (split_constants_module) {
    // Constants are emitted into a separate module to avoid caching them.
    results.llvm_module_constants = std::make_unique<llvm::Module>(
        absl::StrCat(module_name, "_consts"), *llvm_context);
    results.llvm_module_constants->setTargetTriple(llvm::Triple(target_triple));
    results.llvm_module_constants->setDataLayout(data_layout);
  }

  results.use_original_allocations = true;
  results.execution_stream_assignment =
      std::make_unique<NpuExecutionStreamAssignment>(hlo_module);
  return results;
}

std::unique_ptr<mlir::MLIRContext> CreateMlirContext() {
  mlir::DialectRegistry registry;
  // Disable MLIR multi-threading to prevent creating too many threads when
  // compiling XLA executables concurrently (e.g. during auto-tuning).
  auto mlir_context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  mlir_context->getDiagEngine().registerHandler(DiagnosticHandler);
  return mlir_context;
}

std::string Phase(absl::string_view phase_name, const HloModule* module) {
  return absl::StrFormat("%s:#module=%s,program_id=%d#", phase_name,
                         module->name(), module->unique_id());
}

bool UseCache(const DebugOptions& options, bool split_constants_module) {
  return split_constants_module &&
         options.xla_gpu_enable_llvm_module_compilation_parallelism() &&
         !options.xla_gpu_kernel_cache_file().empty();
}

absl::StatusOr<std::unique_ptr<NpuSequentialThunk>> LowerHlo(
    const HloModule* hlo_module, NpuIrEmitterContext& ir_emitter_context,
    llvm::Module* llvm_module_constants, se::Platform::Id platform_id,
    bool use_cache) {
  const DebugOptions& options = hlo_module->config().debug_options();
  ScopedAnnotation annotation(Phase("XlaEmitLlvmIr", hlo_module));
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();

  std::unique_ptr<NpuIrEmitterUnnested> ir_emitter =
      NpuIrEmitterUnnested::Create(&ir_emitter_context);
  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "NpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(
        ir_emitter->EmitHloComputation(hlo_module->entry_computation()));
  }
  return ir_emitter->ConsumeThunkSequence();
}

}  // namespace

absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
    const HloModule* module, const NpuAliasInfo* alias_info,
    BufferValue::SizeFunction buffer_size_bytes_function) {
  if (!module->has_schedule()) {
    const std::string err_msg = "Module need to be scheduled.";
    NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
    return absl::UnavailableError(err_msg);
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module, std::make_unique<SequentialHloOrdering>(module->schedule()),
          std::move(buffer_size_bytes_function), alias_info,
          [](LogicalBuffer::Color) { return 256; },
          /*allocate_buffers_for_constants=*/true));

  VLOG(1) << "Buffer Assignment Stats for " << module->name() << "\n"
          << buffer_assignment->StatsString(alias_info);
  return buffer_assignment;
}

absl::StatusOr<NpuCompileModuleResults> NpuCompileModuleToLlvmIr(
    const HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const se::Platform* platform, const se::DeviceDescription& device_desc,
    const NpuAliasInfo* alias_info,
    BufferValue::SizeFunction buffer_size_bytes_function,
    bool split_constants_module) {
  tsl::profiler::TraceMe traceme("NpuCompileModuleToLlvmIr");
  const bool use_cache =
      UseCache(hlo_module->config().debug_options(), split_constants_module);

  NpuCompileModuleResults results =
      InitializeResults(hlo_module, llvm_context, target_triple, data_layout,
                        split_constants_module);

  TF_ASSIGN_OR_RETURN(
      results.buffer_assignment,
      RunBufferAssignment(hlo_module, alias_info,
                          std::move(buffer_size_bytes_function)));
  TF_ASSIGN_OR_RETURN(
      results.output_info,
      NpuGetOutputInfo(*hlo_module, *results.buffer_assignment));

  // capture the output shape after buffer assignment because it may change
  // during buffer assignment (nevertheless the const hlo_module)
  results.output_shape = hlo_module->result_shape();

  VLOG(1) << "After optimization module fingerprint for " << hlo_module->name()
          << ": " << hlo_module->GetFingerprint128();

  const auto& output_sub_dir_name = NpuTool::generate_unique_folder_name_absl();
  const auto& output_dir = tsl::io::JoinPath(
      NPU_CONFIG_GET_STRING(
          NpuConfig::Key::NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR),
      output_sub_dir_name);
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(output_dir));

  std::unique_ptr<mlir::MLIRContext> mlir_context = CreateMlirContext();
  auto symbolic_expr_context =
      std::make_unique<SymbolicExprContext>(mlir_context.get());
  NpuIrEmitterContext ir_emitter_context(
      output_dir, hlo_module, results.buffer_assignment.get(),
      results.execution_stream_assignment.get(), platform->Name(), device_desc,
      mlir_context.get(), results.llvm_module.get(),
      results.llvm_module_constants.get(),
      /*emit_kernels=*/true);

  TF_ASSIGN_OR_RETURN(
      results.executable,
      LowerHlo(hlo_module, ir_emitter_context,
               results.llvm_module_constants.get(), platform->id(), use_cache));

  results.constants = std::move(ir_emitter_context.constants());

  return results;
}

}  // namespace xla
