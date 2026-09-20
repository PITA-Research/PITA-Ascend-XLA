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

#include "xla/xla_npu/service/npu_compiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "google/protobuf/text_format.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/Support/LLVM.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/core/host_offloading/host_compute_asyncifier.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"
#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/comparison_expander.h"
#include "xla/hlo/transforms/expanders/convolution_4d_expander.h"
#include "xla/hlo/transforms/expanders/convolution_pred_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/logistic_expander.h"
#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"
#include "xla/hlo/transforms/expanders/real_imag_expander.h"
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"
#include "xla/hlo/transforms/expanders/reshape_decomposer.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/transforms/expanders/stable_sort_expander.h"
#include "xla/hlo/transforms/expanders/stochastic_convert_decomposer.h"
#include "xla/hlo/transforms/host_offload_legalize.h"
#include "xla/hlo/transforms/host_offloader.h"
#include "xla/hlo/transforms/host_offloading_prepare.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h"
#include "xla/hlo/transforms/simplifiers/all_gather_permuted_ds_simplifier.h"
#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/dot_merger.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_computation_deduplicator.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization.h"
#include "xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h"
#include "xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_resizer.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_collective_normalization.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/maybe_owning.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_inliner.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/debug/unstable_reduction_detector.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_value.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/scatter_determinism_expander.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_stablehlo_pass.h"
#endif  // PLATFORM_GOOGLE

#include "xla/xla_npu/service/npu_latency_hiding_scheduler.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_alias_info.h"
#include "xla/xla_npu/service/npu_compile_module_to_llvm_ir.h"
#include "xla/xla_npu/service/npu_executable.h"
#include "xla/xla_npu/service/npu_execution_stream_assignment.h"
#include "xla/xla_npu/service/npu_hlo_schedule.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"
#include "xla/xla_npu/service/npu_ir_emitter_unnested.h"
#include "xla/xla_npu/service/transform/npu_pipeline.h"

namespace xla {

namespace {

using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  tsl::profiler::TraceMe traceme("CreateMaybeOwningThreadPool");
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  // CurrentThreadId() returns -1 if the current thread does not belong to the
  // thread pool. If the current thread belongs to the thread pool, we should
  // not be using it, because it can potentially cause deadlocks.
  CHECK(default_thread_pool == nullptr ||
        default_thread_pool->CurrentThreadId() == -1);

  auto create_thread_pool = [&](int num_threads) {
    CHECK_GE(num_threads, 1);
    return std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "",
                                                     num_threads);
  };

  switch (parallelism) {
    case 0:
      if (default_thread_pool == nullptr && default_parallelism > 1) {
        return MaybeOwningThreadPool(create_thread_pool(default_parallelism));
      }
      return MaybeOwningThreadPool(default_thread_pool);
    case 1:
      return MaybeOwningThreadPool(nullptr);
    default:
      return MaybeOwningThreadPool(create_thread_pool(parallelism));
  }
}

}  // namespace

NpuCompiler::NpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 /* default address space */)) {}

namespace {
// Adds the HloVerifier for GPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
  opts.verify_no_collective_deadlocks = true;
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

void CheckNotScheduled(HloModule* hlo_module) {
  if (hlo_module->has_schedule() &&
      !hlo_module->config().debug_options().xla_disable_all_hlo_passes()) {
    LOG(WARNING) << "\nThe current HLO module " << hlo_module->name()
                 << " is scheduled and optimized. \n"
                 << "It is not expected to run optimization passes again.\n"
                    "Use a test method like RunAndCompareNoHloPasses() or "
                 << "the xla_disable_all_hlo_passes flag.";
  }
}

void LogDebugOptions(HloModule* hlo_module) {
  // LOG_LINES is used instead of LOG since the message can exceed the
  // maximum line length, which results in the message being truncated.
  // We are also printing all the fields, instead of the DebugString() because
  // DebugString() does not print the default values of the fields. For example
  // for all boolean fields, the default value is false, it will not be printed
  // by DebugString() if the value is false. We however need that value to be
  // printed, because our "default" values (defined in
  // xla/debug_options_flags.cc) override the default values of the fields. If
  // we set the "default" value of a field to be true in
  // `debug_option_flags.cc`, and it is overridden by XLA_FLAGS to false, we
  // will not see it in the DebugString(). So, we print all the fields.
  if (VLOG_IS_ON(1)) {
    XLA_VLOG_LINES(
        1,
        absl::StrFormat("GpuCompilationEnvironment of hlo_module %s:\n%s",
                        hlo_module->name(),
                        PrintAllFields(hlo_module->config().debug_options())));
  }
}

const HloInstruction* StripCopyAndBitcast(const HloInstruction* instruction) {
  while (instruction->opcode() == HloOpcode::kCopy ||
         instruction->opcode() == HloOpcode::kBitcast) {
    instruction = instruction->operand(0);
  }
  return instruction;
}

bool IsConstantDerivedValue(const HloInstruction* instruction) {
  instruction = StripCopyAndBitcast(instruction);
  if (instruction->opcode() == HloOpcode::kConstant) {
    return true;
  }
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      instruction->operand_count() != 1) {
    return false;
  }
  return StripCopyAndBitcast(instruction->operand(0))->opcode() ==
         HloOpcode::kConstant;
}

bool MayReachWhileThroughTupleLikeUsers(const HloInstruction* instruction) {
  std::vector<const HloInstruction*> stack(instruction->users().begin(),
                                          instruction->users().end());
  absl::flat_hash_set<const HloInstruction*> seen;

  while (!stack.empty()) {
    const HloInstruction* user = stack.back();
    stack.pop_back();
    if (!seen.insert(user).second) {
      continue;
    }

    if (user->opcode() == HloOpcode::kWhile) {
      return true;
    }

    switch (user->opcode()) {
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kCopy:
      case HloOpcode::kBitcast:
        stack.insert(stack.end(), user->users().begin(), user->users().end());
        break;
      default:
        break;
    }
  }

  return false;
}

absl::StatusOr<bool> AddNpuRootCopiesForWhileInitConstants(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    HloInstruction* operand = root->mutable_operand(i);
    if (!operand->shape().IsArray() || !IsConstantDerivedValue(operand) ||
        !MayReachWhileThroughTupleLikeUsers(operand)) {
      continue;
    }

    HloInstruction* copy = entry->AddInstruction(HloInstruction::CreateUnary(
        operand->shape(), HloOpcode::kCopy, operand));
    TF_RETURN_IF_ERROR(root->ReplaceOperandWith(i, copy));
    VLOG(1) << "Inserted NPU root copy for constant-derived while init "
            << operand->name() << " at root operand " << i << " in module "
            << module->name();
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> AddNpuWhileInitCopiesForConstantDerivedValues(
    HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }

      HloInstruction* init = instr->mutable_operand(0);
      if (init->opcode() != HloOpcode::kTuple) {
        continue;
      }

      for (int64_t i = 0; i < init->operand_count(); ++i) {
        HloInstruction* operand = init->mutable_operand(i);
        if (!operand->shape().IsArray() || !IsConstantDerivedValue(operand)) {
          continue;
        }

        HloInstruction* copy =
            computation->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kCopy, operand));
        TF_RETURN_IF_ERROR(init->ReplaceOperandWith(i, copy));
        VLOG(1) << "Inserted NPU while init copy for constant-derived value "
                << operand->name() << " at while " << instr->name()
                << " operand tuple index " << i << " in module "
                << module->name();
        changed = true;
      }
    }
  }

  return changed;
}

absl::Status RunNpuPostSchedulingCopyInsertion(HloModule* module,
                                               const NpuAliasInfo* alias_info) {
  constexpr int64_t kRegionBasedLiveRangeAnalysisLimit = -1;
  const int64_t use_region_based_live_range_analysis =
      module->config()
              .debug_options()
              .xla_gpu_copy_insertion_use_region_analysis()
          ? kRegionBasedLiveRangeAnalysisLimit
          : 0;
  CopyInsertion copy_insertion(alias_info, use_region_based_live_range_analysis);

  TF_RETURN_IF_ERROR(copy_insertion.RemoveUnnecessaryCopies(module));

  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();

  TF_ASSIGN_OR_RETURN(bool added_npu_root_copies,
                      AddNpuRootCopiesForWhileInitConstants(module));
  if (added_npu_root_copies) {
    VLOG(1) << "Added NPU root copies for constant-derived while init values.";
  }

  TF_ASSIGN_OR_RETURN(bool added_npu_while_init_copies,
                      AddNpuWhileInitCopiesForConstantDerivedValues(module));
  if (added_npu_while_init_copies) {
    VLOG(1) << "Added NPU while init copies for constant-derived values.";
  }

  // Re-insert correctness copies that copy elision/live-range analysis alone
  // cannot infer. In particular, constants used as while init values must not
  // stay live out through the same buffer as mutable while carry state.
  TF_RETURN_IF_ERROR(copy_insertion.CopyInsertion::AddSpecialCaseCopies(
      module, /*execution_threads=*/{}));
  TF_RETURN_IF_ERROR(HloDCE().Run(module).status());

  TF_RETURN_IF_ERROR(saved_schedule.Update());
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));
  return absl::OkStatus();
}
}  // namespace

// Returns the TargetConfig, either from the module debug options, or from the
// CompilationOptions, or if both of those are absent, from the attached GPU.
absl::StatusOr<Compiler::GpuTargetConfig> NpuCompiler::GetTargetConfig(
    const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
    se::StreamExecutor* executor) {
  if (executor) {
    Compiler::GpuTargetConfig target_config =
        Compiler::GpuTargetConfig{executor};
    return target_config;
  }
  return absl::InternalError(
      "Either GPU has to be attached, or --xla_gpu_target_config_filename "
      "has to be specified to specify the target to compile for.");
}

absl::StatusOr<std::unique_ptr<HloModule>> NpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  const DebugOptions debug_opts = module->config().debug_options();

  TF_ASSIGN_OR_RETURN(GpuTargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));

  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/tsl::port::MaxParallelism());
  const se::DeviceDescription& npu_device_info =
      gpu_target_config.device_description;
  TF_RETURN_IF_ERROR(
      NpuPipeline(module->config().debug_options(), ShapeSizeBytesFunction(),
                  thread_pool.get_mutable(), npu_device_info,
                  &symbolic_expr_context_)
          .Run(module.get(), {HloInstruction::kMainExecutionThread})
          .status());

  return std::move(module);
}

using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, NpuExecutable::OutputInfo>;

static void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  VLOG(5) << error_string;
}

namespace {

std::unique_ptr<llvm::Module> CopyToContext(const llvm::Module& module,
                                            llvm::LLVMContext& context) {
  llvm::SmallString<0> bitcode;
  llvm::raw_svector_ostream bitcode_ostream(bitcode);
  llvm::WriteBitcodeToFile(module, bitcode_ostream);

  llvm::Expected<std::unique_ptr<llvm::Module>> new_module =
      llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()),
                                "split_module"),
          context);
  CHECK(new_module) << "Failed to parse bitcode "
                    << llvm::toString(new_module.takeError());

  return std::move(new_module.get());
}

}  // namespace

absl::StatusOr<NpuCompiler::BackendCompileResult>
NpuCompiler::CompileSingleModule(
    const HloModuleConfig& module_config,
    const stream_executor::DeviceDescription& device_description,
    const HloModule* debug_module, llvm::Module* llvm_module, bool relocatable,
    const CompileOptions& options, std::optional<int> shard_number) {
  tsl::profiler::TraceMe traceme("CompileSingleModule");
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "NpuCompiler::RunBackend - Running LLVM verifier for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown)")),
        VLOG_IS_ON(4) && !options.is_autotuning_compilation);

    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);

    std::string err;
    llvm::raw_string_ostream err_stream(err);

    // verifyModule() returns true if the module is broken.
    TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR "
           "lowering. Rerun with --xla_dump_to to get the IR"
        << (debug_module
                ? absl::StrCat(" and looks for files with name containing: *",
                               FilenameFor(*debug_module, "", ""), "*")
                : ".");
  }

  TF_ASSIGN_OR_RETURN(
      BackendCompileResult result,
      CompileTargetBinary(module_config, llvm_module, device_description,
                          relocatable, debug_module, options, shard_number));

  const bool should_dump = DumpingEnabledForHloModule(
      debug_module ? debug_module->name() : "", module_config.debug_options());
  if (should_dump) {
    if (debug_module) {
      llvm_ir::DumpIrIfEnabled(
          *debug_module, *llvm_module,
          /*optimized=*/true,
          shard_number.has_value() ? std::to_string(*shard_number) : "");
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }

  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }

  return result;
}

namespace {
int CountFunctions(const llvm::Module& module) {
  int num_functions = 0;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      ++num_functions;
    }
  }
  return num_functions;
}

// Returns the name of the single function in the module or empty string if it's
// not a single-function module.
std::string SingleFunctionName(const llvm::Module& module) {
  std::string name;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      if (name.empty()) {
        // First function in a module: name the module with it.
        name = func.getName().str();
      } else {
        // Not the first function - the module is not cacheable.
        return "";
      }
    }
  }
  return name;
}
}  // namespace

absl::StatusOr<NpuCompiler::CompileResultWithMetadata>
NpuCompiler::CompileToBackendResult(
    HloModule* module, llvm::LLVMContext* llvm_context,
    const CompileOptions& options,
    const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("CompileToBackendResult");
  std::unique_ptr<NpuAliasInfo> alias_info = GetAliasInfo(gpu_device_info);
  TF_ASSIGN_OR_RETURN(
      NpuScheduleMetadata schedule_metadata,
      NpuScheduleModule(module, pointer_size_, gpu_device_info,
                        &symbolic_expr_context_, alias_info.get()));
  TF_RETURN_IF_ERROR(
      RunNpuPostSchedulingCopyInsertion(module, alias_info.get()));
  HloPassPipeline pipeline("scheduled-npu-module");
  AddHloVerifier(&pipeline);
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  absl::StatusOr<se::Platform*> platform =
      se::PlatformManager::PlatformWithId(PlatformId());
  if (!platform.ok()) {
    return absl::Status(
        platform.status().code(),
        absl::StrCat(
            platform.status().message(),
            ". Are you missing gpu_plugin or stream_executor dependency?"));
  }

  const bool use_cache = false;

  NpuCompileModuleResults compile_module_results;

  {
    BufferValue::SizeFunction buffer_size_bytes_function =
        BufferSizeBytesFunction();
    // Compile the module to thnks and llvm IR.
    TF_ASSIGN_OR_RETURN(compile_module_results,
                        NpuCompileModuleToLlvmIr(
                            module, llvm_context, target_triple_, data_layout_,
                            *platform, gpu_device_info, alias_info.get(),
                            std::move(buffer_size_bytes_function),
                            /*split_constants_module=*/use_cache));
  }

  BackendCompileResult backend_result;

  return CompileResultWithMetadata{std::move(backend_result),
                                   std::move(compile_module_results)};
}

absl::StatusOr<std::unique_ptr<Executable>> NpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  tsl::profiler::ScopedAnnotation backend_annotation{[&] {
    return absl::StrFormat("XlaCompileBackend:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  }};

  const DebugOptions& debug_opts = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(GpuTargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));

  if (!options.is_autotuning_compilation) {
    VLOG(1) << "Starting to compile HLO module " << module->name();
  }

  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("NpuCompiler::RunBackend for ", module->name()),
      !options.is_autotuning_compilation);
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name(), " for GPU");
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  if (options.is_autotuning_compilation) {
    if (module->config().debug_options().xla_embed_ir_in_executable()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "xla_embed_ir_in_executable wastes memory!";
    }
  }

  llvm::LLVMContext llvm_context;
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;

  TF_ASSIGN_OR_RETURN(CompileResultWithMetadata res,
                      CompileToBackendResult(module.get(), &llvm_context,
                                             options, gpu_device_info));

  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(*module, "", "thunk_sequence.txt",
                            res.compile_module_results.executable->ToString(0));
  }

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCreateGpuExecutable:#module=%s#",
                           module->name());
  });

  std::unique_ptr<NpuAliasInfo> alias_info = GetAliasInfo(gpu_device_info);
  const NpuAliasInfo* alias_info_ptr = alias_info.get();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<NpuExecutable> npu_executable,
      NpuExecutable::Create(NpuExecutable::Params{
          /*asm_text=*/(options.is_autotuning_compilation &&
                        !res.backend_result.binary.empty())
              ? std::string()
              : std::move(res.backend_result.asm_text),
          /*binary=*/std::move(res.backend_result.binary),
          /*executable=*/std::move(res.compile_module_results.executable),
          /*constants=*/std::move(res.compile_module_results.constants),
          /*output_info=*/std::move(res.compile_module_results.output_info),
          /*module_name=*/std::move(res.compile_module_results.module_name),
          /*program_shape=*/
          module->compute_computation_layout().ComputeProgramShape(),
          /*mlir_allocations=*/
          (res.compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(res.compile_module_results.allocations)),
          /*buffer_assignment=*/
          std::move(res.compile_module_results.buffer_assignment),
          /*alias_info=*/std::move(alias_info), /*debug_options=*/debug_opts,
          /*device_description=*/gpu_device_info,
          /*debug_module=*/options.is_autotuning_compilation
              ? std::unique_ptr<HloModule>()
              : std::move(module),
          /*enable_debug_info_manager=*/!options.is_autotuning_compilation}));

  return static_cast<std::unique_ptr<Executable>>(std::move(npu_executable));
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
NpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                                const AotCompilationOptions& options) {
  return absl::UnimplementedError(
      "NpuCompiler::CompileAheadOfTime is not supported.");
}

HloCostAnalysis::ShapeSizeFunction NpuCompiler::ShapeSizeBytesFunction() const {
  return NpuShapeSizeBytesFunction(pointer_size_, std::nullopt);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>> NpuCompiler::Export(
    Executable* executable) const {
  return absl::UnimplementedError("NpuCompiler::Export is not supported.");
}

HloCostAnalysis::Options CreateHloAnalysisOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    ShapeSizeFn shape_size_fn) {
  tsl::profiler::TraceMe traceme("CreateHloAnalysisOpts");
  HloCostAnalysis::Options hlo_cost_analysis_options;
  hlo_cost_analysis_options.shape_size = shape_size_fn;
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (module.config().debug_options().xla_gpu_enable_host_memory_offloading()) {
    constexpr float kGiga = 1e+9;
    // Fused multiply-add means that these two instructions are computed as
    // one, so for this case the maximum flops is doubled.
    constexpr float kFma = 2;
    float flops_per_sec = gpu_device_info.core_count() *
                          gpu_device_info.fpus_per_core() *
                          gpu_device_info.clock_rate_ghz() * kGiga * kFma;
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    hlo_cost_analysis_options.set_flops_per_second(flops_per_sec);
    hlo_cost_analysis_options.set_transcendentals_per_second(flops_per_sec);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            /*host_memory_space=*/host_memory_space_color,
            gpu_device_info.memory_bandwidth(),
            gpu_device_info.memory_bandwidth());
  }
  return hlo_cost_analysis_options;
}

HloRematerialization::Options CreateRematOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis& hlo_cost_analysis, int64_t scheduler_mem_limit) {
  tsl::profiler::TraceMe traceme("CreateRematOpts");
  bool enable_offloading =
      module.config().debug_options().xla_gpu_enable_host_memory_offloading();
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (enable_offloading) {
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            /*host_memory_space=*/host_memory_space_color,
            gpu_device_info.memory_bandwidth(),
            gpu_device_info.memory_bandwidth());
  }
  HloRematerialization::RematerializationModeConfig
      rematerialization_mode_config(/*recompute=*/true,
                                    /*compress=*/true,
                                    /*host_offload=*/enable_offloading);
  HloRematerialization::Options options(
      hlo_cost_analysis, rematerialization_mode_config,
      // Assume 75% of the total device memory is available for XLA.
      /*memory_limit_bytes=*/scheduler_mem_limit,
      /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
      /*min_remat_size=*/0, /*compact_shape_function=*/nullptr,
      /*host_memory_offload_config=*/offloading_config);
  return options;
}
}  // namespace xla
