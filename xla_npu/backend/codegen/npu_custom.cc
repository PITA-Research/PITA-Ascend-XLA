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

#include "xla/xla_npu/backend/codegen/npu_custom.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/backend/runtime/npu_custom_kernel_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

#if NPU_AUTO_FUSE_BACKEND == 1
#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"
#elif NPU_AUTO_FUSE_BACKEND == 2
#include "xla/xla_npu/service/convert_to_npuir/npuir_backend.h"
#endif

namespace xla {

namespace {

// 编译结果缓存结构 - key 由 fusion 签名和完整 computation 指纹组成
struct CompileCache {
  // Global compile cache (thread safe)
  static std::unordered_map<std::string, NpuCompileResult>& GetCache() {
    static std::unordered_map<std::string, NpuCompileResult> cache;
    return cache;
  }

  // cache lock
  static std::mutex& GetMutex() {
    static std::mutex cache_mutex;
    return cache_mutex;
  }
};

std::string BuildFusionCompileCacheKey(
    const HloComputation* fused_computation,
    const std::unordered_map<std::string, const HloComputation*>&
        computation_map) {
  std::string readable_key =
      absl::StrCat("fusion_", HloOpcodeString(fused_computation->root_instruction()->opcode()),
                   "_", fused_computation->root_instruction()->shape().ToString());
  for (int64_t i = 0;
       i < static_cast<int64_t>(fused_computation->num_parameters()); ++i) {
    readable_key +=
        "_" + fused_computation->parameter_instruction(i)->shape().ToString();
  }

  // Use canonical HLO text for the structural fingerprint so isomorphic
  // fusion bodies with different generated names (e.g. fused_add.17/.18) share
  // one cold compile entry.
  HloPrintOptions print_options = HloPrintOptions::Fingerprint();
  std::string fingerprint_input;

  std::vector<std::string> computation_names;
  computation_names.reserve(computation_map.size());
  for (const auto& entry : computation_map) {
    computation_names.push_back(entry.first);
  }
  absl::c_sort(computation_names);

  for (const std::string& name : computation_names) {
    fingerprint_input += computation_map.at(name)->ToString(print_options);
    fingerprint_input += "\n";
  }

  // Fingerprint() intentionally canonicalizes HLO text to improve reuse across
  // isomorphic fusions with different generated names. That also makes it too
  // coarse for AFIR kernels that bake literal constants into the generated
  // artifact: two same-shape multiply-by-constant fusions with different
  // constant values can otherwise collide and incorrectly share one .so. Fold
  // every constant literal in every participating computation back into the key
  // using a stable post-order walk and a separate hash so we keep the name
  // canonicalization benefit without losing baked-constant correctness.
  std::string constant_fingerprint_input;
  for (const std::string& name : computation_names) {
    const HloComputation* computation = computation_map.at(name);
    for (const HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kConstant) {
        continue;
      }
      const auto* constant = Cast<HloConstantInstruction>(instr);
      absl::StrAppend(&constant_fingerprint_input, "const:",
                      constant->shape().ToString(), ":",
                      constant->literal().ToString(), "\n");
    }
  }

  const std::string structural_hash = absl::StrCat(
      absl::Hex(tsl::Fingerprint64(fingerprint_input), absl::kZeroPad16));
  const std::string constant_hash = absl::StrCat(
      absl::Hex(tsl::Fingerprint64(constant_fingerprint_input),
                absl::kZeroPad16));
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Fusion cache key parts: fusion=%s struct_hash=%s const_hash=%s",
      std::string(fused_computation->name()).c_str(), structural_hash.c_str(),
      constant_hash.c_str());

  return absl::StrCat(readable_key, "_h", structural_hash, "_c",
                      constant_hash);
}

}  // namespace

namespace {

absl::StatusOr<std::unique_ptr<NpuThunk>> BuildCustomKernelThunkForFusion(
    NpuIrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    NpuCustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context.buffer_assignment(),
                          NpuConstant::GetDefaultBufferAlignment(), &fusion));

  NpuThunk::ThunkInfo thunk_info = NpuThunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  return std::make_unique<NpuCustomKernelThunk>(std::move(thunk_info),
                                                std::move(custom_kernel),
                                                std::move(kernel_arguments));
}

}  // namespace

absl::StatusOr<NpuFusionEmissionResult> NpuCustomFusion::Emit(
    NpuIrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloComputation* fused_computation =
      fusion.fused_instructions_computation();

  NpuCompileResult npu_compile_result;
  const std::string entry_computation_key_name =
      NpuConstant::entryComputationKeyName;

  // The current autofuse processing rule is that subcomputations that do not begin with Entry are fused separately.
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Npu auto fuse switch is set open, run fusion compile based on afir.");

  // Build computation map: key is "main" which means entry computation
  // If there is a reduce instruction, add its to_apply computation to the map as well.
  std::unordered_map<std::string, const HloComputation*> computation_map;
  computation_map[entry_computation_key_name] = fused_computation;

  // Traverse all instructions and look for reduce type instructions
  for (const HloInstruction* instr : fused_computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      const HloComputation* reduce_computation = instr->to_apply();
      if (reduce_computation != nullptr) {
        std::string reduce_comp_name = std::string(reduce_computation->name());
        computation_map[reduce_comp_name] = reduce_computation;
        NPU_LOG_FORMAT_PRINT(
            INFO, "Found reduce instruction with to_apply computation: %s",
            reduce_comp_name.c_str());
      }
    }
  }

#if NPU_AUTO_FUSE_BACKEND == 1
  // Generate cache key from both the readable signature and the full fused
  // computation body.  The signature alone is not enough: different fused
  // computations can have the same root opcode and operand/result shapes.
  std::string cache_key =
      BuildFusionCompileCacheKey(fused_computation, computation_map);

  // Check cache
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(CompileCache::GetMutex());
    auto cache_iter = CompileCache::GetCache().find(cache_key);
    if (cache_iter != CompileCache::GetCache().end()) {
      NPU_LOG_FORMAT_PRINT(INFO, "Cache hit for fusion computation: %s",
                           cache_key.c_str());
      npu_compile_result = cache_iter->second;
      cache_hit = true;
    }
  }

  // If cache misses, perform compilation
  if (!cache_hit) {
    NPU_LOG_FORMAT_PRINT(INFO,
                         "Cache miss for fusion computation: %s, compiling...",
                         cache_key.c_str());

    AscirBackend ascir_backend;
    AscirBackendOptions ascir_backend_compile_option;
    ascir_backend_compile_option.output_dir = ir_emitter_context.GetOutputDir();
    auto compile_result =
        ascir_backend.Compile(computation_map, ascir_backend_compile_option);
    if (!compile_result.ok()) {
      return absl::InternalError(absl::StrCat(
          "Autofuse compile failed for fusion computation '", cache_key,
          "'\nOutput dir: ", ascir_backend_compile_option.output_dir, "\n",
          compile_result.status().message()));
    }
    npu_compile_result = std::move(compile_result).value();

    // Update cache
    {
      std::lock_guard<std::mutex> lock(CompileCache::GetMutex());
      CompileCache::GetCache()[cache_key] = npu_compile_result;
      NPU_LOG_FORMAT_PRINT(
          INFO, "Cached compilation result for fusion computation: %s",
          cache_key.c_str());
    }
  }
#elif NPU_AUTO_FUSE_BACKEND == 2
  NPU_LOG_FORMAT_PRINT(
      INFO,
      "Npu auto fuse switch is set open, run fusion compile based on npuir.");
  mlir::hfusion::GraphGenTester::test();
#else
  NPU_LOG_FORMAT_PRINT(
      INFO, "Npu auto fuse switch is set to default, using npuir backend.");
#endif

  auto artifact_infos = npu_compile_result.artifact_infos;

  CHECK_COND_AND_PRINT_NPU_LOG(
      artifact_infos.count(entry_computation_key_name) == 0,
      "Can't find computation %s in compile result",
      entry_computation_key_name);

  auto npu_kernel_config_ptr = std::make_unique<NpuKernelConfig>();
  npu_kernel_config_ptr->npu_kernel_type =
      NpuKernelConfig::NpuKernelType::FUSION;
  npu_kernel_config_ptr->npu_artifact_info =
      artifact_infos[entry_computation_key_name];

  // Store computation HLO text for debugging (avoiding pointer lifetime issues)
  // Use NpuTool::GetDetailedPrintOptions() to get detailed printing options
  HloPrintOptions print_options = NpuTool::GetDetailedPrintOptions();
  for (const auto& [name, computation] : computation_map) {
    npu_kernel_config_ptr->computation_hlo_map[name] =
        computation->ToString(print_options);
  }
  NPU_LOG_FORMAT_PRINT(INFO, "Stored %d computation HLOs in kernel config",
                       npu_kernel_config_ptr->computation_hlo_map.size());
  se::BlockDim block_dims(
      0);  // The block dim of the autofuse operator is calculated by tiling, and the value set here is not used.
  size_t shared_memory_bytes = 0;
  std::string computation_name = std::string(
      fused_computation
          ->name());  // The name of the kernel here is the name of the computation, which is the name followed by calls.
  NpuCustomKernel kernel(std::move(npu_kernel_config_ptr), computation_name,
                         block_dims, shared_memory_bytes);
  TF_ASSIGN_OR_RETURN(
      auto thunk, BuildCustomKernelThunkForFusion(ir_emitter_context, fusion,
                                                  std::move(kernel)));

  NpuFusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

}  // namespace xla
