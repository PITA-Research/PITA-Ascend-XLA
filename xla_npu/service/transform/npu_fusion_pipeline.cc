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

#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"

#include <memory>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

// NPU-specific fusion passes
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_copy_fusion.h"
#include "xla/xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_block_level_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_dispatch_pipeline.h"
#include "xla/xla_npu/service/transform/npu_gemm_fusion_swap_operands.h"
#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"
#include "xla/xla_npu/service/transform/npu_priority_fusion.h"
#include "xla/xla_npu/service/transform/npu_rename_fusions.h"
#include "xla/xla_npu/service/transform/npu_unsupported_fusion_defuser.h"

namespace xla {

HloPassPipeline NpuFusionPipeline(const DebugOptions& debug_options,
                                  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
                                  tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& npu_device_info,
                                  SymbolicExprContext* symbolic_expr_context) {
    HloPassFix<HloPassPipeline> fusion("npu-fusion");

    HloVerifierOpts opts = HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
        LayoutAssignment::InstructionCanChangeLayout);
    fusion.AddInvariantCheckerDebug<HloVerifier>(std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
                                                 "hlo verifier (npufusion)");

    // NPU cost analysis options - use base HloCostAnalysis::Options
    HloCostAnalysis::Options cost_analysis_options{shape_size_bytes_function,
                                                   /*per_second_rates=*/{},
                                                   /*min_latencies_seconds=*/{},
                                                   /*count_multiple_input_accesses=*/true};

    // Print initial HLO
    fusion.AddPass<NpuHloPrinterPass>("Npu fusion initial");

    // NpuGemmFusionSwapOperands - swap GEMM operands for better NPU layout
    // Run BEFORE priority fusion to optimize GEMM operand order
    fusion.AddPass<NpuGemmFusionSwapOperands>();

    // Priority fusion pass for NPU - independent implementation
    fusion.AddPass<NpuPriorityFusion>(thread_pool, npu_device_info, std::move(cost_analysis_options),
                                      symbolic_expr_context);

    // Running CSE affects how many users an op has. This plays a role in what
    // we detect as fusion patterns.
    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();

    // Copy fusion for NPU
    fusion.AddPass<NpuCopyFusion>(npu_device_info);

    fusion.AddPass<HloCSE>(
        /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
        /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
    fusion.AddPass<HloDCE>();

    // Dynamic slice fusion rewriter for NPU
    fusion.AddPass<NpuDynamicSliceFusionRewriter>("NPU");

    // Fusion dynamic memcpy rewriter for NPU
    fusion.AddPass<NpuFusionDynamicMemcpyRewriter>();

    // NpuFusionBlockLevelRewriter - convert loop/input fusions to block-level
    fusion.AddPass<NpuFusionBlockLevelRewriter>(npu_device_info);

    // Rename fusions - use NPU implementation for completeness
    fusion.AddPass<NpuRenameFusions>();

    // Disassemble single-operator fusion that Autofuse does not support (such as transpose-only fused_dot.*)
    // Place it after Rename and before Final HloDCE to ensure the fusion generated by all passes
    // All are checked before codegen, and the dead code after disassembly can be cleaned up by DCE.
    fusion.AddPass<NpuUnsupportedFusionDefuser>();

    // Final HloDCE pass
    fusion.AddPass<HloDCE>();
    fusion.AddPass<NpuHloPrinterPass>("Final HloDCE");

    return std::move(fusion);
}

}  // namespace xla
