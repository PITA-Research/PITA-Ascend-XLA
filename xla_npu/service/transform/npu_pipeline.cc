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

#include "xla/xla_npu/service/transform/npu_pipeline.h"

#include <cstdlib>
#include <memory>
#include <utility>

#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/call_inliner.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/sharding_remover.h"
#include "xla/xla_npu/service/transform/npu_dot_contracting_dims_normalizer.h"
#include "xla/xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fma_addcmul_rewriter.h"
#include "xla/xla_npu/service/transform/npu_fusion_pipeline.h"
#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"
#include "xla/xla_npu/service/transform/npu_entry_param_copy_insertion.h"
#include "xla/xla_npu/service/transform/npu_scatter_copy_insertion.h"
#include "xla/xla_npu/service/transform/npu_scatter_window_expander.h"
#include "xla/xla_npu/service/transform/npu_right_triangular_solve_expander.h"
#include "xla/xla_npu/service/transform/npu_shift_left_decomposition.h"
#include "xla/xla_npu/service/transform/npu_unsigned_to_float_decomposition.h"
#include "xla/xla_npu/service/transform/npu_variadic_reduce_expander.h"
#include "xla/xla_npu/service/transform/npu_tril_rewriter.h"
#include "xla/xla_npu/service/transform/npu_while_body_copy_insertion.h"

namespace xla {

HloPassPipeline NpuPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& npu_device_info,
    SymbolicExprContext* symbolic_expr_context) {
  HloPassFix<HloPassPipeline> pipeline("npu-pipeline");

  HloVerifierOpts opts =
      HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout);
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
      "hlo verifier (npu)");

  pipeline.AddPass<NpuHloPrinterPass>("Npu pipeline initial");

  // Shardy leaves metadata-only custom calls such as
  // xla.sdy.FuncResultSharding in single-device modules. They have identity
  // semantics, but emitting no thunk for them leaves their separately assigned
  // output buffer unwritten. Remove the markers before buffer assignment;
  // ShardingRemover keeps required buffer boundaries as explicit kCopy ops.
  pipeline.AddPass<ShardingRemover>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<NpuHloPrinterPass>("ShardingRemover");

  // Expand kRngBitGenerator (JAX.random.uniform/normal/bernoulli, etc.) into kCall,
  // Threefry/Philox PRNG subcalculation included. Must be executed before FlattenCallGraph,
  // So that subsequent passes can process the expanded basic operators (XOR, Shift, Add, etc.).
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);
  pipeline.AddPass<NpuHloPrinterPass>("RngBitGeneratorExpander");

  pipeline.AddPass<FlattenCallGraph>();
  pipeline.AddPass<NpuHloPrinterPass>("FlattenCallGraph");

  // Inline single-use calls such as the helper computation emitted by
  // jax.random.uniform. Keeping them as runtime NpuCallThunk wrappers left
  // uniform-specific scalar parameter plumbing on a separate call frame and
  // triggered nondeterministic first-element corruption across repeated runs.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<NpuHloPrinterPass>("CallInliner");
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<NpuHloPrinterPass>("HloConstantFolding");
  // Normalize before generic simplification, including rank-0 operands whose
  // scatter otherwise simplifies into an incorrectly shaped scalar add.
  pipeline.AddPass<BatchedGatherScatterNormalizer>();
  pipeline.AddPass<NpuScatterWindowExpander>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<AlgebraicSimplifier>(AlgebraicSimplifierOptions{});
  pipeline.AddPass<NpuHloPrinterPass>("AlgebraicSimplifier");
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<NpuHloPrinterPass>("PostInlineDCE");

  // Preserve coupled value/index semantics before ACLNN dispatch or fusion.
  // Scalar ReduceSum/Amax/Amin cannot execute a variadic tuple reducer.
  pipeline.AddPass<NpuVariadicReduceExpander>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<NpuHloPrinterPass>("VariadicReduceExpander");

  // Normalize batched scatter/gather: rewrite scatter with input_batching_dims
  // into a non-batched version by absorbing batch dimensions into
  // scatter_dims_to_operand_dims and inserted_window_dims, and concatenating
  // iota indices for the batch dimension.
  // CPU/GPU backends both include this pass (cpu_compiler.cc, gpu_compiler.cc);
  // without it, vmap'd scatter retains input_batching_dims with batch_size=1,
  // causing only 1/N batch elements to be written (e.g., qfrc_actuator corruption).
  pipeline.AddPass<BatchedGatherScatterNormalizer>();
  pipeline.AddPass<NpuScatterWindowExpander>();
  // Fold static window indices (common in slice VJPs) once at compile time.
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<NpuHloPrinterPass>("BatchedGatherScatterNormalizer");

  // Insert kCopy for entry parameters that are computed (not passthrough).
  // This prevents buffer assignment from aliasing input parameter buffers
  // with intermediate computation temporaries, which can cause data
  // corruption in scatter/dynamic_update_slice operations on large graphs.
  // Fixes vmap batch bug where qfrc_actuator was corrupted.
  pipeline.AddPass<NpuEntryParamCopyInsertion>();
  pipeline.AddPass<NpuHloPrinterPass>("EntryParamCopyInsertion");

  // Insert copy before scatter operand to prevent result aliasing
  // with entry parameter in buffer assignment.
  pipeline.AddPass<NpuScatterCopyInsertion>();
  pipeline.AddPass<NpuHloPrinterPass>("ScatterCopyInsertion");

  // Break shift_left into multiply + pow(convert(n, F32)),
  // Because CANN does not have aclnnLeftShift API
  pipeline.AddPass<NpuShiftLeftDecomposition>();
  pipeline.AddPass<NpuHloPrinterPass>("ShiftLeftDecomposition");

  // Decompose convert(U16/U32/U64 -> F32/F64) into SIToFP + bias correction,
  // Because aclnnCast maps U32 to ACL_INT32, uint32 values ≥ 2^31
  // It is treated as a negative int32 and converted to a negative float, resulting in incorrect results such as JAX.random.
  // Must be after ShiftLeftDecomposition (shift_left may also produce U32->F32),
  // Executed before fusion (to ensure that the decomposed basic operators can be scheduled correctly).
  pipeline.AddPass<NpuUnsignedToFloatDecomposition>();
  pipeline.AddPass<NpuHloPrinterPass>("UnsignedToFloatDecomposition");

  // Replace tril's HLO decomposition mode with custom-call, mapping to aclnnTril.
  // It needs to be executed before fusion to prevent Iota+Add+Compare+Select from being fused.
  // Negative k constants are lost in AFIR compilation.
  pipeline.AddPass<NpuTrilRewriter>();
  pipeline.AddPass<NpuHloPrinterPass>("TrilRewriter");

  // Insert kCopy in the while body to disconnect the buffer aliasing of the body parameter and operator,
  // Avoid the in-place operation problem of ACLNN operator when input/output share the same buffer
  pipeline.AddPass<NpuWhileBodyCopyInsertion>();
  pipeline.AddPass<NpuHloPrinterPass>("WhileBodyCopyInsertion");

  // Mark the trip count after the while body transformation is completed and before CopyInsertion and fusion.
  // This pass determines the number of iterations of the while loop through pattern matching, annotated to WhileLoopBackendConfig.
  // EmitWhile reads this annotation and skips the condition check for loops (scan/fori_loop) known by trip_count.
  // Directly execute body N times to improve performance and bypass condition race risks.
  pipeline.AddPass<WhileLoopTripCountAnnotator>();

  // Canonicalize dot_general dimension ordering so that the runtime
  // aclnnMatmul path always sees standard layouts:
  //   lhs: [batch..., M..., K...]   (contracting dims at the end)
  //   rhs: [batch..., K..., N...]   (contracting dims right after batch)
  // Inserts kTranspose for non-standard dot_general (e.g., VJP backward
  // pass generates lhs_contracting={0} or rhs_contracting={1}).
  // Must run before CopyInsertion and fusion so that inserted transposes
  // are properly handled by buffer assignment.
  pipeline.AddPass<NpuRightTriangularSolveExpander>();
  pipeline.AddPass<NpuDotContractingDimsNormalizer>();
  pipeline.AddPass<NpuHloPrinterPass>("DotContractingDimsNormalizer");

  // Restore small f32 row-wise vector dot patterns that upstream XLA may have
  // decomposed into multiply + reduce-sum. The runtime kDot path can then use
  // the custom left-order dot ACLNN op for CPU-aligned accumulation.
  pipeline.AddPass<NpuDotFp32LeftOrderRewriter>();
  // The rewriter creates new dots from reduce(multiply), including folded
  // column-major constants. Normalize their physical operands as well.
  pipeline.AddPass<NpuDotContractingDimsNormalizer>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<NpuHloPrinterPass>("DotFp32LeftOrderRewriter");

  // Lower f32 multiply-adds to the custom aclnnFmaAddcmul kernel.  Unlike the
  // former scalar-only Dekker expansion, this preserves the intended FMA
  // rounding for scalar and vector expressions with one runtime thunk. Set
  // NPU_ENABLE_FMA_ADDCMUL=0 to remove this custom-call path for a compiler
  // crash bisect; all multiply-adds then follow the normal NPU lowering.
  const char* enable_fma_addcmul = std::getenv("NPU_ENABLE_FMA_ADDCMUL");
  if (enable_fma_addcmul == nullptr || enable_fma_addcmul[0] != '0' ||
      enable_fma_addcmul[1] != '\0') {
    pipeline.AddPass<NpuFmaAddcmulRewriter>();
    pipeline.AddPass<NpuHloPrinterPass>("FmaAddcmulRewriter");
  }

  // CopyInsertion: handles input-output alias scenarios.
  // When the parameter buffer is also used as an output, the buffer assignment will give the input of the operator operand
  // Allocate new buffer, CopyInsertion insert kCopy before parameter is overwritten to ensure data is available.
  // This is crucial for async mode - without copy, async operations may overwrite input that has not yet been read.
  static AliasInfo default_alias_info;
  pipeline.AddPass<CopyInsertion>(&default_alias_info);
  pipeline.AddPass<NpuHloPrinterPass>("CopyInsertion");

#if NPU_AUTO_FUSE_BACKEND != 0
  pipeline.AddPass(std::make_unique<HloPassPipeline>(
      NpuFusionPipeline(debug_options, std::move(shape_size_bytes_function),
                        thread_pool, npu_device_info, symbolic_expr_context)));
#endif

  return std::move(pipeline);
}

}  // namespace xla
