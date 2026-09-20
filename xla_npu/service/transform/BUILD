# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
)

# NPU Transform Pass collection library
cc_library(
    name = "npu_transform_passes",
    deps = [
        ":npu_pipeline",
        ":npu_fusion_passes",
    ],
)

xla_cc_test(
    name = "npu_fma_addcmul_rewriter_test",
    srcs = ["npu_fma_addcmul_rewriter_test.cc"],
    deps = [
        ":npu_fma_addcmul_rewriter",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "npu_variadic_reduce_expander",
    srcs = ["npu_variadic_reduce_expander.cc"],
    hdrs = ["npu_variadic_reduce_expander.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "//xla:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "npu_variadic_reduce_expander_test",
    srcs = ["npu_variadic_reduce_expander_test.cc"],
    deps = [
        ":npu_variadic_reduce_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_verifier",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# NPU Fusion Pass collection library
cc_library(
    name = "npu_fusion_passes",
    deps = [
        ":npu_fusion_pipeline",
        ":npu_priority_fusion",
        ":npu_copy_fusion",
        ":npu_dynamic_slice_fusion_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        ":npu_fusion_block_level_rewriter",
        ":npu_gemm_fusion_swap_operands",
        ":npu_fusion_dispatch_pipeline",
        ":npu_hlo_printer_pass",
        ":npu_unsupported_fusion_defuser",
    ],
)

# NpuPipeline - Main entry point for all NPU optimization passes
cc_library(
    name = "npu_pipeline",
    srcs = ["npu_pipeline.cc"],
    hdrs = ["npu_pipeline.h"],
    copts = select({
        "//xla/xla_npu/common:npu_auto_fuse_backend_afir": ["-DNPU_AUTO_FUSE_BACKEND=1"],
        "//xla/xla_npu/common:npu_auto_fuse_backend_npuir": ["-DNPU_AUTO_FUSE_BACKEND=2"],
        "//xla/xla_npu/common:npu_ge_backend": ["-DNPU_AUTO_FUSE_BACKEND=3"],
        "//conditions:default": ["-DNPU_AUTO_FUSE_BACKEND=0"],
    }),
    deps = [
        ":npu_dot_contracting_dims_normalizer",
        ":npu_dot_fp32_left_order_rewriter",
        ":npu_entry_param_copy_insertion",
        ":npu_fma_addcmul_rewriter",
        ":npu_scatter_copy_insertion",
        ":npu_scatter_window_expander",
        ":npu_right_triangular_solve_expander",
        ":npu_shift_left_decomposition",
        ":npu_unsigned_to_float_decomposition",
        ":npu_variadic_reduce_expander",
        ":npu_tril_rewriter",
        ":npu_while_body_copy_insertion",
        ":npu_fusion_pipeline",
        ":npu_hlo_printer_pass",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/expanders:rng_bit_generator_expander",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_constant_folding",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms:while_loop_trip_count_annotator",
        "//xla/hlo/analysis:alias_info",
        "//xla/service:batched_gather_scatter_normalizer",
        "//xla/service:copy_insertion",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:hlo_verifier",
        "//xla/service:layout_assignment",
        "//xla/service:sharding_remover",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "npu_fma_addcmul_rewriter",
    srcs = ["npu_fma_addcmul_rewriter.cc"],
    hdrs = ["npu_fma_addcmul_rewriter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionPipeline
cc_library(
    name = "npu_fusion_pipeline",
    srcs = ["npu_fusion_pipeline.cc"],
    hdrs = ["npu_fusion_pipeline.h"],
    deps = [
        ":npu_priority_fusion",
        ":npu_copy_fusion",
        ":npu_dynamic_slice_fusion_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        ":npu_rename_fusions",
        ":npu_hlo_printer_pass",
        ":npu_fusion_block_level_rewriter",
        ":npu_gemm_fusion_swap_operands",
        ":npu_fusion_dispatch_pipeline",
        ":npu_unsupported_fusion_defuser",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/analysis:hlo_dfs_reachability",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:cpu_gpu_shape_verifier",
        "//xla/service:hlo_verifier",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/service:layout_assignment",
        "//xla/service:compilation_stats",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
    ],
)

# NpuHloPrinterPass
cc_library(
    name = "npu_hlo_printer_pass",
    srcs = ["npu_hlo_printer_pass.cc"],
    hdrs = ["npu_hlo_printer_pass.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/common:npu_tool",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
    ],
)

# NpuEntryParamCopyInsertion
cc_library(
    name = "npu_entry_param_copy_insertion",
    srcs = ["npu_entry_param_copy_insertion.cc"],
    hdrs = ["npu_entry_param_copy_insertion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuScatterCopyInsertion
cc_library(
    name = "npu_scatter_copy_insertion",
    srcs = ["npu_scatter_copy_insertion.cc"],
    hdrs = ["npu_scatter_copy_insertion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuScalarFmaCompensation
cc_library(
    name = "npu_scalar_fma_compensation",
    srcs = ["npu_scalar_fma_compensation.cc"],
    hdrs = ["npu_scalar_fma_compensation.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuDotFp32LeftOrderRewriter
cc_library(
    name = "npu_dot_fp32_left_order_rewriter",
    srcs = ["npu_dot_fp32_left_order_rewriter.cc"],
    hdrs = ["npu_dot_fp32_left_order_rewriter.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "npu_scalar_fma_compensation_test",
    srcs = ["npu_scalar_fma_compensation_test.cc"],
    deps = [
        ":npu_scalar_fma_compensation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

# NpuUnsignedToFloatDecomposition - convert(U32, F32) and other unsigned -> floating point decomposition
# Avoid aclnnCast converting U32 to signed INT32 resulting in negative floating point for values ≥ 2^31
cc_library(
    name = "npu_unsigned_to_float_decomposition",
    srcs = ["npu_unsigned_to_float_decomposition.cc"],
    hdrs = ["npu_unsigned_to_float_decomposition.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:literal_util",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
    ],
)

# NpuShiftLeftDecomposition - shift_left decomposes as multiply + pow (CANN without aclnnLeftShift)
cc_library(
    name = "npu_shift_left_decomposition",
    srcs = ["npu_shift_left_decomposition.cc"],
    hdrs = ["npu_shift_left_decomposition.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:literal_util",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuTrilUtils - tril pattern matching helper function (header-only)
cc_library(
    name = "npu_tril_utils",
    hdrs = ["npu_tril_utils.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla:literal_util",
        "@com_google_absl//absl/strings",
    ],
)

# NpuTrilRewriter - tril pattern rewritten as custom-call, mapped to aclnnTril
cc_library(
    name = "npu_tril_rewriter",
    srcs = ["npu_tril_rewriter.cc"],
    hdrs = ["npu_tril_rewriter.h"],
    deps = [
        ":npu_tril_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuWhileBodyCopyInsertion - insert kCopy in while body to break buffer aliasing
cc_library(
    name = "npu_while_body_copy_insertion",
    srcs = ["npu_while_body_copy_insertion.cc"],
    hdrs = ["npu_while_body_copy_insertion.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionWhitelist
cc_library(
    name = "npu_fusion_whitelist",
    hdrs = ["npu_fusion_whitelist.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

# NpuPriorityFusion
cc_library(
    name = "npu_priority_fusion",
    srcs = ["npu_priority_fusion.cc"],
    hdrs = ["npu_priority_fusion.h"],
    deps = [
        ":npu_fusion_whitelist",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
    ],
)

# NpuCopyFusion
cc_library(
    name = "npu_copy_fusion",
    srcs = ["npu_copy_fusion.cc"],
    hdrs = ["npu_copy_fusion.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:call_graph",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuDynamicSliceFusionRewriter
cc_library(
    name = "npu_dynamic_slice_fusion_rewriter",
    srcs = ["npu_dynamic_slice_fusion_rewriter.cc"],
    hdrs = ["npu_dynamic_slice_fusion_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:call_graph",
        "//xla:shape_util",
        "//xla:util",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionDynamicMemcpyRewriter
cc_library(
    name = "npu_fusion_dynamic_memcpy_rewriter",
    srcs = ["npu_fusion_dynamic_memcpy_rewriter.cc"],
    hdrs = ["npu_fusion_dynamic_memcpy_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuRenameFusions
cc_library(
    name = "npu_rename_fusions",
    srcs = ["npu_rename_fusions.cc"],
    hdrs = ["npu_rename_fusions.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",

        "//xla/codegen:ir_emission_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuFusionBlockLevelRewriter
cc_library(
    name = "npu_fusion_block_level_rewriter",
    srcs = ["npu_fusion_block_level_rewriter.cc"],
    hdrs = ["npu_fusion_block_level_rewriter.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:backend_config",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuGemmFusionSwapOperands
cc_library(
    name = "npu_gemm_fusion_swap_operands",
    srcs = ["npu_gemm_fusion_swap_operands.cc"],
    hdrs = ["npu_gemm_fusion_swap_operands.h"],
    deps = [
        "//xla/xla_npu/service:backend_config_cc",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

# NpuFusionDispatchPipeline
cc_library(
    name = "npu_fusion_dispatch_pipeline",
    srcs = ["npu_fusion_dispatch_pipeline.cc"],
    hdrs = ["npu_fusion_dispatch_pipeline.h"],
    deps = [
        ":npu_fusion_block_level_rewriter",
        ":npu_fusion_dynamic_memcpy_rewriter",
        ":npu_fusion_whitelist",
        ":npu_gemm_fusion_swap_operands",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:backend_config_cc",

        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_cse",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:status",
        "@llvm-project//mlir:IR",
    ],
)

# NpuUnsupportedFusionDefuser
cc_library(
    name = "npu_unsupported_fusion_defuser",
    srcs = ["npu_unsupported_fusion_defuser.cc"],
    hdrs = ["npu_unsupported_fusion_defuser.h"],
    deps = [
        ":npu_fusion_whitelist",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# NpuDotContractingDimsNormalizer
cc_library(
    name = "npu_dot_contracting_dims_normalizer",
    srcs = ["npu_dot_contracting_dims_normalizer.cc"],
    hdrs = ["npu_dot_contracting_dims_normalizer.h"],
    deps = [
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "npu_dot_contracting_dims_normalizer_test",
    srcs = ["npu_dot_contracting_dims_normalizer_test.cc"],
    deps = [
        "//xla:literal",
        "//xla/hlo/evaluator:hlo_evaluator",
        ":npu_dot_contracting_dims_normalizer",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_verifier",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "npu_scatter_window_expander",
    srcs = ["npu_scatter_window_expander.cc"],
    hdrs = ["npu_scatter_window_expander.h"],
    deps = [
        "//xla:literal",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "npu_scatter_window_expander_test",
    srcs = ["npu_scatter_window_expander_test.cc"],
    deps = [
        ":npu_scatter_window_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_verifier",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "npu_right_triangular_solve_expander",
    srcs = ["npu_right_triangular_solve_expander.cc"],
    hdrs = ["npu_right_triangular_solve_expander.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)
xla_cc_test(
    name = "npu_right_triangular_solve_expander_test",
    srcs = ["npu_right_triangular_solve_expander_test.cc"],
    deps = [
        ":npu_right_triangular_solve_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_verifier",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
