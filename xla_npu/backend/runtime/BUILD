load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_thunk_id",
    hdrs = ["npu_thunk_id.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/lib/gtl:int_type",
    ],
)

cc_library(
    name = "npu_thunk",
    srcs = ["npu_thunk.cc"],
    hdrs = ["npu_thunk.h"],
    deps = [
        ":npu_thunk_id",

        "//xla/xla_npu/common:npu_tool",
        "//xla/xla_npu/service:npu_buffer_allocations",
        "//xla/xla_npu/service/kernel:npu_kernel_config",

        # "//xla/backends/gpu/runtime:thunk_proto_cc",

        "//xla:executable_run_options",
        "//xla:status_macros",
        "//xla:util",
        # "//xla/backends/gpu/collectives:gpu_clique_key",
        # "//xla/backends/gpu/collectives:gpu_cliques",
        # "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/ffi:execution_context",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/service:executable",
        "//xla/service:global_device_id",
        # "//xla/service/gpu:backend_configs_cc",
        # "//xla/service/gpu:buffer_allocations",
        # "//xla/service/gpu:gpu_executable_run_options",
        # "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "npu_workspace_memory_pool",
    srcs = ["npu_workspace_memory_pool.cc"],
    hdrs = ["npu_workspace_memory_pool.h"],
    deps = [
        "//xla/tsl/platform:statusor",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "npu_stream_ordered_device_memory_pool",
    srcs = ["npu_stream_ordered_device_memory_pool.cc"],
    hdrs = ["npu_stream_ordered_device_memory_pool.h"],
    deps = [
        "//xla:status_macros",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/interface:acl_nn_interface",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "npu_aclnn_thunk",
    srcs = ["npu_aclnn_thunk.cc"],
    hdrs = ["npu_aclnn_thunk.h"],
    deps = [
        ":npu_stream_ordered_device_memory_pool",
        ":npu_thunk",
        ":npu_workspace_memory_pool",
        "//xla/xla_npu/common:npu_tool",
        "//xla/xla_npu/common:npu_operation",
        "//xla/xla_npu/service:npu_buffer_allocations",
        "//xla/xla_npu/dependency/interface:acl_custom_op_interface",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/interface:acl_nn_interface",
        "//xla/xla_npu/dependency/interface:acl_nn_op_interface",

        # "//xla/backends/gpu/runtime:thunk_proto_cc",

        "//xla/service:buffer_assignment",
        "//xla/service:buffer_assignment_proto_cc",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:dnn",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:nvtx_utils",
    ],
)

cc_library(
    name = "npu_custom_kernel_thunk",
    srcs = ["npu_custom_kernel_thunk.cc"],
    hdrs = ["npu_custom_kernel_thunk.h"],
    deps = [
        ":npu_thunk",
        "//xla/xla_npu/common:npu_tool",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service/kernel:npu_custom_kernel",

        # "//xla/backends/gpu/runtime:print_buffer_contents",
        # "//xla/backends/gpu/runtime:thunk_id",

        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        # "//xla/service/gpu:launch_dimensions",
        # "//xla/service/gpu/kernels:custom_kernel",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "npu_call_thunk",
    srcs = ["npu_call_thunk.cc"],
    hdrs = ["npu_call_thunk.h"],
    deps = [
        ":npu_thunk",
        ":npu_sequential_thunk",

        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "npu_sequential_thunk",
    srcs = ["npu_sequential_thunk.cc"],
    hdrs = ["npu_sequential_thunk.h"],
    deps = [
        ":npu_thunk",
        "//xla/xla_npu/dependency/tool:log",

        # "//xla/backends/gpu/runtime:annotation",
        # "//xla/backends/gpu/runtime:thunk",
        # "//xla/backends/gpu/runtime:thunk_proto_cc",

        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "npu_while_thunk",
    srcs = ["npu_while_thunk.cc"],
    hdrs = ["npu_while_thunk.h"],
    deps = [
        ":npu_thunk",

        "//xla/xla_npu/dependency/tool:log",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "npu_conditional_thunk",
    srcs = ["npu_conditional_thunk.cc"],
    hdrs = ["npu_conditional_thunk.h"],
    deps = [
        ":npu_thunk",

        "//xla/xla_npu/dependency/tool:log",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "npu_thunk_pass_pipeline",
    srcs = ["npu_thunk_pass_pipeline.cc"],
    hdrs = ["npu_thunk_pass_pipeline.h"],
    deps = [
        ":npu_sequential_thunk",

        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
