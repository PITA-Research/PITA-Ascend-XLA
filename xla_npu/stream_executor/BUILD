load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_platform_id",
    srcs = [
        "npu_platform_id.cc",
    ],
    hdrs = [
        "npu_platform_id.h",
    ],
    deps = [
        "//xla/stream_executor:platform",
    ],
)

cc_library(
    name = "npu_event",
    hdrs = ["npu_event.h"],
    deps = [
        "//xla/stream_executor:event",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "npu_stream",
    srcs = [
        "npu_stream.cc",
    ],
    hdrs = [
        "npu_stream.h",
    ],
    deps = [
        ":npu_event",
        "//xla/xla_npu/backend/runtime:npu_stream_ordered_device_memory_pool",
        "//xla/xla_npu/backend/runtime:npu_workspace_memory_pool",
        "//xla/xla_npu/common:npu_operation",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:context",
        "@tsl//tsl/platform:denormal",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:setround",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "npu_stream_factory",
    srcs = [
        "npu_stream_factory.cc",
    ],
    hdrs = [
        "npu_stream_factory.h",
    ],
    deps = [
        ":npu_stream",

        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "npu_auto_fuse_kernel",
    srcs = ["npu_auto_fuse_kernel.cc"],
    hdrs = ["npu_auto_fuse_kernel.h"],
    deps = [
        # "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/backend/runtime:npu_workspace_memory_pool",
        "//xla/xla_npu/common:npu_config",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:config_loader",
        "//xla/xla_npu/dependency/tool:function_loader",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/stream_executor:kernel",
        "//xla/stream_executor:stream_executor_h",
    ],
)

cc_library(
    name = "npu_executor",
    srcs = [
        "npu_executor.cc",
    ],
    hdrs = [
        "npu_executor.h",
    ],
    deps = [
        ":npu_auto_fuse_kernel",
        ":npu_event",
        ":npu_stream",
        ":npu_stream_factory",
        "//xla/xla_npu/common:npu_config",
        "//xla/xla_npu/common:npu_operation",
        "//xla/xla_npu/service/kernel:npu_kernel_config",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_common",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "npu_platform",
    srcs = [
        "npu_platform.cc",
    ],
    hdrs = [
        "npu_platform.h",
    ],
    deps = [
        ":npu_executor",
        ":npu_platform_id",
        "//xla/xla_npu/common:npu_constant",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:status",
    ],
)
