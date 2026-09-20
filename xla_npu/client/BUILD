load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_module",
    srcs = ["npu_module.cc"],
    hdrs = ["npu_module.h"],
    deps = [
        "//xla/xla_npu/common:npu_config",
        "//xla/xla_npu/common:npu_constant",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service:npu_compiler",
        "//xla/xla_npu/service:npu_transfer_manager",
        "//xla/xla_npu/stream_executor:npu_platform",
        "//xla/xla_npu/stream_executor:npu_platform_id",

        "//xla/stream_executor:platform_manager",
        "//xla/service:computation_placer",
    ],
)

cc_library(
    name = "npu_stream_client",
    srcs = ["npu_stream_client.cc"],
    hdrs = ["npu_stream_client.h"],
    deps = [
        ":npu_module",
        "//xla/xla_npu/common:npu_constant",
        "//xla/xla_npu/pjrt/npu:npu_pjrt_stream_executor_client",
        "//xla/xla_npu/stream_executor:npu_platform",
        "//xla/xla_npu/stream_executor:npu_platform_id",

        "//xla/client:client_library",
        # "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@tsl//tsl/platform:fingerprint",
    ],
)