load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"], # It is defined as fully visible inside XLA, which is different from the definition of reference cpu. cpu and gpu strictly define which code can be referenced by this implementation.
    licenses = ["notice"],
)

cc_library(
    name = "npu_pjrt_stream_executor_client",
    srcs = ["npu_pjrt_stream_executor_client.cc"],
    hdrs = ["npu_pjrt_stream_executor_client.h"],
    deps = [
        "//xla/xla_npu/pjrt/plugin:npu_topology_description",

        "//xla/pjrt:pjrt_stream_executor_client",
    ],
)
