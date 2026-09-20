load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "npu_topology",
    srcs = ["npu_topology.cc"],
    hdrs = ["npu_topology.h"],
    deps = [
        # ":cpu_topology_proto_cc",
        "//xla/pjrt:pjrt_common",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "npu_device_description",
    srcs = ["npu_device_description.cc"],
    hdrs = ["npu_device_description.h"],
    deps = [
        ":npu_topology",
        "//xla/xla_npu/common:npu_constant",

        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "npu_topology_description",
    srcs = ["npu_topology_description.cc"],
    hdrs = ["npu_topology_description.h"],
    deps = [
        ":npu_device_description",
        ":npu_topology",

        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_device_dimensions",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
