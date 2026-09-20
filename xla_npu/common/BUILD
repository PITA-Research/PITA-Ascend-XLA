load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

# Conditional compilation configuration: ACLNN=0 (default), AFIR=1, NPUIR=2, ge=3
config_setting(
    name = "npu_auto_fuse_backend_afir",
    define_values = {
        "NPU_AUTO_FUSE_BACKEND": "1",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "npu_auto_fuse_backend_npuir",
    define_values = {
        "NPU_AUTO_FUSE_BACKEND": "2",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "npu_ge_backend",
    define_values = {
        "NPU_AUTO_FUSE_BACKEND": "3",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "npu_constant",
    srcs = ["npu_constant.cc"],
    hdrs = ["npu_constant.h"],
    deps = [
        "//xla/codegen/emitters:kernel_arguments",
    ],
)

cc_library(
    name = "npu_config",
    srcs = ["npu_config.cc"],
    hdrs = ["npu_config.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:config_loader",
    ],
)

cc_library(
    name = "npu_operation",
    srcs = ["npu_operation.cc"],
    hdrs = ["npu_operation.h"],
    deps = [
        ":npu_constant",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "npu_tool",
    srcs = ["npu_tool.cc"],
    hdrs = ["npu_tool.h"],
    deps = [
        ":npu_config",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/interface:acl_nn_interface",
        "//xla/xla_npu/dependency/tool:log",

        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
    ],
)