load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "acl_interface",
    srcs = ["acl_interface.cc"],
    hdrs = ["acl_interface.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:function_loader",
        "//xla/xla_npu/dependency/header/acl:acl_hdrs",
    ],
)

cc_library(
    name = "acl_nn_interface",
    srcs = ["acl_nn_interface.cc"],
    hdrs = ["acl_nn_interface.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:function_loader",
        "//xla/xla_npu/dependency/header/acl_nn:acl_nn_hdrs",
    ],
)

cc_library(
    name = "acl_nn_op_interface",
    srcs = ["acl_nn_op_interface.cc"],
    hdrs = ["acl_nn_op_interface.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:function_loader",
        "//xla/xla_npu/dependency/header/acl_nn_op:aclnnop_hdrs",
    ],
)

cc_library(
    name = "acl_custom_op_interface",
    srcs = ["acl_custom_op_interface.cc"],
    hdrs = ["acl_custom_op_interface.h"],
    deps = [
        "//xla/xla_npu/dependency/tool:function_loader",
        "//xla/xla_npu/dependency/header/acl_nn:acl_nn_hdrs",
    ],
)
