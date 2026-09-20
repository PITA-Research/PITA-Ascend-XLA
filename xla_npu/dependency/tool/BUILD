load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "log",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "function_loader",
    srcs = ["function_loader.cc"],
    hdrs = ["function_loader.h"],
    deps = [
        ":log",
    ],
)

cc_library(
    name = "config_loader",
    srcs = ["config_loader.cc"],
    hdrs = ["config_loader.h"],
    deps = [
        ":log",
        "//xla/tsl/util:env_var",
    ],
)