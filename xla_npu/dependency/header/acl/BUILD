load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "acl_hdrs",
    includes = [
        ".",
    ], # Add the directory where the current BUILD file is located to the header file retrieval path
    hdrs = glob([
        "*.h",
        "*.hpp",
        "*.h.inc",
    ]), # Automatically retrieve all header files in the current directory
)