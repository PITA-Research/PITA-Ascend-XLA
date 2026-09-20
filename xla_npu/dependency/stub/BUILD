load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "libge_runner_stub",
    srcs = [
        "ge_api_stub.cc",
    ],
    deps = [
        "//xla/xla_npu/dependency/header/metadef:metadef_hdrs",
        "//xla/xla_npu/dependency/header/ge:ge_hdrs",
    ],
    linkshared = True,
    linkopts = ["-Wl,-soname,libge_runner.so"],
)

cc_import(
    name = "ge_runner_stub",
    shared_library = ":libge_runner_stub",
)

cc_binary(
    name = "libgraph_stub",
    srcs = [
        "graph_stub.cc",
        "operator_stub.cc",
        "operator_factory_stub.cc",
    ],
    deps = [
        "//xla/xla_npu/dependency/header/metadef:metadef_hdrs",
    ],
    linkshared = True,
    linkopts = ["-Wl,-soname,libgraph.so"],
)

cc_import(
    name = "graph_stub",
    shared_library = ":libgraph_stub",
)

cc_library(
    name = "ge_hdrs",
    deps = [
        "//xla/xla_npu/dependency/header/metadef:metadef_hdrs",
        "//xla/xla_npu/dependency/header/ge:ge_hdrs",
        ":ge_runner_stub",
        ":graph_stub",
    ],
)
