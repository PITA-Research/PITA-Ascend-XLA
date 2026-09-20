load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

cc_library(
    name = "pjrt_c_api_npu_internal",
    srcs = ["pjrt_c_api_npu_internal.cc"],
    hdrs = ["pjrt_c_api_npu_internal.h"],
    copts = select({
        "//xla/xla_npu/common:npu_ge_backend": ["-DNPU_AUTO_FUSE_BACKEND=3"],
        "//conditions:default": ["-DNPU_AUTO_FUSE_BACKEND=0"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        # "//xla/xla_npu/client:npu_stream_client",

        # "//xla/xla_npu/pjrt/npu:npu_client", // Not used
        # "//xla/xla_npu/pjrt/plugin:npu_client_options", // Not used
        
        "//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_ffi_internal",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_memory_descriptions_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_phase_compile_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_phase_compile_internal",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ] + select({
        "//xla/xla_npu/common:npu_ge_backend": [
            "//xla/xla_npu/pjrt/air/npu:npu_client"],
        "//conditions:default": ["//xla/xla_npu/client:npu_stream_client"],
    }),
)

cc_library(
    name = "pjrt_c_api_npu",
    srcs = ["pjrt_c_api_npu.cc"],
    hdrs = ["pjrt_c_api_npu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_npu_internal",

        "//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

# PJRT NPU plugin.
xla_cc_binary(
    name = "pjrt_c_api_npu_plugin.so",
    additional_linker_inputs = [
        ":pjrt_c_api_npu_version_script.lds", # Refer to the cpu symbol export file. This file does not actually distinguish between chips. Its function is to export only the GetPjrtApi interface.
    ],
    linkopts = 
        [
            "-Wl,--version-script,$(location :pjrt_c_api_npu_version_script.lds)",
            "-Wl,--no-undefined",
            # AFIR build dependencies
            "-L$(PYTHON_LIBDIR)",
            "-lpython$(PYTHON_VERSION)",
            "-Wl,-rpath,$(PYTHON_LIBDIR)",
    ],
    linkshared = True,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_npu",
    ],
)