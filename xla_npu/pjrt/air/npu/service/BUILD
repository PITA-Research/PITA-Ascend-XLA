# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_buffer_assignment",
    srcs = ["npu_buffer_assignment.cc"],
    hdrs = ["npu_buffer_assignment.h"],
    deps = [
        "//xla/service:buffer_assignment",
    ],
)

cc_library(
    name = "npu_executable",
    srcs = ["npu_executable.cc"],
    hdrs = ["npu_executable.h"],
    deps = [
        ":npu_buffer_assignment",
        "//xla/xla_npu/pjrt/air/npu/ge:graph_data_hdrs",

        # ":cpu_runtime",
        # ":executable_proto_cc",
        "//xla:executable_run_options",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        # "//xla/backends/cpu:constant_allocation",
        # "//xla/backends/cpu/runtime:buffer_allocations",
        # "//xla/backends/cpu/runtime:function_library",
        # "//xla/backends/cpu/runtime:thread_pool_task_runner",
        # "//xla/backends/cpu/runtime:thunk",
        # "//xla/backends/cpu/runtime:thunk_executor",
        # "//xla/backends/cpu/runtime:xfeed_manager",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_status_internal",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_profile_printer_data_cc",
        "//xla/service:hlo_value",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:xla_debug_info_manager",
        # "//xla/stream_executor:device_memory",
        # "//xla/stream_executor:device_memory_allocator",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:denormal",
        "@tsl//tsl/platform:setround",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)