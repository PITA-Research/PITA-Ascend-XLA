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
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "npu_compile_result",
    hdrs = ["npu_compile_result.h"],
    deps = [
    ],
)

cc_library(
    name = "npu_kernel_config",
    hdrs = ["npu_kernel_config.h"],
    deps = [
        ":npu_compile_result",
    ],
)

cc_library(
    name = "npu_custom_kernel",
    srcs = ["npu_custom_kernel.cc"],
    hdrs = ["npu_custom_kernel.h"],
    deps = [
        ":npu_kernel_config",

        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)