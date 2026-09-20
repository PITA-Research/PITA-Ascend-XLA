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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

cc_library(
    name = "graph_convertor",
    srcs = [
        "export_graphdef.cc",
        "status.cc",
        "convert_type.cc",
        "convert_tensor.cc",
    ],
    hdrs = [
        "export_graphdef.h",
        "status.h",
        "convert_type.h",
        "convert_tensor.h",
        "utils.h",
    ],
    deps = [
        "//xla/xla_npu/pjrt/air/mair/ir:air_dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "//xla/xla_npu/dependency/stub:ge_hdrs",
        "//xla/xla_npu/dependency/interface:acl_interface",
        "//xla/xla_npu/dependency/tool:log",
    ],
)