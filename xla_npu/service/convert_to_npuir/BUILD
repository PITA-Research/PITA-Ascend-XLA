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

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "npuir_backend",
    hdrs = [
        "npuir_backend.h",
    ],
    srcs = [
        "npuir_backend.cc",
    ],
    deps = [
        "//xla/xla_npu/dependency/external/npuir:mlir-headers-hfusion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:CAPIArith",
        "@llvm-project//mlir:LinalgDialect",
    ],
)