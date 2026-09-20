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

package(default_visibility = ["//xla:internal"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# Python GIL and Timeout Protection
# ============================================================================

cc_library(
    name = "python_gil_guard",
    srcs = ["python_gil_guard.cc"],
    hdrs = ["python_gil_guard.h"],
    linkstatic = True,
    deps = [
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/strings",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# ============================================================================
# HLO to AFIR Converter
# ============================================================================

# Core conversion library
cc_library(
    name = "convert_to_afir",
    srcs = ["convert_to_afir.cc"],
    hdrs = ["convert_to_afir.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/xla_npu/dependency/external/afir:AFIRDialect",
        "//xla/xla_npu/dependency/external/afir:AFIRHeaders",
        "//xla/xla_npu/dependency/tool:log",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

# AscIR backend library
cc_library(
    name = "ascir_backend",
    srcs = ["ascir_backend.cc"],
    hdrs = ["ascir_backend.h"],
    linkstatic = True,  # Do not pass dependencies externally
    deps = [
        ":convert_to_afir",
        ":python_gil_guard",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/xla_npu/common:npu_config",
        "//xla/xla_npu/common:npu_constant",
        "//xla/xla_npu/dependency/external/afir:AFIRHeaders",
        "//xla/xla_npu/dependency/external/afir:AFIRToASCIRText",
        "//xla/xla_npu/dependency/external/afir:AFIRTransforms",
        "//xla/xla_npu/dependency/tool:log",
        "//xla/xla_npu/service/kernel:npu_compile_result",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# Binary test for AscIR backend
cc_binary(
    name = "ascir_backend_test",
    srcs = ["ascir_backend_test.cc"],
    linkopts = [
        "-L$(PYTHON_LIBDIR)",
        "-lpython$(PYTHON_VERSION)",
        "-Wl,-rpath,$(PYTHON_LIBDIR)",
    ],
    deps = [
        ":ascir_backend",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/xla_npu/common:npu_config",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

# Binary tool for standalone hlo to afir conversion
cc_binary(
    name = "hlo_to_afir_tool",
    srcs = ["hlo_to_afir_tool.cc"],
    deps = [
        ":convert_to_afir",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "//xla/xla_npu/dependency/external/afir:AFIRHeaders",
        "//xla/xla_npu/dependency/external/afir:AFIRTransforms",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
