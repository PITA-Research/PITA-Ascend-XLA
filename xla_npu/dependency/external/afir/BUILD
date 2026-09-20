# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# ============================================================================
# Ascend-MLIR Headers
# ============================================================================

cc_library(
    name = "AFIRHeaders",
    hdrs = glob([
        "Ascend-MLIR/include/Dialect/AFIR/**/*.h",
        "Ascend-MLIR/include/Dialect/AFIR/**/*.hpp",
        "Ascend-MLIR/include/Utils/*.h",
        "Ascend-MLIR/include/Utils/*.hpp",
        "Ascend-MLIR/include/Interface/*.h",
        "Ascend-MLIR/include/Interface/*.hpp",
        "generated/include/Dialect/AFIR/**/*.inc",
        "generated/include/Interface/*.inc",
        "generated/include/Conversion/*.inc",
        "generated/externals/pyasc/include/ascir/**/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
)

# ============================================================================
# Ascend-MLIR Libraries
# ============================================================================

cc_library(
    name = "MLIRAsc",
    srcs = [
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Attributes.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Dialect.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/OpInterfaces.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Ops.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/TypeInterfaces.cpp",
        "Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Types.cpp",
    ],
    hdrs = glob([
        "Ascend-MLIR/externals/pyasc/include/ascir/**/*.h",
        "Ascend-MLIR/externals/pyasc/include/ascir/**/*.hpp",
        "generated/externals/pyasc/include/ascir/**/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "Ascend-MLIR/externals/pyasc/include",
    ],
    deps = [
        ":AFIRHeaders",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRUtils",
    srcs = [
        "Ascend-MLIR/lib/Utils/AFIRUtils.cpp",
    ],
    hdrs = glob([
        "Ascend-MLIR/include/Utils/*.h",
        "Ascend-MLIR/include/Utils/*.hpp",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRHeaders",
        ":MLIRAsc",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRDialect",
    srcs = [
        "Ascend-MLIR/lib/Dialect/AFIR/Dialect.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/concat.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/reduce.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/transpose.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/broadcast.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/cast.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/condition.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/data.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/math.cpp",
    ],
    hdrs = glob([
        "generated/include/Dialect/AFIR/*.inc",
        "generated/include/Interface/*.inc",
    ]),
    includes = [
        "generated/externals/pyasc/include",
        "generated/include",
        "Ascend-MLIR/externals/pyasc/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRHeaders",
        ":AFIRUtils",
        ":MLIRAsc",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "AFIRTransforms",
    srcs = [
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRAddAxis.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRInferShapes.cpp",
        "Ascend-MLIR/lib/Dialect/AFIR/Transforms/Canonicalize.cpp",
    ],
    hdrs = glob([
        "generated/include/Dialect/AFIR/Transforms/*.inc",
    ]),
    includes = [
        "generated/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRDialect",
        ":AFIRHeaders",
        ":AFIRUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

# ============================================================================
# AFIR to ASCIR Text Conversion
# ============================================================================

cc_library(
    name = "AFIRToASCIRText",
    srcs = ["Ascend-MLIR/lib/Conversion/AFIRToASCIRText/AFIRToASCIRText.cpp"],
    hdrs = ["Ascend-MLIR/include/Conversion/AFIRToASCIRText/AFIRToASCIRText.h"],
    includes = [
        "generated/include",
        "Ascend-MLIR/include",
    ],
    deps = [
        ":AFIRHeaders",
        "//xla/xla_npu/common:npu_config",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
