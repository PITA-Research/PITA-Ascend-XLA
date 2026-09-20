load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Symbol Dialect
td_library(
    name = "SymbolTdFiles",
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td",
    ],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ParallelCombiningOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRSymbolIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td",
    tbl_outs = [
        (["-gen-dialect-decls", "-dialect=symbol"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolDialect.h.inc"),
        (["-gen-dialect-defs", "-dialect=symbol"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolDialect.cpp.inc"),
        (["-gen-op-decls"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.h.inc"),
        (["-gen-op-defs"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.cpp.inc"),
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":SymbolTdFiles",
    ],
)

cc_library(
    name = "mlir-headers-symbol",
    hdrs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/Symbol.h",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Utils/Util.h",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Utils/Utils.h",
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/AggregatedOpInterface.h",
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/AggregatedOpInterface.h.inc"
    ],
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolDialect.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolDialect.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.cpp.inc",
        "AscendNPU-IR/bishengir/lib/Dialect/Symbol/IR/SymbolDialect.cpp",
        "AscendNPU-IR/bishengir/lib/Dialect/Symbol/IR/SymbolOps.cpp",
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    deps = [
        ":BiShengIRHFusionAttributesIncGen",
        ":BiShengIRHFusionStructuredOpsIncGen",
        ":BiShengIRHFusionEnumsIncGen",
        ":BiShengIRHFusionOpsIncGen",
        ":MLIRBiShengIREnumsIncGen",        
        ":BiShengIRSymbolIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:FuncDialect",
    ],
)

# MathExt Dialect
td_library(
    name = "MathExtTdFiles",
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td",
    ],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ParallelCombiningOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRMathExtOpsIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td",
    tbl_outs = [
        (["-gen-dialect-decls", "-dialect=mathExt"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOpsDialect.h.inc"),
        (["-gen-dialect-defs", "-dialect=mathExt"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOpsDialect.cpp.inc"),
        (["-gen-op-decls"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.h.inc"),
        (["-gen-op-defs"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.cpp.inc"),
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":MathExtTdFiles",
    ],
)

cc_library(
    name = "mlir-headers-mathExt",
    hdrs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExt.h",
    ],
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOpsDialect.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOpsDialect.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.cpp.inc",
        "AscendNPU-IR/bishengir/lib/Dialect/Math/IR/MathExtDialect.cpp",
        "AscendNPU-IR/bishengir/lib/Dialect/Math/IR/MathExtOps.cpp",
    ],
    # strip_include_prefix = "xla/xla_npu/third_party/npuir/AscendNPU-IR/bishengir/include",
    includes = ["./AscendNPU-IR/bishengir/include"],
    deps = [
        ":BiShengIRMathExtOpsIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MathDialect",
    ],
)

# # BiShengIRAttrs
td_library(
    name = "BiShengIRAttrsTdFiles",
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRAttrs.td",
    ],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ParallelCombiningOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "MLIRBiShengIREnumsIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRAttrs.td",
    tbl_outs = [
        (["-gen-enum-decls"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc"),
        (["-gen-enum-defs"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIREnums.cpp.inc"),
        (["-gen-attrdef-decls", "-attrdefs-dialect=bishengir"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRAttrs.h.inc"),
        (["-gen-attrdef-defs", "-attrdefs-dialect=bishengir"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRAttrs.cpp.inc"),  
        (["-gen-typedef-decls", "-typedefs-dialect=bishengir"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRTypes.h.inc"),
        (["-gen-typedef-defs", "-typedefs-dialect=bishengir"], "AscendNPU-IR/bishengir/include/bishengir/Interfaces/BiShengIRTypes.cpp.inc"),    
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":BiShengIRAttrsTdFiles",
    ],
)

# # HFusion Dialect
td_library(
    name = "HFusionTdFiles",
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/AggregatedOpInterface.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionTraits.td",
    ],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ParallelCombiningOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRHFusionOpsIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td",
    tbl_outs = [
        (["-gen-op-decls"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.h.inc"),
        (["-gen-op-defs"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.cpp.inc"),
        (["-gen-typedef-decls", "-typedefs-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsTypes.h.inc"),
        (["-gen-typedef-defs", "-typedefs-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsTypes.cpp.inc"),    
        (["-gen-dialect-decls", "-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsDialect.h.inc"),
        (["-gen-dialect-defs", "-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsDialect.cpp.inc"),        
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":HFusionTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRHFusionEnumsIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td",
    tbl_outs = [
        (["-gen-enum-decls"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.h.inc"),
        (["-gen-enum-defs"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.cpp.inc"),
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":HFusionTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRHFusionAttributesIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td",
    tbl_outs = [
        (["-gen-attrdef-decls", "-attrdefs-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.h.inc"),
        (["-gen-attrdef-defs", "-attrdefs-dialect=hfusion"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.cpp.inc"),
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":HFusionTdFiles",
    ],
)

gentbl_cc_library(
    name = "BiShengIRHFusionStructuredOpsIncGen",
    td_file = "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td",
    tbl_outs = [
        (["-gen-op-decls"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.h.inc"),
        (["-gen-op-defs"], "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"),
    ],
    includes = ["./AscendNPU-IR/bishengir/include"],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    deps = [
        ":HFusionTdFiles",
    ],
)

cc_library(
    name = "mlir-headers-hfusion",
    hdrs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusion.h",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionImpl.h",
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/AggregatedOpInterface.h",
        # Add missing header files
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Utils/Utils.h",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/Utils/Util.h",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yamlgen.cpp.inc",
    ],
    srcs = [
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsTypes.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsTypes.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsDialect.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOpsDialect.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.cpp.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.h.inc",
        "AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.cpp.inc",
        "AscendNPU-IR/bishengir/lib/Dialect/HFusion/IR/HFusionDialect.cpp",
        "AscendNPU-IR/bishengir/lib/Dialect/HFusion/IR/HFusionOps.cpp",
        "AscendNPU-IR/bishengir/include/bishengir/Interfaces/AggregatedOpInterface.h.inc",
    ],
    defines = ["BISHENGIR_BUILD_STANDALONE_IR_ONLY=1"],
    deps = [
        ":MLIRBiShengIREnumsIncGen",
        ":BiShengIRHFusionOpsIncGen",
        ":BiShengIRHFusionEnumsIncGen",
        ":BiShengIRHFusionAttributesIncGen",
        ":BiShengIRHFusionStructuredOpsIncGen",
        ":mlir-headers-symbol",
        ":mlir-headers-mathExt",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MathDialect",
    ],
)
