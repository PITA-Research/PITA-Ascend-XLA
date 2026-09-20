/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ASC_PASSES_TD
#define ASC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DeclarePyStruct : Pass<"ascendc-declare-py-struct", "ModuleOp"> {
  let summary = "Insert emitasc.declare_py_struct";
  let constructor = "mlir::ascendc::createDeclarePyStructPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
}

def DefineCubeOnly : Pass<"ascendc-define-cube-only", "ModuleOp"> {
  let summary = "Insert emitc.define if CUBE-ONLY";
  let constructor = "mlir::ascendc::createDefineCubeOnlyPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
}

def DetectKernelType : Pass<"ascendc-detect-kernel-type", "ModuleOp"> {
  let summary = "Check whether the kernel is vector-only or mixed";
  let constructor = "mlir::ascendc::createDetectKernelTypePass()";
}

def EraseSync : Pass<"ascendc-erase-sync", "func::FuncOp"> {
  let summary = "Erase intra-core synchronization operations";
  let constructor = "mlir::ascendc::createEraseSyncPass()";
}

def GenerateBoilerplate : Pass<"ascendc-generate-boilerplate", "ModuleOp"> {
  let summary = "Insert emitc.include and additional boilerplate code";
  let constructor = "mlir::ascendc::createGenerateBoilerplatePass()";
  let dependentDialects = ["emitc::EmitCDialect"];
}

def HoistQueBind : Pass<"ascendc-hoist-que-bind", "func::FuncOp"> {
  let summary = "Hoist TQueBind, TQue, TBuf initialization operations";
  let constructor = "mlir::ascendc::createHoistQueBindPass()";
}

def HoistUBAllocation : Pass<"ascendc-hoist-ub-allocation", "func::FuncOp"> {
  let summary = "Hoist tensor allocations to the function root";
  let constructor = "mlir::ascendc::createHoistUBAllocationPass()";
}

def InputOutputTensor : Pass<"ascendc-input-output-tensor", "func::FuncOp"> {
  let summary = "Set input and output operand for local_tensor_auto";
  let constructor = "mlir::ascendc::createInputOutputTensorPass()";
}

def InsertSync : Pass<"ascendc-insert-sync", "func::FuncOp"> {
  let summary = "Insert intra-core synchronization operations";
  let constructor = "mlir::ascendc::createInsertSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def MaterializeTensor : Pass<"ascendc-materialize-tensor", "func::FuncOp"> {
  let summary = "Insert ascendc.tbuf, ascendc.queue and ascendc.alloca for local_tensor_auto";
  let constructor = "mlir::ascendc::createMaterializeTensorPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
}

def LegalizeKernelArgs : Pass<"ascendc-legalize-kernel-args", "ModuleOp"> {
  let summary = "Attach emitasc.kernel_arg attributes and insert operations";
  let constructor = "mlir::ascendc::createLegalizeKernelArgsPass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect",
    "scf::SCFDialect",
  ];
}

def Noop : Pass<"ascendc-noop", "func::FuncOp"> {
  let summary = "This pass does nothing";
  let constructor = "mlir::ascendc::createNoopPass()";
}

def PrivatizeFunc : Pass<"ascendc-privatize-func", "ModuleOp"> {
  let summary = "Mark functions without ascendc.global attribute as private";
  let constructor = "mlir::ascendc::createPrivatizeFuncPass()";
}

def UnifyPipe : Pass<"ascendc-unify-pipe", "func::FuncOp"> {
  let summary = "Unify pipe opertation";
  let constructor = "mlir::ascendc::createUnifyPipePass()";
}

def VerifySync : Pass<"ascendc-verify-sync", "func::FuncOp"> {
  let summary = "Verify TQue synchronization";
  let constructor = "mlir::ascendc::createVerifySyncPass()";
}

def DetectEnableDebug : Pass<"ascendc-detect-enable-debug", "ModuleOp"> {
  let summary = "Check whether the kernel is using debug utils";
  let constructor = "mlir::ascendc::createDetectEnableDebugPass()";
}

#endif // ASC_PASSES_TD
