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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ir/mair_ops.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "transforms/mair_passes.h"

using namespace mlir::stablehlo;
using namespace mlir::air;
namespace cl = llvm::cl;

// Enter stablehlo file
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input stablehlo file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpMair,
};
}  // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMair, "mair",
                          "output the MLIR dump after mair lowering")));

// Read.mlir file or string and convert to mlir::ModuleOp object
static int loadMLIR(mlir::MLIRContext& context,
                    mlir::OwningOpRef<mlir::ModuleOp>& module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can not load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int loadAndProcessMLIR(mlir::MLIRContext& context,
                              mlir::OwningOpRef<mlir::ModuleOp>& module) {
  if (int error = loadMLIR(context, module)) return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToMair = emitAction >= Action::DumpMair;

  if (isLoweringToMair) {
    // Add Stablehlo to Mair dialect conversion pass
    pm.addPass(createLegalizeStablehloToGePass());
  }

  if (failed(pm.run(*module))) return 4;
  return 0;
}

int main(int argc, char** argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ge compiler\n");

  mlir::DialectRegistry registry;
  // mlir::func::registerAllExtensions(registry);
  // mlir::LLVM::registerInlinerInterface(registry);
  registry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  // Load stablehlo and mair Dialect in this MLIR Context.
  context.getOrLoadDialect<StablehloDialect>();
  context.getOrLoadDialect<AirDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  bool isOutputingMLIR = emitAction <= DumpMair;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
