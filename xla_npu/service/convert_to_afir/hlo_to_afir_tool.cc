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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"

ABSL_FLAG(std::string, input, "", "Input HLO file path");
ABSL_FLAG(std::string, output, "", "Output file path (default: stdout)");
ABSL_FLAG(std::string, computation_text, "",
          "HloComputation text to convert directly (overrides --input)");
ABSL_FLAG(bool, preserve_names, true, "Preserve HLO instruction names in MLIR");
ABSL_FLAG(bool, verify, true, "Verify generated MLIR module");
ABSL_FLAG(bool, emit_afir, false,
          "Emit AFIR dialect operations (experimental)");

namespace xla {

namespace {

// Read HLO module from file
absl::StatusOr<std::unique_ptr<HloModule>> LoadHloModule(
    absl::string_view filename) {
  return LoadModuleFromFile(std::string(filename));
}

// Write MLIR module to file
absl::Status WriteMlirModule(mlir::ModuleOp module,
                             absl::string_view filename) {
  if (filename.empty()) {
    // Write to stdout
    module->dump();
    return absl::OkStatus();
  }

  std::string output_file(filename);
  std::ofstream ofs(output_file);
  if (!ofs) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open output file: ", output_file));
  }

  // Print MLIR to string
  std::string mlir_string;
  llvm::raw_string_ostream stream(mlir_string);
  module->print(stream);

  ofs << mlir_string;
  ofs.close();

  return absl::OkStatus();
}

// Run the conversion tool
absl::Status Run(const HloToAfirOptions& options) {
  // Check if computation_text is provided
  std::string computation_text = absl::GetFlag(FLAGS_computation_text);

  // Create MLIR context and load required dialects
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::afir::AFIRDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;

  // Keep the module alive if we're parsing computation_text
  std::unique_ptr<HloModule> temp_module;

  if (!computation_text.empty()) {  // convert single hlo computation
    // Parse computation from text
    std::cout << "Parsing HloComputation from text..." << std::endl;

    // Wrap the computation text in a simple HloModule for parsing
    std::string wrapped_text =
        absl::StrCat("HloModule TempModule\n\n", computation_text);

    // Create a temporary module and keep it alive
    HloModuleConfig config;
    HloParserOptions parser_options;
    TF_ASSIGN_OR_RETURN(temp_module, ParseAndReturnUnverifiedModule(
                                         wrapped_text, config, parser_options));

    const HloComputation* computation = temp_module->entry_computation();

    std::cout << "Parsed computation: " << computation->name() << std::endl;
    std::cout << "  Instructions: " << computation->instruction_count()
              << std::endl;

    // Convert HloComputation to AFIR
    TF_ASSIGN_OR_RETURN(mlir_module, ConvertHloComputationToAfir(
                                         *computation, &context, options));
  } else {  // convert entire hlo module
    // Get input file
    std::string input_file = absl::GetFlag(FLAGS_input);
    if (input_file.empty()) {
      return absl::InvalidArgumentError(
          "Either --input or --computation_text must be specified");
    }

    // Load HLO module
    TF_ASSIGN_OR_RETURN(auto hlo_module, LoadHloModule(input_file));

    std::cout << "Loaded HLO module: " << hlo_module->name() << std::endl;
    std::cout << "  Computations: " << hlo_module->computation_count()
              << std::endl;
    std::cout << "  Entry Computation Instructions: "
              << hlo_module->entry_computation()->instruction_count()
              << std::endl;

    // Convert to AFIR
    TF_ASSIGN_OR_RETURN(mlir_module,
                        ConvertHloToAfir(*hlo_module, &context, options));
  }

  std::cout << "Conversion successful!" << std::endl;

  // Write output
  std::string output_file = absl::GetFlag(FLAGS_output);
  TF_RETURN_IF_ERROR(WriteMlirModule(*mlir_module, output_file));

  if (!output_file.empty()) {
    std::cout << "Output written to: " << output_file << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace

}  // namespace xla

int main(int argc, char** argv) {
  // Parse command-line flags
  absl::ParseCommandLine(argc, argv);

  // Set up conversion options
  xla::HloToAfirOptions options;
  options.preserve_names = absl::GetFlag(FLAGS_preserve_names);
  options.verify_module = absl::GetFlag(FLAGS_verify);
  options.emit_afir_dialect = absl::GetFlag(FLAGS_emit_afir);

  // Run conversion
  absl::Status status = xla::Run(options);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
