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

#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"
#include "xla/xla_npu/common/npu_config.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/xla_data.pb.h"

ABSL_FLAG(std::string, hlo_text, "", "HLO text to compile");
ABSL_FLAG(std::string, hlo_file, "", "HLO file to compile");
ABSL_FLAG(std::string, output_dir, "./test_ascir_output", "Output directory");

// Simple HLO module for testing
const char kSimpleHLO[] = R"(
HloModule SimpleAdd

ENTRY main {
  %param0 = f32[4] parameter(0)
  %param1 = f32[4] parameter(1)
  %add = f32[4] add(%param0, %param1)
  %sub = f32[4] subtract(%param0, %param1)
  %mul = f32[4] multiply(%add, %sub)
  ROOT %result = f32[4] add(%add, %mul)
}
)";

// More complex HLO module for testing (from test_simple.hlo)
const char kComplexHLO[] = R"(
HloModule TensorFlowMatMul

ENTRY %main {
  %arg0 = f32[4,4] parameter(0)
  %arg1 = f32[4,4] parameter(1)
  %constant = f32[] constant(0.01)
  %broadcast = f32[4,4] broadcast(%constant), dimensions={}
  %multiply.1 = f32[4,4] multiply(%arg1, %broadcast)
  %dot = f32[4,4] dot(%arg0, %multiply.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %dot.1 = f32[4,4] dot(%dot, %arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

absl::StatusOr<std::unique_ptr<xla::HloModule>> ParseHloString(
    absl::string_view hlo_text,
    const std::string& module_name = "test_module") {
  xla::HloModuleConfig config;
  xla::HloParserOptions options;

  return xla::ParseAndReturnUnverifiedModule(hlo_text, config, options);
}

/*
Parse HLO text to HloModule.

This function parses HLO text (which may contain multiple computations)
and returns the HloModule along with a pointer to the entry computation.

The HloModule contains ALL computations from the HLO text:
- Entry computation (accessible via module->entry_computation())
- Sub-computations (e.g., reduce computations, fusion computations)
  All computations can be accessed via module->computations()

Input format example:
1. Full HLO module with multiple computations:
   HloModule MyModule

   ENTRY %main {
     %param0 = f32[4] parameter(0)
     ROOT %add = f32[4] add(%param0, %param0)
   }

2. Pure computation body (will be wrapped as entry computation):
   {
     %param0 = f32[4] parameter(0)
     %param1 = f32[4] parameter(1)
     ROOT %add = f32[4] add(%param0, %param1)
   }

Returns: pair<module, entry_computation*>
Note: The module must maintain its lifecycle until compile is completed.
 */
absl::StatusOr<std::pair<std::unique_ptr<xla::HloModule>, xla::HloComputation*>>
ParseHloComputationString(
    absl::string_view computation_text,
    const std::string& computation_name = "parsed_computation") {
  xla::HloModuleConfig config;
  xla::HloParserOptions options;

  // Process the computation text to add ENTRY prefix to fusion computation
  std::string processed_text = std::string(computation_text);

  // Find lines starting with "%fused" and add "ENTRY" prefix
  // This handles fusion computations like %fused_reduce, %fused_computation, etc.
  size_t pos = 0;
  while ((pos = processed_text.find("%fused", pos)) != std::string::npos) {
    // Check if this is at the start of a line (or after whitespace)
    if (pos == 0 || processed_text[pos - 1] == '\n') {
      // Insert "ENTRY " before "%fused"
      processed_text.insert(pos, "ENTRY ");
      pos += 7;  // Skip "ENTRY %fused"
    } else {
      pos += 1;  // Move forward
    }
  }

  // Wrap as HloModule
  std::string wrapped_hlo =
      absl::StrCat("HloModule temp_module\n\n", processed_text);

  TF_ASSIGN_OR_RETURN(auto module, xla::ParseAndReturnUnverifiedModule(
                                       wrapped_hlo, config, options));

  // Find the fusion computation (starting with "fused") to use as entry
  // If no fusion computation found, use the default entry computation
  xla::HloComputation* entry_computation = nullptr;

  for (const auto* computation : module->computations()) {
    std::string comp_name = std::string(computation->name());
    if (comp_name.find("fused") == 0) {
      entry_computation = const_cast<xla::HloComputation*>(computation);
      std::cout << "  Found fusion computation as entry: " << comp_name
                << std::endl;
      break;
    }
  }

  // Fallback to default entry if no fusion computation found
  if (entry_computation == nullptr) {
    entry_computation = module->entry_computation();
    std::cout << "  Using default entry computation: "
              << entry_computation->name() << std::endl;
  }

  return std::make_pair(std::move(module), entry_computation);
}

absl::Status RunTest(absl::string_view hlo_text,
                     const std::string& output_dir) {
  std::cout << "=== HLO to AscIR Backend Test ===" << std::endl;
  std::cout << "\n[Step 1] Parsing HLO text..." << std::endl;

  // Parse HLO text to HloModule and HloComputation
  // Note: The module must maintain its lifecycle until compile is completed, otherwise, a coredump will occur.
  TF_ASSIGN_OR_RETURN(auto parsed, ParseHloComputationString(hlo_text));
  auto& hlo_module = parsed.first;
  xla::HloComputation* entry_computation = parsed.second;

  std::cout << " HLO module parsed: " << entry_computation->name() << std::endl;
  std::cout << "     Entry computation instructions: "
            << entry_computation->instruction_count() << std::endl;

  // Collect all computations from the module (entry + sub-computations)
  std::unordered_map<std::string, const xla::HloComputation*> computations;

  int computation_count = 0;
  for (const auto* computation : hlo_module->computations()) {
    std::string key = computation->IsEntryComputation()
                          ? "main"
                          : std::string(computation->name());
    computations[key] = computation;
    computation_count++;
  }

  std::cout << "**** Parse hlo successfully! ****" << std::endl;
  std::cout << "     Total computations in module: " << computation_count
            << std::endl;
  for (const auto& [key, comp] : computations) {
    std::cout << "       - " << key << " (" << comp->instruction_count()
              << " instructions)" << std::endl;
  }

  std::cout << "\n[Step 2] Configuring AscIR Backend..." << std::endl;

  xla::AscirBackendOptions options;
  options.output_dir = output_dir;
  options.preserve_intermediates = true;
  // Read soc_version from environment variable SOC_VERSION, default to "Ascend910B1" if not set
  const char* env_soc_version = std::getenv("SOC_VERSION");
  options.soc_version =
      env_soc_version ? std::string(env_soc_version) : "Ascend910B1";

  std::cout << "  Output directory: " << options.output_dir << std::endl;
  std::cout << "  SOC version: " << options.soc_version << std::endl;

  std::cout << "\n[Step 3] Compiling HLO to AscIR Kernel..." << std::endl;

  xla::AscirBackend backend;
  TF_ASSIGN_OR_RETURN(auto result, backend.Compile(computations, options));

  std::cout << "**** Compilation successfully! ****" << std::endl;

  std::cout << "\n[Step 4] Generated Artifacts:" << std::endl;
  std::cout << "  Total artifacts compiled: " << result.artifact_infos.size()
            << std::endl;

  int idx = 0;
  for (const auto& [artifact_name, artifact_info] : result.artifact_infos) {
    std::cout << "\n  Artifact " << (++idx) << ": " << artifact_name
              << std::endl;
    std::cout << "    Name:      " << artifact_info.name << std::endl;
    std::cout << "    Directory: " << artifact_info.dir << std::endl;
  }

  // Verify directories exist
  std::cout << "\n[Step 5] Verifying generated directories..." << std::endl;

  int total_dirs = 0;
  int existing_dirs = 0;

  for (const auto& [artifact_name, artifact_info] : result.artifact_infos) {
    total_dirs++;
    std::string check_cmd = "test -d " + artifact_info.dir;
    int dir_exists = (system(check_cmd.c_str()) == 0);
    if (dir_exists) {
      std::cout << "  Directory exists for " << artifact_name << " ("
                << artifact_info.dir << ")" << std::endl;
      existing_dirs++;
    } else {
      std::cout << "  Directory missing for " << artifact_name << " ("
                << artifact_info.dir << ")" << std::endl;
    }
  }

  std::cout << "\n  Directories verified: " << existing_dirs << "/"
            << total_dirs << std::endl;
  std::cout << "\n=== Test Completed Successfully ===" << std::endl;

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Initialize NPU config
  xla::NpuConfig::GetInstance().Init();

  std::cout << "========================================" << std::endl;
  std::cout << "HLO to AscIR Backend Test" << std::endl;
  std::cout << "========================================" << std::endl;

  std::string hlo_text;
  std::string source_desc;

  // Determine HLO source
  if (!absl::GetFlag(FLAGS_hlo_file).empty()) {
    // Read from file
    std::string hlo_file = absl::GetFlag(FLAGS_hlo_file);
    std::cout << "Reading HLO from file: " << hlo_file << std::endl;

    std::ifstream file(hlo_file);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file: " << hlo_file << std::endl;
      return EXIT_FAILURE;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    hlo_text = buffer.str();
    source_desc = absl::StrCat("file: ", hlo_file);
  } else if (!absl::GetFlag(FLAGS_hlo_text).empty()) {
    // Use provided HLO text
    hlo_text = absl::GetFlag(FLAGS_hlo_text);
    source_desc = "command line text";
  } else {
    // Use default test HLO
    std::cout << "Using default test HLO" << std::endl;
    hlo_text = kComplexHLO;
    source_desc = "built-in test";
  }

  std::cout << "HLO source: " << source_desc << std::endl;
  std::cout << "HLO text length: " << hlo_text.length() << " bytes"
            << std::endl;
  std::cout << std::endl;

  // Run test
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);

  auto status = RunTest(hlo_text, output_dir);
  if (!status.ok()) {
    std::cerr << "\n❌ Test Failed: " << status.message() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
