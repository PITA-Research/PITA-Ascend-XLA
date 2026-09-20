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

#ifndef XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_
#define XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_npu/service/kernel/npu_compile_result.h"

namespace xla {

// Configuration for AscIR backend compilation
struct AscirBackendOptions {
  // Output directory for generated files
  std::string output_dir = "./ascir_output";

  // Whether to preserve intermediate files
  bool preserve_intermediates = true;

  // SOC version (e.g., "Ascend910B1")
  // Can be overridden by SOC_VERSION environment variable
  std::string soc_version = []() {
    const char* env_soc_version = std::getenv("SOC_VERSION");
    return env_soc_version ? std::string(env_soc_version)
                           : std::string("Ascend910B1");
  }();
};

// Result of compiling a single HLO computation
struct SingleComputationResult {
  // Computation name
  std::string computation_name;

  // Path to generated host file
  std::string host_file;

  // Path to generated device file
  std::string device_file;

  // Path to compiled device kernel and tiling so (.so)
  std::string device_binary_file;

  // Path to generated tiling data file
  std::string tiling_data_file;

  // Path to generated ascir protobuf text file
  std::string ascir_file;

  // Graph name extracted from ascir text
  std::string graph_name;

  // Tiling data bytes
  std::vector<uint8_t> tiling_data;
};

// AscIR Backend Interface
// This class provides an interface to compile HLO modules to AscIR kernels
// through the following pipeline
// 1. HLO -> AFIR (using ConvertHloToAfir)
// 2. AFIR -> AscIR (using cpp interface)
// 3. AscIR -> Kernel (using Ascend runtime)
class AscirBackend {
 public:
  AscirBackend();
  ~AscirBackend();

  // Compile an HLO module to AscIR kernel
  // Args: computations: Map of computation name to HloComputation pointer.
  //                 Key "main" represents the entry computation.
  //                 Other keys (e.g., "region_0.1") represent sub-computations
  //                 used by instructions like reduce (to_apply).
  //       options: Compilation options
  // Returns: NpuCompileResult containing paths to generated files
  absl::StatusOr<NpuCompileResult> Compile(
      const std::unordered_map<std::string, const HloComputation*>&
          computations,
      const AscirBackendOptions& options = AscirBackendOptions());

 private:
  // Step 1: Convert HLO to AFIR MLIR module (in-memory)
  // Returns a vector of AFIR strings, one per computation
  absl::StatusOr<std::vector<std::string>> ConvertHloToAfirModule(
      const std::unordered_map<std::string, const HloComputation*>&
          computations,
      const AscirBackendOptions& options);

  // Step 2: Convert AFIR to AscIR
  absl::StatusOr<std::string> ConvertAfirToAscir(
      const std::string& afir_string, const AscirBackendOptions& options);

  // Step 3: Call Ascend runtime to generate kernel and tiling for single computation
  absl::StatusOr<SingleComputationResult> GenerateKernel(
      const std::string& ascir_text, const std::string& computation_name,
      const AscirBackendOptions& options);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_ASCIR_BACKEND_H_
