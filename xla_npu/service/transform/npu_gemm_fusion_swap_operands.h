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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_

#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/service/npu_backend_config.pb.h"

namespace xla {

// Memory layout format specified by NPU
enum class NpuLayoutFormat {
  kDefault,     // Default layout (Row-major)
  kNC1HWC0,     // NPU dedicated convolution layout
  kFRACTAL_NZ,  // Efficient layout for NPU matrix calculations
  kNDHWC,       // 3D convolution layout
  kNCHW,        // Channel first layout
};

// Swap GEMM operands for better NPU layout/tile performance
// This PASS analyzes GEMM operations and may swap operand order to optimize NPU memory layout.
// The NPU has a specific access mode for the FRACTAL_NZ layout, and swapping operands can:
// 1. Make better use of the NPU's block computing capabilities
// 2. Reduce memory access latency
// 3. Improve cache utilization
// Conversion example: Before: dot(A, B) where A.shape = [M, K], B.shape = [K, N]
// After:  dot(transpose(B), transpose(A)) with adjusted dimension mapping
// Note: Swapping operands requires adjusting the dimension mapping in dot_dimension_numbers accordingly.
class NpuGemmFusionSwapOperands : public HloModulePass {
 public:
  NpuGemmFusionSwapOperands() = default;
  ~NpuGemmFusionSwapOperands() override = default;

  absl::string_view name() const override {
    return "npu-gemm-fusion-swap-operands";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Run on a single computation
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  // Check if GEMM operands should be swapped
  bool ShouldSwapOperands(const HloDotInstruction* dot);

  // Get the preferred layout format for a given tensor
  NpuLayoutFormat GetPreferredLayoutFormat(const Shape& shape, bool is_lhs);

  // Check if the shape is compatible with a given layout format
  bool IsLayoutCompatible(const Shape& shape, NpuLayoutFormat format);

  // Swap the operands of a GEMM instruction
  absl::StatusOr<bool> SwapGemmOperands(HloDotInstruction* dot);

  // Create a transpose instruction for layout conversion
  absl::StatusOr<HloInstruction*> CreateLayoutTranspose(
      HloInstruction* operand, const std::vector<int64_t>& permutation);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_GEMM_FUSION_SWAP_OPERANDS_H_
