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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_DOT_CONTRACTING_DIMS_NORMALIZER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_DOT_CONTRACTING_DIMS_NORMALIZER_H_

#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {

// Canonicalize dot_general dimension ordering for the NPU runtime.
//
// The NPU runtime (aclnnMatmul) uses C-order strides and assumes operands
// are in canonical layout: lhs [batch, M, K], rhs [batch, K, N].
// When JAX's VJP generates non-standard contracting dims (e.g.,
// lhs_contracting={0} or rhs_contracting={1}), the physical data layout
// doesn't match, producing incorrect results.
//
// This pass inserts kTranspose instructions to reorder dimensions so that:
//   lhs: [batch..., keep(M)..., contract(K)...]
//   rhs: [batch..., contract(K)..., keep(N)...]
//   batch dims are always leading.
//
// Only inserts transposes when the current ordering is not already canonical.
// Non-C-order constant operands are physically relaid out in private C-order
// literals, including canonical dots whose runtime view adds singleton axes or
// flattens batch/contracting dimensions. Other users retain the original literal.
// It also normalizes every kTranspose and kDot result to C-order because the
// ACLNN runtime materializes these outputs as contiguous C-order buffers.
// Tuple/GTE aliases inherit their operands' updated layouts, including nested
// tuple results, so output transfers do not reinterpret C-order buffers.
class NpuDotContractingDimsNormalizer : public HloModulePass {
 public:
  NpuDotContractingDimsNormalizer() = default;
  ~NpuDotContractingDimsNormalizer() override = default;

  absl::string_view name() const override {
    return "npu-dot-contracting-dims-normalizer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  absl::StatusOr<bool> NormalizeDot(HloDotInstruction* dot);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_DOT_CONTRACTING_DIMS_NORMALIZER_H_
