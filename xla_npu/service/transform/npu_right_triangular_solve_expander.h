/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_RIGHT_TRIANGULAR_SOLVE_EXPANDER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_RIGHT_TRIANGULAR_SOLVE_EXPANDER_H_
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
// Convert real X op(A) = B to op(A)^T X^T = B^T for the left-only ACLNN API.
class NpuRightTriangularSolveExpander : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-right-triangular-solve-expander";
  }
 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};
}  // namespace xla
#endif
