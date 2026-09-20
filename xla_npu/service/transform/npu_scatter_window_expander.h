/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SCATTER_WINDOW_EXPANDER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SCATTER_WINDOW_EXPANDER_H_

#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {
// Normalize every single-output scatter-add to bounded rank-1 point updates.
// Run after BatchedGatherScatterNormalizer; interpret window, inserted and
// index-grid dimensions uniformly, including scalar and length-one windows.
// Invalid windows go to a discarded sink element. The generated scatter is
// marked to make the pass idempotent; input shape is never a bypass condition.
class NpuScatterWindowExpander : public HloModulePass {
 public:
  absl::string_view name() const override { return "npu-scatter-window-expander"; }
 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};
}  // namespace xla
#endif
