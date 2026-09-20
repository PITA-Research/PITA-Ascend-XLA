/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_VARIADIC_REDUCE_EXPANDER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_VARIADIC_REDUCE_EXPANDER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// ACLNN's scalar reductions cannot implement a coupled tuple reducer. Expand
// scalar elementwise tuple reducers (including JAX argmin/argmax) into an
// adjacent-pair tree, cloning the actual reducer rather than guessing its kind.
class NpuVariadicReduceExpander : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-variadic-reduce-expander";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_VARIADIC_REDUCE_EXPANDER_H_
