/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FMA_ADDCMUL_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FMA_ADDCMUL_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Lowers f32 value * tensor1 * tensor2 + self to aclnnFmaAddcmul.
class NpuFmaAddcmulRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "npu-fma-addcmul-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FMA_ADDCMUL_REWRITER_H_
