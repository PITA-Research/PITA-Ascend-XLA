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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSUPPORTED_FUSION_DEFUSER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSUPPORTED_FUSION_DEFUSER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Disassemble fusions that Autofuse is not suitable for processing (such as transpose-only, constant-only,
// or graphs containing skip-fusion ops).
// The upstream XLA pass may generate fusion such as fused_dot.*, and only kTranspose remains after DCE.
// Or kBroadcast/kConstant and other trivial diagrams, which may also be included in sign/i32 divide
// Wait for the op of ACLNN to be used. Dot/Scatter/Reduce/Cholesky/While hero fusion is also here to unpack the normal HLO.
class NpuUnsupportedFusionDefuser : public HloModulePass {
 public:
  NpuUnsupportedFusionDefuser() = default;

  absl::string_view name() const override {
    return "npu-unsupported-fusion-defuser";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSUPPORTED_FUSION_DEFUSER_H_
