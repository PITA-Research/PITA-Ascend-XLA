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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// FusionDynamicMemcpyRewriter-like pass for NPU.
// Detects fusions that can be emitted using DynamicMemcpyFusion and annotates them accordingly.
class NpuFusionDynamicMemcpyRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-fusion-dynamic-memcpy-rewriter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DYNAMIC_MEMCPY_REWRITER_H_
