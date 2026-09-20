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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_RENAME_FUSIONS_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_RENAME_FUSIONS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// An HLO pass that gives fusions and fused computations descriptive names.
// The name is based on hero instructions and the fusion kind, i.e.
// Fusions get name "<fusion kind>_<hero instructions>_fusion",
// and fused computations get name "fused_<hero instructions>".
// In the case of multiple roots, the hero instructions in the name are
// underscore-separated and alphabetically sorted.

class NpuRenameFusions : public HloModulePass {
  absl::string_view name() const override { return "npu-rename-fusions"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_RENAME_FUSIONS_H_
