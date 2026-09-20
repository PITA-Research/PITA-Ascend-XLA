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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SCATTER_COPY_INSERTION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SCATTER_COPY_INSERTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Inserts a copy before scatter's data operand (operand 0) to break
// parameter aliasing in buffer assignment.
//
// Scatter writes to its output buffer, which buffer assignment may alias
// with the input parameter. When the output allocation is an entry
// computation parameter, the executor treats it as passthrough and
// copies from the original argument buffer instead of the computed
// buffer, discarding scatter's result.
//
// By inserting copy(x) before scatter(x, indices, updates), the scatter
// aliases with the copy's buffer rather than the parameter, so the
// executor correctly reads the scatter-modified output.
class NpuScatterCopyInsertion : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-scatter-copy-insertion";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SCATTER_COPY_INSERTION_H_
