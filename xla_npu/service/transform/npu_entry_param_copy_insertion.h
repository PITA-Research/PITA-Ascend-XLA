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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_ENTRY_PARAM_COPY_INSERTION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_ENTRY_PARAM_COPY_INSERTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Inserts kCopy after entry computation parameters to prevent buffer aliasing
// between input parameters and computed outputs.
//
// In vmap-batched computations, the large HLO graph can cause the buffer
// assignment to alias input parameter buffers with intermediate computation
// temporaries. When scatter/dynamic_update_slice operations modify these
// aliased buffers, they corrupt data that is still needed by downstream
// operations (e.g., the qfrc_actuator computation corrupting qvel or ctrl).
//
// This pass ensures that entry parameters get their own buffer by inserting
// a kCopy immediately after the parameter, so that all computation reads
// from the copy's buffer rather than the original parameter buffer.
class NpuEntryParamCopyInsertion : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-entry-param-copy-insertion";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_ENTRY_PARAM_COPY_INSERTION_H_
