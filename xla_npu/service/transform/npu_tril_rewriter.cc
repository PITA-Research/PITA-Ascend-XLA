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

#include "xla/xla_npu/service/transform/npu_tril_rewriter.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/transform/npu_tril_utils.h"

namespace xla {

absl::StatusOr<bool> NpuTrilRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // Collect Select instructions matching tril pattern (collect first and then modify to avoid iterator failure)
    struct TrilMatch {
      HloInstruction* select;
      int64_t k;
      HloInstruction* input;
    };
    std::vector<TrilMatch> matches;

    for (HloInstruction* instr : computation->instructions()) {
      int64_t k = 0;
      const HloInstruction* input = nullptr;
      if (tril::MatchTrilSelect(instr, &k, &input)) {
        matches.push_back(
            {instr, k, const_cast<HloInstruction*>(input)});
      }
    }

    for (const auto& match : matches) {
      NPU_LOG_FORMAT_PRINT(
          INFO,
          "Rewriting tril pattern at '%s' (k=%lld) → custom-call aclnnTril",
          std::string(match.select->name()).c_str(), (long long)match.k);

      // Create CustomCall, opaque encoded k value
      Shape output_shape = match.select->shape();
      HloInstruction* custom_call = computation->AddInstruction(
          HloInstruction::CreateCustomCall(
              output_shape, {match.input}, "tril",
              /*opaque=*/absl::StrCat("k=", match.k)));

      custom_call->set_metadata(match.select->metadata());

      TF_RETURN_IF_ERROR(match.select->ReplaceAllUsesWith(custom_call));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(match.select));

      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
