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

#include "xla/xla_npu/service/transform/npu_while_body_copy_insertion.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::StatusOr<bool> NpuWhileBodyCopyInsertion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Collect all while body calculations
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) continue;

      HloComputation* body = instr->while_body();
      HloInstruction* body_param = body->parameter_instruction(0);

      // Check whether the root of body directly/indirectly uses body_param
      // If the buffer of body root and body param may aliasing,
      // Insert kCopy between body param and its direct consumer to break aliasing
      bool body_modified = false;
      std::vector<HloInstruction*> users_to_redirect;
      for (HloInstruction* user : body_param->users()) {
        // kCopy / kBitcast / kGetTupleElement does not require copy insertion
        if (user->opcode() == HloOpcode::kCopy ||
            user->opcode() == HloOpcode::kBitcast ||
            user->opcode() == HloOpcode::kGetTupleElement) {
          continue;
        }
        users_to_redirect.push_back(user);
      }

      if (users_to_redirect.empty()) continue;

      // Insert a kCopy for each user that needs to be redirected
      for (HloInstruction* user : users_to_redirect) {
        HloInstruction* copy_instr =
            body->AddInstruction(HloInstruction::CreateUnary(
                body_param->shape(), HloOpcode::kCopy, body_param));

        // Replace body_param in user with copy
        for (int64_t i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == body_param) {
            TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, copy_instr));
          }
        }

        NPU_LOG_FORMAT_PRINT(
            WARN,
            "Inserted kCopy in while body '%s': param[%s] -> copy[%s] -> %s[%s]",
            std::string(body->name()).c_str(),
            std::string(body_param->name()).c_str(),
            std::string(copy_instr->name()).c_str(),
            HloOpcodeString(user->opcode()),
            std::string(user->name()).c_str());
        body_modified = true;
      }

      if (body_modified) {
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
