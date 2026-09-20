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

#include "xla/xla_npu/service/transform/npu_scatter_copy_insertion.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::StatusOr<bool> NpuScatterCopyInsertion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads)  {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // Collect scatter instructions first; inserting copies during iteration
    // would invalidate the post-order list.
    std::vector<HloInstruction*> scatter_instrs;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kScatter) {
        scatter_instrs.push_back(instr);
      }
    }

    for (HloInstruction* instr : scatter_instrs) {
      HloInstruction* data_operand = instr->mutable_operand(0);

      // Insert copy of data_operand before scatter.  AddInstruction places
      // the copy after its operands, so the copy is inserted between
      // data_operand and scatter — exactly where it belongs.
      HloInstruction* copy = computation->AddInstruction(
    HloInstruction::CreateUnary(
        data_operand->shape(),
        HloOpcode::kCopy,
        data_operand));

      TF_RETURN_IF_ERROR(instr->ReplaceOperandWith(0, copy));

      std::string data_operand_name = std::string(data_operand->name());
std::string data_operand_opcode =
    std::string(HloOpcodeString(data_operand->opcode()));
std::string scatter_name = std::string(instr->name());

      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
