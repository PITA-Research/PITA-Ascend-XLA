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

#include "xla/xla_npu/service/transform/npu_scalar_fma_compensation.h"

#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace {

bool IsF32Scalar(const HloInstruction* instr) {
  return instr->shape().element_type() == F32 &&
         ShapeUtil::IsScalar(instr->shape());
}

bool IsScalarF32Multiply(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kMultiply && IsF32Scalar(instr);
}

HloInstruction* AddBinary(HloComputation* computation, HloOpcode opcode,
                          HloInstruction* lhs, HloInstruction* rhs) {
  return computation->AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), opcode, lhs, rhs));
}

HloInstruction* AddScalarF32Constant(HloComputation* computation,
                                     float value) {
  return computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(value)));
}

HloInstruction* AddSplitHigh(HloComputation* computation,
                             HloInstruction* splitter,
                             HloInstruction* value) {
  HloInstruction* tmp =
      AddBinary(computation, HloOpcode::kMultiply, splitter, value);
  return AddBinary(
      computation, HloOpcode::kSubtract, tmp,
      AddBinary(computation, HloOpcode::kSubtract, tmp, value));
}

HloInstruction* AddCompensatedMultiplyAdd(HloInstruction* add) {
  HloComputation* computation = add->parent();
  HloInstruction* mul = add->mutable_operand(0);
  HloInstruction* other = add->mutable_operand(1);
  if (!IsScalarF32Multiply(mul)) {
    std::swap(mul, other);
  }

  HloInstruction* lhs = mul->mutable_operand(0);
  HloInstruction* rhs = mul->mutable_operand(1);
  HloInstruction* splitter = AddScalarF32Constant(computation, 4097.0f);

  HloInstruction* lhs_hi = AddSplitHigh(computation, splitter, lhs);
  HloInstruction* lhs_lo =
      AddBinary(computation, HloOpcode::kSubtract, lhs, lhs_hi);
  HloInstruction* rhs_hi = AddSplitHigh(computation, splitter, rhs);
  HloInstruction* rhs_lo =
      AddBinary(computation, HloOpcode::kSubtract, rhs, rhs_hi);

  HloInstruction* rounded =
      AddBinary(computation, HloOpcode::kAdd, mul, other);
  HloInstruction* err = AddBinary(
      computation, HloOpcode::kSubtract,
      AddBinary(computation, HloOpcode::kMultiply, lhs_hi, rhs_hi), mul);
  err = AddBinary(computation, HloOpcode::kAdd, err,
                  AddBinary(computation, HloOpcode::kMultiply, lhs_hi,
                            rhs_lo));
  err = AddBinary(computation, HloOpcode::kAdd, err,
                  AddBinary(computation, HloOpcode::kMultiply, lhs_lo,
                            rhs_hi));
  err = AddBinary(computation, HloOpcode::kAdd, err,
                  AddBinary(computation, HloOpcode::kMultiply, lhs_lo,
                            rhs_lo));

  return AddBinary(computation, HloOpcode::kAdd, rounded, err);
}

bool ShouldRewrite(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kAdd || !IsF32Scalar(instr)) {
    return false;
  }
  return IsScalarF32Multiply(instr->operand(0)) ||
         IsScalarF32Multiply(instr->operand(1));
}

}  // namespace

absl::StatusOr<bool> NpuScalarFmaCompensation::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    std::vector<HloInstruction*> candidates;
    for (HloInstruction* instr : computation->instructions()) {
      if (ShouldRewrite(instr)) {
        candidates.push_back(instr);
      }
    }

    for (HloInstruction* instr : candidates) {
      HloInstruction* result = AddCompensatedMultiplyAdd(instr);
      NPU_LOG_FORMAT_PRINT(
          INFO,
          "Rewrote scalar f32 multiply-add %s with product residual "
          "compensation",
          std::string(instr->name()).c_str());
      TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, result));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
