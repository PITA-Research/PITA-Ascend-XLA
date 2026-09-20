/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */

#include "xla/xla_npu/service/transform/npu_fma_addcmul_rewriter.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace {
constexpr char kFmaAddcmulTarget[] = "__npu_fma_addcmul";

bool IsF32Array(const HloInstruction* instr) {
  return instr->shape().IsArray() && instr->shape().element_type() == F32;
}

// NpuTool::CreateAclTensorPtr describes every ACLNN tensor as dense C-order
// ND. Passing a non-C-order XLA buffer through FmaAddcmul would therefore make
// the Optry kernel interpret the allocation with the wrong strides. Leave
// such expressions on the normal lowering path until this custom call has
// explicit layout-aware descriptors.
bool IsCOrderF32Array(const HloInstruction* instr) {
  return IsF32Array(instr) && instr->shape().has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout());
}

bool GetF32ScalarConstant(const HloInstruction* instr, float* value) {
  const HloInstruction* constant = instr;
  if (instr->opcode() == HloOpcode::kBroadcast && instr->operand_count() == 1) {
    constant = instr->operand(0);
  }
  if (constant->opcode() != HloOpcode::kConstant ||
      constant->shape().element_type() != F32 ||
      !ShapeUtil::IsScalar(constant->shape())) {
    return false;
  }
  *value = constant->literal().Get<float>({});
  return std::isfinite(*value);
}

struct Match {
  HloInstruction* self;
  HloInstruction* tensor1;
  HloInstruction* tensor2;
  float value;
};

bool MatchFmaAddcmul(HloInstruction* add, Match* match) {
  if (add->opcode() != HloOpcode::kAdd || !IsCOrderF32Array(add)) return false;
  HloInstruction* product = add->mutable_operand(0);
  HloInstruction* self = add->mutable_operand(1);
  if (product->opcode() != HloOpcode::kMultiply) std::swap(product, self);
  if (product->opcode() != HloOpcode::kMultiply ||
      !IsCOrderF32Array(product) || !IsCOrderF32Array(self) ||
      product->shape() != add->shape() ||
      self->shape() != add->shape()) return false;

  HloInstruction* tensor1 = product->mutable_operand(0);
  HloInstruction* tensor2 = product->mutable_operand(1);
  float value = 1.0f;
  for (int i = 0; i != 2; ++i) {
    HloInstruction* scaled = product->mutable_operand(i);
    HloInstruction* other = product->mutable_operand(1 - i);
    if (scaled->opcode() != HloOpcode::kMultiply) continue;
    for (int j = 0; j != 2; ++j) {
      float candidate = 0.0f;
      if (GetF32ScalarConstant(scaled->operand(j), &candidate)) {
        tensor1 = scaled->mutable_operand(1 - j);
        tensor2 = other;
        value = candidate;
        goto matched;
      }
    }
  }
matched:
  if (!IsCOrderF32Array(tensor1) || !IsCOrderF32Array(tensor2) ||
      tensor1->shape() != add->shape() || tensor2->shape() != add->shape()) return false;
  *match = {self, tensor1, tensor2, value};
  return true;
}

std::string EncodeValue(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return absl::StrFormat("value_bits=%u", bits);
}
}  // namespace

absl::StatusOr<bool> NpuFmaAddcmulRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // Do not retain Match operands across replacements. In an accumulation
    // chain, a later candidate can use an earlier candidate as `self`. Once
    // the earlier add is replaced, its HloInstruction is destroyed and any
    // cached operand pointer becomes dangling (for example
    // add(add(a * b, c), d * e)). Retain only the candidate instruction, then
    // re-match against the current graph immediately before replacement.
    std::vector<HloInstruction*> candidates;
    for (HloInstruction* instr : computation->instructions()) {
      Match match;
      if (MatchFmaAddcmul(instr, &match)) candidates.push_back(instr);
    }
    for (HloInstruction* add : candidates) {
      Match match;
      if (!MatchFmaAddcmul(add, &match)) continue;
      HloInstruction* custom_call = computation->AddInstruction(
          HloInstruction::CreateCustomCall(
              add->shape(), {match.self, match.tensor1, match.tensor2},
              kFmaAddcmulTarget, EncodeValue(match.value)));
      custom_call->set_metadata(add->metadata());
      NPU_LOG_FORMAT_PRINT(INFO, "Rewriting f32 multiply-add '%s' to aclnnFmaAddcmul",
                           std::string(add->name()).c_str());
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(add, custom_call));
      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
