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

#include "xla/xla_npu/service/transform/npu_unsigned_to_float_decomposition.h"

#include <cmath>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

absl::StatusOr<bool> NpuUnsignedToFloatDecomposition::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // Collect all convert instructions that need to be decomposed (collect first and then modify to avoid iterator failure)
    std::vector<HloInstruction*> convert_instrs;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kConvert) continue;
      PrimitiveType src_type = instr->operand(0)->shape().element_type();
      PrimitiveType dst_type = instr->shape().element_type();
      // Only handles unsigned integer -> floating point conversions
      if ((src_type == U16 || src_type == U32 || src_type == U64) &&
          (dst_type == F32 || dst_type == F64)) {
        convert_instrs.push_back(instr);
      }
    }

    for (HloInstruction* instr : convert_instrs) {
      HloInstruction* value = instr->mutable_operand(0);
      PrimitiveType src_type = value->shape().element_type();
      PrimitiveType dst_type = instr->shape().element_type();

      // U32 -> F32 不能通过 "bitcast to S32 -> convert -> add 2^32"
      // 实现。S32 -> F32 和随后的加法会在不同 binade 中各舍入一次，
      // 这与原始 U32 -> F32 的单次 round-to-nearest-even 不等价。
      //
      // 将 value 拆为两个最多 16 bit 的非负分量：
      //   value = high * 2^16 + low
      // high/low 到 F32 都是精确转换，乘以 2^16 也精确，因此只有最后
      // 一次 Add 发生 F32 舍入，保留原始 convert 的语义。
      if (src_type == U32 && dst_type == F32) {
        const Shape& value_shape = value->shape();
        const Shape s32_shape =
            ShapeUtil::ChangeElementType(value_shape, S32);
        const Shape f32_shape =
            ShapeUtil::ChangeElementType(value_shape, F32);

        HloInstruction* shift_scalar = computation->AddInstruction(
            HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(16)));
        HloInstruction* shift = computation->AddInstruction(
            HloInstruction::CreateBroadcast(value_shape, shift_scalar, {}));
        HloInstruction* high_u32 = computation->AddInstruction(
            HloInstruction::CreateBinary(value_shape,
                                         HloOpcode::kShiftRightLogical, value,
                                         shift));

        HloInstruction* low_mask_scalar = computation->AddInstruction(
            HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(0xFFFFu)));
        HloInstruction* low_mask = computation->AddInstruction(
            HloInstruction::CreateBroadcast(value_shape, low_mask_scalar,
                                            {}));
        HloInstruction* low_u32 = computation->AddInstruction(
            HloInstruction::CreateBinary(value_shape, HloOpcode::kAnd, value,
                                         low_mask));

        HloInstruction* high_s32 = computation->AddInstruction(
            HloInstruction::CreateBitcastConvert(s32_shape, high_u32));
        HloInstruction* low_s32 = computation->AddInstruction(
            HloInstruction::CreateBitcastConvert(s32_shape, low_u32));
        HloInstruction* high_f32 = computation->AddInstruction(
            HloInstruction::CreateConvert(f32_shape, high_s32));
        HloInstruction* low_f32 = computation->AddInstruction(
            HloInstruction::CreateConvert(f32_shape, low_s32));

        HloInstruction* scale_scalar = computation->AddInstruction(
            HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<float>(65536.0f)));
        HloInstruction* scale = computation->AddInstruction(
            HloInstruction::CreateBroadcast(f32_shape, scale_scalar, {}));
        HloInstruction* scaled_high = computation->AddInstruction(
            HloInstruction::CreateBinary(f32_shape, HloOpcode::kMultiply,
                                         high_f32, scale));
        HloInstruction* result = computation->AddInstruction(
            HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd,
                                         scaled_high, low_f32));

        NPU_LOG_FORMAT_PRINT(
            INFO,
            "Decomposed convert(U32, F32) %s into exact 16-bit halves",
            std::string(instr->name()).c_str());
        TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(result));
        changed = true;
        continue;
      }

      // Determine signed correspondence type and bit width
      PrimitiveType signed_type;
      int bit_width;
      switch (src_type) {
        case U16:
          signed_type = S16;
          bit_width = 16;
          break;
        case U32:
          signed_type = S32;
          bit_width = 32;
          break;
        case U64:
          signed_type = S64;
          bit_width = 64;
          break;
        default:
          continue;
      }

      const Shape& value_shape = value->shape();
      Shape signed_shape =
          ShapeUtil::ChangeElementType(value_shape, signed_type);
      Shape float_shape =
          ShapeUtil::ChangeElementType(value_shape, dst_type);

      // --- Decomposition begins ---

      // 1. value_s = BitcastConvert(value, signed_type)
      //    Reinterpret unsigned bit patterns as signed (zero overhead, no changes to data in memory)
      HloInstruction* value_s = computation->AddInstruction(
          HloInstruction::CreateBitcastConvert(signed_shape, value));

      // 2. value_f = Convert(value_s, dst_type)
      //    SIToFP: correctly generates positive floating point numbers for values < 2^(bit_width-1),
      //    Produces negative floating point numbers for values ≥ 2^(bit_width-1) (subject to subsequent correction)
      HloInstruction* value_f = computation->AddInstruction(
          HloInstruction::CreateConvert(float_shape, value_s));

      // 3. zero = 0.0 is broadcast to the shape of value
      Literal zero_lit = (dst_type == F64)
                             ? LiteralUtil::CreateR0<double>(0.0)
                             : LiteralUtil::CreateR0<float>(0.0f);
      HloInstruction* zero_scalar = computation->AddInstruction(
          HloInstruction::CreateConstant(std::move(zero_lit)));
      HloInstruction* zero = computation->AddInstruction(
          HloInstruction::CreateBroadcast(float_shape, zero_scalar, {}));

      // 4. is_neg = Compare(value_f, zero, LT)
      //    Check whether the high bit of the original uint is set (SIToFP produces negative values)
      Shape pred_shape = ShapeUtil::ChangeElementType(float_shape, PRED);
      HloInstruction* is_neg = computation->AddInstruction(
          HloInstruction::CreateCompare(pred_shape, value_f, zero,
                                        ComparisonDirection::kLt));

      // 5. bias = 2^bit_width is broadcast to the shape of value
      double bias_val = std::ldexp(1.0, bit_width);
      Literal bias_lit = (dst_type == F64)
                             ? LiteralUtil::CreateR0<double>(bias_val)
                             : LiteralUtil::CreateR0<float>(
                                   static_cast<float>(bias_val));
      HloInstruction* bias_scalar = computation->AddInstruction(
          HloInstruction::CreateConstant(std::move(bias_lit)));
      HloInstruction* bias = computation->AddInstruction(
          HloInstruction::CreateBroadcast(float_shape, bias_scalar, {}));

      // 6. correction = Select(is_neg, bias, zero)
      //    When the original uint high bit is set, offset 2^bit_width to offset the negative value of SIToFP
      HloInstruction* correction = computation->AddInstruction(
          HloInstruction::CreateTernary(float_shape, HloOpcode::kSelect,
                                        is_neg, bias, zero));

      // 7. result = Add(value_f, correction)
      //    value_f + 0 (positive value) or value_f + 2^bit_width (negative value corrected to positive value)
      HloInstruction* result = computation->AddInstruction(
          HloInstruction::CreateBinary(float_shape, HloOpcode::kAdd,
                                       value_f, correction));

      NPU_LOG_FORMAT_PRINT(
          INFO,
          "Decomposed convert(%s, %s) %s into SIToFP + bias correction",
          PrimitiveType_Name(src_type).c_str(),
          PrimitiveType_Name(dst_type).c_str(),
          std::string(instr->name()).c_str());

      TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(result));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
