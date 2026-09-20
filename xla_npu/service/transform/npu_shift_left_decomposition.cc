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

#include "xla/xla_npu/service/transform/npu_shift_left_decomposition.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

namespace {

// Extract the underlying constant from the instruction chain: if instr is constant or broadcast(constant),
// Returns the constant directive and broadcast dimensions; otherwise returns nullopt.
struct ConstantInfo {
  HloInstruction* constant;
  absl::Span<const int64_t> bcast_dims;
};
std::optional<ConstantInfo> GetConstantInfo(HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kConstant) {
    return ConstantInfo{instr, {}};
  }
  if (instr->opcode() == HloOpcode::kBroadcast &&
      instr->operand(0)->opcode() == HloOpcode::kConstant) {
    return ConstantInfo{instr->mutable_operand(0), instr->dimensions()};
  }
  return std::nullopt;
}

// Computes 2^value element-wise on a constant literal, with the result type being the same as the input.
absl::StatusOr<Literal> ComputePowerOfTwo(const Literal& lit) {
  PrimitiveType type = lit.shape().element_type();
  Literal result(Literal::CreateFromShape(lit.shape()));
  int64_t count = lit.element_count();

  switch (type) {
    case U8: {
      auto src = lit.data<uint8_t>();
      auto dst = result.data<uint8_t>();
      for (int64_t i = 0; i < count; ++i) {
        dst[i] = (src[i] < 8) ? static_cast<uint8_t>(1u << src[i]) : 0;
      }
      break;
    }
    case U16: {
      auto src = lit.data<uint16_t>();
      auto dst = result.data<uint16_t>();
      for (int64_t i = 0; i < count; ++i) {
        dst[i] = (src[i] < 16) ? static_cast<uint16_t>(1u << src[i]) : 0;
      }
      break;
    }
    case U32: {
      auto src = lit.data<uint32_t>();
      auto dst = result.data<uint32_t>();
      for (int64_t i = 0; i < count; ++i) {
        dst[i] = (src[i] < 32) ? (1u << src[i]) : 0u;
      }
      break;
    }
    case U64: {
      auto src = lit.data<uint64_t>();
      auto dst = result.data<uint64_t>();
      for (int64_t i = 0; i < count; ++i) {
        dst[i] = (src[i] < 64) ? (1ull << src[i]) : 0ull;
      }
      break;
    }
    case S8: {
      auto src = lit.data<int8_t>();
      auto dst = result.data<int8_t>();
      for (int64_t i = 0; i < count; ++i) {
        int shift = static_cast<int>(src[i]);
        dst[i] = (shift >= 0 && shift < 8)
                     ? static_cast<int8_t>(1 << shift)
                     : 0;
      }
      break;
    }
    case S16: {
      auto src = lit.data<int16_t>();
      auto dst = result.data<int16_t>();
      for (int64_t i = 0; i < count; ++i) {
        int shift = static_cast<int>(src[i]);
        dst[i] = (shift >= 0 && shift < 16)
                     ? static_cast<int16_t>(1 << shift)
                     : 0;
      }
      break;
    }
    case S32: {
      auto src = lit.data<int32_t>();
      auto dst = result.data<int32_t>();
      for (int64_t i = 0; i < count; ++i) {
        int shift = src[i];
        dst[i] = (shift >= 0 && shift < 32) ? (1 << shift) : 0;
      }
      break;
    }
    case S64: {
      auto src = lit.data<int64_t>();
      auto dst = result.data<int64_t>();
      for (int64_t i = 0; i < count; ++i) {
        int64_t shift = src[i];
        dst[i] = (shift >= 0 && shift < 64) ? (1LL << shift) : 0;
      }
      break;
    }
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported integer type for shift_left: %s",
                          PrimitiveType_Name(type)));
  }
  return result;
}

}  // namespace

absl::StatusOr<bool> NpuShiftLeftDecomposition::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    std::vector<HloInstruction*> shift_left_instrs;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kShiftLeft) {
        shift_left_instrs.push_back(instr);
      }
    }

    for (HloInstruction* instr : shift_left_instrs) {
      HloInstruction* x = instr->mutable_operand(0);
      HloInstruction* n = instr->mutable_operand(1);
      PrimitiveType elem_type = instr->shape().element_type();

      HloInstruction* result = nullptr;

      // Try the constant optimization path: compute 2^n directly in the integer domain
      auto const_info = GetConstantInfo(n);
      if (const_info.has_value()) {
        TF_ASSIGN_OR_RETURN(
            Literal power_lit,
            ComputePowerOfTwo(const_info->constant->literal()));

        HloInstruction* power_const = computation->AddInstruction(
            HloInstruction::CreateConstant(std::move(power_lit)));

        if (n->opcode() == HloOpcode::kBroadcast) {
          // n is broadcast(constant), do the same for power_const broadcast
          HloInstruction* power_bcast = computation->AddInstruction(
              HloInstruction::CreateBroadcast(
                  instr->shape(), power_const,
                  std::vector<int64_t>(const_info->bcast_dims.begin(),
                                       const_info->bcast_dims.end())));
          result = computation->AddInstruction(
              HloInstruction::CreateBinary(instr->shape(),
                                           HloOpcode::kMultiply, x,
                                           power_bcast));
        } else {
          // n is a scalar constant, multiply directly
          result = computation->AddInstruction(
              HloInstruction::CreateBinary(instr->shape(),
                                           HloOpcode::kMultiply, x,
                                           power_const));
        }

        NPU_LOG_FORMAT_PRINT(
            INFO,
            "Decomposed shift_left %s into multiply(x, 2^n) [constant path]",
            std::string(instr->name()).c_str());
      } else {
        // Non-const paths: use pow(2, convert(n, F32))
        // For unsigned types, first bitcast to signed types and then convert to F32.
        // Avoid convert(U32, F32) taking the host fallback path.
        Shape n_f32_shape = ShapeUtil::ChangeElementType(n->shape(), F32);
        HloInstruction* n_for_convert = n;

        if (elem_type == U32 || elem_type == U64 || elem_type == U16) {
          PrimitiveType signed_type =
              (elem_type == U64) ? S64 : (elem_type == U16) ? S16 : S32;
          Shape n_signed_shape =
              ShapeUtil::ChangeElementType(n->shape(), signed_type);
          n_for_convert = computation->AddInstruction(
              HloInstruction::CreateBitcastConvert(n_signed_shape, n));
        }

        HloInstruction* n_f32 = computation->AddInstruction(
            HloInstruction::CreateConvert(n_f32_shape, n_for_convert));

        HloInstruction* two_scalar = computation->AddInstruction(
            HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<float>(2.0f)));
        HloInstruction* two_bcast = computation->AddInstruction(
            HloInstruction::CreateBroadcast(n_f32_shape, two_scalar, {}));

        HloInstruction* power_f32 = computation->AddInstruction(
            HloInstruction::CreateBinary(n_f32_shape, HloOpcode::kPower,
                                         two_bcast, n_f32));

        // Convert pow result back to original type
        HloInstruction* power_orig = nullptr;
        if (elem_type == U32 || elem_type == U64 || elem_type == U16) {
          // First convert F32 -> signed type (aclnnCast supports), then bitcast to unsigned
          PrimitiveType signed_type =
              (elem_type == U64) ? S64 : (elem_type == U16) ? S16 : S32;
          Shape power_signed_shape =
              ShapeUtil::ChangeElementType(instr->shape(), signed_type);
          HloInstruction* power_signed = computation->AddInstruction(
              HloInstruction::CreateConvert(power_signed_shape, power_f32));
          power_orig = computation->AddInstruction(
              HloInstruction::CreateBitcastConvert(instr->shape(),
                                                    power_signed));
        } else {
          power_orig = computation->AddInstruction(
              HloInstruction::CreateConvert(instr->shape(), power_f32));
        }

        result = computation->AddInstruction(
            HloInstruction::CreateBinary(instr->shape(), HloOpcode::kMultiply,
                                         x, power_orig));

        NPU_LOG_FORMAT_PRINT(
            INFO,
            "Decomposed shift_left %s into multiply(x, pow(2, "
            "convert(%s, F32)))",
            std::string(instr->name()).c_str(),
            std::string(n->name()).c_str());
      }

      TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(result));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
