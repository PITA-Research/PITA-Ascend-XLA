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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

// NpuFusionWhitelist manages a whitelist of HLO opcodes that should skip fusion.
// Certain operations like Dot and Reduce are excluded from fusion to preserve
// their optimized implementations.
class NpuFusionWhitelist {
 public:
  static bool IsAfirArithmeticOp(HloOpcode opcode) {
    return opcode == HloOpcode::kAdd ||
           opcode == HloOpcode::kSubtract ||
           opcode == HloOpcode::kMultiply ||
           opcode == HloOpcode::kDivide;
  }

  static bool IsAfirSupportedArithmeticType(HloOpcode opcode,
                                            PrimitiveType type) {
    switch (opcode) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
        return type == F16 || type == F32 || type == S16 || type == S32;
      case HloOpcode::kDivide:
        return type == F16 || type == F32;
      default:
        return true;
    }
  }

  static bool HasUnsupportedArithmeticDtype(const HloInstruction* instr) {
    if (instr == nullptr || !IsAfirArithmeticOp(instr->opcode())) {
      return false;
    }

    if (!IsAfirSupportedArithmeticType(instr->opcode(),
                                       instr->shape().element_type())) {
      return true;
    }

    for (const HloInstruction* operand : instr->operands()) {
      if (!IsAfirSupportedArithmeticType(instr->opcode(),
                                         operand->shape().element_type())) {
        return true;
      }
    }

    return false;
  }

  // Returns true if given instruction should skip fusion.
  static bool ShouldSkipFusion(const HloInstruction* instr) {
    if (instr == nullptr) {
      return false;
    }
    return HasUnsupportedArithmeticDtype(instr) ||
           ShouldSkipFusion(instr->opcode());
  }

  // Returns true if given opcode should skip fusion.
  static bool ShouldSkipFusion(HloOpcode opcode) {
    // Hardcoded whitelist of opcodes to skip fusion
    // Note: Exp, Add, and Subtract are added to avoid numerical instability in fused kernels
    static const absl::flat_hash_set<HloOpcode> kSkipFusionOpcodes = {
        HloOpcode::kDot,
        HloOpcode::kCopy,     // Copy handled via ACLNN thunk (aclrtMemcpy), not AFIR
        HloOpcode::kScatter,  // Scatter handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kReduce,   // Reduce handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kCompare,  // Compare not yet supported in AFIR converter
        HloOpcode::kSelect,   // Select not yet supported in AFIR converter
        HloOpcode::kRemainder,
        HloOpcode::kGather,   // Gather handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kIsFinite, // IsFinite handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kAbs,      // Abs handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kSqrt,     // Sqrt handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kLog,      // Log handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kLog1p,    // Log1p handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kBitcastConvert,  // BitcastConvert handled via aclrtMemcpy
        HloOpcode::kSign,     // Sign handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kSin,      // Sin handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kCos,      // Cos handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kAtan2,    // Atan2 handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kPower,    // Power handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kNot,      // Not (BitwiseNot) handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kAnd,      // And (BitwiseAnd) handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kOr,       // Or (BitwiseOr) handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kXor,      // Xor (BitwiseXor) handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kMaximum,  // Maximum handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kMinimum,  // Minimum handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kSort,     // Sort handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kTopK,     // TopK handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kRoundNearestAfz,  // Round handled via ACLNN thunks
        HloOpcode::kRoundNearestEven, // Round handled via ACLNN thunks
        HloOpcode::kIota,     // Iota handled via ACLNN thunks (host fill), not AFIR fusion
        HloOpcode::kConvert,  // Convert handled via ACLNN thunks (host cast), not AFIR fusion
        HloOpcode::kReduceWindow,  // ReduceWindow handled via ACLNN thunks (aclnnMaxPool), not AFIR fusion
        HloOpcode::kReverse,   // Reverse handled via ACLNN thunks (aclnnFlip), not AFIR fusion
        HloOpcode::kPad,       // Pad handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kConcatenate,  // Concat handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kCholesky,  // Cholesky handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kTriangularSolve,  // TriangularSolve via ACLNN thunks
        HloOpcode::kShiftRightLogical,    // RightShift handled via aclnnRightShift
        HloOpcode::kShiftRightArithmetic, // RightShift handled via aclnnRightShift
        // HloOpcode::kDivide,     // Divide handled via ACLNN thunks (aclnnDiv), not AFIR fusion
        // HloOpcode::kMultiply,  // Multiply handled via ACLNN thunks (aclnnMul)
        // HloOpcode::kAdd,       // Add handled via ACLNN thunks, not AFIR fusion
        // HloOpcode::kSubtract,  // Subtract handled via ACLNN thunks, not AFIR fusion
        HloOpcode::kWhile,     // While is a control flow op, handled via NpuWhileThunk
        HloOpcode::kConditional,  // Conditional is a control flow op, handled via NpuConditionalThunk
        HloOpcode::kTuple,     // Tuple state is represented by buffers, not AFIR
        HloOpcode::kGetTupleElement,  // Tuple projection is not an AFIR op
        HloOpcode::kSlice,          // Slice: ASCIR serialization bug for fused slice, use ACLNN
        HloOpcode::kDynamicSlice,   // DynamicSlice: same ASCIR bug, use ACLNN
        HloOpcode::kDynamicUpdateSlice,  // DUS: use ACLNN thunk
        HloOpcode::kNegate,         // Negate: use ACLNN thunk (aclnnNeg)
        HloOpcode::kExp,            // Exp: ASCIR bug with non-F32 types, use ACLNN
        HloOpcode::kExpm1,          // Expm1: not in AFIR converter, use ACLNN
        HloOpcode::kRsqrt,          // Rsqrt: not in AFIR converter, use ACLNN
        HloOpcode::kClamp,          // Clamp: use ACLNN thunk
    };
    return kSkipFusionOpcodes.contains(opcode);
  }
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_WHITELIST_H_
