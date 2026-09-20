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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_TRIL_UTILS_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_TRIL_UTILS_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"

namespace xla::tril {

// Extract a scalar constant value from HLO (automatically strips the Broadcast wrapper).
// Supports S32/S64/U32/U64 types. Returns true on success and sets *value.
inline bool ExtractConstantValue(const HloInstruction* hlo, int64_t* value) {
  const HloInstruction* core = hlo;
  while (core->opcode() == HloOpcode::kBroadcast) {
    core = core->operand(0);
  }
  if (core->opcode() != HloOpcode::kConstant) return false;

  const Literal& lit = core->literal();
  if (!ShapeUtil::IsScalar(lit.shape())) {
    return false;
  }
  switch (lit.shape().element_type()) {
    case PrimitiveType::S32:
      *value = static_cast<int64_t>(lit.Get<int32_t>({}));
      return true;
    case PrimitiveType::S64:
      *value = lit.Get<int64_t>({});
      return true;
    case PrimitiveType::U32:
      *value = static_cast<int64_t>(lit.Get<uint32_t>({}));
      return true;
    case PrimitiveType::U64:
      *value = static_cast<int64_t>(lit.Get<uint64_t>({}));
      return true;
    default:
      return false;
  }
}

// Extract valid iota dimensions from HLO.
// Two modes are handled:
//   Iota(dim)                                    → iota_dim = dim
//   Broadcast(Iota(dim), dimensions={...}) -> iota_dim = broadcast mapped dimensions
// Returns true on success and sets *iota_dim.
inline bool ExtractIotaDim(const HloInstruction* hlo, int64_t* iota_dim) {
  if (hlo->opcode() == HloOpcode::kIota) {
    *iota_dim = hlo->dimensions(0);
    return true;
  }
  // Broadcast(Iota): 1D iota(dim=d) is broadcast to ND,
  // broadcast.dimensions() maps input dimension d to output dimension broadcast.dimensions(d)
  if (hlo->opcode() == HloOpcode::kBroadcast &&
      hlo->operand(0)->opcode() == HloOpcode::kIota) {
    int64_t src_dim = hlo->operand(0)->dimensions(0);
    *iota_dim = hlo->dimensions(src_dim);
    return true;
  }
  return false;
}

// Starting from the HLO instruction, extract the iota dimension and offset value k.
//
// Supported modes:
//   Iota(dim)                                          → k=0
//   Broadcast(Iota(dim), dimensions={d})               → k=0, effective_dim=d
//   Add(Iota_like, Constant_like) -> k=constant
//   Add(Constant_like, Iota_like) -> k=constant
//   Broadcast (any of the above) -> recursive expansion
//
// where Iota_like = Iota(dim) or Broadcast(Iota(dim), dimensions={d})
//      Constant_like = Constant(k) or Broadcast(Constant(k))
//
// A successful match returns true and sets iota_dim / k_value.
inline bool TraceIotaOffset(const HloInstruction* hlo, int64_t* iota_dim,
                            int64_t* k_value) {
  // Broadcast wrapper: if it is Broadcast(Iota), calculate the effective dimensions, otherwise expand recursively
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    if (ExtractIotaDim(hlo, iota_dim)) {
      *k_value = 0;
      return true;
    }
    return TraceIotaOffset(hlo->operand(0), iota_dim, k_value);
  }

  // Add mode: iota-like on one side (possibly wrapped by Broadcast), constant on the other
  if (hlo->opcode() == HloOpcode::kAdd) {
    const HloInstruction* lhs = hlo->operand(0);
    const HloInstruction* rhs = hlo->operand(1);

    int64_t dim = -1;
    int64_t const_val = 0;

    if (ExtractIotaDim(lhs, &dim) && ExtractConstantValue(rhs, &const_val)) {
      *iota_dim = dim;
      *k_value = const_val;
      return true;
    }
    if (ExtractConstantValue(lhs, &const_val) && ExtractIotaDim(rhs, &dim)) {
      *iota_dim = dim;
      *k_value = const_val;
      return true;
    }
    return false;
  }

  // Direct Iota (k=0)
  if (hlo->opcode() == HloOpcode::kIota) {
    *k_value = 0;
    *iota_dim = hlo->dimensions(0);
    return true;
  }

  return false;
}

// Test whether the Select directive matches the tril pattern:
//   select(compare(iota(dim=0)+k, iota(dim=1), GE), input, zeros)
//
// When the match is successful, the diagonal offset is returned through out_k, and out_input returns the input tensor operand.
inline bool MatchTrilSelect(const HloInstruction* select, int64_t* out_k,
                            const HloInstruction** out_input = nullptr) {
  if (select->opcode() != HloOpcode::kSelect) return false;
  if (select->operand_count() != 3) return false;

  // on_false must be all zeros: broadcast(constant(0))
  const HloInstruction* zero_src = select->operand(2);
  while (zero_src->opcode() == HloOpcode::kBroadcast) {
    zero_src = zero_src->operand(0);
  }
  if (zero_src->opcode() != HloOpcode::kConstant) return false;
  if (!ShapeUtil::IsScalar(zero_src->literal().shape())) {
    return false;
  }
  if (!zero_src->literal().IsZero({})) return false;

  // Predicate must be Compare(GE), possibly wrapped by Broadcast/BroadcastInDim (vmap path)
  const HloInstruction* pred = select->operand(0);
  while (pred->opcode() == HloOpcode::kBroadcast) {
    pred = pred->operand(0);
  }
  if (pred->opcode() != HloOpcode::kCompare) return false;
  if (pred->comparison_direction() != ComparisonDirection::kGe) return false;

  // Tracking compares iota dimensions and offsets on both sides
  int64_t lhs_dim = -1, lhs_k = 0;
  int64_t rhs_dim = -1, rhs_k = 0;
  if (!TraceIotaOffset(pred->operand(0), &lhs_dim, &lhs_k)) return false;
  if (!TraceIotaOffset(pred->operand(1), &rhs_dim, &rhs_k)) return false;

  // Determine which is dim=0 (with k offset) and which is dim=1 (no offset)
  int64_t k = 0;
  if (lhs_dim == 0 && rhs_dim == 1) {
    k = lhs_k;
  } else if (lhs_dim == 1 && rhs_dim == 0) {
    k = rhs_k;
  } else {
    return false;
  }

  *out_k = k;
  if (out_input != nullptr) {
    *out_input = select->operand(1);
  }
  return true;
}

// Convenient encapsulation: detect whether the fusion calculation matches the tril pattern.
// Equivalent to MatchTrilSelect(fused_computation->root_instruction(), out_k).
inline bool MatchTrilFusion(const HloComputation* fused_computation,
                            int64_t* out_k) {
  return MatchTrilSelect(fused_computation->root_instruction(), out_k);
}

}  // namespace xla::tril

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_TRIL_UTILS_H_
