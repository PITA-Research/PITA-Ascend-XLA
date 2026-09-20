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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSIGNED_TO_FLOAT_DECOMPOSITION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSIGNED_TO_FLOAT_DECOMPOSITION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Decompose convert(U16/U32/U64 -> F32/F64) into available operations on the device side.
//
// Root cause: aclnnCast API does not support UINT32/UINT16/UINT64 input types,
// ConvertXlaType2AclDataType maps U32 to ACL_INT32 (signed),
// Causes uint32 values ≥ 2^31 to be treated as negative int32 and converted to negative float.
//
// Decomposition strategy (taking U32->F32 as an example):
//   value_s = BitcastConvert(value, S32) // Reinterpret bit pattern
//   value_f = Convert(value_s, F32) // SIToFP (the high-order value may be negative)
//   is_neg = Compare(value_f, 0.0, LT) // Check whether the high bit of the original uint is set
//   bias = Select(is_neg, 4294967296.0, 0) // Compensate for negative values 2^32
//   result = Add(value_f, bias) // correct uint32 -> float32
//
// Mathematical equivalence:
//   For value ∈ [0, 2^31): SIToFP is correct -> value_f ≥ 0 -> bias=0 -> result=value_f
//   For value ∈ [2^31, 2^32): SIToFP negative value value_f -> bias=2^32 -> result=value_f+2^32
//   Precision: 2^32 can be expressed exactly as float32 (power of 2), and addition introduces no additional precision loss.
//
// The same applies to U16->{F32,F64} (bias=2^16) and U64->{F32,F64} (bias=2^64).
class NpuUnsignedToFloatDecomposition : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-unsigned-to-float-decomposition";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_UNSIGNED_TO_FLOAT_DECOMPOSITION_H_
