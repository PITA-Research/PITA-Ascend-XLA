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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SHIFT_LEFT_DECOMPOSITION_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SHIFT_LEFT_DECOMPOSITION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// CANN does not have aclnnLeftShift API and needs to decompose shift_left into equivalent operations.
//
// Constant displacement path (shift amount is a compile-time constant):
//   Compute 2^n directly in the integer domain as a constant, then multiply(x, 2^n)
//   Intermediate conversions to floating point are avoided and all integer types (including U32/U64) are handled correctly.
//
// Non-constant displacement path (shift amount is a runtime parameter):
//   shift_left(x, n) = x * pow(2, convert(n, F32))
//   For unsigned types, first use BitcastConvert to reinterpret them into signed types and then convert to F32.
//   Avoid convert(U32, F32) taking the host fallback path and causing timing problems.
class NpuShiftLeftDecomposition : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "npu-shift-left-decomposition";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_SHIFT_LEFT_DECOMPOSITION_H_
