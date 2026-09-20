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

#ifndef XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_
#define XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_npu/backend/codegen/npu_fusion_emitter.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

// A wrapper for fusions implemented using the mechanism in
// xla/service/gpu/kernels. See custom_kernel_fusion.h in that folder for
// details.
class NpuCustomFusion : public NpuFusionInterface {
 public:
  absl::StatusOr<NpuFusionEmissionResult> Emit(
      NpuIrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_CODEGEN_NPU_CUSTOM_H_
