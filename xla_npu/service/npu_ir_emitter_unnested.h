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

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/npu_ir_emitter.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {
class NpuIrEmitterUnnested : public NpuIrEmitter {
 public:
  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  using ValueVector3 = std::array<llvm::Value*, 3>;
  using ValueVector2 = std::array<llvm::Value*, 2>;

  using ConstantGenerator = std::function<llvm::Value*(int64_t)>;

  NpuIrEmitterUnnested(const NpuIrEmitterUnnested&) = delete;
  NpuIrEmitterUnnested& operator=(const NpuIrEmitterUnnested&) = delete;

  static std::unique_ptr<NpuIrEmitterUnnested> Create(
      NpuIrEmitterContext* ir_emitter_context);

  // Transfers the ownership of thunk_sequence_ out.
  std::unique_ptr<NpuSequentialThunk> ConsumeThunkSequence(
      NpuThunk::ThunkInfo thunk_info = NpuThunk::ThunkInfo{}) {
    return std::make_unique<NpuSequentialThunk>(thunk_info,
                                                std::move(thunk_sequence_));
  }

  // Emits code for the given HLO computation.
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  absl::Status EmitHloComputation(const HloComputation* computation);

 private:
  explicit NpuIrEmitterUnnested(NpuIrEmitterContext* ir_emitter_context);

  // NpuIrEmitterUnnested handles the following instructions differently from
  // NpuIrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  absl::Status EmitCall(const HloInstruction* instr);

  absl::Status EmitWhile(const HloInstruction* instr);

  absl::Status EmitConditional(const HloInstruction* instr);

  absl::Status EmitConstant(const HloConstantInstruction* instr);

  absl::Status EmitNpuAclnnThunk(const HloInstruction* instr);

  absl::Status EmitFmaAddcmulThunk(const HloInstruction* instr, float value);

  absl::Status HandleGather(const HloInstruction* instr);

  absl::Status EmitFusion(const HloFusionInstruction* instr);

  // NpuAclnnThunk that builds aclnnTril for use by CustomCall and Fusion paths.
  absl::Status EmitTrilThunk(const HloInstruction* instr, int64_t diagonal);

  absl::Status EmitHloInstruction(const HloInstruction* instr);

  absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Add a owning NpuThunk object to the thunk sequence.
  void AddThunkToThunkSequence(std::unique_ptr<NpuThunk> thunk) {
    if (emit_group_thunks_) {
      scoped_thunk_sequence_.emplace_back(std::move(thunk));
      return;
    }
    thunk_sequence_.emplace_back(std::move(thunk));
  }

  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;
  NpuThunkSequence thunk_sequence_;
  NpuThunkSequence scoped_thunk_sequence_;
  bool emit_group_thunks_ = false;

  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_UNNESTED_H_
