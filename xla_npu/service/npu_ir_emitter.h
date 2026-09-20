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

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_H_

#include <vector>

#include "absl/status/status.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"
#include "xla/xla_npu/service/npu_hlo_to_ir_bindings.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {
class NpuIrEmitter : public DfsHloVisitorWithDefault,
                     public IrBuilderMixin<NpuIrEmitter> {
 public:
  NpuIrEmitter(const NpuIrEmitter&) = delete;
  NpuIrEmitter& operator=(const NpuIrEmitter&) = delete;

  absl::Status DefaultAction(HloInstruction* hlo) override;
  absl::Status HandleConstant(HloInstruction* constant) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleFft(HloInstruction* fft) override;
  absl::Status HandleAllReduce(HloInstruction* crs) override;
  absl::Status HandleInfeed(HloInstruction* infeed) override;
  absl::Status HandleOutfeed(HloInstruction* outfeed) override;
  absl::Status HandleSend(HloInstruction* send) override;
  absl::Status HandleSendDone(HloInstruction* send_done) override;
  absl::Status HandleRecv(HloInstruction* recv) override;
  absl::Status HandleRecvDone(HloInstruction* recv_done) override;
  absl::Status HandleParameter(HloInstruction* parameter) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleScatter(HloInstruction* scatter) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleCustomCall(HloInstruction* custom_call) override;
  absl::Status HandleBatchNormInference(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm) override;
  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;

  absl::Status FinishVisit(HloInstruction* root) override {
    return absl::OkStatus();
  }

  llvm::IRBuilderBase* builder() { return &b_; }

 protected:
  // Constructs an NpuIrEmitter with the given NpuIrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the NpuIrEmitter
  // object.
  explicit NpuIrEmitter(NpuIrEmitterContext* ir_emitter_context,
                        bool is_nested);

  // Helper for calling NpuHloToIrBindings::GetIrArray.
  // Gets the IrArray which contains inst.  This array has metadata that makes
  // it valid only within the IR that implements consumer.  If you are
  // implementing an HLO and want to get its own output buffer, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& inst,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {}) {
    return bindings_.GetIrArray(inst, consumer, shape_index);
  }
  // A convenient helper for calling NpuHloToIrBindings::GetBasePointer.
  llvm::Value* GetBasePointer(const HloInstruction& inst,
                              ShapeIndexView shape_index = {}) const {
    return bindings_.GetBasePointer(inst, shape_index);
  }

  // Generates the IrArray for each output of an hlo instruction and returns
  // a vector containing such IrArrays.
  std::vector<llvm_ir::IrArray> ConstructIrArrayForOutputs(
      const HloInstruction& hlo);

  // Emit a single-threaded or multi-threaded loop that computes every element
  // in the result of the given HLO instruction. This produces a series of
  // nested loops (e.g. one for each dimension of the `hlo`'s shape). The body
  // of the inner-most loop is provided by the body_emitter function.
  virtual absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) = 0;

  NpuIrEmitterContext* ir_emitter_context_;
  llvm::Module* module_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> b_;

  // // Mapping from HLO to its underlying LLVM value.
  NpuHloToIrBindings bindings_;

  // Bind all argument IrArrays of `fusion` to `fused_emitter`.
  void BindFusionArguments(const HloInstruction* fusion,
                           FusedIrEmitter* fused_emitter);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMITTER_H_
