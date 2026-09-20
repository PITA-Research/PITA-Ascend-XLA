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

#include "xla/xla_npu/service/npu_ir_emitter.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/tuple_ops.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_npu/service/npu_ir_emitter_context.h"

namespace xla {

NpuIrEmitter::NpuIrEmitter(NpuIrEmitterContext* ir_emitter_context,
                           bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext()),
      bindings_(&b_, module_, is_nested) {}

absl::Status NpuIrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArray(*operand, *hlo)
          .EmitReadArrayElement(index, &b_, operand->name());
    };
  }
  return EmitTargetElementLoop(
      *hlo, ElementalIrEmitter(module_, &b_)
                .MakeElementGenerator(hlo, operand_to_generator));
}

absl::Status NpuIrEmitter::HandleConstant(HloInstruction* constant) {
  return absl::OkStatus();
}

absl::Status NpuIrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  VLOG(2) << "NpuIrEmitter::HandleAddDependency: "
          << add_dependency->ToString();
  const HloInstruction* operand = add_dependency->operand(0);
  // Add_Dependency is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  }
  return absl::OkStatus();
}

absl::Status NpuIrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->operand(0);
  CHECK(bindings_.BoundToIrValue(*operand));
  bindings_.BindHloToIrValue(
      *get_tuple_element,
      llvm_ir::EmitGetTupleElement(
          get_tuple_element->shape(), get_tuple_element->tuple_index(),
          /*alignment=*/1, GetBasePointer(*operand),
          llvm_ir::ShapeToIrType(operand->shape(), module_->getContext()),
          &b_));
  return absl::OkStatus();
}

absl::Status NpuIrEmitter::HandleSend(HloInstruction*) {
  return Unimplemented("Send is not implemented on GPU");
}

absl::Status NpuIrEmitter::HandleSendDone(HloInstruction*) {
  return Unimplemented("Send-Done is not implemented on GPU");
}

absl::Status NpuIrEmitter::HandleRecv(HloInstruction*) {
  return Unimplemented("Recv is not implemented on GPU");
}

absl::Status NpuIrEmitter::HandleRecvDone(HloInstruction*) {
  return Unimplemented("Recv-done is not implemented on GPU");
}

absl::Status NpuIrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter is not implemented on GPUs.");
}

absl::Status NpuIrEmitter::HandleTuple(HloInstruction* tuple) {
  std::vector<llvm::Value*> base_ptrs;
  for (const HloInstruction* operand : tuple->operands()) {
    base_ptrs.push_back(GetBasePointer(*operand));
  }
  llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
  return absl::OkStatus();
}

absl::Status NpuIrEmitter::HandleConvolution(HloInstruction* convolution) {
  if (ShapeUtil::IsZeroElementArray(convolution->shape())) {
    // Emit no code for an empty output.
    return absl::OkStatus();
  }
  return Unimplemented(
      "Hit a case for convolution that is not implemented on GPU.");
}

absl::Status NpuIrEmitter::HandleFft(HloInstruction* fft) {
  if (ShapeUtil::IsZeroElementArray(fft->shape())) {
    // Emit no code for an empty output.
    return absl::OkStatus();
  }
  return Unimplemented("Hit a case for fft that is not implemented on GPU.");
}

absl::Status NpuIrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented(
      "AllReduce cannot be nested inside of fusion, map, etc.");
}

absl::Status NpuIrEmitter::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status NpuIrEmitter::HandleCall(HloInstruction* call) {
  return absl::UnimplementedError("NpuIrEmitter::HandleCall is not supported.");
}

absl::Status NpuIrEmitter::HandleCustomCall(HloInstruction*) {
  return Unimplemented("custom-call");
}

absl::Status NpuIrEmitter::HandleInfeed(HloInstruction*) {
  return Unimplemented("Infeed is not supported on GPU.");
}

absl::Status NpuIrEmitter::HandleOutfeed(HloInstruction*) {
  return Unimplemented("Outfeed is not supported on GPU.");
}

absl::Status NpuIrEmitter::HandleBatchNormInference(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormInference directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

absl::Status NpuIrEmitter::HandleBatchNormTraining(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormTraining directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

absl::Status NpuIrEmitter::HandleBatchNormGrad(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormGrad directly.  It should "
      "be lowered before IR emission to HLO-soup using BatchNormRewriter.");
}

std::vector<llvm_ir::IrArray> NpuIrEmitter::ConstructIrArrayForOutputs(
    const HloInstruction& hlo) {
  std::vector<llvm_ir::IrArray> output_arrays;
  if (hlo.shape().IsTuple()) {
    int64_t number_outputs = ShapeUtil::TupleElementCount(hlo.shape());
    output_arrays.reserve(number_outputs);
    for (int64_t i = 0; i < number_outputs; ++i) {
      output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
    }
  } else {
    output_arrays.push_back(GetIrArray(hlo, hlo));
  }
  return output_arrays;
}

void NpuIrEmitter::BindFusionArguments(const HloInstruction* fusion,
                                       FusedIrEmitter* fused_emitter) {
  for (int i = 0; i < fusion->operand_count(); i++) {
    const HloInstruction* operand = fusion->operand(i);
    fused_emitter->BindGenerator(
        *fusion->fused_parameter(i),
        [this, operand, fusion](llvm_ir::IrArray::Index index) {
          return GetIrArray(*operand, *fusion)
              .EmitReadArrayElement(index, &b_, operand->name());
        });
  }
}

}  // namespace xla
