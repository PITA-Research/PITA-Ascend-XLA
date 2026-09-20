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

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

LogicalResult eraseUnusedOp(Operation *op, PatternRewriter &rewriter) {
  if (!op->getUses().empty()) {
    return failure();
  }
  rewriter.eraseOp(op);
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// GlobalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalTensorOp::canonicalize(GlobalTensorOp op,
                                           PatternRewriter &rewriter) {
  return eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// LocalTensorOp
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorOp::canonicalize(LocalTensorOp op,
                                          PatternRewriter &rewriter) {
  return eraseUnusedOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// PipeBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult PipeBarrierOp::canonicalize(PipeBarrierOp op,
                                          PatternRewriter &rewriter) {
  Block *block = op->getBlock();
  auto nextIt = std::next(Block::iterator(op));
  if (nextIt == block->end()) return failure();
  if (auto nextOp = dyn_cast<ascendc::PipeBarrierOp>(*nextIt)) {
    if (op.getPipe() == Pipe::PIPE_ALL) {
      rewriter.eraseOp(nextOp);
      return success();
    }
    if (op->getAttrs() == nextOp->getAttrs() ||
        nextOp.getPipe() == Pipe::PIPE_ALL) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

bool LocalTensorReinterpretCastOp::areCastCompatible(TypeRange inputs,
                                                     TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         isa<LocalTensorType>(inputs[0]) && isa<LocalTensorType>(outputs[0]);
}

OpFoldResult LocalTensorReinterpretCastOp::fold(FoldAdaptor adaptor) {
  Value in = getIn();
  return in.getType() == getType() ? in : nullptr;
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void AscendCDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/Asc/IR/AscendCOps.cpp.inc"
      >();
}
