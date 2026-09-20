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

#ifndef ASCIR_TARGET_ASC_MLIR_ARITH_H
#define ASCIR_TARGET_ASC_MLIR_ARITH_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {

template <typename BinaryOpType>
auto printOperation(CodeEmitter &emitter, BinaryOpType op) -> LogicalResultForT<
    BinaryOpType, arith::AddIOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp,
    arith::SubIOp, arith::CeilDivSIOp, arith::AndIOp, arith::OrIOp,
    arith::ShLIOp, arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
    arith::MinNumFOp, arith::AddFOp, arith::DivFOp, arith::ShRSIOp,
    arith::ShRUIOp, arith::SubFOp, arith::MaxSIOp, arith::MulFOp,
    arith::MinSIOp, arith::XOrIOp, arith::DivUIOp> {
  if (failed(isScalarOperation(op)) ||
      failed(emitter.emitAssignPrefix(*op.getOperation()))) {
    return failure();
  }
  auto &os = emitter.ostream();
  if constexpr (std::is_same_v<BinaryOpType, arith::CeilDivSIOp>) {
    os << "(" << emitter.getOrCreateName(op.getLhs()) << " + "
       << emitter.getOrCreateName(op.getRhs()) << " - 1) / "
       << emitter.getOrCreateName(op.getRhs());
    return success();
  }
  if constexpr (llvm::is_one_of<BinaryOpType, arith::MaximumFOp,
                                arith::MaxNumFOp, arith::MinimumFOp,
                                arith::MinNumFOp, arith::MinSIOp,
                                arith::MaxSIOp>::value) {
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    os << "((" << lhs;
    if constexpr (llvm::is_one_of<BinaryOpType, arith::MaximumFOp,
                                  arith::MaxNumFOp, arith::MaxSIOp>::value) {
      os << " > ";
    } else {
      os << " < ";
    }
    os << rhs << ") ? (" << lhs << ") : (" << rhs << "))";
    return success();
  }
  os << emitter.getOrCreateName(op.getLhs());
  if constexpr (llvm::is_one_of<BinaryOpType, arith::AddIOp,
                                arith::AddFOp>::value) {
    os << " + ";
  } else if constexpr (llvm::is_one_of<BinaryOpType, arith::SubIOp,
                                       arith::SubFOp>::value) {
    os << " - ";
  } else if constexpr (llvm::is_one_of<BinaryOpType, arith::MulIOp,
                                       arith::MulFOp>::value) {
    os << " * ";
  } else if constexpr (llvm::is_one_of<BinaryOpType, arith::DivSIOp,
                                       arith::DivUIOp, arith::DivFOp>::value) {
    os << " / ";
  } else if constexpr (std::is_same_v<BinaryOpType, arith::RemSIOp>) {
    os << " % ";
  } else if constexpr (std::is_same_v<BinaryOpType, arith::AndIOp>) {
    os << " & ";
  } else if constexpr (std::is_same_v<BinaryOpType, arith::OrIOp>) {
    os << " | ";
  } else if constexpr (std::is_same_v<BinaryOpType, arith::ShLIOp>) {
    os << " << ";
  } else if constexpr (llvm::is_one_of<BinaryOpType, arith::ShRSIOp,
                                       arith::ShRUIOp>::value) {
    os << " >> ";
  } else if constexpr (std::is_same_v<BinaryOpType, arith::XOrIOp>) {
    os << " ^ ";
  } else {
    llvm_unreachable("not implemented");
  }
  os << emitter.getOrCreateName(op.getRhs());
  return success();
}

template <typename CastOpType>
auto printOperation(CodeEmitter &emitter, CastOpType op)
    -> LogicalResultForT<CastOpType, arith::ExtUIOp, arith::ExtSIOp,
                         arith::ExtFOp, arith::TruncIOp, arith::TruncFOp,
                         arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp,
                         arith::UIToFPOp> {
  FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
  auto &os = emitter.ostream();
  os << "static_cast<";
  FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
  os << ">(" << emitter.getOrCreateName(op.getIn()) << ")";
  return success();
}

LogicalResult printOperation(CodeEmitter &emitter,
                             arith::ConstantOp constantOp);

LogicalResult printOperation(CodeEmitter &emitter, arith::MulUIExtendedOp op);

LogicalResult printOperation(CodeEmitter &emitter, arith::CmpIOp op);

LogicalResult printOperation(CodeEmitter &emitter, arith::CmpFOp op);

LogicalResult printOperation(CodeEmitter &emitter, arith::BitcastOp op);

LogicalResult printOperation(CodeEmitter &emitter, arith::SelectOp op);

LogicalResult printOperation(CodeEmitter &emitter, arith::IndexCastOp op);

}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_MLIR_ARITH_H
