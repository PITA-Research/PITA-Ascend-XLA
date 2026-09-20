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

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::APIOp::getAPIName() {
      return getImpl()->getAPIName(getImpl(), getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::APIOp::getComment() {
      return getImpl()->getComment(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::BinaryL3Op::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::BinaryL3Op::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::BinaryL3Op::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src0` parameter
::mlir::Value mlir::ascendc::BinaryL3Op::getSrc0() {
      return getImpl()->implBinaryOp->getSrc0(getImpl()->implBinaryOp, getOperation());
  }

/// Obtain `src1` parameter
::mlir::Value mlir::ascendc::BinaryL3Op::getSrc1() {
      return getImpl()->implBinaryOp->getSrc1(getImpl()->implBinaryOp, getOperation());
  }

/// Obtain `src0` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getSrc0() {
      return getImpl()->getSrc0(getImpl(), getOperation());
  }

/// Obtain `src1` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getSrc1() {
      return getImpl()->getSrc1(getImpl(), getOperation());
  }

/// Obtain `sharedTmpBuffer` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getSharedTmpBuffer() {
      return getImpl()->getSharedTmpBuffer(getImpl(), getOperation());
  }

/// Obtain `calCount` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getCalCount() {
      return getImpl()->getCalCount(getImpl(), getOperation());
  }

/// Obtain `isReuseSource` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getIsReuseSource() {
      return getImpl()->getIsReuseSource(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::BinaryMathOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::BinaryMathOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::BinaryMathOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src0` parameter
::mlir::Value mlir::ascendc::BinaryOp::getSrc0() {
      return getImpl()->getSrc0(getImpl(), getOperation());
  }

/// Obtain `src1` parameter
::mlir::Value mlir::ascendc::BinaryOp::getSrc1() {
      return getImpl()->getSrc1(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::BinaryOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::BinaryOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::BinaryOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::DataConversionOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::DataConversionOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::DataCopyOp::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Get copy direction
::mlir::ascendc::CopyDirection mlir::ascendc::DataCopyOp::getDirection() {
      return getImpl()->getDirection(getImpl(), getOperation());
  }

/// Set src operand
void mlir::ascendc::DataCopyOp::setSrc(Value src) {
      return getImpl()->setSrc(getImpl(), getOperation(), src);
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::DataCopyOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::DataCopyOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::DataCopyOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::MathLibraryOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::MathLibraryOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::MathLibraryOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::OpWithDst::getDst() {
      return getImpl()->getDst(getImpl(), getOperation());
  }

/// Obtain `mask` parameter
::mlir::Value mlir::ascendc::UnaryL0Op::getMask() {
      return getImpl()->getMask(getImpl(), getOperation());
  }

/// Obtain `repeatTimes` parameter
::mlir::Value mlir::ascendc::UnaryL0Op::getRepeatTimes() {
      return getImpl()->getRepeatTimes(getImpl(), getOperation());
  }

/// Obtain `repeatParams` parameter
::mlir::Value mlir::ascendc::UnaryL0Op::getRepeatParams() {
      return getImpl()->getRepeatParams(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::UnaryL0Op::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::UnaryL0Op::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::UnaryL0Op::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::UnaryL0Op::getSrc() {
      return getImpl()->implUnaryOp->getSrc(getImpl()->implUnaryOp, getOperation());
  }

/// Obtain `calCount` parameter
::mlir::Value mlir::ascendc::UnaryL2Op::getCalCount() {
      return getImpl()->getCalCount(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::UnaryL2Op::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::UnaryL2Op::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::UnaryL2Op::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::UnaryL2Op::getSrc() {
      return getImpl()->implUnaryOp->getSrc(getImpl()->implUnaryOp, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::UnaryMathOp::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Obtain `sharedTmpBuffer` parameter
::mlir::Value mlir::ascendc::UnaryMathOp::getSharedTmpBuffer() {
      return getImpl()->getSharedTmpBuffer(getImpl(), getOperation());
  }

/// Obtain `calCount` parameter
::mlir::Value mlir::ascendc::UnaryMathOp::getCalCount() {
      return getImpl()->getCalCount(getImpl(), getOperation());
  }

/// Obtain `isReuseSource` parameter
::mlir::Value mlir::ascendc::UnaryMathOp::getIsReuseSource() {
      return getImpl()->getIsReuseSource(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::UnaryMathOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::UnaryMathOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::UnaryMathOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::UnaryOp::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::UnaryOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::UnaryOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::UnaryOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `mask` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getMask() {
      return getImpl()->getMask(getImpl(), getOperation());
  }

/// Obtain `repeatTimes` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getRepeatTimes() {
      return getImpl()->getRepeatTimes(getImpl(), getOperation());
  }

/// Obtain `repeatParams` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getRepeatParams() {
      return getImpl()->getRepeatParams(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::VecScalarL0Op::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::VecScalarL0Op::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getSrc() {
      return getImpl()->implVecScalarOp->getSrc(getImpl()->implVecScalarOp, getOperation());
  }

/// Obtain `scalar` parameter
::mlir::Value mlir::ascendc::VecScalarL0Op::getScalar() {
      return getImpl()->implVecScalarOp->getScalar(getImpl()->implVecScalarOp, getOperation());
  }

/// Obtain `calCount` parameter
::mlir::Value mlir::ascendc::VecScalarL2Op::getCalCount() {
      return getImpl()->getCalCount(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::VecScalarL2Op::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::VecScalarL2Op::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::VecScalarL2Op::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::VecScalarL2Op::getSrc() {
      return getImpl()->implVecScalarOp->getSrc(getImpl()->implVecScalarOp, getOperation());
  }

/// Obtain `scalar` parameter
::mlir::Value mlir::ascendc::VecScalarL2Op::getScalar() {
      return getImpl()->implVecScalarOp->getScalar(getImpl()->implVecScalarOp, getOperation());
  }

/// Obtain `src` parameter
::mlir::Value mlir::ascendc::VecScalarOp::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Obtain `scalar` parameter
::mlir::Value mlir::ascendc::VecScalarOp::getScalar() {
      return getImpl()->getScalar(getImpl(), getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::VecScalarOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::VecScalarOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }

/// Obtain `dst` parameter
::mlir::Value mlir::ascendc::VecScalarOp::getDst() {
      return getImpl()->implOpWithDst->getDst(getImpl()->implOpWithDst, getOperation());
  }

/// Obtain Ascend C library name
::llvm::StringRef mlir::ascendc::VectorOp::getAPIName() {
      return getImpl()->implAPIOp->getAPIName(getImpl()->implAPIOp, getOperation());
  }

/// Obtain describing comment
::llvm::StringRef mlir::ascendc::VectorOp::getComment() {
      return getImpl()->implAPIOp->getComment(getImpl()->implAPIOp, getOperation());
  }
