/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::afir {

class AbsOp;

} // namespace mlir::afir
namespace mlir::afir {

class AddOp;

} // namespace mlir::afir
namespace mlir::afir {

class AllOp;

} // namespace mlir::afir
namespace mlir::afir {

class AnyOp;

} // namespace mlir::afir
namespace mlir::afir {

class BitwiseAndOp;

} // namespace mlir::afir
namespace mlir::afir {

class BroadcastOp;

} // namespace mlir::afir
namespace mlir::afir {

class CastOp;

} // namespace mlir::afir
namespace mlir::afir {

class ClipByValue;

} // namespace mlir::afir
namespace mlir::afir {

class ConcatOp;

} // namespace mlir::afir
namespace mlir::afir {

class DivOp;

} // namespace mlir::afir
namespace mlir::afir {

class EqOp;

} // namespace mlir::afir
namespace mlir::afir {

class ErfOp;

} // namespace mlir::afir
namespace mlir::afir {

class ExpOp;

} // namespace mlir::afir
namespace mlir::afir {

class FloorDivOp;

} // namespace mlir::afir
namespace mlir::afir {

class GeOp;

} // namespace mlir::afir
namespace mlir::afir {

class GeluOp;

} // namespace mlir::afir
namespace mlir::afir {

class GtOp;

} // namespace mlir::afir
namespace mlir::afir {

class IndexExprOp;

} // namespace mlir::afir
namespace mlir::afir {

class IsFiniteOp;

} // namespace mlir::afir
namespace mlir::afir {

class IsnanOp;

} // namespace mlir::afir
namespace mlir::afir {

class LeOp;

} // namespace mlir::afir
namespace mlir::afir {

class LeakyReluOp;

} // namespace mlir::afir
namespace mlir::afir {

class LnOp;

} // namespace mlir::afir
namespace mlir::afir {

class LoadOp;

} // namespace mlir::afir
namespace mlir::afir {

class LogicalAndOp;

} // namespace mlir::afir
namespace mlir::afir {

class LogicalNotOp;

} // namespace mlir::afir
namespace mlir::afir {

class LogicalOrOp;

} // namespace mlir::afir
namespace mlir::afir {

class LtOp;

} // namespace mlir::afir
namespace mlir::afir {

class MaxOp;

} // namespace mlir::afir
namespace mlir::afir {

class MaximumOp;

} // namespace mlir::afir
namespace mlir::afir {

class MeanOp;

} // namespace mlir::afir
namespace mlir::afir {

class MinOp;

} // namespace mlir::afir
namespace mlir::afir {

class MinimumOp;

} // namespace mlir::afir
namespace mlir::afir {

class MulOp;

} // namespace mlir::afir
namespace mlir::afir {

class NeOp;

} // namespace mlir::afir
namespace mlir::afir {

class NegOp;

} // namespace mlir::afir
namespace mlir::afir {

class PowOp;

} // namespace mlir::afir
namespace mlir::afir {

class ProdOp;

} // namespace mlir::afir
namespace mlir::afir {

class ReciprocalOp;

} // namespace mlir::afir
namespace mlir::afir {

class ReluOp;

} // namespace mlir::afir
namespace mlir::afir {

class RsqrtOp;

} // namespace mlir::afir
namespace mlir::afir {

class ScalarOp;

} // namespace mlir::afir
namespace mlir::afir {

class SelectOp;

} // namespace mlir::afir
namespace mlir::afir {

class SigmoidOp;

} // namespace mlir::afir
namespace mlir::afir {

class SignOp;

} // namespace mlir::afir
namespace mlir::afir {

class SqrtOp;

} // namespace mlir::afir
namespace mlir::afir {

class StoreOp;

} // namespace mlir::afir
namespace mlir::afir {

class SubOp;

} // namespace mlir::afir
namespace mlir::afir {

class SumOp;

} // namespace mlir::afir
namespace mlir::afir {

class TanhOp;

} // namespace mlir::afir
namespace mlir::afir {

class TrueDivOp;

} // namespace mlir::afir
namespace mlir::afir {

class WhereOp;

} // namespace mlir::afir
namespace mlir::afir {

class TransposeOp;

} // namespace mlir::afir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::AbsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AbsOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AbsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.abs", odsAttrs.getContext());
  }

  AbsOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class AbsOpGenericAdaptor : public detail::AbsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AbsOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  AbsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AbsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AbsOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  AbsOpGenericAdaptor(RangeT values, const AbsOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AbsOp, typename = std::enable_if_t<std::is_same_v<LateInst, AbsOp>>>
  AbsOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AbsOpAdaptor : public AbsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AbsOpGenericAdaptor::AbsOpGenericAdaptor;
  AbsOpAdaptor(AbsOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AbsOp : public ::mlir::Op<AbsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AbsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AbsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.abs");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AbsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AbsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AbsOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::AddOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AddOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.add", odsAttrs.getContext());
  }

  AddOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class AddOpGenericAdaptor : public detail::AddOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AddOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  AddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AddOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  AddOpGenericAdaptor(RangeT values, const AddOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AddOp, typename = std::enable_if_t<std::is_same_v<LateInst, AddOp>>>
  AddOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AddOpAdaptor : public AddOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AddOpGenericAdaptor::AddOpGenericAdaptor;
  AddOpAdaptor(AddOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AddOp : public ::mlir::Op<AddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AddOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AddOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<AddOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<AddOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.add");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AddOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::AllOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AllOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.all", odsAttrs.getContext());
  }

  AllOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class AllOpGenericAdaptor : public detail::AllOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  AllOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AllOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  AllOpGenericAdaptor(RangeT values, const AllOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllOp, typename = std::enable_if_t<std::is_same_v<LateInst, AllOp>>>
  AllOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllOpAdaptor : public AllOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllOpGenericAdaptor::AllOpGenericAdaptor;
  AllOpAdaptor(AllOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllOp : public ::mlir::Op<AllOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<AllOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<AllOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.all");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AllOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::AnyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AnyOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AnyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.any", odsAttrs.getContext());
  }

  AnyOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class AnyOpGenericAdaptor : public detail::AnyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AnyOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  AnyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AnyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AnyOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  AnyOpGenericAdaptor(RangeT values, const AnyOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AnyOp, typename = std::enable_if_t<std::is_same_v<LateInst, AnyOp>>>
  AnyOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AnyOpAdaptor : public AnyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AnyOpGenericAdaptor::AnyOpGenericAdaptor;
  AnyOpAdaptor(AnyOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AnyOp : public ::mlir::Op<AnyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AnyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AnyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<AnyOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<AnyOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.any");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AnyOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AnyOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AnyOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::BitwiseAndOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class BitwiseAndOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  BitwiseAndOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.bitwise_and", odsAttrs.getContext());
  }

  BitwiseAndOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class BitwiseAndOpGenericAdaptor : public detail::BitwiseAndOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BitwiseAndOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  BitwiseAndOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  BitwiseAndOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : BitwiseAndOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  BitwiseAndOpGenericAdaptor(RangeT values, const BitwiseAndOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = BitwiseAndOp, typename = std::enable_if_t<std::is_same_v<LateInst, BitwiseAndOp>>>
  BitwiseAndOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BitwiseAndOpAdaptor : public BitwiseAndOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BitwiseAndOpGenericAdaptor::BitwiseAndOpGenericAdaptor;
  BitwiseAndOpAdaptor(BitwiseAndOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class BitwiseAndOp : public ::mlir::Op<BitwiseAndOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BitwiseAndOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BitwiseAndOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<BitwiseAndOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<BitwiseAndOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.bitwise_and");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BitwiseAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BitwiseAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::BitwiseAndOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::BroadcastOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class BroadcastOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  BroadcastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.broadcast", odsAttrs.getContext());
  }

  BroadcastOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class BroadcastOpGenericAdaptor : public detail::BroadcastOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BroadcastOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  BroadcastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  BroadcastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : BroadcastOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  BroadcastOpGenericAdaptor(RangeT values, const BroadcastOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = BroadcastOp, typename = std::enable_if_t<std::is_same_v<LateInst, BroadcastOp>>>
  BroadcastOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BroadcastOpAdaptor : public BroadcastOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BroadcastOpGenericAdaptor::BroadcastOpGenericAdaptor;
  BroadcastOpAdaptor(BroadcastOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class BroadcastOp : public ::mlir::Op<BroadcastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BroadcastOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BroadcastOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.broadcast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BroadcastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BroadcastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BroadcastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static BroadcastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BroadcastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BroadcastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BroadcastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static BroadcastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BroadcastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BroadcastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Broadcast;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::BroadcastOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::CastOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CastOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  CastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.cast", odsAttrs.getContext());
  }

  CastOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class CastOpGenericAdaptor : public detail::CastOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CastOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  CastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CastOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  CastOpGenericAdaptor(RangeT values, const CastOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CastOp, typename = std::enable_if_t<std::is_same_v<LateInst, CastOp>>>
  CastOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CastOpAdaptor : public CastOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CastOpGenericAdaptor::CastOpGenericAdaptor;
  CastOpAdaptor(CastOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CastOp : public ::mlir::Op<CastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CastOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CastOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.cast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static CastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static CastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static CastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static CastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static CastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static CastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static CastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static CastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CastOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CastOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::CastOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ClipByValue declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ClipByValueGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ClipByValueGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.clip_by_value", odsAttrs.getContext());
  }

  ClipByValueGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ClipByValueGenericAdaptor : public detail::ClipByValueGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ClipByValueGenericAdaptorBase;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
public:
  ClipByValueGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ClipByValueGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ClipByValueGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ClipByValueGenericAdaptor(RangeT values, const ClipByValueGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ClipByValue, typename = std::enable_if_t<std::is_same_v<LateInst, ClipByValue>>>
  ClipByValueGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput1() {
    return (*getODSOperands(0).begin());
  }

  ValueT getInput2() {
    return (*getODSOperands(1).begin());
  }

  ValueT getInput3() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ClipByValueAdaptor : public ClipByValueGenericAdaptor<::mlir::ValueRange> {
public:
  using ClipByValueGenericAdaptor::ClipByValueGenericAdaptor;
  ClipByValueAdaptor(ClipByValue op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ClipByValue : public ::mlir::Op<ClipByValue, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClipByValueAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ClipByValueGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
  using ::mlir::InferTypeOpInterface::Trait<ClipByValue>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<ClipByValue>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.clip_by_value");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput1() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput2() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput3() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getInput1Mutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput2Mutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput3Mutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClipByValue create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClipByValue create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ClipByValue)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ConcatOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConcatOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ConcatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.concat", odsAttrs.getContext());
  }

  ConcatOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getConcatAxisAttr();
  uint32_t getConcatAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ConcatOpGenericAdaptor : public detail::ConcatOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConcatOpGenericAdaptorBase;
  static constexpr int odsIndex_inputs = 0;
public:
  ConcatOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConcatOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConcatOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ConcatOpGenericAdaptor(RangeT values, const ConcatOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConcatOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConcatOp>>>
  ConcatOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConcatOpAdaptor : public ConcatOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConcatOpGenericAdaptor::ConcatOpGenericAdaptor;
  ConcatOpAdaptor(ConcatOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConcatOp : public ::mlir::Op<ConcatOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConcatOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConcatOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_inputs = 0;
  using ::mlir::InferTypeOpInterface::Trait<ConcatOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<ConcatOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("concat_axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getConcatAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getConcatAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.concat");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getInputs() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getConcatAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getConcatAxisAttrName()));
  }

  uint32_t getConcatAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setConcatAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getConcatAxisAttrName(), attr);
  }

  void setConcatAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::IntegerAttr concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, uint32_t concat_axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConcatOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConcatOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ConcatOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::DivOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DivOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  DivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.div", odsAttrs.getContext());
  }

  DivOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class DivOpGenericAdaptor : public detail::DivOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DivOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  DivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DivOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  DivOpGenericAdaptor(RangeT values, const DivOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DivOp, typename = std::enable_if_t<std::is_same_v<LateInst, DivOp>>>
  DivOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DivOpAdaptor : public DivOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DivOpGenericAdaptor::DivOpGenericAdaptor;
  DivOpAdaptor(DivOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DivOp : public ::mlir::Op<DivOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DivOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DivOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<DivOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<DivOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.div");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::DivOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::EqOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class EqOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  EqOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.eq", odsAttrs.getContext());
  }

  EqOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class EqOpGenericAdaptor : public detail::EqOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EqOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  EqOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  EqOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : EqOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  EqOpGenericAdaptor(RangeT values, const EqOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = EqOp, typename = std::enable_if_t<std::is_same_v<LateInst, EqOp>>>
  EqOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class EqOpAdaptor : public EqOpGenericAdaptor<::mlir::ValueRange> {
public:
  using EqOpGenericAdaptor::EqOpGenericAdaptor;
  EqOpAdaptor(EqOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class EqOp : public ::mlir::Op<EqOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EqOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EqOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<EqOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<EqOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.eq");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EqOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static EqOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::EqOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ErfOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ErfOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ErfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.erf", odsAttrs.getContext());
  }

  ErfOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ErfOpGenericAdaptor : public detail::ErfOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ErfOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  ErfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ErfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ErfOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ErfOpGenericAdaptor(RangeT values, const ErfOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ErfOp, typename = std::enable_if_t<std::is_same_v<LateInst, ErfOp>>>
  ErfOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ErfOpAdaptor : public ErfOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ErfOpGenericAdaptor::ErfOpGenericAdaptor;
  ErfOpAdaptor(ErfOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ErfOp : public ::mlir::Op<ErfOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ErfOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ErfOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.erf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErfOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErfOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ErfOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ExpOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExpOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ExpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.exp", odsAttrs.getContext());
  }

  ExpOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ExpOpGenericAdaptor : public detail::ExpOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExpOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  ExpOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExpOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExpOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ExpOpGenericAdaptor(RangeT values, const ExpOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExpOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExpOp>>>
  ExpOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExpOpAdaptor : public ExpOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExpOpGenericAdaptor::ExpOpGenericAdaptor;
  ExpOpAdaptor(ExpOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExpOp : public ::mlir::Op<ExpOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExpOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExpOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.exp");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExpOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExpOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ExpOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::FloorDivOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class FloorDivOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  FloorDivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.floor_div", odsAttrs.getContext());
  }

  FloorDivOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class FloorDivOpGenericAdaptor : public detail::FloorDivOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FloorDivOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  FloorDivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FloorDivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FloorDivOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  FloorDivOpGenericAdaptor(RangeT values, const FloorDivOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = FloorDivOp, typename = std::enable_if_t<std::is_same_v<LateInst, FloorDivOp>>>
  FloorDivOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FloorDivOpAdaptor : public FloorDivOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FloorDivOpGenericAdaptor::FloorDivOpGenericAdaptor;
  FloorDivOpAdaptor(FloorDivOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class FloorDivOp : public ::mlir::Op<FloorDivOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FloorDivOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FloorDivOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<FloorDivOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<FloorDivOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.floor_div");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FloorDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FloorDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::FloorDivOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::GeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  GeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.ge", odsAttrs.getContext());
  }

  GeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class GeOpGenericAdaptor : public detail::GeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GeOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  GeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  GeOpGenericAdaptor(RangeT values, const GeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GeOp, typename = std::enable_if_t<std::is_same_v<LateInst, GeOp>>>
  GeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GeOpAdaptor : public GeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GeOpGenericAdaptor::GeOpGenericAdaptor;
  GeOpAdaptor(GeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GeOp : public ::mlir::Op<GeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<GeOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<GeOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.ge");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::GeOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::GeluOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GeluOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  GeluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.gelu", odsAttrs.getContext());
  }

  GeluOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class GeluOpGenericAdaptor : public detail::GeluOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GeluOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  GeluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GeluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GeluOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  GeluOpGenericAdaptor(RangeT values, const GeluOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GeluOp, typename = std::enable_if_t<std::is_same_v<LateInst, GeluOp>>>
  GeluOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GeluOpAdaptor : public GeluOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GeluOpGenericAdaptor::GeluOpGenericAdaptor;
  GeluOpAdaptor(GeluOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GeluOp : public ::mlir::Op<GeluOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GeluOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GeluOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<GeluOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<GeluOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.gelu");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GeluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::GeluOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::GtOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GtOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  GtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.gt", odsAttrs.getContext());
  }

  GtOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class GtOpGenericAdaptor : public detail::GtOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GtOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  GtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GtOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  GtOpGenericAdaptor(RangeT values, const GtOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GtOp, typename = std::enable_if_t<std::is_same_v<LateInst, GtOp>>>
  GtOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GtOpAdaptor : public GtOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GtOpGenericAdaptor::GtOpGenericAdaptor;
  GtOpAdaptor(GtOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GtOp : public ::mlir::Op<GtOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GtOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GtOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<GtOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<GtOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.gt");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::GtOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::IndexExprOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class IndexExprOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  IndexExprOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.index_expr", odsAttrs.getContext());
  }

  IndexExprOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class IndexExprOpGenericAdaptor : public detail::IndexExprOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IndexExprOpGenericAdaptorBase;
public:
  IndexExprOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  IndexExprOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : IndexExprOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  IndexExprOpGenericAdaptor(RangeT values, const IndexExprOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = IndexExprOp, typename = std::enable_if_t<std::is_same_v<LateInst, IndexExprOp>>>
  IndexExprOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IndexExprOpAdaptor : public IndexExprOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IndexExprOpGenericAdaptor::IndexExprOpGenericAdaptor;
  IndexExprOpAdaptor(IndexExprOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class IndexExprOp : public ::mlir::Op<IndexExprOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IndexExprOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IndexExprOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.index_expr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IndexExprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IndexExprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IndexExprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IndexExprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IndexExprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IndexExprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IndexExprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IndexExprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IndexExprOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IndexExprOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 0; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Invalid;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::IndexExprOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::IsFiniteOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class IsFiniteOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  IsFiniteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.isfinite", odsAttrs.getContext());
  }

  IsFiniteOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class IsFiniteOpGenericAdaptor : public detail::IsFiniteOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IsFiniteOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  IsFiniteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  IsFiniteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : IsFiniteOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  IsFiniteOpGenericAdaptor(RangeT values, const IsFiniteOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = IsFiniteOp, typename = std::enable_if_t<std::is_same_v<LateInst, IsFiniteOp>>>
  IsFiniteOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IsFiniteOpAdaptor : public IsFiniteOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IsFiniteOpGenericAdaptor::IsFiniteOpGenericAdaptor;
  IsFiniteOpAdaptor(IsFiniteOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class IsFiniteOp : public ::mlir::Op<IsFiniteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IsFiniteOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IsFiniteOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.isfinite");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsFiniteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsFiniteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsFiniteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsFiniteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsFiniteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsFiniteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsFiniteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsFiniteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IsFiniteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IsFiniteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::IsFiniteOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::IsnanOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class IsnanOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  IsnanOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.isnan", odsAttrs.getContext());
  }

  IsnanOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class IsnanOpGenericAdaptor : public detail::IsnanOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IsnanOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  IsnanOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  IsnanOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : IsnanOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  IsnanOpGenericAdaptor(RangeT values, const IsnanOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = IsnanOp, typename = std::enable_if_t<std::is_same_v<LateInst, IsnanOp>>>
  IsnanOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IsnanOpAdaptor : public IsnanOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IsnanOpGenericAdaptor::IsnanOpGenericAdaptor;
  IsnanOpAdaptor(IsnanOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class IsnanOp : public ::mlir::Op<IsnanOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IsnanOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IsnanOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.isnan");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsnanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsnanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsnanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static IsnanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsnanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsnanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsnanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static IsnanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IsnanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IsnanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::IsnanOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.le", odsAttrs.getContext());
  }

  LeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LeOpGenericAdaptor : public detail::LeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LeOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  LeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LeOpGenericAdaptor(RangeT values, const LeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LeOp, typename = std::enable_if_t<std::is_same_v<LateInst, LeOp>>>
  LeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LeOpAdaptor : public LeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LeOpGenericAdaptor::LeOpGenericAdaptor;
  LeOpAdaptor(LeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LeOp : public ::mlir::Op<LeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<LeOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<LeOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.le");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LeOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LeakyReluOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LeakyReluOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LeakyReluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.leaky_relu", odsAttrs.getContext());
  }

  LeakyReluOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::FloatAttr getNegativeSlopAttr();
  ::llvm::APFloat getNegativeSlop();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LeakyReluOpGenericAdaptor : public detail::LeakyReluOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LeakyReluOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  LeakyReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LeakyReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LeakyReluOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LeakyReluOpGenericAdaptor(RangeT values, const LeakyReluOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LeakyReluOp, typename = std::enable_if_t<std::is_same_v<LateInst, LeakyReluOp>>>
  LeakyReluOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LeakyReluOpAdaptor : public LeakyReluOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LeakyReluOpGenericAdaptor::LeakyReluOpGenericAdaptor;
  LeakyReluOpAdaptor(LeakyReluOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LeakyReluOp : public ::mlir::Op<LeakyReluOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LeakyReluOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LeakyReluOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("negative_slop"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getNegativeSlopAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getNegativeSlopAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.leaky_relu");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::FloatAttr getNegativeSlopAttr() {
    return ::llvm::cast<::mlir::FloatAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNegativeSlopAttrName()));
  }

  ::llvm::APFloat getNegativeSlop();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setNegativeSlopAttr(::mlir::FloatAttr attr) {
    (*this)->setAttr(getNegativeSlopAttrName(), attr);
  }

  void setNegativeSlop(::llvm::APFloat attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::FloatAttr negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::APFloat negative_slop, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeakyReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LeakyReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LeakyReluOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LnOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LnOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.ln", odsAttrs.getContext());
  }

  LnOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LnOpGenericAdaptor : public detail::LnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LnOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  LnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LnOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LnOpGenericAdaptor(RangeT values, const LnOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LnOp, typename = std::enable_if_t<std::is_same_v<LateInst, LnOp>>>
  LnOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LnOpAdaptor : public LnOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LnOpGenericAdaptor::LnOpGenericAdaptor;
  LnOpAdaptor(LnOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LnOp : public ::mlir::Op<LnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.ln");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LnOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoadOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.load", odsAttrs.getContext());
  }

  LoadOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LoadOpGenericAdaptor : public detail::LoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  LoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LoadOpGenericAdaptor(RangeT values, const LoadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoadOp>>>
  LoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoadOpAdaptor : public LoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadOpGenericAdaptor::LoadOpGenericAdaptor;
  LoadOpAdaptor(LoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoadOp : public ::mlir::Op<LoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Load;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::MTE2;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LoadOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LogicalAndOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LogicalAndOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LogicalAndOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.logical_and", odsAttrs.getContext());
  }

  LogicalAndOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LogicalAndOpGenericAdaptor : public detail::LogicalAndOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LogicalAndOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  LogicalAndOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LogicalAndOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LogicalAndOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LogicalAndOpGenericAdaptor(RangeT values, const LogicalAndOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LogicalAndOp, typename = std::enable_if_t<std::is_same_v<LateInst, LogicalAndOp>>>
  LogicalAndOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LogicalAndOpAdaptor : public LogicalAndOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LogicalAndOpGenericAdaptor::LogicalAndOpGenericAdaptor;
  LogicalAndOpAdaptor(LogicalAndOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LogicalAndOp : public ::mlir::Op<LogicalAndOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LogicalAndOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LogicalAndOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<LogicalAndOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<LogicalAndOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.logical_and");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalAndOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalAndOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LogicalAndOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LogicalNotOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LogicalNotOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LogicalNotOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.logical_not", odsAttrs.getContext());
  }

  LogicalNotOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LogicalNotOpGenericAdaptor : public detail::LogicalNotOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LogicalNotOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  LogicalNotOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LogicalNotOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LogicalNotOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LogicalNotOpGenericAdaptor(RangeT values, const LogicalNotOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LogicalNotOp, typename = std::enable_if_t<std::is_same_v<LateInst, LogicalNotOp>>>
  LogicalNotOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LogicalNotOpAdaptor : public LogicalNotOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LogicalNotOpGenericAdaptor::LogicalNotOpGenericAdaptor;
  LogicalNotOpAdaptor(LogicalNotOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LogicalNotOp : public ::mlir::Op<LogicalNotOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LogicalNotOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LogicalNotOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.logical_not");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalNotOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalNotOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LogicalNotOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LogicalOrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LogicalOrOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LogicalOrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.logical_or", odsAttrs.getContext());
  }

  LogicalOrOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LogicalOrOpGenericAdaptor : public detail::LogicalOrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LogicalOrOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  LogicalOrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LogicalOrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LogicalOrOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LogicalOrOpGenericAdaptor(RangeT values, const LogicalOrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LogicalOrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LogicalOrOp>>>
  LogicalOrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LogicalOrOpAdaptor : public LogicalOrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LogicalOrOpGenericAdaptor::LogicalOrOpGenericAdaptor;
  LogicalOrOpAdaptor(LogicalOrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LogicalOrOp : public ::mlir::Op<LogicalOrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LogicalOrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LogicalOrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<LogicalOrOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<LogicalOrOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.logical_or");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalOrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LogicalOrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LogicalOrOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::LtOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LtOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  LtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.lt", odsAttrs.getContext());
  }

  LtOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class LtOpGenericAdaptor : public detail::LtOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LtOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  LtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LtOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  LtOpGenericAdaptor(RangeT values, const LtOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LtOp, typename = std::enable_if_t<std::is_same_v<LateInst, LtOp>>>
  LtOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LtOpAdaptor : public LtOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LtOpGenericAdaptor::LtOpGenericAdaptor;
  LtOpAdaptor(LtOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LtOp : public ::mlir::Op<LtOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LtOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LtOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<LtOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<LtOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.lt");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::LtOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MaxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MaxOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.max", odsAttrs.getContext());
  }

  MaxOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MaxOpGenericAdaptor : public detail::MaxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MaxOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  MaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MaxOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MaxOpGenericAdaptor(RangeT values, const MaxOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MaxOp, typename = std::enable_if_t<std::is_same_v<LateInst, MaxOp>>>
  MaxOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MaxOpAdaptor : public MaxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MaxOpGenericAdaptor::MaxOpGenericAdaptor;
  MaxOpAdaptor(MaxOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MaxOp : public ::mlir::Op<MaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MaxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<MaxOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MaxOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaxOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaxOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MaxOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MaximumOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MaximumOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MaximumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.maximum", odsAttrs.getContext());
  }

  MaximumOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MaximumOpGenericAdaptor : public detail::MaximumOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MaximumOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  MaximumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MaximumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MaximumOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MaximumOpGenericAdaptor(RangeT values, const MaximumOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MaximumOp, typename = std::enable_if_t<std::is_same_v<LateInst, MaximumOp>>>
  MaximumOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MaximumOpAdaptor : public MaximumOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MaximumOpGenericAdaptor::MaximumOpGenericAdaptor;
  MaximumOpAdaptor(MaximumOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MaximumOp : public ::mlir::Op<MaximumOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MaximumOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MaximumOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<MaximumOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MaximumOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.maximum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaximumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MaximumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MaximumOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MeanOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MeanOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MeanOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.mean", odsAttrs.getContext());
  }

  MeanOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MeanOpGenericAdaptor : public detail::MeanOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MeanOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  MeanOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MeanOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MeanOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MeanOpGenericAdaptor(RangeT values, const MeanOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MeanOp, typename = std::enable_if_t<std::is_same_v<LateInst, MeanOp>>>
  MeanOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MeanOpAdaptor : public MeanOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MeanOpGenericAdaptor::MeanOpGenericAdaptor;
  MeanOpAdaptor(MeanOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MeanOp : public ::mlir::Op<MeanOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MeanOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MeanOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<MeanOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MeanOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.mean");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MeanOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MeanOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MeanOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MinOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MinOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.min", odsAttrs.getContext());
  }

  MinOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MinOpGenericAdaptor : public detail::MinOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MinOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  MinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MinOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MinOpGenericAdaptor(RangeT values, const MinOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MinOp, typename = std::enable_if_t<std::is_same_v<LateInst, MinOp>>>
  MinOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MinOpAdaptor : public MinOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MinOpGenericAdaptor::MinOpGenericAdaptor;
  MinOpAdaptor(MinOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MinOp : public ::mlir::Op<MinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MinOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MinOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<MinOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MinOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.min");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MinOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MinimumOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MinimumOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MinimumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.minimum", odsAttrs.getContext());
  }

  MinimumOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MinimumOpGenericAdaptor : public detail::MinimumOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MinimumOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  MinimumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MinimumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MinimumOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MinimumOpGenericAdaptor(RangeT values, const MinimumOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MinimumOp, typename = std::enable_if_t<std::is_same_v<LateInst, MinimumOp>>>
  MinimumOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MinimumOpAdaptor : public MinimumOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MinimumOpGenericAdaptor::MinimumOpGenericAdaptor;
  MinimumOpAdaptor(MinimumOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MinimumOp : public ::mlir::Op<MinimumOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MinimumOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MinimumOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<MinimumOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MinimumOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.minimum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinimumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MinimumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MinimumOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::MulOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MulOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.mul", odsAttrs.getContext());
  }

  MulOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class MulOpGenericAdaptor : public detail::MulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MulOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  MulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MulOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  MulOpGenericAdaptor(RangeT values, const MulOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MulOp, typename = std::enable_if_t<std::is_same_v<LateInst, MulOp>>>
  MulOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MulOpAdaptor : public MulOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MulOpGenericAdaptor::MulOpGenericAdaptor;
  MulOpAdaptor(MulOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MulOp : public ::mlir::Op<MulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<MulOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<MulOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.mul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::MulOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::NeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class NeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  NeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.ne", odsAttrs.getContext());
  }

  NeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class NeOpGenericAdaptor : public detail::NeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NeOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  NeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  NeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : NeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  NeOpGenericAdaptor(RangeT values, const NeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = NeOp, typename = std::enable_if_t<std::is_same_v<LateInst, NeOp>>>
  NeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NeOpAdaptor : public NeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NeOpGenericAdaptor::NeOpGenericAdaptor;
  NeOpAdaptor(NeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class NeOp : public ::mlir::Op<NeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<NeOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<NeOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.ne");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


    int getNumberOfOperands() { return 2; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::NeOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::NegOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class NegOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  NegOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.neg", odsAttrs.getContext());
  }

  NegOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class NegOpGenericAdaptor : public detail::NegOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::NegOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  NegOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  NegOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : NegOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  NegOpGenericAdaptor(RangeT values, const NegOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = NegOp, typename = std::enable_if_t<std::is_same_v<LateInst, NegOp>>>
  NegOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class NegOpAdaptor : public NegOpGenericAdaptor<::mlir::ValueRange> {
public:
  using NegOpGenericAdaptor::NegOpGenericAdaptor;
  NegOpAdaptor(NegOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class NegOp : public ::mlir::Op<NegOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = NegOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = NegOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.neg");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NegOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static NegOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::NegOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::PowOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class PowOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  PowOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.pow", odsAttrs.getContext());
  }

  PowOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class PowOpGenericAdaptor : public detail::PowOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PowOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  PowOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PowOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PowOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  PowOpGenericAdaptor(RangeT values, const PowOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PowOp, typename = std::enable_if_t<std::is_same_v<LateInst, PowOp>>>
  PowOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PowOpAdaptor : public PowOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PowOpGenericAdaptor::PowOpGenericAdaptor;
  PowOpAdaptor(PowOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PowOp : public ::mlir::Op<PowOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PowOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PowOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<PowOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<PowOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.pow");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::PowOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ProdOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ProdOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ProdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.prod", odsAttrs.getContext());
  }

  ProdOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ProdOpGenericAdaptor : public detail::ProdOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ProdOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  ProdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ProdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ProdOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ProdOpGenericAdaptor(RangeT values, const ProdOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ProdOp, typename = std::enable_if_t<std::is_same_v<LateInst, ProdOp>>>
  ProdOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ProdOpAdaptor : public ProdOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ProdOpGenericAdaptor::ProdOpGenericAdaptor;
  ProdOpAdaptor(ProdOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ProdOp : public ::mlir::Op<ProdOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ProdOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ProdOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<ProdOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<ProdOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.prod");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ProdOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ReciprocalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ReciprocalOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ReciprocalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.reciprocal", odsAttrs.getContext());
  }

  ReciprocalOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ReciprocalOpGenericAdaptor : public detail::ReciprocalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReciprocalOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  ReciprocalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReciprocalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ReciprocalOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ReciprocalOpGenericAdaptor(RangeT values, const ReciprocalOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ReciprocalOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReciprocalOp>>>
  ReciprocalOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReciprocalOpAdaptor : public ReciprocalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReciprocalOpGenericAdaptor::ReciprocalOpGenericAdaptor;
  ReciprocalOpAdaptor(ReciprocalOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ReciprocalOp : public ::mlir::Op<ReciprocalOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReciprocalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReciprocalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.reciprocal");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReciprocalOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReciprocalOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ReciprocalOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ReluOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ReluOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ReluOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.relu", odsAttrs.getContext());
  }

  ReluOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ReluOpGenericAdaptor : public detail::ReluOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReluOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  ReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReluOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ReluOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ReluOpGenericAdaptor(RangeT values, const ReluOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ReluOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReluOp>>>
  ReluOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReluOpAdaptor : public ReluOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReluOpGenericAdaptor::ReluOpGenericAdaptor;
  ReluOpAdaptor(ReluOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ReluOp : public ::mlir::Op<ReluOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReluOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReluOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.relu");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReluOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ReluOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ReluOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::RsqrtOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RsqrtOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  RsqrtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.rsqrt", odsAttrs.getContext());
  }

  RsqrtOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class RsqrtOpGenericAdaptor : public detail::RsqrtOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RsqrtOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  RsqrtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RsqrtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : RsqrtOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  RsqrtOpGenericAdaptor(RangeT values, const RsqrtOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RsqrtOp, typename = std::enable_if_t<std::is_same_v<LateInst, RsqrtOp>>>
  RsqrtOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RsqrtOpAdaptor : public RsqrtOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RsqrtOpGenericAdaptor::RsqrtOpGenericAdaptor;
  RsqrtOpAdaptor(RsqrtOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RsqrtOp : public ::mlir::Op<RsqrtOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RsqrtOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RsqrtOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.rsqrt");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RsqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RsqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::RsqrtOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::ScalarOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ScalarOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ScalarOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.scalar", odsAttrs.getContext());
  }

  ScalarOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::Attribute getValueAttr();
  ::mlir::Attribute getValue();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class ScalarOpGenericAdaptor : public detail::ScalarOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ScalarOpGenericAdaptorBase;
public:
  ScalarOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ScalarOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ScalarOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  ScalarOpGenericAdaptor(RangeT values, const ScalarOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ScalarOp, typename = std::enable_if_t<std::is_same_v<LateInst, ScalarOp>>>
  ScalarOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ScalarOpAdaptor : public ScalarOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ScalarOpGenericAdaptor::ScalarOpGenericAdaptor;
  ScalarOpAdaptor(ScalarOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ScalarOp : public ::mlir::Op<ScalarOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalarOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ScalarOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers"), ::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getValueAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.scalar");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::Attribute getValueAttr() {
    return ::llvm::cast<::mlir::Attribute>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()));
  }

  ::mlir::Attribute getValue();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setValueAttr(::mlir::Attribute attr) {
    (*this)->setAttr(getValueAttrName(), attr);
  }

  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ScalarOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ScalarOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ScalarOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static ScalarOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ScalarOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ScalarOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ScalarOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static ScalarOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ScalarOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ScalarOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 0; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Invalid;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Scalar;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ScalarOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SelectOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SelectOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SelectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.select", odsAttrs.getContext());
  }

  SelectOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SelectOpGenericAdaptor : public detail::SelectOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SelectOpGenericAdaptorBase;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
public:
  SelectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SelectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SelectOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SelectOpGenericAdaptor(RangeT values, const SelectOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SelectOp, typename = std::enable_if_t<std::is_same_v<LateInst, SelectOp>>>
  SelectOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput1() {
    return (*getODSOperands(0).begin());
  }

  ValueT getInput2() {
    return (*getODSOperands(1).begin());
  }

  ValueT getInput3() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SelectOpAdaptor : public SelectOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SelectOpGenericAdaptor::SelectOpGenericAdaptor;
  SelectOpAdaptor(SelectOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SelectOp : public ::mlir::Op<SelectOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SelectOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SelectOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.select");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput1() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput2() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput3() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getInput1Mutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput2Mutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput3Mutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SelectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SelectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SelectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SelectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SelectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SelectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SelectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SelectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SelectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SelectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 3; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SelectOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SigmoidOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SigmoidOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SigmoidOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.sigmoid", odsAttrs.getContext());
  }

  SigmoidOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SigmoidOpGenericAdaptor : public detail::SigmoidOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SigmoidOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  SigmoidOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SigmoidOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SigmoidOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SigmoidOpGenericAdaptor(RangeT values, const SigmoidOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SigmoidOp, typename = std::enable_if_t<std::is_same_v<LateInst, SigmoidOp>>>
  SigmoidOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SigmoidOpAdaptor : public SigmoidOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SigmoidOpGenericAdaptor::SigmoidOpGenericAdaptor;
  SigmoidOpAdaptor(SigmoidOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SigmoidOp : public ::mlir::Op<SigmoidOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SigmoidOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SigmoidOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.sigmoid");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SigmoidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SigmoidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SigmoidOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SignOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SignOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SignOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.sign", odsAttrs.getContext());
  }

  SignOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SignOpGenericAdaptor : public detail::SignOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SignOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  SignOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SignOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SignOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SignOpGenericAdaptor(RangeT values, const SignOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SignOp, typename = std::enable_if_t<std::is_same_v<LateInst, SignOp>>>
  SignOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SignOpAdaptor : public SignOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SignOpGenericAdaptor::SignOpGenericAdaptor;
  SignOpAdaptor(SignOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SignOp : public ::mlir::Op<SignOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SignOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SignOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<SignOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<SignOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.sign");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SignOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SignOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SignOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SqrtOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SqrtOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SqrtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.sqrt", odsAttrs.getContext());
  }

  SqrtOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SqrtOpGenericAdaptor : public detail::SqrtOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SqrtOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  SqrtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SqrtOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SqrtOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SqrtOpGenericAdaptor(RangeT values, const SqrtOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SqrtOp, typename = std::enable_if_t<std::is_same_v<LateInst, SqrtOp>>>
  SqrtOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SqrtOpAdaptor : public SqrtOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SqrtOpGenericAdaptor::SqrtOpGenericAdaptor;
  SqrtOpAdaptor(SqrtOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SqrtOp : public ::mlir::Op<SqrtOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SqrtOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SqrtOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.sqrt");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqrtOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqrtOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SqrtOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::StoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class StoreOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  StoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.store", odsAttrs.getContext());
  }

  StoreOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class StoreOpGenericAdaptor : public detail::StoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::StoreOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  StoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  StoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : StoreOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  StoreOpGenericAdaptor(RangeT values, const StoreOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = StoreOp, typename = std::enable_if_t<std::is_same_v<LateInst, StoreOp>>>
  StoreOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class StoreOpAdaptor : public StoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using StoreOpGenericAdaptor::StoreOpGenericAdaptor;
  StoreOpAdaptor(StoreOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class StoreOp : public ::mlir::Op<StoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = StoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static StoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static StoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static StoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static StoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static StoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static StoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static StoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static StoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StoreOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static StoreOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 1; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Store;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::MTE3;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::StoreOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SubOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SubOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SubOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.sub", odsAttrs.getContext());
  }

  SubOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SubOpGenericAdaptor : public detail::SubOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SubOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  SubOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SubOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SubOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SubOpGenericAdaptor(RangeT values, const SubOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SubOp, typename = std::enable_if_t<std::is_same_v<LateInst, SubOp>>>
  SubOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SubOpAdaptor : public SubOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SubOpGenericAdaptor::SubOpGenericAdaptor;
  SubOpAdaptor(SubOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SubOp : public ::mlir::Op<SubOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SubOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<SubOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<SubOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.sub");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SubOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SubOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SubOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::SumOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SumOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  SumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.sum", odsAttrs.getContext());
  }

  SumOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr();
  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class SumOpGenericAdaptor : public detail::SumOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SumOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  SumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SumOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SumOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  SumOpGenericAdaptor(RangeT values, const SumOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SumOp, typename = std::enable_if_t<std::is_same_v<LateInst, SumOp>>>
  SumOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SumOpAdaptor : public SumOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SumOpGenericAdaptor::SumOpGenericAdaptor;
  SumOpAdaptor(SumOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SumOp : public ::mlir::Op<SumOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SumOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SumOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<SumOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<SumOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.sum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()));
  }

  uint32_t getAxis();
  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getAxisAttrName(), attr);
  }

  void setAxis(uint32_t attrValue);
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t axis, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SumOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SumOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::SumOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::TanhOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TanhOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TanhOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.tanh", odsAttrs.getContext());
  }

  TanhOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class TanhOpGenericAdaptor : public detail::TanhOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TanhOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  TanhOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TanhOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TanhOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  TanhOpGenericAdaptor(RangeT values, const TanhOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TanhOp, typename = std::enable_if_t<std::is_same_v<LateInst, TanhOp>>>
  TanhOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TanhOpAdaptor : public TanhOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TanhOpGenericAdaptor::TanhOpGenericAdaptor;
  TanhOpAdaptor(TanhOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TanhOp : public ::mlir::Op<TanhOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TanhOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TanhOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.tanh");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TanhOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TanhOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::TanhOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::TrueDivOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TrueDivOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TrueDivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.truediv", odsAttrs.getContext());
  }

  TrueDivOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class TrueDivOpGenericAdaptor : public detail::TrueDivOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TrueDivOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  TrueDivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TrueDivOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TrueDivOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  TrueDivOpGenericAdaptor(RangeT values, const TrueDivOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TrueDivOp, typename = std::enable_if_t<std::is_same_v<LateInst, TrueDivOp>>>
  TrueDivOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TrueDivOpAdaptor : public TrueDivOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TrueDivOpGenericAdaptor::TrueDivOpGenericAdaptor;
  TrueDivOpAdaptor(TrueDivOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TrueDivOp : public ::mlir::Op<TrueDivOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TrueDivOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TrueDivOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using ::mlir::InferTypeOpInterface::Trait<TrueDivOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<TrueDivOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.truediv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TrueDivOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TrueDivOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::TrueDivOp)

namespace mlir::afir {


//===----------------------------------------------------------------------===//
// ::mlir::afir::WhereOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WhereOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  WhereOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.where", odsAttrs.getContext());
  }

  WhereOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
};

} // namespace detail
template <typename RangeT>
class WhereOpGenericAdaptor : public detail::WhereOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WhereOpGenericAdaptorBase;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
public:
  WhereOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WhereOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WhereOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  WhereOpGenericAdaptor(RangeT values, const WhereOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WhereOp, typename = std::enable_if_t<std::is_same_v<LateInst, WhereOp>>>
  WhereOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput1() {
    return (*getODSOperands(0).begin());
  }

  ValueT getInput2() {
    return (*getODSOperands(1).begin());
  }

  ValueT getInput3() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WhereOpAdaptor : public WhereOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WhereOpGenericAdaptor::WhereOpGenericAdaptor;
  WhereOpAdaptor(WhereOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WhereOp : public ::mlir::Op<WhereOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WhereOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WhereOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input1 = 0;
  static constexpr int odsIndex_input2 = 1;
  static constexpr int odsIndex_input3 = 2;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.where");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput1() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput2() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput3() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getInput1Mutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput2Mutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInput3Mutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static WhereOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static WhereOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static WhereOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static WhereOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static WhereOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static WhereOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static WhereOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static WhereOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::Value input3, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/uint32_t loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WhereOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WhereOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
    int getNumberOfOperands() { return 3; }
    int getNumberOfResults() { return 1; }

  afir::ComputeType getComputeType() {
    return afir::ComputeType::Elementwise;
  }
  afir::ComputeUnit getComputeUnit() {
    return afir::ComputeUnit::Vector;
  }
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::WhereOp)

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// ::mlir::afir::TransposeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TransposeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("afir.transpose", odsAttrs.getContext());
  }

  TransposeOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndexingMapsAttr();
  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::DictionaryAttr getIrAttrDefAttr();
  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::IntegerAttr getLoopAxisAttr();
  uint32_t getLoopAxis();
  ::mlir::ArrayAttr getOutputsAttr();
  ::mlir::ArrayAttr getOutputs();
  ::mlir::ArrayAttr getPermutationAttr();
  ::mlir::ArrayAttr getPermutation();
  ::mlir::ArrayAttr getTmpBuffersAttr();
  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
};

} // namespace detail
template <typename RangeT>
class TransposeOpGenericAdaptor : public detail::TransposeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TransposeOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TransposeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  TransposeOpGenericAdaptor(RangeT values, const TransposeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TransposeOp, typename = std::enable_if_t<std::is_same_v<LateInst, TransposeOp>>>
  TransposeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TransposeOpAdaptor : public TransposeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TransposeOpGenericAdaptor::TransposeOpGenericAdaptor;
  TransposeOpAdaptor(TransposeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::InferShapedTypeOpInterface::Trait, ::mlir::OpTrait::InferShapedTypeOpAdaptor, ::mlir::OpTrait::InferTensorType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TransposeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using ::mlir::InferTypeOpInterface::Trait<TransposeOp>::refineReturnTypes;
  using ::mlir::InferShapedTypeOpInterface::Trait<TransposeOp>::reifyReturnTypeShapes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indexing_maps"), ::llvm::StringRef("ir_attr_def"), ::llvm::StringRef("loop_axis"), ::llvm::StringRef("outputs"), ::llvm::StringRef("permutation"), ::llvm::StringRef("tmp_buffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndexingMapsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndexingMapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIrAttrDefAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIrAttrDefAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLoopAxisAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLoopAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getPermutationAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getPermutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTmpBuffersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTmpBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("afir.transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
  }

  ::mlir::ArrayAttr getIndexingMapsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexingMapsAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getIndexingMaps();
  ::mlir::DictionaryAttr getIrAttrDefAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIrAttrDefAttrName()));
  }

  ::mlir::DictionaryAttr getIrAttrDef();
  ::mlir::IntegerAttr getLoopAxisAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLoopAxisAttrName()));
  }

  uint32_t getLoopAxis();
  ::mlir::ArrayAttr getOutputsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOutputsAttrName()));
  }

  ::mlir::ArrayAttr getOutputs();
  ::mlir::ArrayAttr getPermutationAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPermutationAttrName()));
  }

  ::mlir::ArrayAttr getPermutation();
  ::mlir::ArrayAttr getTmpBuffersAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTmpBuffersAttrName()));
  }

  ::std::optional< ::mlir::ArrayAttr > getTmpBuffers();
  void setIndexingMapsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getIndexingMapsAttrName(), attr);
  }

  void setIrAttrDefAttr(::mlir::DictionaryAttr attr) {
    (*this)->setAttr(getIrAttrDefAttrName(), attr);
  }

  void setLoopAxisAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLoopAxisAttrName(), attr);
  }

  void setLoopAxis(::std::optional<uint32_t> attrValue);
  void setOutputsAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getOutputsAttrName(), attr);
  }

  void setPermutationAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getPermutationAttrName(), attr);
  }

  void setTmpBuffersAttr(::mlir::ArrayAttr attr) {
    (*this)->setAttr(getTmpBuffersAttrName(), attr);
  }

  ::mlir::Attribute removeIndexingMapsAttr() {
  return (*this)->removeAttr(getIndexingMapsAttrName());
  }

  ::mlir::Attribute removeIrAttrDefAttr() {
  return (*this)->removeAttr(getIrAttrDefAttrName());
  }

  ::mlir::Attribute removeLoopAxisAttr() {
  return (*this)->removeAttr(getLoopAxisAttrName());
  }

  ::mlir::Attribute removeTmpBuffersAttr() {
  return (*this)->removeAttr(getTmpBuffersAttrName());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static TransposeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr permutation, /*optional*/::mlir::ArrayAttr indexing_maps, /*optional*/::mlir::IntegerAttr loop_axis, /*optional*/::mlir::DictionaryAttr ir_attr_def, /*optional*/::mlir::ArrayAttr tmp_buffers, ::mlir::ArrayAttr outputs = nullptr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TransposeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TransposeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TransposeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TransposeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
      static ::llvm::LogicalResult
      inferReturnTypeComponents(::mlir::MLIRContext *context,
                              std::optional<::mlir::Location> location,
                              Adaptor adaptor,
                              ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &inferredReturnShapes);


  /// Returns when two result types are compatible for this op; method used by
  /// InferTypeOpInterface.
  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::TransposeOp)


#endif // GET_OP_CLASSES

