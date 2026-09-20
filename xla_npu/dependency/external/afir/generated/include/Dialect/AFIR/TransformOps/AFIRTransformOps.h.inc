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
|* From: AFIRTransformOps.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::transform {

/// Add index arguments to a function
/// This transform takes a function operation and adds a specified number
///     of index type arguments to its argument list. The new arguments are
///     appended to end of the existing argument list.
/// 
///     The transform returns:
///     - The modified function operation
///     - Handles to each of the newly added index arguments
/// 
///     This transform will emit a silenceable failure if:
///      - The handle does not contain exactly one function operation
///      - The function operation is not a func.func operation
/// 
///     This transform will emit a definite failure if:
///      - Failed to modify the function signature
class AddIndexArgsOp;

} // namespace mlir::transform
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::transform {


//===----------------------------------------------------------------------===//
// ::mlir::transform::AddIndexArgsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AddIndexArgsOpGenericAdaptorBase {
public:
  struct Properties {
    using num_argsTy = ::mlir::IntegerAttr;
    num_argsTy num_args;

    auto getNumArgs() const {
      auto &propStorage = this->num_args;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setNumArgs(const ::mlir::IntegerAttr &propValue) {
      this->num_args = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.num_args == this->num_args &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AddIndexArgsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("transform.func.add_index_args", odsAttrs.getContext());
  }

  AddIndexArgsOpGenericAdaptorBase(AddIndexArgsOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getNumArgsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().num_args);
    return attr;
  }

  uint64_t getNumArgs();
};

} // namespace detail
template <typename RangeT>
class AddIndexArgsOpGenericAdaptor : public detail::AddIndexArgsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AddIndexArgsOpGenericAdaptorBase;
  static constexpr int odsIndex_target = 0;
public:
  AddIndexArgsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AddIndexArgsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AddIndexArgsOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AddIndexArgsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AddIndexArgsOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AddIndexArgsOpGenericAdaptor(RangeT values, const AddIndexArgsOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AddIndexArgsOp, typename = std::enable_if_t<std::is_same_v<LateInst, AddIndexArgsOp>>>
  AddIndexArgsOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTarget() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AddIndexArgsOpAdaptor : public AddIndexArgsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AddIndexArgsOpGenericAdaptor::AddIndexArgsOpGenericAdaptor;
  AddIndexArgsOpAdaptor(AddIndexArgsOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AddIndexArgsOp : public ::mlir::Op<AddIndexArgsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AddIndexArgsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AddIndexArgsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_target = 0;
  using Properties = FoldAdaptor::Properties;
  using ::mlir::transform::TransformOpInterface::Trait<AddIndexArgsOp>::allowsRepeatedHandleOperands;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("num_args")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNumArgsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNumArgsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.func.add_index_args");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> getTarget() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getTargetMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> getTransformed() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
  }

  ::mlir::Operation::result_range getNewArgs() {
    return getODSResults(1);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getNumArgsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().num_args);
  }

  uint64_t getNumArgs();
  void setNumArgsAttr(::mlir::IntegerAttr attr) {
    getProperties().num_args = attr;
  }

  void setNumArgs(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, uint64_t num_args);
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, uint64_t num_args);
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type transformed, ::mlir::TypeRange new_args, ::mlir::Value target, uint64_t num_args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_args);
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_args);
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AddIndexArgsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AddIndexArgsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter &rewriter, ::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::transform
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::AddIndexArgsOp)


#endif // GET_OP_CLASSES

