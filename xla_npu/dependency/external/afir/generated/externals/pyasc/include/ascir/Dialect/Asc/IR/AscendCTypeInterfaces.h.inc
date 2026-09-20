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
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::ascendc {

class BaseQueueType;

} // namespace mlir::ascendc
namespace mlir::ascendc {

class BaseTensorType;

} // namespace mlir::ascendc
namespace mlir::ascendc {

class BaseLoadDataParamsType;

} // namespace mlir::ascendc
namespace mlir::ascendc {

namespace detail {
struct BaseQueueTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseQueueType;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseQueueType;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct BaseQueueTypeTrait;

} // namespace detail
class BaseQueueType : public ::mlir::TypeInterface<BaseQueueType, detail::BaseQueueTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<BaseQueueType, detail::BaseQueueTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::BaseQueueTypeTrait<ConcreteType> {};
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

namespace detail {
struct BaseTensorTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::ShapedType::Concept *implShapedType = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implShapedType = interfaceMap.lookup<::mlir::ShapedType>();
      assert(implShapedType && "`::mlir::ascendc::BaseTensorType` expected its base interface `::mlir::ShapedType` to be registered");
    }
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseTensorType;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseTensorType;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct BaseTensorTypeTrait;

} // namespace detail
class BaseTensorType : public ::mlir::TypeInterface<BaseTensorType, detail::BaseTensorTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<BaseTensorType, detail::BaseTensorTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::BaseTensorTypeTrait<ConcreteType> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ShapedType
  //===----------------------------------------------------------------===//

  operator ::mlir::ShapedType () const {
    if (!*this) return nullptr;
    return ::mlir::ShapedType(*this, getImpl()->implShapedType);
  }


  /// Returns a clone of this type with the given shape and element type.
  /// 
  /// If no shape is provided, the shape of this type is used. In that case, if
  /// this type is unranked, so is the resulting type.
  /// 
  /// If a shape is provided, the resulting type is always ranked, even if this
  /// type is unranked.
  ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;

  /// Returns the element type of this shaped type.
  ::mlir::Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this type if it is ranked, otherwise asserts.
  ::llvm::ArrayRef<int64_t> getShape() const;

    static constexpr int64_t kDynamic =
        std::numeric_limits<int64_t>::min();

    /// Whether the given dimension size indicates a dynamic dimension.
    static constexpr bool isDynamic(int64_t dValue) {
      return dValue == kDynamic;
    }

    /// Whether the given shape has any size that indicates a dynamic dimension.
    static bool isDynamicShape(ArrayRef<int64_t> dSizes) {
      return llvm::any_of(dSizes, isDynamic);
    }

    /// Whether the given dimension size indicates a statically-sized dimension.
    static constexpr bool isStatic(int64_t dValue) {
      return dValue != kDynamic;
    }

    /// Whether the given shape has static dimensions only.
    static bool isStaticShape(ArrayRef<int64_t> dSizes) {
      return llvm::all_of(dSizes, isStatic);
    }

    /// Return the number of elements present in the given shape.
    static int64_t getNumElements(ArrayRef<int64_t> shape);

    /// Return a clone of this type with the given new shape and element type.
    /// The returned type is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape. The returned type
    /// is ranked, even if this type is unranked.
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return cloneWith(shape, getElementType());
    }

    /// Return a clone of this type with the given new element type. The
    /// returned type is ranked if and only if this type is ranked. In that
    /// case, the returned type has the same shape as this type.
    auto clone(::mlir::Type elementType) {
      return (*this).cloneWith(/*shape=*/std::nullopt, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return (*this).getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert((*this).hasRank() && "cannot query rank of unranked shaped type");
      return (*this).getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements((*this).getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic((*this).getShape()[idx]);
    }

    /// Returns true if this dimension has a static size (for ranked types);
    /// aborts for unranked types.
    bool isStaticDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isStatic((*this).getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return (*this).hasRank() &&
             ::mlir::ShapedType::isStaticShape((*this).getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && (*this).getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    size_t getNumDynamicDims() const {
      return llvm::count_if((*this).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return (*this).getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if((*this).getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

namespace detail {
struct BaseLoadDataParamsTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseLoadDataParamsType;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BaseLoadDataParamsType;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct BaseLoadDataParamsTypeTrait;

} // namespace detail
class BaseLoadDataParamsType : public ::mlir::TypeInterface<BaseLoadDataParamsType, detail::BaseLoadDataParamsTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<BaseLoadDataParamsType, detail::BaseLoadDataParamsTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::BaseLoadDataParamsTypeTrait<ConcreteType> {};
};

} // namespace mlir::ascendc
namespace mlir::ascendc::detail {

  template <typename ConcreteType>
  struct BaseQueueTypeTrait : public ::mlir::TypeInterface<BaseQueueType, detail::BaseQueueTypeInterfaceTraits>::Trait<ConcreteType> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteType>
  struct BaseTensorTypeTrait : public ::mlir::TypeInterface<BaseTensorType, detail::BaseTensorTypeInterfaceTraits>::Trait<ConcreteType> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteType>
  struct BaseLoadDataParamsTypeTrait : public ::mlir::TypeInterface<BaseLoadDataParamsType, detail::BaseLoadDataParamsTypeInterfaceTraits>::Trait<ConcreteType> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc {


} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
