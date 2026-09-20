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

/// Base interface for operations representing Ascend C API
class APIOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Operation with `dst` operand (typically tensor)
class OpWithDst;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Base interface for vector operations
class VectorOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Data copy operation
class DataCopyOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation
class UnaryOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation (L0 API)
class UnaryL0Op;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation (L2 API)
class UnaryL2Op;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector binary operation
class BinaryOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector binary operation (L3 API)
class BinaryL3Op;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation
class VecScalarOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation (L0 API)
class VecScalarL0Op;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation (L2 API)
class VecScalarL2Op;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Math library operation
class MathLibraryOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Unary math library operation
class UnaryMathOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Binary math library operation
class BinaryMathOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Data conversion operation
class DataConversionOp;

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Base interface for operations representing Ascend C API
namespace detail {
struct APIOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::StringRef (*getAPIName)(const Concept *impl, ::mlir::Operation *);
    ::llvm::StringRef (*getComment)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::APIOp;
    Model() : Concept{getAPIName, getComment} {}

    static inline ::llvm::StringRef getAPIName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getComment(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::APIOp;
    FallbackModel() : Concept{getAPIName, getComment} {}

    static inline ::llvm::StringRef getAPIName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getComment(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct APIOpTrait;

} // namespace detail
class APIOp : public ::mlir::OpInterface<APIOp, detail::APIOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<APIOp, detail::APIOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::APIOpTrait<ConcreteOp> {};

  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Operation with `dst` operand (typically tensor)
namespace detail {
struct OpWithDstInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getDst)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::OpWithDst;
    Model() : Concept{getDst} {}

    static inline ::mlir::Value getDst(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::OpWithDst;
    FallbackModel() : Concept{getDst} {}

    static inline ::mlir::Value getDst(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct OpWithDstTrait;

} // namespace detail
class OpWithDst : public ::mlir::OpInterface<OpWithDst, detail::OpWithDstInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpWithDst, detail::OpWithDstInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OpWithDstTrait<ConcreteOp> {};

  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Base interface for vector operations
namespace detail {
struct VectorOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::VectorOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::VectorOp;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::VectorOp;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct VectorOpTrait;

} // namespace detail
class VectorOp : public ::mlir::OpInterface<VectorOp, detail::VectorOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorOp, detail::VectorOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VectorOpTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Data copy operation
namespace detail {
struct DataCopyOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ascendc::CopyDirection (*getDirection)(const Concept *impl, ::mlir::Operation *);
    void (*setSrc)(const Concept *impl, ::mlir::Operation *, Value);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::DataCopyOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::DataCopyOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::DataCopyOp;
    Model() : Concept{getSrc, getDirection, setSrc} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ascendc::CopyDirection getDirection(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value src);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::DataCopyOp;
    FallbackModel() : Concept{getSrc, getDirection, setSrc} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ascendc::CopyDirection getDirection(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value src);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::ascendc::CopyDirection getDirection(::mlir::Operation *tablegen_opaque_val) const;
    void setSrc(::mlir::Operation *tablegen_opaque_val, Value src) const;
  };
};
template <typename ConcreteOp>
struct DataCopyOpTrait;

} // namespace detail
class DataCopyOp : public ::mlir::OpInterface<DataCopyOp, detail::DataCopyOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<DataCopyOp, detail::DataCopyOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DataCopyOpTrait<ConcreteOp> {};

  /// Obtain `src` parameter
  ::mlir::Value getSrc();

  /// Get copy direction
  ::mlir::ascendc::CopyDirection getDirection();

  /// Set src operand
  void setSrc(Value src);
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation
namespace detail {
struct UnaryOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::UnaryOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::UnaryOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::UnaryOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryOp;
    Model() : Concept{getSrc} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryOp;
    FallbackModel() : Concept{getSrc} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct UnaryOpTrait;

} // namespace detail
class UnaryOp : public ::mlir::OpInterface<UnaryOp, detail::UnaryOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<UnaryOp, detail::UnaryOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::UnaryOpTrait<ConcreteOp> {};

  /// Obtain `src` parameter
  ::mlir::Value getSrc();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation (L0 API)
namespace detail {
struct UnaryL0OpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getMask)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getRepeatTimes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getRepeatParams)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::UnaryOp::Concept *implUnaryOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::UnaryL0Op` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::UnaryL0Op` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::UnaryL0Op` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implUnaryOp = interfaceMap.lookup<::mlir::ascendc::UnaryOp>();
      assert(implUnaryOp && "`::mlir::ascendc::UnaryL0Op` expected its base interface `::mlir::ascendc::UnaryOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryL0Op;
    Model() : Concept{getMask, getRepeatTimes, getRepeatParams} {}

    static inline ::mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryL0Op;
    FallbackModel() : Concept{getMask, getRepeatTimes, getRepeatParams} {}

    static inline ::mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct UnaryL0OpTrait;

} // namespace detail
class UnaryL0Op : public ::mlir::OpInterface<UnaryL0Op, detail::UnaryL0OpInterfaceTraits> {
public:
  using ::mlir::OpInterface<UnaryL0Op, detail::UnaryL0OpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::UnaryL0OpTrait<ConcreteOp> {};

  /// Obtain `mask` parameter
  ::mlir::Value getMask();

  /// Obtain `repeatTimes` parameter
  ::mlir::Value getRepeatTimes();

  /// Obtain `repeatParams` parameter
  ::mlir::Value getRepeatParams();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::UnaryOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::UnaryOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::UnaryOp(*this, getImpl()->implUnaryOp);
  }


  /// Obtain `src` parameter
  ::mlir::Value getSrc();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector unary operation (L2 API)
namespace detail {
struct UnaryL2OpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getCalCount)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::UnaryOp::Concept *implUnaryOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::UnaryL2Op` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::UnaryL2Op` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::UnaryL2Op` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implUnaryOp = interfaceMap.lookup<::mlir::ascendc::UnaryOp>();
      assert(implUnaryOp && "`::mlir::ascendc::UnaryL2Op` expected its base interface `::mlir::ascendc::UnaryOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryL2Op;
    Model() : Concept{getCalCount} {}

    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryL2Op;
    FallbackModel() : Concept{getCalCount} {}

    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct UnaryL2OpTrait;

} // namespace detail
class UnaryL2Op : public ::mlir::OpInterface<UnaryL2Op, detail::UnaryL2OpInterfaceTraits> {
public:
  using ::mlir::OpInterface<UnaryL2Op, detail::UnaryL2OpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::UnaryL2OpTrait<ConcreteOp> {};

  /// Obtain `calCount` parameter
  ::mlir::Value getCalCount();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::UnaryOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::UnaryOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::UnaryOp(*this, getImpl()->implUnaryOp);
  }


  /// Obtain `src` parameter
  ::mlir::Value getSrc();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector binary operation
namespace detail {
struct BinaryOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc0)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getSrc1)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::BinaryOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::BinaryOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::BinaryOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryOp;
    Model() : Concept{getSrc0, getSrc1} {}

    static inline ::mlir::Value getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryOp;
    FallbackModel() : Concept{getSrc0, getSrc1} {}

    static inline ::mlir::Value getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct BinaryOpTrait;

} // namespace detail
class BinaryOp : public ::mlir::OpInterface<BinaryOp, detail::BinaryOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<BinaryOp, detail::BinaryOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BinaryOpTrait<ConcreteOp> {};

  /// Obtain `src0` parameter
  ::mlir::Value getSrc0();

  /// Obtain `src1` parameter
  ::mlir::Value getSrc1();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector binary operation (L3 API)
namespace detail {
struct BinaryL3OpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::BinaryOp::Concept *implBinaryOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::BinaryL3Op` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::BinaryL3Op` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::BinaryL3Op` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implBinaryOp = interfaceMap.lookup<::mlir::ascendc::BinaryOp>();
      assert(implBinaryOp && "`::mlir::ascendc::BinaryL3Op` expected its base interface `::mlir::ascendc::BinaryOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryL3Op;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryL3Op;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct BinaryL3OpTrait;

} // namespace detail
class BinaryL3Op : public ::mlir::OpInterface<BinaryL3Op, detail::BinaryL3OpInterfaceTraits> {
public:
  using ::mlir::OpInterface<BinaryL3Op, detail::BinaryL3OpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BinaryL3OpTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::BinaryOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::BinaryOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::BinaryOp(*this, getImpl()->implBinaryOp);
  }


  /// Obtain `src0` parameter
  ::mlir::Value getSrc0();

  /// Obtain `src1` parameter
  ::mlir::Value getSrc1();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation
namespace detail {
struct VecScalarOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getScalar)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::VecScalarOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::VecScalarOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::VecScalarOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarOp;
    Model() : Concept{getSrc, getScalar} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarOp;
    FallbackModel() : Concept{getSrc, getScalar} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct VecScalarOpTrait;

} // namespace detail
class VecScalarOp : public ::mlir::OpInterface<VecScalarOp, detail::VecScalarOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<VecScalarOp, detail::VecScalarOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VecScalarOpTrait<ConcreteOp> {};

  /// Obtain `src` parameter
  ::mlir::Value getSrc();

  /// Obtain `scalar` parameter
  ::mlir::Value getScalar();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation (L0 API)
namespace detail {
struct VecScalarL0OpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getMask)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getRepeatTimes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getRepeatParams)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::VecScalarOp::Concept *implVecScalarOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::VecScalarL0Op` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::VecScalarL0Op` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::VecScalarL0Op` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implVecScalarOp = interfaceMap.lookup<::mlir::ascendc::VecScalarOp>();
      assert(implVecScalarOp && "`::mlir::ascendc::VecScalarL0Op` expected its base interface `::mlir::ascendc::VecScalarOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarL0Op;
    Model() : Concept{getMask, getRepeatTimes, getRepeatParams} {}

    static inline ::mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarL0Op;
    FallbackModel() : Concept{getMask, getRepeatTimes, getRepeatParams} {}

    static inline ::mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct VecScalarL0OpTrait;

} // namespace detail
class VecScalarL0Op : public ::mlir::OpInterface<VecScalarL0Op, detail::VecScalarL0OpInterfaceTraits> {
public:
  using ::mlir::OpInterface<VecScalarL0Op, detail::VecScalarL0OpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VecScalarL0OpTrait<ConcreteOp> {};

  /// Obtain `mask` parameter
  ::mlir::Value getMask();

  /// Obtain `repeatTimes` parameter
  ::mlir::Value getRepeatTimes();

  /// Obtain `repeatParams` parameter
  ::mlir::Value getRepeatParams();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VecScalarOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VecScalarOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VecScalarOp(*this, getImpl()->implVecScalarOp);
  }


  /// Obtain `src` parameter
  ::mlir::Value getSrc();

  /// Obtain `scalar` parameter
  ::mlir::Value getScalar();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Vector-scalar operation (L2 API)
namespace detail {
struct VecScalarL2OpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getCalCount)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::VecScalarOp::Concept *implVecScalarOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::VecScalarL2Op` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::VecScalarL2Op` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::VecScalarL2Op` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implVecScalarOp = interfaceMap.lookup<::mlir::ascendc::VecScalarOp>();
      assert(implVecScalarOp && "`::mlir::ascendc::VecScalarL2Op` expected its base interface `::mlir::ascendc::VecScalarOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarL2Op;
    Model() : Concept{getCalCount} {}

    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::VecScalarL2Op;
    FallbackModel() : Concept{getCalCount} {}

    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct VecScalarL2OpTrait;

} // namespace detail
class VecScalarL2Op : public ::mlir::OpInterface<VecScalarL2Op, detail::VecScalarL2OpInterfaceTraits> {
public:
  using ::mlir::OpInterface<VecScalarL2Op, detail::VecScalarL2OpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::VecScalarL2OpTrait<ConcreteOp> {};

  /// Obtain `calCount` parameter
  ::mlir::Value getCalCount();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VecScalarOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VecScalarOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VecScalarOp(*this, getImpl()->implVecScalarOp);
  }


  /// Obtain `src` parameter
  ::mlir::Value getSrc();

  /// Obtain `scalar` parameter
  ::mlir::Value getScalar();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Math library operation
namespace detail {
struct MathLibraryOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::MathLibraryOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::MathLibraryOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::MathLibraryOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::MathLibraryOp;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::MathLibraryOp;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MathLibraryOpTrait;

} // namespace detail
class MathLibraryOp : public ::mlir::OpInterface<MathLibraryOp, detail::MathLibraryOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<MathLibraryOp, detail::MathLibraryOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MathLibraryOpTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Unary math library operation
namespace detail {
struct UnaryMathOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getSharedTmpBuffer)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getCalCount)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getIsReuseSource)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::MathLibraryOp::Concept *implMathLibraryOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::UnaryMathOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::UnaryMathOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::UnaryMathOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implMathLibraryOp = interfaceMap.lookup<::mlir::ascendc::MathLibraryOp>();
      assert(implMathLibraryOp && "`::mlir::ascendc::UnaryMathOp` expected its base interface `::mlir::ascendc::MathLibraryOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryMathOp;
    Model() : Concept{getSrc, getSharedTmpBuffer, getCalCount, getIsReuseSource} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::UnaryMathOp;
    FallbackModel() : Concept{getSrc, getSharedTmpBuffer, getCalCount, getIsReuseSource} {}

    static inline ::mlir::Value getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct UnaryMathOpTrait;

} // namespace detail
class UnaryMathOp : public ::mlir::OpInterface<UnaryMathOp, detail::UnaryMathOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<UnaryMathOp, detail::UnaryMathOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::UnaryMathOpTrait<ConcreteOp> {};

  /// Obtain `src` parameter
  ::mlir::Value getSrc();

  /// Obtain `sharedTmpBuffer` parameter
  ::mlir::Value getSharedTmpBuffer();

  /// Obtain `calCount` parameter
  ::mlir::Value getCalCount();

  /// Obtain `isReuseSource` parameter
  ::mlir::Value getIsReuseSource();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::MathLibraryOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::MathLibraryOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::MathLibraryOp(*this, getImpl()->implMathLibraryOp);
  }

};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Binary math library operation
namespace detail {
struct BinaryMathOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSrc0)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getSrc1)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getSharedTmpBuffer)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getCalCount)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getIsReuseSource)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;
    const ::mlir::ascendc::VectorOp::Concept *implVectorOp = nullptr;
    const ::mlir::ascendc::OpWithDst::Concept *implOpWithDst = nullptr;
    const ::mlir::ascendc::MathLibraryOp::Concept *implMathLibraryOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::BinaryMathOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
      implVectorOp = interfaceMap.lookup<::mlir::ascendc::VectorOp>();
      assert(implVectorOp && "`::mlir::ascendc::BinaryMathOp` expected its base interface `::mlir::ascendc::VectorOp` to be registered");
      implOpWithDst = interfaceMap.lookup<::mlir::ascendc::OpWithDst>();
      assert(implOpWithDst && "`::mlir::ascendc::BinaryMathOp` expected its base interface `::mlir::ascendc::OpWithDst` to be registered");
      implMathLibraryOp = interfaceMap.lookup<::mlir::ascendc::MathLibraryOp>();
      assert(implMathLibraryOp && "`::mlir::ascendc::BinaryMathOp` expected its base interface `::mlir::ascendc::MathLibraryOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryMathOp;
    Model() : Concept{getSrc0, getSrc1, getSharedTmpBuffer, getCalCount, getIsReuseSource} {}

    static inline ::mlir::Value getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::BinaryMathOp;
    FallbackModel() : Concept{getSrc0, getSrc1, getSharedTmpBuffer, getCalCount, getIsReuseSource} {}

    static inline ::mlir::Value getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct BinaryMathOpTrait;

} // namespace detail
class BinaryMathOp : public ::mlir::OpInterface<BinaryMathOp, detail::BinaryMathOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<BinaryMathOp, detail::BinaryMathOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BinaryMathOpTrait<ConcreteOp> {};

  /// Obtain `src0` parameter
  ::mlir::Value getSrc0();

  /// Obtain `src1` parameter
  ::mlir::Value getSrc1();

  /// Obtain `sharedTmpBuffer` parameter
  ::mlir::Value getSharedTmpBuffer();

  /// Obtain `calCount` parameter
  ::mlir::Value getCalCount();

  /// Obtain `isReuseSource` parameter
  ::mlir::Value getIsReuseSource();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::VectorOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::VectorOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::VectorOp(*this, getImpl()->implVectorOp);
  }

  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::OpWithDst
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::OpWithDst () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::OpWithDst(*this, getImpl()->implOpWithDst);
  }


  /// Obtain `dst` parameter
  ::mlir::Value getDst();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::MathLibraryOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::MathLibraryOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::MathLibraryOp(*this, getImpl()->implMathLibraryOp);
  }

};

} // namespace mlir::ascendc
namespace mlir::ascendc {

/// Data conversion operation
namespace detail {
struct DataConversionOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::ascendc::APIOp::Concept *implAPIOp = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implAPIOp = interfaceMap.lookup<::mlir::ascendc::APIOp>();
      assert(implAPIOp && "`::mlir::ascendc::DataConversionOp` expected its base interface `::mlir::ascendc::APIOp` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ascendc::DataConversionOp;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ascendc::DataConversionOp;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DataConversionOpTrait;

} // namespace detail
class DataConversionOp : public ::mlir::OpInterface<DataConversionOp, detail::DataConversionOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<DataConversionOp, detail::DataConversionOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DataConversionOpTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::ascendc::APIOp
  //===----------------------------------------------------------------===//

  operator ::mlir::ascendc::APIOp () const {
    if (!*this) return nullptr;
    return ::mlir::ascendc::APIOp(*this, getImpl()->implAPIOp);
  }


  /// Obtain Ascend C library name
  ::llvm::StringRef getAPIName();

  /// Obtain describing comment
  ::llvm::StringRef getComment();
};

} // namespace mlir::ascendc
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct APIOpTrait : public ::mlir::OpInterface<APIOp, detail::APIOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct OpWithDstTrait : public ::mlir::OpInterface<OpWithDst, detail::OpWithDstInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct VectorOpTrait : public ::mlir::OpInterface<VectorOp, detail::VectorOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct DataCopyOpTrait : public ::mlir::OpInterface<DataCopyOp, detail::DataCopyOpInterfaceTraits>::Trait<ConcreteOp> {

    /// Get copy direction
    ::mlir::ascendc::CopyDirection getDirection() {
      auto dstType = (*static_cast<ConcreteOp *>(this)).getDst().getType();
    auto srcType = (*static_cast<ConcreteOp *>(this)).getSrc().getType();
    if (isa<GlobalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::gm_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::gm_ubuf;
    } else if (isa<LocalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::ubuf_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::ubuf_ubuf;
    }
    return CopyDirection::Unknown;
    }

    /// Set src operand
    void setSrc(Value src) {
      (*static_cast<ConcreteOp *>(this)).getSrcMutable().assign(src);
    }
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct UnaryOpTrait : public ::mlir::OpInterface<UnaryOp, detail::UnaryOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct UnaryL0OpTrait : public ::mlir::OpInterface<UnaryL0Op, detail::UnaryL0OpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct UnaryL2OpTrait : public ::mlir::OpInterface<UnaryL2Op, detail::UnaryL2OpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct BinaryOpTrait : public ::mlir::OpInterface<BinaryOp, detail::BinaryOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct BinaryL3OpTrait : public ::mlir::OpInterface<BinaryL3Op, detail::BinaryL3OpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct VecScalarOpTrait : public ::mlir::OpInterface<VecScalarOp, detail::VecScalarOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct VecScalarL0OpTrait : public ::mlir::OpInterface<VecScalarL0Op, detail::VecScalarL0OpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct VecScalarL2OpTrait : public ::mlir::OpInterface<VecScalarL2Op, detail::VecScalarL2OpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct MathLibraryOpTrait : public ::mlir::OpInterface<MathLibraryOp, detail::MathLibraryOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct UnaryMathOpTrait : public ::mlir::OpInterface<UnaryMathOp, detail::UnaryMathOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct BinaryMathOpTrait : public ::mlir::OpInterface<BinaryMathOp, detail::BinaryMathOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc::detail {

  template <typename ConcreteOp>
  struct DataConversionOpTrait : public ::mlir::OpInterface<DataConversionOp, detail::DataConversionOpInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::ascendc::detail
namespace mlir::ascendc {

template<typename ConcreteOp>
::llvm::StringRef detail::APIOpInterfaceTraits::Model<ConcreteOp>::getAPIName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAPIName();
}
template<typename ConcreteOp>
::llvm::StringRef detail::APIOpInterfaceTraits::Model<ConcreteOp>::getComment(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getComment();
}
template<typename ConcreteOp>
::llvm::StringRef detail::APIOpInterfaceTraits::FallbackModel<ConcreteOp>::getAPIName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAPIName(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::StringRef detail::APIOpInterfaceTraits::FallbackModel<ConcreteOp>::getComment(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getComment(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::OpWithDstInterfaceTraits::Model<ConcreteOp>::getDst(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDst();
}
template<typename ConcreteOp>
::mlir::Value detail::OpWithDstInterfaceTraits::FallbackModel<ConcreteOp>::getDst(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDst(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::DataCopyOpInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::ascendc::CopyDirection detail::DataCopyOpInterfaceTraits::Model<ConcreteOp>::getDirection(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDirection();
}
template<typename ConcreteOp>
void detail::DataCopyOpInterfaceTraits::Model<ConcreteOp>::setSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value src) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setSrc(src);
}
template<typename ConcreteOp>
::mlir::Value detail::DataCopyOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ascendc::CopyDirection detail::DataCopyOpInterfaceTraits::FallbackModel<ConcreteOp>::getDirection(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDirection(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::DataCopyOpInterfaceTraits::FallbackModel<ConcreteOp>::setSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value src) {
  return static_cast<const ConcreteOp *>(impl)->setSrc(tablegen_opaque_val, src);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ascendc::CopyDirection detail::DataCopyOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDirection(::mlir::Operation *tablegen_opaque_val) const {
auto dstType = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDst().getType();
    auto srcType = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc().getType();
    if (isa<GlobalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::gm_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::gm_ubuf;
    } else if (isa<LocalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::ubuf_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::ubuf_ubuf;
    }
    return CopyDirection::Unknown;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::DataCopyOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setSrc(::mlir::Operation *tablegen_opaque_val, Value src) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrcMutable().assign(src);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::UnaryOpInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::Model<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMask();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::Model<ConcreteOp>::getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRepeatTimes();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::Model<ConcreteOp>::getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRepeatParams();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMask(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRepeatTimes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRepeatParams(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::UnaryL2OpInterfaceTraits::Model<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCalCount();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryL2OpInterfaceTraits::FallbackModel<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCalCount(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::BinaryOpInterfaceTraits::Model<ConcreteOp>::getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc0();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryOpInterfaceTraits::Model<ConcreteOp>::getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc1();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc0(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc1(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::VecScalarOpInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarOpInterfaceTraits::Model<ConcreteOp>::getScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getScalar();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarOpInterfaceTraits::FallbackModel<ConcreteOp>::getScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getScalar(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::Model<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMask();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::Model<ConcreteOp>::getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRepeatTimes();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::Model<ConcreteOp>::getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRepeatParams();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMask(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getRepeatTimes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRepeatTimes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL0OpInterfaceTraits::FallbackModel<ConcreteOp>::getRepeatParams(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRepeatParams(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::VecScalarL2OpInterfaceTraits::Model<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCalCount();
}
template<typename ConcreteOp>
::mlir::Value detail::VecScalarL2OpInterfaceTraits::FallbackModel<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCalCount(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::Model<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::Model<ConcreteOp>::getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSharedTmpBuffer();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::Model<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCalCount();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::Model<ConcreteOp>::getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIsReuseSource();
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSharedTmpBuffer(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCalCount(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::UnaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIsReuseSource(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::Model<ConcreteOp>::getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc0();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::Model<ConcreteOp>::getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSrc1();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::Model<ConcreteOp>::getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSharedTmpBuffer();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::Model<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCalCount();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::Model<ConcreteOp>::getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIsReuseSource();
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc0(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc0(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getSrc1(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSrc1(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getSharedTmpBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSharedTmpBuffer(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getCalCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCalCount(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::BinaryMathOpInterfaceTraits::FallbackModel<ConcreteOp>::getIsReuseSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIsReuseSource(tablegen_opaque_val);
}

} // namespace mlir::ascendc
namespace mlir::ascendc {


} // namespace mlir::ascendc
