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
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Dialect.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::afir {

/// Ascend Frontend IR Dialect
/// AFIR (Ascend Frontend IR) is a high-level dialect designed for
///     representing computational operations targeting Ascend hardware.
///     It provides a bridge between high-level ML frameworks and lower-level
///     hardware-specific dialects like ASC-IR.
/// 
///     Features:
///     - High-level tensor operations
///     - Type-safe representations
///     - Hardware-agnostic abstractions
///     - Extensible attribute system
///     - 1:1 mapping with AscGraph protobuf definitions
class AFIRDialect : public ::mlir::Dialect {
  explicit AFIRDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~AFIRDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("afir");
  }

    // Type parsing and printing
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;

    // Attribute parsing and printing
    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
  };

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AFIRDialect)
