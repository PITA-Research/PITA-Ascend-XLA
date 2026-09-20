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

//===- AFIRDialect.cpp - AFIR dialect implementation ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "Dialect/AFIR/Dialect.cpp.inc"

// Include generated enum definitions
#include "Dialect/AFIR/Enums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.cpp.inc"

// Include generated operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.cpp.inc"

void AFIRDialect::initialize() {
  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/AFIR/Attrs.cpp.inc"
      >();

  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Dialect/AFIR/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

Type AFIRDialect::parseType(DialectAsmParser &parser) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated parser:
  // return generatedTypeParser(parser);
  return Type();
}

void AFIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated printer:
  // (void)generatedTypePrinter(type, printer);
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute AFIRDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrType;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrType, type, attr);
  if (parseResult.has_value() && succeeded(parseResult.value())) {
    return attr;
  }
  return Attribute();
}

void AFIRDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (failed(generatedAttributePrinter(attr, printer))) {
    return;
  }
}
