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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ascir/Dialect/Asc/IR/AscendCEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// TPositionAttr
//===----------------------------------------------------------------------===//

ParseResult mlir::ascendc::parsePrettyTPosition(AsmParser &odsParser,
                                                TPositionAttr &attr) {
  StringRef pos;
  if (odsParser.parseKeyword(&pos)) return failure();
  if (auto maybePos = symbolizeTPosition(pos)) {
    attr = TPositionAttr::get(odsParser.getContext(), maybePos.value());
    return success();
  }
  return odsParser.emitError(odsParser.getNameLoc())
         << "position is not recognized: " << pos;
}

void mlir::ascendc::printPrettyTPosition(AsmPrinter &odsPrinter,
                                         const TPositionAttr &attr) {
  odsPrinter << stringifyTPosition(attr.getValue());
}

//===----------------------------------------------------------------------===//
// CubeFormatAttr
//===----------------------------------------------------------------------===//

ParseResult ascendc::parsePrettyCubeFormat(AsmParser &odsParser,
                                           CubeFormatAttr &attr) {
  StringRef pos;
  if (odsParser.parseKeyword(&pos)) return failure();
  if (auto maybePos = symbolizeCubeFormat(pos)) {
    attr = CubeFormatAttr::get(odsParser.getContext(), maybePos.value());
    return success();
  }
  return odsParser.emitError(odsParser.getNameLoc())
         << "cube format is not recognized: " << pos;
}

void ascendc::printPrettyCubeFormat(AsmPrinter &odsPrinter,
                                    const CubeFormatAttr &attr) {
  odsPrinter << stringifyCubeFormat(attr.getValue());
}

//===----------------------------------------------------------------------===//
// LayoutModeAttr
//===----------------------------------------------------------------------===//

ParseResult ascendc::parsePrettyLayoutMode(AsmParser &odsParser,
                                           LayoutModeAttr &attr) {
  StringRef pos;
  if (odsParser.parseKeyword(&pos)) return failure();
  if (auto maybePos = symbolizeLayoutMode(pos)) {
    attr = LayoutModeAttr::get(odsParser.getContext(), maybePos.value());
    return success();
  }
  return odsParser.emitError(odsParser.getNameLoc())
         << "layout mode is not recognized: " << pos;
}

void ascendc::printPrettyLayoutMode(AsmPrinter &odsPrinter,
                                    const LayoutModeAttr &attr) {
  odsPrinter << stringifyLayoutMode(attr.getValue());
}

//===----------------------------------------------------------------------===//
// CO2LayoutAttr
//===----------------------------------------------------------------------===//

ParseResult mlir::ascendc::parsePrettyCO2Layout(AsmParser &odsParser,
                                                CO2LayoutAttr &attr) {
  StringRef pos;
  if (odsParser.parseKeyword(&pos)) return failure();
  if (auto maybePos = symbolizeCO2Layout(pos)) {
    attr = CO2LayoutAttr::get(odsParser.getContext(), maybePos.value());
    return success();
  }
  return odsParser.emitError(odsParser.getNameLoc())
         << "CO2Layout is not recognized: " << pos;
}

void mlir::ascendc::printPrettyCO2Layout(AsmPrinter &odsPrinter,
                                         const CO2LayoutAttr &attr) {
  odsPrinter << stringifyCO2Layout(attr.getValue());
}

//===----------------------------------------------------------------------===//
// AscendCDialect
//===----------------------------------------------------------------------===//

void AscendCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ascir/Dialect/Asc/IR/AscendCAttributes.cpp.inc"
      >();
}
