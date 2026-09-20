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

#ifndef ASCIR_DIALECT_ASC_IR_ASC_H
#define ASCIR_DIALECT_ASC_IR_ASC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ascir/Dialect/Asc/IR/AscFuncTrait.h"

#include "ascir/Dialect/Asc/IR/AscendCDialect.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCEnums.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCTypeInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCTypes.h.inc"

#include "ascir/Dialect/Asc/IR/AscendCOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "ascir/Dialect/Asc/IR/AscendCOps.h.inc"

namespace mlir {
namespace ascendc {

ParseResult parsePrettyTPosition(AsmParser &odsParser, TPositionAttr &attr);
void printPrettyTPosition(AsmPrinter &odsPrinter, const TPositionAttr &attr);

ParseResult parsePrettyCubeFormat(AsmParser &odsParser, CubeFormatAttr &attr);
void printPrettyCubeFormat(AsmPrinter &odsPrinter, const CubeFormatAttr &attr);

ParseResult parsePrettyLayoutMode(AsmParser &odsParser, LayoutModeAttr &attr);
void printPrettyLayoutMode(AsmPrinter &odsPrinter, const LayoutModeAttr &attr);

ParseResult parsePrettyCO2Layout(AsmParser &odsParser, CO2LayoutAttr &attr);
void printPrettyCO2Layout(AsmPrinter &odsPrinter, const CO2LayoutAttr &attr);

ParseResult parsePrettyAippInputFormat(AsmParser &odsParser,
                                       AippInputFormatAttr &attr);
void printPrettyAippInputFormat(AsmPrinter &odsPrinter,
                                const AippInputFormatAttr &attr);

void registerExternalModels(DialectRegistry &registry);

ParseResult parsePrettyCmpMode(AsmParser &odsParser, CMPMODEAttr &attr);
void printPrettyCmpMode(AsmPrinter &odsPrinter, const CMPMODEAttr &attr);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_DIALECT_ASC_IR_ASC_H
