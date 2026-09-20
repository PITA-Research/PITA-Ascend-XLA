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

#ifndef ASC_DIALECT_TD
#define ASC_DIALECT_TD

include "mlir/IR/OpBase.td"

def AscendC_Dialect : Dialect {
  let name = "ascendc";
  let summary = "A special dialect to support Ascend C API";

  let description = [{
    The AscendC dialect provides a number of operations that represent Ascend C
    library API for Huawei NPU.
  }];

  let cppNamespace = "::mlir::ascendc";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
    void registerOps();
  }];
}

#endif // ASC_DIALECT_TD
