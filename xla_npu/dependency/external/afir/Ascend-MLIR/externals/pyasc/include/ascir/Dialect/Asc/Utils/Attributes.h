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

#ifndef ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char*

namespace mlir {
namespace ascendc {

namespace attr {
LITERAL aicore = "ascendc.aicore";
LITERAL api = "ascendc.api";
LITERAL compile_mix = "asc.compile_mix";
LITERAL emitAsUnsigned = "ascendc.emit_as_unsigned";
LITERAL global = "ascendc.global";
LITERAL enable_debug = "asc.enable_debug";
LITERAL matmulCubeOnly = "asc.matmul_cube_only";
}  // namespace attr

}  // namespace ascendc
}  // namespace mlir

#undef LITERAL
#endif  // ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
