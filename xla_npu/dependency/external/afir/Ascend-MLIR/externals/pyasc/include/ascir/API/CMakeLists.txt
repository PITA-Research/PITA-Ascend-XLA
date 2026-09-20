# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Types.td)
tablegen(AscIR Types.h.inc -gen-api-types)
tablegen(AscIR Types.td.inc -gen-api-typedefs)
add_public_tablegen_target(AscIRAPITypesIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
tablegen(AscIR AscTypeBindings.h.inc -gen-pybind-defs-types)
add_public_tablegen_target(AscTypesPybindGen)
