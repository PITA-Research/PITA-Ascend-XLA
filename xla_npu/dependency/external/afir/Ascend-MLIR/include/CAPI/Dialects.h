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

#ifndef AFIR_INTERGRATIONS_DIALECT_H
#define AFIR_INTERGRATIONS_DIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "Dialect/AFIR/AFIR.h"
#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AFIR, afir);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFIRPositionConfigAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirPositionConfigAttrGet(MlirContext mlirCtx, mlir::afir::Position position,
                          uint32_t depth, bool is_double_buffer);

MLIR_CAPI_EXPORTED mlir::afir::Position mlirPositionConfigAttrGetPosition(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED uint32_t mlirPositionConfigAttrGetDepth(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirPositionConfigAttrGetIsDoubleBuffer(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFIRAscTensorGroupsAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAscTensorGroupsAttrGet(
    MlirContext mlirCtx, std::vector<int64_t> vectorized_axis,
    std::vector<MlirAttribute> vectorized_strides, int64_t tensor_id,
    int64_t reuse_id, MlirAttribute position_config, int64_t position_id);

MLIR_CAPI_EXPORTED std::vector<int64_t> mlirAscTensorGroupsAttrGetVectorizeAxis(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED std::vector<MlirAttribute>
mlirAscTensorGroupsAttrGetVectorizeStrides(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
mlirAscTensorGroupsAttrGetTensorId(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
mlirAscTensorGroupsAttrGetReuseId(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirAscTensorGroupsAttrGePositionConfig(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
mlirAscTensorGroupsAttrGetPositionId(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFIRAxisAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAxisAttrGet(
    MlirContext mlirCtx, int64_t id, MlirAttribute name,
    mlir::afir::AxisType axisType, bool bind_block, MlirAttribute size,
    MlirAttribute align, std::vector<int64_t> from);

MLIR_CAPI_EXPORTED int64_t mlirAxisAttrGetId(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAxisAttrGetName(MlirAttribute attr);

MLIR_CAPI_EXPORTED mlir::afir::AxisType mlirAxisAttrGetAxisType(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAxisAttrGetBindBlock(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAxisAttrGetSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAxisAttrGetAlign(MlirAttribute attr);

MLIR_CAPI_EXPORTED std::vector<int64_t> mlirAxisAttrGetFrom(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAscGraphAttrGroupsAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirAscGraphAttrGroupsAttrGet(
    MlirContext mlirCtx, int64_t tiling_key, std::vector<MlirAttribute> axes,
    mlir::afir::AscGraphType type, std::vector<MlirAttribute> size_var);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_CHECK_DIALECT_H