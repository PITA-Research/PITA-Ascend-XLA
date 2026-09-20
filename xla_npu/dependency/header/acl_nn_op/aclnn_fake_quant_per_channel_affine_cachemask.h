/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_fake_quant_per_channel_affine_cachemask.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_
#define OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFakeQuantPerChannelAffineCachemask calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function:
 * For the input data self, use scale and zero_point to pseudo-quantize the input self on the specified axis, and quantize it according to quant_min and quant_max.
 * The pseudo-quantization output is updated in the value range, and the result out and the corresponding position mask are finally returned.
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C1(l0op::Transpose)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A2[(scale)] -->B2(l0op::Contiguous)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A3[(zeroPoint)]-->B3(l0op::Contiguous)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A4((quantMin)) --> D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A5((quantMax)) --> D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A6((axis))-->C1(l0op::Transpose)
 *   D(l0op::FakeQuantPerChannelAffineCachemask)-->C2(l0op::Transpose)-->E1(l0op::ViewCopy)-->F1[(out)]
 *   D(l0op::FakeQuantPerChannelAffineCachemask)-->C3(l0op::Transpose)-->EF2(l0op::ViewCopy)-->F2[(mask)]
 * ```
 * @param [in] self:
 * The data type of aclTensor on the Device side supports FLOAT16 and FLOAT32. Supports non-continuous Tensor, [data format](../../../docs/zh/context/data format.md) supports ND.
 * @param [in] scale: aclTensor on the Device side, indicating the input pseudo-quantization scaling coefficient. The data type supports FLOAT16 and FLOAT32, and the size is 1.
 * @param [in] zeroPoint: aclTensor on the Device side, indicating the zero reference parameter of the input pseudo quantization. The data type supports INT32, and the size is 1.
 * @param [in] axis: Integer type on the Host side, indicating the calculation dimension, and the data type supports INT.
 * @param [in] quantMin: Integer type on the Host side, indicating the minimum value after pseudo-quantization of the input data. The data type supports INT.
 * @param [in] quantMax: Integer type on the Host side, indicating the maximum value after pseudo-quantization of the input data. The data type supports INT.
 * @param [out] out:
 * The aclTensor on the Device side supports LOAT16, FLOAT32 data types, non-continuous Tensor, and [data format] (../../../docs/zh/context/data format.md) supports ND.
 * @param [out] mask: aclTensor on the Device side. The data type supports BOOL and non-continuous Tensor. [Data format] (../../../docs/zh/context/data format.md) supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint,
    int64_t axis, int64_t quantMin, int64_t quantMax, aclTensor* out,
    aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFakeQuantPerChannelAffineCachemask's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnFakeQuantPerChannelAffineCachemask.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_
