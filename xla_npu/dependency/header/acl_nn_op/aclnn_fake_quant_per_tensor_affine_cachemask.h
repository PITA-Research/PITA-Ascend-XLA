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
 * \file aclnn_fake_quant_per_tensor_affine_cachemask.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_
#define OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFakeQuantPerTensorAffineCachemask calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function:
 * 1) fake_quant_enabled >= 1: For the input data self, use scale and zero_point to perform pseudo quantization processing on the input self
 * And update the value range of the pseudo-quantization output according to quant_min and quant_max, and finally return the result out and the corresponding position mask mask.
 * 2) fake_quant_enabled < 1: The returned result out is the self.clone() object, and the mask mask is all True.
 *
 * Calculation diagram: as follows
 * Scenario 1: When fake_quant_enabled is less than 1, use ViewCopy to calculate out and Fill to calculate mask.
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C1(l0op::ViewCopy)-->D1[(out)]
 *   A1[(self)] -->B1(l0op::Contiguous)-->D(l0op::Fill)-->C2(l0op::ViewCopy)-->D2[(mask)]
 * ```
 *
 * Scenario 2: When fake_quant_enabled is greater than or equal to 1, because the sizes of scale and zero_point are both 1, broadcast first and then use the FakeQuantPerChannelAffineCachemask operator to complete the calculation.
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A2[(scale)] -->B2(l0op::Contiguous)-->F1(l0op::BroadcastTo)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A3[(zeroPoint)]-->B3(l0op::Contiguous)-->F2(l0op::BroadcastTo)-->C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A4((quantMin)) --> C(l0op::FakeQuantPerChannelAffineCachemask)
 *   A5((quantMax)) --> C(l0op::FakeQuantPerChannelAffineCachemask)
 *   C(l0op::FakeQuantPerChannelAffineCachemask)-->D1(l0op::ViewCopy)-->E1[(out)]
 *   C(l0op::FakeQuantPerChannelAffineCachemask)-->D2(l0op::ViewCopy)-->E2[(mask)]
 * ```
 *
 * @param [in] self:
 * The data type of aclTensor on the Device side supports FLOAT16 and FLOAT32. Supports non-continuous Tensor, [data format](../../../docs/zh/context/data format.md) supports ND.
 * @param [in] scale: aclTensor on the Device side, indicating the input pseudo-quantization scaling coefficient. The data type supports FLOAT16 and FLOAT32, and the size is 1.
 * @param [in] zeroPoint: aclTensor on the Device side, indicating the zero reference parameter of the input pseudo quantization. The data type supports INT32, and the size is 1.
 * @param [in] fake_quant_enabled: Floating point type on the Host side, indicating whether to perform pseudo quantification calculation. The data type supports FLOAT.
 * @param [in] quantMin: Integer type on the Host side, indicating the minimum value after pseudo-quantization of the input data. The data type supports INT.
 * @param [in] quantMax: Integer type on the Host side, indicating the maximum value after pseudo-quantization of the input data. The data type supports INT.
 * @param [out] out:
 * The aclTensor on the Device side supports LOAT16, FLOAT32 data types, non-continuous Tensor, and [data format] (../../../docs/zh/context/data format.md) supports ND.
 * @param [out] mask: aclTensor on the Device side. The data type supports BOOL and non-continuous Tensor. [Data format] (../../../docs/zh/context/data format.md) supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint,
    float fakeQuantEnbled, int64_t quantMin, int64_t quantMax, aclTensor* out,
    aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFakeQuantPerTensorAffineCachemask's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnFakeQuantPerTensorAffineCachemask.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS_H_
