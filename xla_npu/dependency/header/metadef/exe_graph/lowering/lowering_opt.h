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

/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_LOWERING_OPT_H_
#define INC_EXTERNAL_LOWERING_OPT_H_

#include <cstdint>

namespace gert {
struct LoweringOption {
  /**
   * Do you believe the shape on the output tensor passed in by the user? If this option is turned on, you can omit the InferShape of the output node on the calculation graph and improve the Host scheduling performance.
   * At the same time, the ability to verify the length of the externally passed output Tensor Shape and TensorData will also be lost.
   *
   * Constraints:
   * 1. If a node has multiple outputs, and some of the outputs are not the output of the network,
   * Then the InferShape of this node will not be omitted, which means that this option will be ignored on this node.
   * 2. If a node does not have an InferShape function, such as the third and fourth types of operators,
   * If Shape needs to be copied back from Device, this option will be ignored on this node.
   * 3. This option is a load-time option. Once selected, it means that each subsequent call to this model requires the user to pass in the output shape, otherwise the execution may fail.
   */
  bool trust_shape_on_out_tensor = false;

  /**
   * Always zero-copy switch, off by default. If this switch is turned on, it means that the external caller is always guaranteed to correctly apply for output memory, including:
   * 1. The requested output memory is greater than or equal to the output shape, so the calculated Tensor size
   * 2. The placement of the output memory is correct
   *
   * After turning on this switch, the Host scheduling performance can be improved a little. At the same time, rollback processing for zero-copy failures will no longer be performed.
   * When the externally requested output memory is incorrect or the output memory is not requested, an error is reported.
   */
  bool always_zero_copy = false;

  /**
   * Always use external allocator switch, off by default. If this switch is turned on, it means that external callers are always guaranteed to pass in all allocators, including:
   * 1. Create all allocators required in the loading/execution phase and pass them in
   * 2. Since the external allocator is always trusted, once it is turned on, if it fails to obtain the external allocator during the loading/execution phase, an error will be reported.
   *
   * After turning on this switch, there is no need to create an allocator inside the actuator, reducing resource waste.
   */
  bool always_external_allocator = false;

  /**
   * Enable single stream, disabled by default. If this switch is turned on, the dynamic root graph task will be delivered on a stream during execution.
   * This switch is used by the user of rt2 (ACL/hybrid model) to decide whether to use only one stream resource based on whether the device's upstream stream is sufficient.
   *
   */
  bool enable_single_stream = false;

  /**
   * Binary compatible reserved fields, when adding option, correspondingly reduce and delete the reserved length.
   */
  uint8_t reserved[4U + 8U] = {0U};
};
}  // namespace gert

#endif  // INC_EXTERNAL_GRAPH_GRAPH_H_
