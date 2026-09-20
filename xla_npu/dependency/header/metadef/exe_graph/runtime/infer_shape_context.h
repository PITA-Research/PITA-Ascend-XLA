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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
#include <type_traits>
#include "shape.h"
#include "tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
#include "graph/inference_context.h"
namespace gert {
/**
 * The index of the extended input after the node input. If extension is required, please add an enumeration type.
 */
enum class InputExternLayout : uint32_t {
  kInferShapeFunc =
      1,  // only exe runtime need infer shape func, compile stage need set to null
  kInferenceContext =
      2,  // only resource op in compile stage need inference context, exe runtime need set to null
};
/**
 * InferShape kernel context
 */
class InferShapeContext : public ExtendedKernelContext {
 public:
  /**
   * According to the input index, get the input shape pointer
   * @param index input index
   * @return Enter the shape pointer. When the index is illegal, a null pointer is returned.
   */
  const Shape *GetInputShape(const size_t index) const {
    return GetInputPointer<Shape>(index);
  }
  /**
   * According to the input index, get the output tensor pointer
   *
   * **Note: Only when `IMPL_OP` implements the operator and sets the corresponding input to data dependency, can this interface be called to obtain the tensor, otherwise the behavior is undefined. **
   * @param index input index
   * @return Enter the tensor pointer. When the index is illegal, a null pointer is returned.
   */
  const Tensor *GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }

  /**
   * Based on the operator IR prototype definition, obtain the input tensor pointer of type `OPTIONAL_INPUT`
   * @param ir_index index in IR prototype definition
   * @return tensor pointer, index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const Tensor *GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * Based on the operator IR prototype definition, obtain the input shape pointer of type `OPTIONAL_INPUT`
   * @param ir_index index in IR prototype definition
   * @return shape pointer, index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const Shape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input shape pointer of type `DYNAMIC_INPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return shape pointer, when index or relative_index is illegal, return a null pointer
   */
  const Shape *GetDynamicInputShape(const size_t ir_index,
                                    const size_t relative_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, relative_index);
  }

  /**
   * According to the output index, obtain the output shape pointer
   * @param index output index
   * @return Outputs the shape pointer. When the index is illegal, a null pointer is returned.
   */
  Shape *GetOutputShape(const size_t index) {
    return GetOutputPointer<Shape>(index);
  }

  /**
   * Get the InferenceContext pointer
   * @param NA
   * @return Outputs the InferenceContext pointer. The pointer is only used in compiled state after the node input address.
   */
  const ge::InferenceContext *GetInferenceContextPtr() const {
    const auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(
        GetContext()->compute_node_info);
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    const size_t offset =
        compute_node_info->GetInputsNum() +
        static_cast<size_t>(InputExternLayout::kInferenceContext);
    if (GetContext()->input_size < offset) {
      return nullptr;
    }
    return GetInputPointer<ge::InferenceContext>(offset - 1UL);
  }
};
static_assert(std::is_standard_layout<InferShapeContext>::value,
              "The class InferShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
