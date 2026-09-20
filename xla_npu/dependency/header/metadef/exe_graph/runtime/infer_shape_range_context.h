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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#include <type_traits>
#include "range.h"
#include "tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
namespace gert {
using TensorRange = Range<Tensor>;
/**
 * InferShapeRange kernel context
 */
class InferShapeRangeContext : public ExtendedKernelContext {
 public:
  /**
   * According to the input index, obtain the input shape range pointer
   * @param index input index
   * @return Enter the shape range pointer. When the index is illegal, a null pointer is returned.
   */
  const Range<Shape> *GetInputShapeRange(const size_t index) const {
    return GetInputPointer<Range<Shape>>(index);
  }
  /**
   * According to the input index, get the output tensor pointer
   *
   * **Note: Only when `IMPL_OP` implements the operator and sets the corresponding input to data dependency, can this interface be called to obtain the tensor, otherwise the behavior is undefined. **
   * @param index input index
   * @return Enter the tensor pointer. When the index is illegal, a null pointer is returned.
   */
  const TensorRange *GetInputTensorRange(const size_t index) const {
    return GetInputPointer<TensorRange>(index);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input tensor range pointer of type `OPTIONAL_INPUT`
   * @param ir_index index in IR prototype definition
   * @return tensor range pointer, index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const TensorRange *GetOptionalInputTensorRange(const size_t ir_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, 0);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input tensor pointer of type `DYNAMIC_INPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return tensor range pointer, when index or relative_index is illegal, return a null pointer
   */
  const TensorRange *GetDynamicInputTensorRange(
      const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, relative_index);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input shape range pointer of type `OPTIONAL_INPUT`
   * @param ir_index index in IR prototype definition
   * @return shape range pointer, index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const Range<Shape> *GetOptionalInputShapeRange(const size_t ir_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, 0);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input shape pointer of type `DYNAMIC_INPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return shape range pointer, when index or relative_index is illegal, return a null pointer
   */
  const Range<Shape> *GetDynamicInputShapeRange(
      const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, relative_index);
  }

  /**
   * According to the output index, obtain the output shape range pointer
   * @param index output index
   * @return Outputs the shape range pointer. When the index is illegal, a null pointer is returned.
   */
  Range<Shape> *GetOutputShapeRange(const size_t index) {
    return GetOutputPointer<Range<Shape>>(index);
  }
};
static_assert(std::is_standard_layout<InferShapeRangeContext>::value,
              "The class InferShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
