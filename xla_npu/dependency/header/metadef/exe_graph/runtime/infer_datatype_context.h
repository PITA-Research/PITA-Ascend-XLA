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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_DATATYPE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_DATATYPE_CONTEXT_H_
#include <type_traits>
#include "tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
#include "graph/types.h"

namespace gert {
/**
 * InferDataType kernel context
 */
class InferDataTypeContext : public ExtendedKernelContext {
 public:
  /**
   * According to the input index, get the input DataType
   * @param index input index
   * @return When the input datatype and index are illegal, DT_UNDEFINED is returned.
   */
  ge::DataType GetInputDataType(const size_t index) const {
    const auto in_datatype = GetInputPointer<ge::DataType>(index);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }

  /**
   * Based on the operator IR prototype definition, obtain the input DataType of the `OPTIONAL_INPUT` type
   * @param ir_index index in IR prototype definition
   * @return in_datatype, when the index is illegal, or the INPUT is not instantiated, DT_UNDEFINED is returned.
   */
  ge::DataType GetOptionalInputDataType(const size_t index) const {
    const auto in_datatype = GetDynamicInputPointer<ge::DataType>(index, 0);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }
  /**
   * Based on the operator IR prototype definition, obtain the input DataType of the `DYNAMIC_INPUT` type
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return When datatype, index or relative_index is illegal, DT_UNDEFINED is returned
   */
  ge::DataType GetDynamicInputDataType(const size_t index,
                                       const size_t relative_index) const {
    const auto in_datatype =
        GetDynamicInputPointer<ge::DataType>(index, relative_index);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }

  /**
   * According to the output index, get the output DataType
   * @param index output index
   * @return Outputs the shape pointer. When the index is illegal, DT_UNDEFINED is returned.
   */
  ge::DataType GetOutputDataType(const size_t index) const {
    const auto datatype_ptr = GetOutputPointer<ge::DataType>(index);
    if (datatype_ptr == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *datatype_ptr;
  }

  /**
   * Set the output DataType according to the output index
   * @param index output index
   * @param datatype output datatype
   * @return Set the result. When the index is illegal, the return fails.
   */
  ge::graphStatus SetOutputDataType(const size_t index,
                                    const ge::DataType datatype) {
    const auto output_dtype = GetOutputPointer<ge::DataType>(index);
    if (output_dtype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    *output_dtype = datatype;
    return ge::GRAPH_SUCCESS;
  }
};
static_assert(std::is_standard_layout<InferDataTypeContext>::value,
              "The class InferDataTypeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
