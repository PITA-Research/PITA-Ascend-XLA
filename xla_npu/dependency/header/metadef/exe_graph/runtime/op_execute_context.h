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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_OP_EXECUTE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_OP_EXECUTE_CONTEXT_H_
#include <type_traits>
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "ge/ge_allocator.h"

namespace gert {
using rtStream = void *;
struct OpExecuteOptions {
  int32_t precision_mode;  // precision mode
  int32_t deterministic;   // deterministic calculation
  char allow_hf32[3UL];    // hf32
  char reserve[53];        // reserved
};

enum class OpExecuteInputExtendIndex {
  kAllocate,
  kStream,
  kExecuteOption,
  kExecuteFunc,
  // add new extend input here
  kNum
};

enum class OpExecuteOutputIndex {
  kBlockMemory,
  // add new extend output here
  kNum
};

/**
 * ACLNN kernel context
 */
class OpExecuteContext : public ExtendedKernelContext {
 public:
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
   * Based on the operator IR prototype definition, obtain the input Tensor pointer of type `DYNAMIC_INPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return tensor pointer, when index or relative_index is illegal, return null pointer
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index,
                                      const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * According to the output index, get the output tensor pointer
   *
   * **Note: Only when `IMPL_OP` implements the operator and sets the corresponding input to data dependency, can this interface be called to obtain the tensor, otherwise the behavior is undefined. **
   * @param index output index
   * @return Outputs the tensor pointer. When the index is illegal, a null pointer is returned.
   */
  const Tensor *GetOutputTensor(const size_t index) const {
    const size_t input_num = GetComputeNodeInputNum();
    return GetInputPointer<Tensor>(input_num + index);
  }

  /**
   * Based on the operator IR prototype definition, obtain the input Tensor pointer of type `DYNAMIC_OUTPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_OUTPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return tensor pointer, when index or relative_index is illegal, return null pointer
   */
  const Tensor *GetDynamicOutputTensor(const size_t ir_index,
                                       const size_t relative_index) const {
    const size_t input_num = GetComputeNodeInputNum();
    return GetDynamicInputPointer<Tensor>(input_num + ir_index, relative_index);
  }

  /**
   * Get stream
   * @return rtStream, the stream issued by the ACLNN operator, returns nullptr in abnormal circumstances
   */
  rtStream GetStream() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    auto stream = GetInputPointer<rtStream>(
        input_num + output_num +
        static_cast<size_t>(OpExecuteInputExtendIndex::kStream));
    if (stream == nullptr) {
      return nullptr;
    }
    return *stream;
  }

  /**
   * Get ACLNN interface
   * @return void *, ACLNN interface pointer, returns nullptr in abnormal situations
   */
  void *GetOpExecuteFunc() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    auto op_execute_func = GetInputPointer<void *>(
        input_num + output_num +
        static_cast<size_t>(OpExecuteInputExtendIndex::kExecuteFunc));
    if (op_execute_func == nullptr) {
      return nullptr;
    }
    return *op_execute_func;
  }

  /**
   * Apply for workspace memory size
   * @param size The size of the requested memory
   * @return void *, memory address, return nullptr in abnormal circumstances
   */
  void *MallocWorkspace(const size_t size);

  /**
   * Release workspace memory
   */
  void FreeWorkspace();

  /**
   * Get deterministic computing mode
   * @return bool, whether to enable deterministic calculation, return false by default in abnormal situations
   */
  bool GetDeterministic() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions *options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num +
        static_cast<size_t>(OpExecuteInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return false;
    }
    return (options->deterministic != 0);
  }

  /**
   * Get allow_hf32
   * @return string, whether to enable hf32, normally return four strings: 01, 00, 10, 11
   * The first character indicates whether the Conv class operator supports hf32
   * The second character indicates whether the MatMul operator supports hf32. In abnormal cases, nullptr is returned.
   */
  const char *GetAllowHf32() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions *options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num +
        static_cast<size_t>(OpExecuteInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return nullptr;
    }
    return options->allow_hf32;
  }

  /**
   * Get precision mode
   * @return int32, precision mode, abnormal situation returns a maximum value of int32
   */
  int32_t GetPrecisionMode() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions *options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num +
        static_cast<size_t>(OpExecuteInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return options->precision_mode;
  }
};
static_assert(std::is_standard_layout<OpExecuteContext>::value,
              "The class OpExecuteContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_OP_EXECUTE_CONTEXT_H_
