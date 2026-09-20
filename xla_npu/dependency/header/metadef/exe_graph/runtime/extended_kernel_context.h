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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
#include <type_traits>
#include "kernel_context.h"
#include "context_extend.h"

namespace gert {
class ExtendedKernelContext : protected KernelContext {
 public:
  /**
   * Get the Tensor description of the node input according to index
   * @param index input index
   * @return Enter the pointer of TensorDesc. When the index is illegal, a null pointer is returned.
   */
  const CompileTimeTensorDesc *GetInputDesc(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetInputTdInfo(index);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input CompileTimeTensorDesc pointer of type `OPTIONAL_INPUT`
   * @param ir_index index in IR prototype definition
   * @return CompileTimeTensorDesc pointer, the index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const CompileTimeTensorDesc *GetOptionalInputDesc(
      const size_t ir_index) const {
    return GetDynamicInputDesc(ir_index, 0);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input CompileTimeTensorDesc pointer of type `DYNAMIC_INPUT`
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return CompileTimeTensorDesc pointer, when index or relative_index is illegal, return a null pointer
   */
  const CompileTimeTensorDesc *GetDynamicInputDesc(
      const size_t ir_index, const size_t relative_index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    const auto ins_info = GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_index) {
      return nullptr;
    }
    return compute_node_info->GetInputTdInfo(ins_info->GetInstanceStart() +
                                             relative_index);
  }
  /**
   * Get the Tensor description output by the node according to index
   * @param index output index
   * @return Outputs the pointer of TensorDesc. When the index is illegal, a null pointer is returned.
   */
  const CompileTimeTensorDesc *GetOutputDesc(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetOutputTdInfo(index);
  }
  /**
   * According to the index in the IR prototype, obtain the instantiation information on the node
   * @param ir_index index defined in IR prototype
   * @return instantiation information
   */
  const AnchorInstanceInfo *GetIrInputInstanceInfo(
      const size_t ir_index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetInputInstanceInfo(ir_index);
  }
  /**
   * Get the input number of the compute node
   * @return Calculate the number of inputs to the node
   */
  size_t GetComputeNodeInputNum() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return 0;
    }
    return compute_node_info->GetInputsNum();
  }
  /**
   * Get the output number of the compute node
   * @return Calculate the number of output nodes
   */
  size_t GetComputeNodeOutputNum() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return 0;
    }
    return compute_node_info->GetOutputsNum();
  }
  /**
   * Get the attributes of the calculation node. Only the attributes defined in the IR prototype can be obtained.
   * @return the properties of the computed node
   */
  const RuntimeAttrs *GetAttrs() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetAttrs();
  }
  /**
   * Get the type of compute node
   * @return type of calculation node
   */
  const ge::char_t *GetNodeType() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetNodeType();
  }
  /**
   * Get the name of the compute node
   * @return the name of the calculation node
   */
  const ge::char_t *GetNodeName() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetNodeName();
  }
  /**
   * Get information about the computing nodes corresponding to this kernel
   * @return Compute node information
   */
  const ComputeNodeInfo *GetComputeNodeInfo() const {
    return static_cast<const ComputeNodeInfo *>(GetComputeNodeExtend());
  }
  /**
   * Get the name of this kernel
   * @return the name of this kernel
   */
  const ge::char_t *GetKernelName() const {
    const auto extend_info = GetExtendInfo();
    if (extend_info == nullptr) {
      return nullptr;
    }
    return extend_info->GetKernelName();
  }
  /**
   * Get the type of this kernel
   * @return The type of this kernel
   */
  const ge::char_t *GetKernelType() const {
    const auto extend_info = GetExtendInfo();
    if (extend_info == nullptr) {
      return nullptr;
    }
    return extend_info->GetKernelType();
  }
  /**
   * Get extended information of this kernel
   * @return Extended information of this kernel
   */
  const KernelExtendInfo *GetExtendInfo() const {
    return static_cast<const KernelExtendInfo *>(GetKernelExtend());
  }

 protected:
  template <typename T, size_t Offset = 0>
  const T *GetDynamicInputPointer(size_t ir_index,
                                  size_t relative_ins_index) const {
    const auto ins_info = GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_ins_index) {
      return nullptr;
    }
    return GetInputPointer<T>(Offset + ins_info->GetInstanceStart() +
                              relative_ins_index);
  }
};
static_assert(std::is_standard_layout<ExtendedKernelContext>::value,
              "The class ExtendedKernelRunContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
