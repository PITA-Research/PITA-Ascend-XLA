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

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#include "storage_shape.h"
#include "tensor.h"
#include "continuous_vector.h"
#include "extended_kernel_context.h"
#include "tiling_data.h"
#include "external/ge_common/ge_api_error_codes.h"

namespace fe {
class PlatFormInfos;
}  // namespace fe

namespace gert {
/**
 * tiling kernel context
 */
class TilingContext : public ExtendedKernelContext {
 public:
  const void *GetCompileInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index =
        compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<void *>();
  }
  /**
   * GetCompileInfo
   * @tparam T CompileInfo type
   * @return CompileInfo pointer
   */
  template <typename T>
  const T *GetCompileInfo() const {
    return reinterpret_cast<const T *>(GetCompileInfo());
  }
  /**
   * Get the input shape, which contains the original shape and the runtime shape.
   * @param index input index
   * @return Enter the shape pointer. If the index is illegal, a null pointer is returned.
   */
  const StorageShape *GetInputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }
    return GetInputPointer<StorageShape>(index);
  }
  /**
   * Get input tensor
   *
   * **Note: Only when `IMPL_OP` implements the operator and sets the corresponding input to data dependency, can this interface be called to obtain the tensor, otherwise the behavior is undefined. **
   * @param index input index
   * @return Enter the tensor pointer, and return a null pointer when the index is illegal.
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
   * @return Tensor pointer, when index or relative_index is illegal, a null pointer is returned
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index,
                                      const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input shape pointer of the `OPTIONAL_INPUT` type. The shape contains the original shape and the runtime shape.
   * @param ir_index index in IR prototype definition
   * @return shape pointer, index is illegal, or when the INPUT is not instantiated, a null pointer is returned
   */
  const StorageShape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
  }
  /**
   * Based on the operator IR prototype definition, obtain the input shape pointer of the `DYNAMIC_INPUT` type. The shape contains the original shape and the runtime shape.
   * @param ir_index index in IR prototype definition
   * @param relative_index The relative index after the input is instantiated. For example, if a DYNAMIC_INPUT instantiates 3 inputs, then the valid range of relative_index is [0,2]
   * @return shape pointer, when index or relative_index is illegal, return a null pointer
   */
  const StorageShape *GetDynamicInputShape(const size_t ir_index,
                                           const size_t relative_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, relative_index);
  }
  /**
   * According to the output index, obtain the output shape pointer, which contains the original shape and runtime shape.
   * @param index output index
   * @return Outputs the shape pointer. When the index is illegal, a null pointer is returned.
   */
  const StorageShape *GetOutputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    const size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }

  /*
   * outputs, the outputs of tiling are arranged in the following order:
   * outputs[0]: tiling-key
   * outputs[1]: block-dim
   * outputs[2]: atomic-clean-flag
   * outputs[3]: tiling-data
   * outputs[4]: workspace sizes
   * outputs[5]: tiling condition
   * outputs[6]: schedule mode
   */
  enum TilingOutputIndex : uint32_t {
    kOutputTilingKey,
    kOutputBlockDim,
    kOutputAtomicCleanFlag,
    kOutputTilingData,
    kOutputWorkspace,
    kOutputTilingCond,
    kOutputScheduleMode,
    // add new output definitions here
    kOutputNum
  };

  /*
  * outputs[0]: fallible tiling condition
  */
  enum FallibleTilingOutputIndex : uint32_t {
    kTilingStatus = TilingOutputIndex::kOutputNum,
    // add new output definitions here
    kFallibleOutputNum
  };

  /**
   * Set tiling key
   * @param tiling_key tiling key
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus SetTilingKey(const uint64_t tiling_key) {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_key;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Get tiling key
   * @return tiling key, when acquisition fails
   */
  uint64_t GetTilingKey() const {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return std::numeric_limits<uint64_t>::max();
    }
    return *p;
  }

  /**
   * Set schedule_mode
   * @param schedule_mode schedule_mode
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus SetScheduleMode(const uint32_t schedule_mode) {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = schedule_mode;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Get settings schedule_mode
   * @return Set schedule_mode, when acquisition fails
   */
  uint32_t GetScheduleMode() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return 0U;
    }
    return *p;
  }

  /**
   * Set block dim
   * @param block_dim block dim
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus SetBlockDim(const uint32_t block_dim) {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = block_dim;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Get block dim
   * @return block dim
   */
  uint32_t GetBlockDim() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }
  /**
   * Set tiling cond
   * @param tiling_cond tiling condition
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus SetTilingCond(int32_t tiling_cond) {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_cond;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Get tiling cond
   * @return tiling cond: The valid tiling_cond is greater than or equal to 0. If the value is invalid, -1 is returned.
   */
  int32_t GetTilingCond() const {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return -1;
    }
    return *p;
  }
  /**
   * Set whether atomic clean is required
   * @param atomic true/false represents whether atomic clean needs to be done
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus SetNeedAtomic(const bool atomic) {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = atomic;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Get whether atomic clean is required
   * @return true/false
   */
  bool NeedAtomic() const {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return false;
    }
    return *p;
  }
  /**
   * Get a typed tiling data pointer
   * @tparam T tiling data type, sizeof(T) cannot be greater than the maximum tiling data length specified in the compilation result
   * @return tiling data pointer, return null pointer on failure
   */
  template <typename T>
  auto GetTilingData() -> T * {
    auto tiling_data = GetRawTilingData();
    if (tiling_data == nullptr) {
      return nullptr;
    }
    if (tiling_data->GetCapacity() < sizeof(T)) {
      return nullptr;
    }
    tiling_data->SetDataSize(sizeof(T));
    return static_cast<T *>(tiling_data->GetData());
  }
  /**
   * Get untyped tiling data code stream
   * @return tiling data pointer, return null pointer on failure
   */
  TilingData *GetRawTilingData() {
    return *GetOutputPointer<TilingData *>(kOutputTilingData);
  }
  /**
   * Get the workspace sizes pointer
   * @param workspace_count The number of workspaces. The number of workspaces passed in cannot exceed the maximum number of workspaces specified at compile time.
   * @return workspace sizes pointer
   */
  size_t *GetWorkspaceSizes(const size_t workspace_count) {
    const auto workspace =
        GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return nullptr;
    }
    if (workspace->SetSize(workspace_count) != ge::SUCCESS) {
      return nullptr;
    }
    return workspace->MutableData();
  }
  /**
   * Get the number of workspaces
   * @return the number of workspaces
   */
  size_t GetWorkspaceNum() const {
    const auto workspace =
        GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return 0U;
    }
    return workspace->GetSize();
  }
  /**
   * Get fe::PlatFormInfos pointer
   * @return fe::PlatFormInfos pointer
   */
  fe::PlatFormInfos *GetPlatformInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index =
        compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index + 1U);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<fe::PlatFormInfos *>();
  }

  /**
   * Get deterministic calculation variables
   * @return int32 variable
   */
  int32_t GetDeterministic() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    const size_t index =
        compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    // Here, according to tiling memory arrangement, deterministic calculation fields are added in
    // inputshape outputshape compileinfo platform tiling_func after
    const auto av = GetInput(index + 3U);
    if (av == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return av->GetValue<int32_t>();
  }
};
static_assert(std::is_standard_layout<TilingContext>::value,
              "The class TilingContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
