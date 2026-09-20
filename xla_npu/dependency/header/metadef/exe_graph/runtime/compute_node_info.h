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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include "graph/types.h"
#include "storage_format.h"
#include "runtime_attrs.h"
namespace gert {
class AnchorInstanceInfo {
 public:
  AnchorInstanceInfo() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  };
  AnchorInstanceInfo(const uint32_t instance_start,
                     const uint32_t instantiation_num)
      : instance_start_(instance_start), instantiation_num_(instantiation_num) {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  /**
   * Get the number of instantiations of this input/output
   * @return the number of instantiations
   */
  size_t GetInstanceNum() const { return instantiation_num_; }

  /**
   * Get the index of the first instantiated Anchor of this input/output
   * @return the index of the first instantiated Anchor
   */
  size_t GetInstanceStart() const { return instance_start_; }

  /**
   * Set the index of the first instantiated Anchor of this input/output
   * @param instance_start The index of the first instantiated Anchor
   */
  void SetInstanceStart(const uint32_t instance_start) {
    instance_start_ = instance_start;
  }

  /**
   * Set the number of instantiations of this input/output
   * @param instantiation_num Number of instantiations
   */
  void SetInstantiationNum(const uint32_t instantiation_num) {
    instantiation_num_ = instantiation_num;
  }

 private:
  uint32_t instance_start_;
  uint32_t instantiation_num_;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<AnchorInstanceInfo>::value,
              "The class AnchorInstanceInfo must be a POD");

class CompileTimeTensorDesc {
 public:
  CompileTimeTensorDesc() {
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }
  /**
   * GetDataType
   * @return DataType
   */
  ge::DataType GetDataType() const { return data_type_; }
  /**
   * Get format
   * @return format
   */
  const StorageFormat &GetFormat() const { return storage_format_; }
  /**
   * Get the original format
   * @return original format
   */
  ge::Format GetOriginFormat() const {
    return storage_format_.GetOriginFormat();
  }
  /**
   * Get runtime format
   * @return runtime format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_.GetStorageFormat();
  }
  /**
   * Obtain the supplementary dimension rules when converting the original format to the runtime format
   * @return Supplementary dimension rules
   */
  ExpandDimsType GetExpandDimsType() const {
    return storage_format_.GetExpandDimsType();
  }
  /**
   * Set data type
   * @param data_type data type
   */
  void SetDataType(const ge::DataType data_type) { data_type_ = data_type; }
  /**
   * Set runtime format
   * @param format runtime format
   */
  void SetStorageFormat(const ge::Format format) {
    storage_format_.SetStorageFormat(format);
  }
  /**
   * Set original format
   * @param format original format
   */
  void SetOriginFormat(const ge::Format format) {
    storage_format_.SetOriginFormat(format);
  }
  /**
   * Set the supplementary dimension rules when converting the original format to the runtime format
   * @param expand_dims_type Supplementary dimension rules
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    storage_format_.SetExpandDimsType(expand_dims_type);
  }

 private:
  ge::DataType data_type_;
  StorageFormat storage_format_;
  uint8_t reserved_
      [40];  // Reservd field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<CompileTimeTensorDesc>::value,
              "The class CompileTimeTensorDesc must be a POD");

class ComputeNodeInfo {
 public:
  /**
   * Get the node type of the computing node
   * @return node type
   */
  const ge::char_t *GetNodeType() const { return node_type_; }
  /**
   * Get the node name of the computing node
   * @return node name
   */
  const ge::char_t *GetNodeName() const { return node_name_; }
  /**
   * Get the number of inputs in the operator IR prototype definition
   * @return The number of inputs defined in the IR prototype
   */
  size_t GetIrInputsNum() const { return ir_inputs_num_; }
  /**
   * Get the number of outputs in the operator IR prototype definition
   * @return The number of outputs defined in the IR prototype
   */
  size_t GetIrOutputsNum() const { return ir_outputs_num_; }
  /**
   * Get the input number of the computing node
   * @return The input number of the calculation node
   */
  size_t GetInputsNum() const { return inputs_num_; }
  /**
   * Get the output number of the computing node
   * @return Calculate the number of output nodes
   */
  size_t GetOutputsNum() const { return outputs_num_; }
  /**
   * According to the input index in the IR prototype, obtain the corresponding instantiation information
   * @param ir_index input index in IR prototype definition
   * @return input instantiation information
   */
  const AnchorInstanceInfo *GetInputInstanceInfo(const size_t ir_index) const {
    if (ir_index >= ir_inputs_num_) {
      return nullptr;
    }
    const auto inputs =
        reinterpret_cast<const AnchorInstanceInfo *>(&place_holder);
    return inputs + ir_index;
  }
  /**
  * According to the output index in the IR prototype, obtain the corresponding instantiation information
  * @param ir_index output index in IR prototype definition
  * @return output instantiation information
  */
  const AnchorInstanceInfo *GetOutputInstanceInfo(const size_t ir_index) const {
    if (ir_index >= ir_outputs_num_) {
      return nullptr;
    }
    const auto outputs = reinterpret_cast<const AnchorInstanceInfo *>(
        reinterpret_cast<const uint8_t *>(&place_holder) +
        sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
        sizeof(CompileTimeTensorDesc) * (inputs_num_ + outputs_num_) +
        runtime_attr_size_);
    return outputs + ir_index;
  }
  /**
   * Get the Tensor description input by the computing node. Note that shape information that cannot be determined at compile time is not in the Tensor description.
   * @param index The input index of the calculation node
   * @return Tensor description
   */
  const CompileTimeTensorDesc *GetInputTdInfo(const size_t index) const {
    if (index >= inputs_num_) {
      return nullptr;
    }
    const auto inputs = reinterpret_cast<const CompileTimeTensorDesc *>(
        reinterpret_cast<const uint8_t *>(&place_holder) +
        sizeof(AnchorInstanceInfo) * ir_inputs_num_);
    return inputs + index;
  }
  /**
   * Get the Tensor description output by the computing node. Note that shape information that cannot be determined at compile time is not included in the Tensor description.
   * @param index Calculate the output index of the node
   * @return Tensor description
   */
  const CompileTimeTensorDesc *GetOutputTdInfo(const size_t index) const {
    if (index >= outputs_num_) {
      return nullptr;
    }
    const auto outputs = reinterpret_cast<const CompileTimeTensorDesc *>(
        reinterpret_cast<const uint8_t *>(&place_holder) +
        sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
        sizeof(CompileTimeTensorDesc) * inputs_num_);
    return outputs + index;
  }
  /**
   * Get the attribute values on the compute node. Only the attribute values defined by IR will be returned, and other attribute values will be discarded.
   * @return All attribute values defined by the IR prototype. The attribute values are saved in the order defined by the IR prototype.
   */
  const RuntimeAttrs *GetAttrs() const {
    return reinterpret_cast<const RuntimeAttrs *>(
        reinterpret_cast<const uint8_t *>(&place_holder) +
        sizeof(AnchorInstanceInfo) * ir_inputs_num_ +
        sizeof(CompileTimeTensorDesc) * (inputs_num_ + outputs_num_));
  }
  /**
   * Set the node type of the computing node
   * @param node_type node type of the calculation node
   */
  void SetNodeType(const ge::char_t *node_type) { node_type_ = node_type; }
  /**
   * Set the node name of the computing node
   * @param node_name node name of the computing node
   */
  void SetNodeName(const ge::char_t *node_name) { node_name_ = node_name; }
  /**
   * According to the input index in the IR prototype, obtain the corresponding instantiation information
   * @param ir_index input index in IR prototype definition
   * @return input instantiation information
   */
  AnchorInstanceInfo *MutableInputInstanceInfo(const size_t ir_index) const;
  /**
   * According to the output index in the IR prototype, obtain the corresponding instantiation information
   * @param ir_index output index in IR prototype definition
   * @return output instantiation information
   */
  AnchorInstanceInfo *MutableOutputInstanceInfo(const size_t ir_index) const;
  /**
   * Get the Tensor description input by the computing node. Note that shape information that cannot be determined at compile time is not in the Tensor description.
   * @param index The input index of the calculation node
   * @return Tensor description
   */
  CompileTimeTensorDesc *MutableInputTdInfo(const size_t index) const;
  /**
   * Get the Tensor description output by the computing node. Note that shape information that cannot be determined at compile time is not included in the Tensor description.
   * @param index Calculate the output index of the node
   * @return Tensor description
   */
  CompileTimeTensorDesc *MutableOutputTdInfo(const size_t index) const;
  /**
   * Get the attribute values on the compute node. Only the attribute values defined by IR will be returned, and other attribute values will be discarded.
   * @return All attribute values defined by the IR prototype. The attribute values are saved in the order defined by the IR prototype.
   */
  RuntimeAttrs *MutableAttrs() const;
  static ge::graphStatus CalcSize(const size_t ir_inputs_num,
                                  const size_t inputs_num,
                                  const size_t outputs_num, size_t &total_size);
  void Init(const size_t ir_inputs_num, const size_t inputs_num,
            const size_t outputs_num, const ge::char_t *node_name,
            const ge::char_t *node_type);
  static ge::graphStatus CalcSize(const size_t ir_inputs_num,
                                  const size_t ir_outputs_num,
                                  const size_t inputs_num,
                                  const size_t outputs_num, size_t &total_size);
  void Init(const size_t ir_inputs_num, const size_t ir_outputs_num,
            const size_t inputs_num, const size_t outputs_num,
            const size_t attr_size, const ge::char_t *node_name,
            const ge::char_t *node_type);

  ComputeNodeInfo() = delete;
  ComputeNodeInfo(const ComputeNodeInfo &) = delete;
  ComputeNodeInfo(ComputeNodeInfo &&) = delete;
  ComputeNodeInfo &operator=(const ComputeNodeInfo &) = delete;
  ComputeNodeInfo &operator=(ComputeNodeInfo &&) = delete;

 private:
  const ge::char_t *node_type_;
  const ge::char_t *node_name_;
  size_t ir_inputs_num_;
  size_t inputs_num_;
  size_t outputs_num_;
  size_t ir_outputs_num_;
  size_t runtime_attr_size_;
  uint8_t reserved_
      [24];  // Reserved field, 24+8, do not directly use when only 8-byte left
  // following by inputs-AnchorInstanceInfo, inputs-outputs-CompileTimeTensorDesc, RuntimeAttrs,
  // outputs-AnchorInstanceInfo
  uint64_t place_holder;
};
static_assert(std::is_standard_layout<ComputeNodeInfo>::value,
              "The class ComputeNodeInfo must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_COMPUTE_NODE_INFO_H_
