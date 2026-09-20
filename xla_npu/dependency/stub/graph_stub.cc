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

#include <iostream>

#include "xla/xla_npu/dependency/header/metadef/graph/graph.h"
#include "xla/xla_npu/dependency/header/metadef/graph/ascend_string.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"

namespace ge {
Graph::Graph(const char* name) {
  std::cout << "[STUB] ge Graph constructor" << std::endl;
}

Graph graph("test");
GNode gnode;

graphStatus Graph::AddOp(const ge::Operator& op) { return 0; }

Graph& Graph::SetInputs(const std::vector<Operator>& inputs) { return graph; }

Graph& Graph::SetOutputs(const std::vector<Operator>& outputs) { return graph; }

Graph& Graph::SetOutputs(
    const std::vector<std::pair<Operator, std::vector<size_t>>>&
        output_indexs) {
  return graph;
}

graphStatus Graph::GetAllOpName(std::vector<AscendString>& names) const {
  return 0;
}

std::vector<GNode> Graph::GetAllNodes() const { return {}; };

GNode AddNodeByOp(const Operator& op) { return gnode; }
AscendString::AscendString(const char* name) {}
bool AscendString::operator<(const AscendString& d) const { return true; }
const char* AscendString::GetString() const { return 0; }
bool AscendString::operator==(const AscendString& d) const { return true; }

TensorDesc::TensorDesc() {}
TensorDesc::TensorDesc(const TensorDesc& desc) {}
TensorDesc::TensorDesc(TensorDesc&& desc) {}
TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}
void TensorDesc::SetShape(const Shape& shape) {}
void TensorDesc::SetPlacement(Placement placement) {};
Placement TensorDesc::GetPlacement() const { return {}; }
void TensorDesc::SetFormat(Format format) {};
void TensorDesc::SetOriginFormat(Format origin_format) {};
void TensorDesc::SetOriginShape(const Shape& origin_shape) {};
void TensorDesc::SetDataType(DataType dt) {};
Shape TensorDesc::GetShape() const { return {}; }
DataType TensorDesc::GetDataType() const { return DataType::DT_FLOAT; }
TensorDesc& TensorDesc::operator=(TensorDesc&& desc){};
void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer,
                  const size_t& const_data_len) {}

/* Shape */
Shape::Shape() {}
Shape::Shape(const std::vector<int64_t>& dims) {}
int64_t Shape::GetDim(size_t idx) const { return 0; }
graphStatus Shape::SetDim(size_t idx, int64_t value) { return 0; }
std::vector<int64_t> Shape::GetDims() const { return {}; }
int64_t Shape::GetShapeSize() const { return 0; }

Tensor::Tensor() {}
Tensor::Tensor(const TensorDesc& tensor_desc, const uint8_t* data,
               size_t size) {}
Tensor::Tensor(const TensorDesc& tensor_desc) {}
graphStatus Tensor::SetData(const uint8_t* data, size_t size) { return 0; }
graphStatus Tensor::SetData(const char_t* data) { return 0; }
graphStatus Tensor::SetTensorDesc(const TensorDesc& tensor_desc) { return 0; }
const uint8_t* Tensor::GetData() const { return nullptr; }
uint8_t* Tensor::GetData() { return nullptr; }
size_t Tensor::GetSize() const { return 0; }
TensorDesc Tensor::GetTensorDesc() const { return {}; }
graphStatus Tensor::ResetData(uint8_t* data, size_t size,
                              const Tensor::DeleteFunc& deleter_func) {
  return 0;
}
graphStatus Tensor::SetShapeDimNum(const size_t dim_num) { return 0; }
graphStatus Tensor::SetShapeDim(const size_t idx, const int64_t dim_value) {
  return 0;
}

std::vector<GNode> Graph::GetDirectNode() const { return {}; }

GNode::GNode() {}
graphStatus GNode::GetType(AscendString& type) const { return 0; }
graphStatus GNode::SetAttr(const AscendString& name,
                           int64_t& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name,
                           int32_t& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, float& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name,
                           AscendString& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, bool& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name,
                           std::vector<int64_t>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name,
                           std::vector<float>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name,
                           std::vector<bool>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<std::vector<int64_t>>& attr_value) const {
  return 0;
}

graphStatus GNode::GetName(AscendString& name) const { return 0; }
}  // namespace ge
