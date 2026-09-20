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

#include "./convert_tensor.h"
#include "./convert_type.h"

#include "./utils.h"

namespace ge {
static const int32_t kGeSizeFloat = sizeof(float);
static const int32_t kGeSizeHalfFloat = sizeof(float) / 2;
static const int32_t kGeSizeInt8 = sizeof(int8_t);
static const int32_t kGeSizeInt16 = sizeof(int16_t);
static const int32_t kGeSizeInt32 = sizeof(int32_t);
static const int32_t kGeSizeInt64 = sizeof(int64_t);
static const int32_t kGeSizeUint8 = sizeof(uint8_t);
static const int32_t kGeSizeBool = sizeof(bool);
static const int32_t kGeSizeDouble = sizeof(double);
static const int32_t kGeSizeUint64 = sizeof(uint64_t);
static const int32_t kGeSizeUint16 = sizeof(uint16_t);
static const int32_t kGeSizeUint32 = sizeof(uint32_t);

static std::map<ge::DataType, int32_t> CONST_OPDATA_TYPE_SIZE_MAP = {
    {ge::DT_FLOAT, kGeSizeFloat},   {ge::DT_FLOAT16, kGeSizeHalfFloat},
    {ge::DT_INT8, kGeSizeInt8},     {ge::DT_INT16, kGeSizeInt16},
    {ge::DT_INT32, kGeSizeInt32},   {ge::DT_INT64, kGeSizeInt64},
    {ge::DT_UINT8, kGeSizeUint8},   {ge::DT_UINT16, kGeSizeUint16},
    {ge::DT_UINT32, kGeSizeUint32}, {ge::DT_UINT64, kGeSizeUint64},
    {ge::DT_DOUBLE, kGeSizeDouble}, {ge::DT_BOOL, kGeSizeBool}};

}  // namespace ge

namespace mlir::air {

template <typename T>
std::string ConvertToString(std::vector<T> x) {
  if (x.size() == 0) return "";
  std::string res;
  res.reserve(x.size() * sizeof(std::string));
  res = "[";
  for (auto e : x) {
    res += std::to_string(e) + " ";
  }
  res += "]";
  return res;
}

template <typename T>
void GetTensorData(Tensor* tensor, std::vector<T>* data) {
  DataType dtype = tensor->GetTensorDesc().GetDataType();
  if (sizeof(T) != CONST_OPDATA_TYPE_SIZE_MAP[dtype]) {
  }
  auto dsize = tensor->GetTensorDesc().GetShape().GetShapeSize();
  const T* p = const_cast<T*>(reinterpret_cast<const T*>(tensor->GetData()));
  for (int64_t i = 0; i < dsize; i++) {
    data->push_back(p[i]);
  }
}

// Converts an MLIR elements attribute and adds it to specified repeated field.
template <typename T>
void ConvertElementsAttr(const mlir::DenseElementsAttr attr, Tensor* output) {
  if (attr.isSplat()) {
    auto raw_data = attr.getRawData().data();
    auto ele_size = attr.getRawData().size();
    auto dsize = attr.size();
    auto padding_data = new uint8_t[ele_size * dsize];
    for (auto i = 0; i < dsize; ++i) {
      memcpy(padding_data + i * ele_size, raw_data, ele_size);
    }
    auto status = output->SetData(
        reinterpret_cast<const uint8_t*>(padding_data), dsize * ele_size);
    delete[] padding_data;
    if (status != ge::GRAPH_SUCCESS) {
    }
  } else {
    auto raw_data = attr.getRawData();
    auto status = output->SetData(
        reinterpret_cast<const uint8_t*>(&raw_data.data()[0]), raw_data.size());
    if (status != ge::GRAPH_SUCCESS) {
    }
  }
}

void ConvertToTensorShape(ArrayRef<int64_t> shape, Shape* output_shape) {
  std::vector<int64_t> dims;
  for (auto d : shape) {
    dims.push_back(d);
  }
  *output_shape = ge::Shape(dims);
}

Status ConvertToTensor(const mlir::ElementsAttr attr, Tensor* output) {
  auto type = attr.getType();
  auto shape = mlir::dyn_cast<mlir::ShapedType>(type).getShape();
  DataType output_dtype;
  ConvertToDataType(type, &output_dtype);
  auto output_shape = ge::Shape();
  ConvertToTensorShape(shape, &output_shape);
  auto output_format = FORMAT_ND;
  TensorDesc output_desc(output_shape, output_format, output_dtype);
  output->SetTensorDesc(output_desc);
  if (mlir::isa<SparseElementsAttr>(attr)) {
    return Status::OK();
  }

  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr) {
    return Status::OK();
  }

  switch (output_dtype) {
    case DT_DOUBLE:
      ConvertElementsAttr<double>(dense_attr, output);
      break;
    case DT_FLOAT16:
      // patch: DT_FLOAT16
      ConvertElementsAttr<int16_t>(dense_attr, output);  // ge::fp16_t
      break;
    case DT_FLOAT:
      ConvertElementsAttr<float>(dense_attr, output);
      break;
    case DT_QUINT8:
    case DT_INT8:
      ConvertElementsAttr<int8_t>(dense_attr, output);
      break;
    case DT_QUINT16:
    case DT_INT16:
      ConvertElementsAttr<int16_t>(dense_attr, output);
      break;
    case DT_INT32:
      ConvertElementsAttr<int32_t>(dense_attr, output);
      break;
    case DT_INT64:
      ConvertElementsAttr<int64_t>(dense_attr, output);
      break;
    case DT_UINT8:
      ConvertElementsAttr<uint8_t>(dense_attr, output);
      break;
    case DT_UINT16:
      ConvertElementsAttr<uint16_t>(dense_attr, output);
      break;
    case DT_UINT32:
      ConvertElementsAttr<uint32_t>(dense_attr, output);
      break;
    case DT_UINT64:
      ConvertElementsAttr<uint64_t>(dense_attr, output);
      break;
    default:
      return Status::OK();
  }

  return Status::OK();
}

}  // namespace mlir::air
