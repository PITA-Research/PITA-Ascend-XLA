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

#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_

#include <cstring>

#include "xla/xla_npu/dependency/header/metadef/graph/ge_error_codes.h"
#include "storage_shape.h"
#include "storage_format.h"
#include "tensor_data.h"

namespace gert {
using TensorAddress = void *;            ///<Tensor address
using ConstTensorAddress = void *const;  ///<Tensor address

class Tensor {
 public:
  // memse function misra alarm shielding
  Tensor() {
    (void)memset(reserved_, 0, sizeof(reserved_));
    (void)memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format,
         const TensorPlacement placement, const ge::DataType data_type,
         TensorAddress addr)
      : Tensor(storage_shape, storage_format, placement, data_type, addr,
               nullptr) {}
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format,
         ge::DataType data_type)
      : storage_shape_(storage_shape),
        storage_format_(storage_format),
        data_type_(data_type) {
    (void)memset(reserved_, 0, sizeof(reserved_));
    (void)memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format,
         const TensorPlacement placement, const ge::DataType data_type,
         TensorAddress addr, TensorAddrManager manager)
      : storage_shape_(storage_shape),
        storage_format_(storage_format),
        data_type_(data_type),
        tensor_data_(
            addr, manager,
            static_cast<size_t>(ge::GetSizeInBytes(GetShapeSize(), data_type_)),
            placement) {
    (void)memset(reserved_, 0, sizeof(reserved_));
    (void)memset(reserved_field_, 0, sizeof(reserved_field_));
  }
  /**
   * Get the shape size. The so-called shape size refers to the number of elements contained in this shape.
   * @return shape size
   */
  int64_t GetShapeSize() const {
    return storage_shape_.GetStorageShape().GetShapeSize();
  }
  /**
   * Get the data address of Tensor
   * @tparam T data type
   * @return data address
   */
  template <class T>
  const T *GetData() const {
    return static_cast<const T *>(GetAddr());
  }
  /**
   * Get the data address of Tensor
   * @tparam T data type
   * @return data address
   */
  template <class T>
  auto GetData() -> T * {
    return static_cast<T *>(GetAddr());
  }
  /**
   * Set Tensor data
   * @param data data
   */
  void SetData(TensorData &&data) { tensor_data_ = std::move(data); }
  /**
   * Get the data address of Tensor
   * @return data address
   */
  const void *GetAddr() const {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<const void *>(
          reinterpret_cast<const uint8_t *>(this) + sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * Get the data address of Tensor
   * @return data address
   */
  void *GetAddr() {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(this) +
                                      sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * Get the memory size of Tensor
   * @return memory size
   */
  size_t GetSize() const { return tensor_data_.GetSize(); }

  /**
   * Set the memory size of Tensor
   * @param Tensor memory size
   */
  void SetSize(const size_t size) { tensor_data_.SetSize(size); }

  /**
   * Get the data type of Tensor
   * @return data type
   */
  ge::DataType GetDataType() const { return data_type_; }
  /**
   * Set the data type of Tensor
   * @param data_type data type
   */
  void SetDataType(const ge::DataType data_type) { data_type_ = data_type; }
  /**
   * Create a Tensor, and the tensor data is continuously arranged behind the Tensor object.
   * @param shape_size number of elements
   * @param dt data type
   * @param total_size The length of the created Tensor in memory
   * @return the created Tensor pointer
   */
  static std::unique_ptr<uint8_t[]> CreateFollowing(const int64_t shape_size,
                                                    const ge::DataType dt,
                                                    size_t &total_size) {
    total_size = static_cast<size_t>(ge::GetSizeInBytes(shape_size, dt));
    return NewFollowingTensor(dt, total_size);
  }
  /**
   * Create a Tensor, and the tensor data is continuously arranged behind the Tensor object.
   * @param tensor_size tensor length
   * @param dt data type
   * @param total_size The length of the created Tensor in memory
   * @return the created Tensor pointer
   */
  static std::unique_ptr<uint8_t[]> CreateFollowing(const ge::DataType dt,
                                                    const size_t tensor_size,
                                                    size_t &total_size) {
    total_size = tensor_size;
    if (ge::AddOverflow(total_size, sizeof(Tensor), total_size)) {
      return nullptr;
    }
    auto holder =
        std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    auto tensor = reinterpret_cast<Tensor *>(holder.get());
    new (holder.get()) Tensor({}, {}, dt);
    tensor->SetPlacement(kFollowing);
    tensor->tensor_data_ =
        TensorData(nullptr, nullptr, total_size - sizeof(Tensor), kFollowing);
    return holder;
  }
  /**
   * Get running shape
   * @return read-only runtime shape reference
   */
  const Shape &GetStorageShape() const {
    return storage_shape_.GetStorageShape();
  }
  /**
   * Get running shape
   * @return reference to shape at runtime
   */
  Shape &MutableStorageShape() { return storage_shape_.MutableStorageShape(); }
  /**
   * Get the original shape
   * @return read-only original shape reference
   */
  const Shape &GetOriginShape() const {
    return storage_shape_.GetOriginShape();
  }
  /**
   * Get the original shape
   * @return original shape reference
   */
  Shape &MutableOriginShape() { return storage_shape_.MutableOriginShape(); }
  /**
   * Get shape, including running and original shape
   * @return read-only shape reference
   */
  const StorageShape &GetShape() const { return storage_shape_; }
  /**
   * Get shape, including running and original shape
   * @return shape reference
   */
  StorageShape &GetShape() { return storage_shape_; }
  /**
   * Get runtime format
   * @return runtime format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_.GetStorageFormat();
  }
  /**
   * Set runtime format
   * @param storage_format runtime format
   */
  void SetStorageFormat(const ge::Format storage_format) {
    storage_format_.SetStorageFormat(storage_format);
  }
  /**
   * Get the original format
   * @return original format
   */
  ge::Format GetOriginFormat() const {
    return storage_format_.GetOriginFormat();
  }
  /**
   * Set original format
   * @param origin_format original format
   */
  void SetOriginFormat(const ge::Format origin_format) {
    storage_format_.SetOriginFormat(origin_format);
  }
  /**
   * Get format, format contains runtime format and original format
   * @return read-only format reference
   */
  const StorageFormat &GetFormat() const { return storage_format_; }
  /**
   * Get format, format contains runtime format and original format
   * @return format reference
   */
  StorageFormat &MutableFormat() { return storage_format_; }
  /**
   * Get supplementary dimension rules
   * @return Supplementary dimension rules
   */
  ExpandDimsType GetExpandDimsType() const {
    return storage_format_.GetExpandDimsType();
  }
  /**
   * Set dimension compensation rules
   * @param expand_dims_type Supplementary dimension rules
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    storage_format_.SetExpandDimsType(expand_dims_type);
  }
  /**
   * Get tensor's placement
   * @return tensor placement
   */
  TensorPlacement GetPlacement() const { return tensor_data_.GetPlacement(); }
  /**
   * Set tensor placement
   * @param tensor placement
   */
  void SetPlacement(const TensorPlacement placement) {
    tensor_data_.SetPlacement(placement);
  }
  /**
   * Get tensor data
   * @return read-only tensor data reference
   */
  const TensorData &GetTensorData() const { return tensor_data_; }
  /**
   * Get tensor data
   * @return writable tensor data reference
   */
  TensorData &MutableTensorData() { return tensor_data_; }

 private:
  static std::unique_ptr<uint8_t[]> NewFollowingTensor(const ge::DataType dt,
                                                       size_t &total_size) {
    if (ge::AddOverflow(total_size, sizeof(Tensor), total_size)) {
      return nullptr;
    }
    auto holder =
        std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    auto tensor = reinterpret_cast<Tensor *>(holder.get());
    new (holder.get()) Tensor({}, {}, dt);
    tensor->SetPlacement(kFollowing);
    tensor->tensor_data_ =
        TensorData(nullptr, nullptr, total_size - sizeof(Tensor), kFollowing);
    return holder;
  }

 private:
  StorageShape storage_shape_;
  StorageFormat storage_format_;
  uint8_t reserved_[4];  // Reserved field, 4-byte aligned
  ge::DataType data_type_;
  TensorData tensor_data_;
  uint8_t reserved_field_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<Tensor>::value,
              "The class Tensor must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
