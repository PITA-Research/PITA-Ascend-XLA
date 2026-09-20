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

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <securec.h>
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/ge_error_codes.h"
#include "utils/extern_math_util.h"

namespace gert {
enum class AttrDataType : int32_t {
  kBool = 0,
  kString,
  kInt32,
  kInt64,
  kUint32,
  kFloat32,
  kFloat16,
  kListBool,
  kListString,
  kListInt32,
  kListInt64,
  kListUint32,
  kListFloat32,
  kListFloat16,
  kListListInt32,
  kListListInt64,
  kBfloat16,
  kInt8,
  kInt16,
  kListInt8,
  kListInt16,
  kListListBool,
  kListListFloat16,
  kListBfloat16,
  kListListBfloat16,
  kListListFloat32,
  kListListInt8,
  kListListInt16,
  kUint8,
  kListUint8,
  kListListUint8,
  kUint16,
  kListUint16,
  kListListUint16,
  kListListUint32,
  kUint64,
  kListUint64,
  kListListUint64,
  kTypeEnd
};
class TilingData {
 public:
  /**
   * Get the maximum tiling data length that this instance can accommodate
   * @return maximum tiling data length
   */
  size_t GetCapacity() const { return capacity_; }
  /**
   * Get tiling data length
   * @return tiling data length
   */
  size_t GetDataSize() const { return data_size_; }
  /**
   * Set tiling data length
   * @param size tiling data length
   */
  void SetDataSize(const size_t size) { data_size_ = size; }
  /**
   * Get data pointer
   * @return data pointer
   */
  void *GetData() { return data_; }
  /**
   * Get data pointer
   * @return data pointer
   */
  const void *GetData() const { return data_; }
  /**
   * Add tiling data backwards. If the addition exceeds the maximum length that can be accommodated, the addition will fail.
   * @tparam T The type of tiling data added
   * @param data added tiling data instance
   * @return successfully returns ge::GRAPH_SUCCESS
   */
  template <typename T, typename std::enable_if<
                            std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T &data) {
    size_t after_size;
    if (ge::AddOverflow(data_size_, sizeof(data), after_size)) {
      return ge::GRAPH_FAILED;
    }
    if (after_size > capacity_) {
      return ge::GRAPH_FAILED;
    }
    *reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(data_) + GetDataSize()) =
        data;
    data_size_ = after_size;
    return ge::GRAPH_SUCCESS;
  }

  template <typename T, typename std::enable_if<
                            std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T *data, size_t append_num) {
    size_t append_size;
    if (ge::MulOverflow(sizeof(T), append_num, append_size)) {
      return ge::GRAPH_MUL_OVERFLOW;
    }
    size_t after_size;
    if (ge::AddOverflow(data_size_, append_size, after_size)) {
      return ge::GRAPH_ADD_OVERFLOW;
    }
    const auto ret = memcpy_s(reinterpret_cast<uint8_t *>(data_) + data_size_,
                              capacity_ - data_size_, data, append_size);
    if (ret != EOK) {
      return ge::GRAPH_MEMCPY_FAILED;
    }
    data_size_ = after_size;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * Create a TilingData class instance with the maximum capacity
   * @param cap_size maximum capacity in bytes
   * @return instance pointer
   */
  static std::unique_ptr<uint8_t[]> CreateCap(const size_t cap_size) {
    size_t total_size;
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return nullptr;
    }
    auto td_buf =
        std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]());
    if (td_buf == nullptr) {
      return nullptr;
    }
    const auto td = reinterpret_cast<TilingData *>(td_buf.get());
    td->Init(cap_size, td_buf.get() + sizeof(TilingData));
    return td_buf;
  }
  /**
   * Calculate the memory space occupied by the TilingData instance based on the maximum capacity
   * @param cap_size maximum capacity in bytes
   * @param total_size memory space, unit is bytes
   * @return successfully returns ge::GRAPH_SUCCESS
   */
  static ge::graphStatus CalcTotalSize(const size_t cap_size,
                                       size_t &total_size) {
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  /**
   * InitializeTilingData
   * @param cap_size maximum capacity
   * @param data address of tiling data
   */
  void Init(const size_t cap_size, void *const data) {
    capacity_ = cap_size;
    data_size_ = 0UL;
    data_ = data;
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  ge::graphStatus AppendConvertedAttrVal(const RuntimeAttrs *attrs,
                                         const size_t attr_index,
                                         const AttrDataType src_type,
                                         const AttrDataType dst_type);
  TilingData(const TilingData &) = delete;
  TilingData(TilingData &&) = delete;
  TilingData operator=(const TilingData &) = delete;
  TilingData operator=(TilingData &&) = delete;

 private:
  TilingData() = default;
  size_t capacity_;
  size_t data_size_;
  void *data_;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};

/**
 * Add tiling data backwards. If the addition exceeds the maximum length that can be accommodated, this operation will be ignored.
 * @tparam T The type of tiling data added
 * @param out TilingData class instance
 * @param data instance of added tiling data
 * @return
 */
template <typename T>
TilingData &operator<<(TilingData &out, const T &data) {
  out.Append(
      data);  // we can not throw exception, so callers can not get the error information
  return out;
}
static_assert(std::is_standard_layout<TilingData>::value,
              "The class TilingData must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
