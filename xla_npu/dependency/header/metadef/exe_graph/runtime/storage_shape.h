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

#ifndef METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#include <type_traits>
#include "shape.h"

namespace gert {
struct StorageShape {
 public:
  StorageShape() {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  };
  /**
   * Construct a runtime shape instance
   * @param origin_shape original shape
   * @param storage_shape runtime shape
   */
  StorageShape(const std::initializer_list<int64_t> &origin_shape,
               const std::initializer_list<int64_t> &storage_shape)
      : origin_shape_(origin_shape), storage_shape_(storage_shape) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }
  /**
   * Get the original shape
   * @return original shape
   */
  const Shape &GetOriginShape() const { return origin_shape_; }
  /**
   * Get runtime shape
   * @return runtime shape
   */
  const Shape &GetStorageShape() const { return storage_shape_; }
  /**
   * Get the writable original shape
   * @return writable original shape
   */
  Shape &MutableOriginShape() { return origin_shape_; }
  /**
   * Get a writable runtime shape
   * @return writable runtime shape
   */
  Shape &MutableStorageShape() { return storage_shape_; }
  /**
   * Determine whether shapes are equal
   * @param other another shape
   * @return true means equal
   */
  bool operator==(const StorageShape &other) const {
    return origin_shape_ == other.origin_shape_ &&
           storage_shape_ == other.storage_shape_;
  }
  /**
   * Determine whether shapes are not equal
   * @param other another shape
   * @return true means not equal
   */
  bool operator!=(const StorageShape &other) const { return !(*this == other); }

 private:
  Shape origin_shape_;
  Shape storage_shape_;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<StorageShape>::value,
              "The class must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
