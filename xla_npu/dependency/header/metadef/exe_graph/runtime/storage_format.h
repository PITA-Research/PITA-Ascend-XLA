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

#ifndef METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
#define METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
#include <memory>
#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "expand_dims_type.h"
namespace gert {
struct StorageFormat {
 public:
  StorageFormat() {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  };
  /**
   * Construct a format, including original format, runtime format, and supplementary dimension rules
   * @param origin_format original format
   * @param storage_format runtime format
   * @param expand_dims_type Supplementary dimension rules
   */
  StorageFormat(const ge::Format origin_format, const ge::Format storage_format,
                const ExpandDimsType &expand_dims_type)
      : origin_format_(origin_format),
        storage_format_(storage_format),
        expand_dims_type_(expand_dims_type) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }
  /**
   * Get the original format
   * @return original format
   */
  ge::Format GetOriginFormat() const { return origin_format_; }
  /**
   * Set original format
   * @param origin_format original format
   */
  void SetOriginFormat(const ge::Format origin_format) {
    origin_format_ = origin_format;
  }
  /**
   * Get runtime format
   * @return runtime format
   */
  ge::Format GetStorageFormat() const { return storage_format_; }
  /**
   * Set runtime format
   * @param storage_format runtime format
   */
  void SetStorageFormat(const ge::Format storage_format) {
    storage_format_ = storage_format;
  }
  /**
   * Get supplementary dimension rules
   * @return Supplementary dimension rules
   */
  ExpandDimsType GetExpandDimsType() const { return expand_dims_type_; }
  /**
   * Set dimension compensation rules
   * @param expand_dims_type Supplementary dimension rules
   */
  void SetExpandDimsType(const ExpandDimsType &expand_dims_type) {
    expand_dims_type_ = expand_dims_type;
  }
  /**
   * Get writable supplementary dimension rules
   * @return Supplementary dimension rule reference
   */
  ExpandDimsType &MutableExpandDimsType() { return expand_dims_type_; }
  /**
   * Determine whether the formats are equal
   * @param other another format
   * @return true means equal
   */
  bool operator==(const StorageFormat &other) const {
    return origin_format_ == other.origin_format_ &&
           storage_format_ == other.storage_format_ &&
           expand_dims_type_ == other.expand_dims_type_;
  }
  /**
   * Determine whether the formats are not equal
   * @param other another format
   * @return true means not equal
   */
  bool operator!=(const StorageFormat &other) const {
    return !(*this == other);
  }

 private:
  ge::Format origin_format_;
  ge::Format storage_format_;
  ExpandDimsType expand_dims_type_;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<StorageFormat>::value,
              "The class StorageFormat must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_STORAGE_FORMAT_H_
