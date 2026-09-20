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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
#include <cstdint>
#include <cstddef>
#include "xla/xla_npu/dependency/header/metadef/graph/ge_error_codes.h"
#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "shape.h"

namespace gert {
/**
 * This class is based on the shape after dimensionality complementation and describes the dimensionality complementation rules. This class is designed to have the same length as a uint64_t, so it can be copied at a small cost.
 *
 * The complementary dimension is similar to the ExpandDims operator. On the basis of the original shape, one or more dimensions are added. For example, the original shape[2,2] has two axes, then the shape after adding two dimensions between the two axes is [2,1,1,2].
 * The 0th and 3rd axes of the shape after complementation are called the original axes, and the 1st and 2nd axes are called the complementation axes.
 *
 * This class describes the complementary dimension rules through 1 and 0. 1 represents the current axis as the complementary dimension axis, 0 represents the current axis as the original axis, and represents the source of each axis of the current shape from left to right, for example:
 * | Dimension complement rules | Shape before dimension complement | Shape after dimension complement |
 * | -------- | ----------- | ------------------------------------------------------------ |
 * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
 * | 100      | [2, 3]      | [1, 2, 3]                                                    |
 * | 1000 | [2, 3] | The dimension complementation rule does not match the shape before dimensionality complementation. The rule specifies that the original axis has 3 axes, but the original shape only has 2 axes. An error is reported when the dimensionality complementation is performed. |
 *
 */
class ExpandDimsType {
 public:
  using AxisIndex = uint64_t;
  static constexpr size_t kMaxExpandSize = 56;

  ExpandDimsType() : size_(0U), mask_(0U) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memse function misra alarm shielding
  }

  /**
   * Create a complementary dimension rule from a string
   * @param expand_dims_type Complementary dimension rules described by string
   */
  explicit ExpandDimsType(const ge::char_t *const expand_dims_type)
      : ExpandDimsType() {
    if (expand_dims_type == nullptr) {
      return;
    }
    const auto size = strlen(expand_dims_type);
    if (size > kMaxExpandSize) {
      // log error
      return;
    }

    size_ = size;
    for (AxisIndex i = 0; i < size; ++i) {
      if (expand_dims_type[i] == '1') {
        SetExpandIndex(i);
      }
    }
  }

  /**
   * Create a complementary dimension rule through int. The bit field in int is defined as
   * | Field | Type | Meaning |
   * | -------- | ------- | ---------------------- |
   * | High 8 bits | uint8_t | Complementary dimension rule length |
   * | Lower 56 bits | Bit field | Complementary dimension rules described using 0 and 1 |
   *
   * For the sake of simplicity, the order of the dimension complementation rules is opposite to that of the string. For example, if the dimensionality complementation rule described by the string is "1100", then the corresponding dimensionality complementation rule is "0011" which is converted into a number of 3.
   * @param expand_dims_type Supplementary dimension rules
   */
  explicit ExpandDimsType(const int64_t reshape_type_mask) : ExpandDimsType() {
    if (reshape_type_mask == 0) {
      return;
    }
    size_ = static_cast<uint64_t>(static_cast<uint64_t>(reshape_type_mask) >>
                                  kMaxExpandSize);
    if (size_ > kMaxExpandSize) {
      return;
    }
    mask_ = static_cast<uint64_t>(static_cast<uint64_t>(reshape_type_mask) &
                                  0xffULL);
  }
  /**
   * Determine whether the supplementary dimension rules are consistent
   * @param other another instance
   * @return true/false
   */
  bool operator==(const ExpandDimsType &other) const {
    return (size_ == other.size_) && (mask_ == other.mask_);
  }
  /**
   * Get the number of dims after supplementary dimensions
   * @return
   */
  AxisIndex GetFullSize() const { return size_; }
  /**
   * Set the complementary axis
   * @param index The index-th axis is the complementary dimension axis
   */
  void SetExpandIndex(const AxisIndex index) { mask_ |= (1UL << index); }
  /**
   * Based on the shape after the supplementary dimension, determine whether the index root axis is the supplementary dimension axis.
   * @param index index axis
   * @return true means the complementary dimension axis, false means the axis of the original shape
   */
  bool IsExpandIndex(const AxisIndex index) const {
    return static_cast<bool>((1UL << index) & mask_);
  }
  /**
   * Add dimensions to the shape and write the result to out_shape
   * @param shape input shape, shape before supplementary dimensions
   * @param out_shape output shape, shape after supplementary dimensions
   * @return Dimension addition successfully returns ge::GRAPH_SUCCESS
   */
  ge::graphStatus Expand(const Shape &shape, Shape &out_shape) const {
    if (shape.GetDimNum() == size_) {
      out_shape = shape;
      return ge::GRAPH_SUCCESS;
    }
    size_t shape_pos = 0;
    out_shape.SetDimNum(0);
    for (size_t out_shape_pos = 0; out_shape_pos < size_; ++out_shape_pos) {
      if (!IsExpandIndex(out_shape_pos)) {
        if (shape_pos >= shape.GetDimNum()) {
          return ge::GRAPH_FAILED;
        }
        (void)out_shape.AppendDim(shape.GetDim(shape_pos++));
      } else {
        (void)out_shape.AppendDim(1);
      }
    }

    for (; shape_pos < shape.GetDimNum(); ++shape_pos) {
      (void)out_shape.AppendDim(shape.GetDim(shape_pos));
    }
    return ge::GRAPH_SUCCESS;
  }
  /**
   * Replenishing maintenance in place
   * @param shape directly adds dimensions to this shape
   * @return Dimension addition successfully returns ge::GRAPH_SUCCESS
   */
  ge::graphStatus Expand(Shape &shape) const {
    // full_size:4, shape:[A,B], reshape_type:1010
    // shape:[A,B] + full_size:4 -> [A,B,1,1]
    if (shape.GetDimNum() == size_) {
      return ge::GRAPH_SUCCESS;
    }
    size_t dim_size = shape.GetDimNum();
    size_t cur_dim_size = dim_size;
    while (cur_dim_size++ < size_) {
      (void)shape.AppendDim(1);
    }

    // shape:[A,B,1,1] + 1010 -> [A,1,B,1]
    for (int32_t i = static_cast<int32_t>(size_ - 1UL); i >= 0; --i) {
      if (!IsExpandIndex(static_cast<AxisIndex>(i))) {
        if ((dim_size > 0) && (dim_size - 1UL < static_cast<size_t>(i))) {
          shape.SetDim(static_cast<size_t>(i), shape.GetDim(dim_size - 1));
          shape.SetDim(dim_size - 1UL, 1);
          dim_size--;
        }
      }
    }
    return ge::GRAPH_SUCCESS;
  }

 private:
  uint64_t size_ : 8;
  uint64_t mask_ : kMaxExpandSize;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
