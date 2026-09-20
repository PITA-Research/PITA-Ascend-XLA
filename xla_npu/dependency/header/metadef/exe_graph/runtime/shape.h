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

#ifndef METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_

#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <limits>
#include "xla/xla_npu/dependency/header/metadef/utils/extern_math_util.h"

namespace gert {
struct Shape {
 public:
  static constexpr size_t kMaxDimNum = 25;
  static constexpr int64_t kInvalidDimValue =
      std::numeric_limits<int64_t>::min();

 public:
  /**
   * A shape is constructed by default. In the shape instance constructed by default, the length of dim_num is 0.
   */
  Shape() : dim_num_(0), dims_{0} {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }

  /**
   * Construct a shape through dims values, for example: Shape({8,3,224,224}) creates a Shape instance with 4 dimensions, and the values of each dimension are 8,3,224,224 respectively.
   * @param dims All dim values of shape
   */
  Shape(const std::initializer_list<int64_t> &args) : Shape() {
    if (args.size() > kMaxDimNum) {
      return;
    }
    dim_num_ = args.size();
    size_t i = 0;
    for (const int64_t arg : args) {
      dims_[i++] = arg;
    }
  }

  /**
   * copy construction
   * @param other source object
   * In order to improve performance, the space of dims_ exceeding dim_num_ is not copied, and there may be dirty data.
   */
  Shape(const Shape &other) {
    dim_num_ = other.dim_num_;
    for (size_t i = 0U; i < dim_num_; ++i) {
      dims_[i] = other.dims_[i];
    }
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }

  /**
   * copy assignment
   * @param other
   * @return
   * In order to improve performance, the space of dims_ exceeding dim_num_ is not copied, and there may be dirty data.
   */
  Shape &operator=(const Shape &other) {
    if (&other != this) {
      dim_num_ = other.dim_num_;
      for (size_t i = 0U; i < dim_num_; ++i) {
        dims_[i] = other.dims_[i];
      }
    }
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
    return *this;
  }

  /**
   * Determine whether it is equal to another shape object. If the dim num of two shapes and the value of each dim in dim num are equal, then the two shapes are considered equal.
   * @param rht Another Shape object
   * @return true/false
   */
  bool operator==(const Shape &rht) const {
    if (this->dim_num_ != rht.dim_num_) {
      return false;
    }
    for (size_t i = 0; i < this->dim_num_; i++) {
      if (this->dims_[i] != rht.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Determine whether it is not equal to another Shape object
   * @param rht Another Shape object
   * @return true/false
   */
  bool operator!=(const Shape &rht) const { return !(*this == rht); }

  /**
   * Get the shape size. The so-called shape size refers to how many elements there are in the shape.
   * @return shape-size, when overflow occurs, return `kInvalidDimValue`
   */
  int64_t GetShapeSize() const {
    int64_t shape_size = 1;
    for (size_t i = 0; i < dim_num_; ++i) {
      if (ge::MulOverflow(shape_size, dims_[i], shape_size)) {
        return kInvalidDimValue;
      }
    }
    return shape_size;
  }

  /**
   * Determine whether this shape is a scalar. The so-called scalar means that GetDimNum() is 0
   * @return true/false
   */
  bool IsScalar() const { return dim_num_ == 0L; }

  /**
   * Set shape to scalar
   * @param none
   */
  void SetScalar() { dim_num_ = 0; }

  /**
   * Get dim num
   * @return
   */
  size_t GetDimNum() const { return dim_num_; }

  /**
   * Set dim num
   * @param dim_num
   */
  void SetDimNum(const size_t dim_num) { this->dim_num_ = dim_num; }

  /**
   * Get dim value
   * @param idx dim index, the caller needs to ensure that the index is legal
   * @return dim value, when idx exceeds MaxDimNum, return `kInvalidDimValue`
   */
  int64_t GetDim(const size_t idx) const {
    if (idx >= kMaxDimNum) {
      return kInvalidDimValue;
    }
    return dims_[idx];
  }

  /**
   * Get dim value
   * @param idx dim index, the caller needs to ensure that the index is legal
   * @return dim value, behavior is undefined
   */
  const int64_t &operator[](const size_t idx) const { return dims_[idx]; }

  /**
   * Get dim value
   * @param idx dim index, the caller needs to ensure that the index is legal
   * @return dim value, when idx exceeds MaxDimNum, the behavior is undefined
   */
  int64_t &operator[](const size_t idx) { return dims_[idx]; }

  /**
   * Set dim value
   * @param idx dim index, the caller needs to ensure that the index is legal
   * @return
   */
  void SetDim(size_t idx, const int64_t dim_value) {
    if (idx >= kMaxDimNum) {
      return;
    }
    dims_[idx] = dim_value;
    this->dim_num_ = (this->dim_num_ < idx) ? idx : this->dim_num_;
  }

  /**
   * Extend a dim value backwards. If the number of extended dims exceeds the maximum limit of Shape, then this function does nothing.
   * @param extended dim value
   * @return this reference
   */
  Shape &AppendDim(const int64_t value) {
    if (this->dim_num_ >= kMaxDimNum) {
      return *this;
    }
    dims_[this->dim_num_++] = value;
    return *this;
  }

 private:
  size_t dim_num_;
  int64_t dims_[kMaxDimNum];
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<Shape>::value,
              "The class Shape must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
