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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RANGE_H_
#define METADEF_CXX_INC_EXE_GRAPH_RANGE_H_

#include <array>
#include <iostream>
#include "utils/extern_math_util.h"
#include "shape.h"

namespace gert {
template <typename T>
class Range {
 public:
  /**
   * Constructs a range by default
   */
  Range() : min_(nullptr), max_(nullptr) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }

  /**
   * Construct range from minimum and maximum T object pointers
   * @param min range minimum value pointer
   * @param max range maximum value pointer
   */
  Range(T *min, T *max) : min_(min), max_(max) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }

  /**
   * Construct a range through a T pointer, indicating that the maximum and minimum values are the same
   * @param same_ele T pointer
   */
  explicit Range(T *same_ele) : min_(same_ele), max_(same_ele) {
    (void)memset(reserved_, 0, sizeof(reserved_));  // memset function misra alarm shielding
  }

  /**
   * Determine whether it is equal to another range object. If the minimum and maximum element pointers of the two ranges are equal respectively, then the two ranges are considered equal.
   * If pointers are not equal, compare the T objects to see if they are equal.
   * @param rht Another Range object
   * @return true/false
   */
  bool operator==(const Range<T> &rht) const {
    if ((this->min_ == rht.min_) && (this->max_ == rht.max_)) {
      return true;
    } else {
      return (*this->min_ == *rht.min_) && (*this->max_ == *rht.max_);
    }
  }

  /**
   * Set the smallest T object pointer
   * @param min smallest T object pointer
   */
  void SetMin(T *min) { min_ = min; }

  /**
   * Set the largest T object pointer
   * @param max The largest T object pointer
   */
  void SetMax(T *max) { max_ = max; }

  /**
   * Get the smallest T object pointer
   * @return
   */
  const T *GetMin() const { return min_; }

  /**
   * Get the largest T object pointer
   * @return
   */
  const T *GetMax() const { return max_; }

  /**
   * Get the smallest T object pointer
   * @return
   */
  T *GetMin() { return min_; }

  /**
   * Get the largest T object pointer
   * @return
   */
  T *GetMax() { return max_; }

 private:
  T *min_;
  T *max_;
  uint8_t reserved_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RANGE_H_
