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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
#include <cstdint>
#include <type_traits>
#include <cstddef>
#include "continuous_vector.h"
#include "tensor.h"

namespace gert {
class RuntimeAttrs {
 public:
  /**
   * Get properties
   * @tparam T attribute type
   * @param index attribute index
   * @return pointer to property
   */
  template <typename T>
  const T *GetAttrPointer(size_t index) const {
    return reinterpret_cast<const T *>(GetPointerByIndex(index));
  }
  /**
   * Get int type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const int64_t *GetInt(const size_t index) const {
    return GetAttrPointer<int64_t>(index);
  }
  /**
   * Get list int type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const TypedContinuousVector<int64_t> *GetListInt(const size_t index) const {
    return GetAttrPointer<TypedContinuousVector<int64_t>>(index);
  }
  /**
   * Get list int type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const ContinuousVectorVector *GetListListInt(const size_t index) const {
    return GetAttrPointer<ContinuousVectorVector>(index);
  }
  /**
   * Get string type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const char *GetStr(const size_t index) const {
    return GetAttrPointer<char>(index);
  }
  /**
   * Get tensor type attributes
   * @param index attribute index
   * @return pointer to attribute value
   */
  const Tensor *GetTensor(const size_t index) const {
    return GetAttrPointer<Tensor>(index);
  }
  /**
   * Get float type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const float *GetFloat(const size_t index) const {
    return GetAttrPointer<float>(index);
  }
  /**
   * Get bool type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const bool *GetBool(const size_t index) const {
    return GetAttrPointer<bool>(index);
  }

  /**
   * Get list_float32 type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const TypedContinuousVector<float> *GetListFloat(const size_t index) const {
    return GetAttrPointer<TypedContinuousVector<float>>(index);
  }

  /**
   * Get list_float32 type attribute
   * @param index attribute index
   * @return pointer to attribute value
   */
  const ContinuousVectorVector *GetListListFloat(const size_t index) const {
    return GetAttrPointer<ContinuousVectorVector>(index);
  }

  /**
   * Get the number of attributes
   * @return number of attributes
   */
  size_t GetAttrNum() const;

  RuntimeAttrs() = delete;
  RuntimeAttrs(const RuntimeAttrs &) = delete;
  RuntimeAttrs(RuntimeAttrs &&) = delete;
  RuntimeAttrs &operator=(const RuntimeAttrs &) = delete;
  RuntimeAttrs &operator=(RuntimeAttrs &&) = delete;

 private:
  const void *GetPointerByIndex(size_t index) const;

  uint64_t placeholder_;
};
static_assert(std::is_standard_layout<RuntimeAttrs>::value,
              "This class must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
