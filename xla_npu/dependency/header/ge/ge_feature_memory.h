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

#ifndef INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
#define INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H

#include <memory>
#include <vector>
#include "ge_api_types.h"
#include "ge_error_codes.h"

namespace ge {
class GE_FUNC_VISIBILITY FeatureMemory {
 public:
  class Builder;

  class FeatureMemoryData;

  ~FeatureMemory();

  FeatureMemory &operator=(const FeatureMemory &) & = delete;
  FeatureMemory(const FeatureMemory &) = delete;

  ///
  /// @brief get memory type
  /// @return memory type
  ///
  MemoryType GetType() const;

  ///
  /// @brief get memory size
  /// @return memory size
  ///
  size_t GetSize() const;

  ///
  /// @brief is fixed feature memory
  /// @return true if feature memory is fixed, otherwise false
  ///
  bool IsFixed() const;

 private:
  FeatureMemory() = default;
  std::shared_ptr<FeatureMemoryData> data_{nullptr};
};
using FeatureMemoryPtr = std::shared_ptr<FeatureMemory>;
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_FEATURE_MEMORY_H
