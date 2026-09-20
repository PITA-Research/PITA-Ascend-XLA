/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_OPDEV_OP_CONFIG_H
#define OP_API_OP_API_COMMON_INC_OPDEV_OP_CONFIG_H

#include <cstdint>

namespace op {

struct OpConfigInfo {
  OpConfigInfo() = default;
  OpConfigInfo(const OpConfigInfo &rhs) {
    isDeterministicOn_ = rhs.isDeterministicOn_;
    aicNum_ = rhs.aicNum_;
    aivNum_ = rhs.aivNum_;
  }
  OpConfigInfo &operator=(const OpConfigInfo &rhs) {
    if (this != &rhs) {
      isDeterministicOn_ = rhs.isDeterministicOn_;
      aicNum_ = rhs.aicNum_;
      aivNum_ = rhs.aivNum_;
    }
    return *this;
  }

  uint32_t aicNum_{0};
  uint32_t aivNum_{0};
  bool isDeterministicOn_{false};
  uint8_t reserved[7] = {0};
};

}  // namespace op

#endif  // OP_API_OP_API_COMMON_INC_OPDEV_OP_CONFIG_H