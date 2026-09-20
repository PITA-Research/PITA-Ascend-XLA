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

#ifndef INC_EXTERNAL_REGISTER_OP_TILING_ATTR_UTILS_H_
#define INC_EXTERNAL_REGISTER_OP_TILING_ATTR_UTILS_H_

#include <memory>
#include "external/graph/operator.h"

namespace optiling {
class AttrData;
using AttrDataPtr = std::shared_ptr<AttrData>;

class AttrData {
 public:
  AttrData() {}
  virtual ~AttrData() {}
  virtual size_t GetSize() const = 0;
  virtual const std::uint8_t *GetData() = 0;
};

ge::graphStatus GetOperatorAttrValue(const ge::Operator &op,
                                     const char *attr_name,
                                     const char *attr_dtype,
                                     AttrDataPtr &attr_data_ptr,
                                     const char *target_dtype = nullptr);

}  // namespace optiling
#endif  // INC_EXTERNAL_REGISTER_OP_TILING_ATTR_UTILS_H_
