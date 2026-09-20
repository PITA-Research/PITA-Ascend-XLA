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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_
#include <type_traits>
#include <memory>
#include "compute_node_info.h"

namespace gert {
class KernelExtendInfo {
 public:
  /**
   * Get kernel name
   * @return kernel name
   */
  const ge::char_t *GetKernelName() const { return kernel_name_; }
  /**
   * Set kernel name
   * @param kernel_name kernel name
   */
  void SetKernelName(const ge::char_t *kernel_name) {
    kernel_name_ = kernel_name;
  }
  /**
   * Get kernel type
   * @return kernel type
   */
  const ge::char_t *GetKernelType() const { return kernel_type_; }
  /**
   * Set kernel type
   * @param kernel_type kernel type
   */
  void SetKernelType(const ge::char_t *const kernel_type) {
    (void)reserved_;
    kernel_type_ = kernel_type;
  }

  KernelExtendInfo() = delete;
  KernelExtendInfo(const KernelExtendInfo &) = delete;
  KernelExtendInfo(KernelExtendInfo &&) = delete;
  KernelExtendInfo &operator=(const KernelExtendInfo &) = delete;
  KernelExtendInfo &operator=(KernelExtendInfo &&) = delete;

 private:
  const ge::char_t *kernel_name_;
  const ge::char_t *kernel_type_;
  uint8_t reserved_
      [56];  // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<KernelExtendInfo>::value,
              "The class KernelExtendInfo must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTEXT_EXTEND_H_
