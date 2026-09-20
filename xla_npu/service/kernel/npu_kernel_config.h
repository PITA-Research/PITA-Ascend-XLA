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

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_

#include <string>
#include <unordered_map>

#include "xla/xla_npu/service/kernel/npu_compile_result.h"

namespace xla {

class NpuKernelConfig {
 public:
  enum class NpuKernelType {
    FUSION = 0,
  };

  NpuKernelType npu_kernel_type;
  NpuArtifactInfo npu_artifact_info;

  // Stores all relevant computation HLO text, the key is the computation name
  std::unordered_map<std::string, std::string> computation_hlo_map;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_KERNEL_CONFIG_H_