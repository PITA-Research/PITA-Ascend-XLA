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

#ifndef XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
#define XLA_XLA_NPU_COMMON_NPU_CONFIG_H_

#include <cstdlib>
#include <string>
#include "xla/xla_npu/dependency/tool/config_loader.h"

namespace xla {

class NpuConfig {
 public:
  enum class Key {
    NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR,  // Fusion operator compilation product output position
    USE_OLD_CANN_STYLE,  // CANN 8.5.0 and below need to turn on this switch when performing operator fusion based on the AFIR backend.
    XLA_NPU_MAX_PRINT_COUNT,  // Control the number of input and output printing elements of the fusion operator and ACLNN operator
    XLA_NPU_DEVICE_ID,          // Specify the NPU device ID used
    ENABLE_SYNC_AFTER_EXECUTE,  // Whether to synchronize after operator execution (for custom kernel, independent of ENABLE_PRINT_IO_VALUES)
    ENABLE_PRINT_IO_VALUES,  // Whether to print input and output values (including D2H operations, which affects performance; the stream will be automatically synchronized, no additional setting of ENABLE_SYNC_AFTER_EXECUTE is required)
    XLA_NPU_ENABLE_DIM_SCATTER,  // Whether to enable dim-based Scatter (aclnnScatter/aclnnScatterAdd), turned off by default to avoid LinearIndex->ScatterElementsV2->561103
    // Due to the problem of config_loader and log circular dependency,
    // The XLA_NPU_LOG_LEVEL (controlling the display log level) environment variable is directly defined in the log class
  };

  NpuConfig(const NpuConfig&) = delete;
  NpuConfig& operator=(const NpuConfig&) = delete;

  static NpuConfig& GetInstance() {
    static NpuConfig instance;
    return instance;
  }

  void Init();

 private:
  NpuConfig() = default;
  ~NpuConfig() = default;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONFIG_H_
