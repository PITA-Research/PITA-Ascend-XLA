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

#include "xla/xla_npu/common/npu_config.h"

namespace xla {

void NpuConfig::Init() {
  dep::ConfigLoader::ConfigDescMap desc_map = {
      {static_cast<int>(Key::NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR),
       {"NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR",
        dep::ConfigLoader::DefaultValueType(std::string("")),
        dep::ConfigLoader::ValueType::STRING}},
      {static_cast<int>(Key::USE_OLD_CANN_STYLE),
       {"USE_OLD_CANN_STYLE", dep::ConfigLoader::DefaultValueType(false),
        dep::ConfigLoader::ValueType::BOOL}},
      {static_cast<int>(Key::XLA_NPU_MAX_PRINT_COUNT),
       {"XLA_NPU_MAX_PRINT_COUNT", dep::ConfigLoader::DefaultValueType(8),
        dep::ConfigLoader::ValueType::INT}},
      {static_cast<int>(Key::XLA_NPU_DEVICE_ID),
       {"XLA_NPU_DEVICE_ID", dep::ConfigLoader::DefaultValueType(0),
        dep::ConfigLoader::ValueType::INT}},
      {static_cast<int>(Key::ENABLE_SYNC_AFTER_EXECUTE),
       {"ENABLE_SYNC_AFTER_EXECUTE", dep::ConfigLoader::DefaultValueType(false),
        dep::ConfigLoader::ValueType::BOOL}},
      {static_cast<int>(Key::ENABLE_PRINT_IO_VALUES),
       {"ENABLE_PRINT_IO_VALUES", dep::ConfigLoader::DefaultValueType(false),
        dep::ConfigLoader::ValueType::BOOL}},
      {static_cast<int>(Key::XLA_NPU_ENABLE_DIM_SCATTER),
       {"XLA_NPU_ENABLE_DIM_SCATTER", dep::ConfigLoader::DefaultValueType(false),
        dep::ConfigLoader::ValueType::BOOL}},
  };
  dep::ConfigLoader::GetInstance().Register(desc_map);
}

}  // namespace xla
