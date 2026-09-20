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

#include "xla/xla_npu/dependency/tool/config_loader.h"

#include "xla/tsl/util/env_var.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace dep {

void ConfigLoader::Register(const ConfigDescMap& desc_map) {
  NPU_LOG_FORMAT_PRINT(INFO, "Register config for ConfigLoader.");

  for (const auto& [key, desc] : desc_map) {
    absl::Status status = absl::OkStatus();

    switch (desc.type) {
      case ValueType::STRING: {
        std::string str_val;
        status = tsl::ReadStringFromEnvVar(
            desc.str_desc, std::get<std::string>(desc.default_value), &str_val);
        if (status.ok()) {
          value_map_[key] = str_val;
          NPU_LOG_FORMAT_PRINT(
              INFO, "Set config key - '%s', value - '%s', type - '%d' success.",
              desc.str_desc, str_val, static_cast<int>(desc.type));
        }
        break;
      }
      case ValueType::BOOL: {
        bool bool_val;
        status = tsl::ReadBoolFromEnvVar(
            desc.str_desc, std::get<bool>(desc.default_value), &bool_val);
        if (status.ok()) {
          value_map_[key] = bool_val;
          NPU_LOG_FORMAT_PRINT(
              INFO, "Set config key - '%s', value - '%v', type - '%d' success.",
              desc.str_desc, bool_val, static_cast<int>(desc.type));
        }
        break;
      }
      case ValueType::INT: {
        int64_t int_val;
        status = tsl::ReadInt64FromEnvVar(
            desc.str_desc, std::get<int>(desc.default_value), &int_val);
        if (status.ok()) {
          value_map_[key] = static_cast<int>(int_val);
          NPU_LOG_FORMAT_PRINT(
              INFO, "Set config key - '%s', value - '%d', type - '%d' success.",
              desc.str_desc, int_val, static_cast<int>(desc.type));
        }
        break;
      }
      default:
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "Set config key - '%s' failed, config type - '%d' not support.",
            desc.str_desc, static_cast<int>(desc.type));
    }

    if (!status.ok()) {
      NPU_LOG_FORMAT_PRINT(
          WARN, "Set config key - '%s', type - '%d' failed, error status: %s.",
          desc.str_desc, static_cast<int>(desc.type),
          status.ToString().c_str());
    }
  }
}

void ConfigLoader::Register(int key, DefaultValueType value) {
  value_map_[key] = value;
}

ConfigLoader::Desc ConfigLoader::GetDesc(int key) const {
  NPU_LOG_FORMAT_PRINT(ERROR, "Config key - '%d' not found in desc map.", key);
  throw std::out_of_range("Config key not found");
}

ConfigLoader::DefaultValueType ConfigLoader::GetConfigValue(int key) const {
  auto it = value_map_.find(key);
  if (it != value_map_.end()) {
    return it->second;
  }
  NPU_LOG_FORMAT_PRINT(WARN, "Config key - '%d' not found in value map.", key);
  throw std::out_of_range("Config key not found");
}

template std::string ConfigLoader::GetTypedValue<std::string>(int key) const;
template bool ConfigLoader::GetTypedValue<bool>(int key) const;
template int ConfigLoader::GetTypedValue<int>(int key) const;

}  // namespace dep
}  // namespace xla
