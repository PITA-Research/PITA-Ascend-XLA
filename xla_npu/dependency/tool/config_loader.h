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

#ifndef XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_
#define XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_

#include <string>
#include <unordered_map>
#include <variant>

namespace xla {
namespace dep {

class ConfigLoader {
 public:
  enum class ValueType { STRING, BOOL, INT };
  using DefaultValueType = std::variant<std::string, bool, int>;
  struct Desc {
    std::string str_desc;
    DefaultValueType default_value;
    ValueType type;
  };

  using ConfigDescMap = std::unordered_map<int, Desc>;

  ConfigLoader(const ConfigLoader&) = delete;
  ConfigLoader& operator=(const ConfigLoader&) = delete;

  static ConfigLoader& GetInstance() {
    static ConfigLoader instance;
    return instance;
  }

  void Register(const ConfigDescMap& desc_map);
  void Register(int key, DefaultValueType value);

  Desc GetDesc(int key) const;
  DefaultValueType GetConfigValue(int key) const;
  template <typename T>
  T GetTypedValue(int key) const;

 private:
  ConfigLoader() = default;
  ~ConfigLoader() = default;

  std::unordered_map<int, DefaultValueType> value_map_;
};

template <typename T>
T ConfigLoader::GetTypedValue(int key) const {
  DefaultValueType val = GetConfigValue(key);
  if (auto* typed_val = std::get_if<T>(&val)) {
    return *typed_val;
  }
  const Desc& desc = GetDesc(key);
  if (auto* default_val = std::get_if<T>(&desc.default_value)) {
    return *default_val;
  }
  throw std::bad_variant_access();
}

}  // namespace dep
}  // namespace xla

#define NPU_CONFIG_GET_STRING(key)                                    \
  ::xla::dep::ConfigLoader::GetInstance().GetTypedValue<std::string>( \
      static_cast<int>(key))

#define NPU_CONFIG_GET_BOOL(key)                               \
  ::xla::dep::ConfigLoader::GetInstance().GetTypedValue<bool>( \
      static_cast<int>(key))

#define NPU_CONFIG_GET_INT(key)                               \
  ::xla::dep::ConfigLoader::GetInstance().GetTypedValue<int>( \
      static_cast<int>(key))

#endif  // XLA_XLA_NPU_DEPENDENCY_TOOL_CONFIG_LOADER_H_
