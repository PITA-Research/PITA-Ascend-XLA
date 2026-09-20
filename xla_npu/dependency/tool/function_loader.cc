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

#include "xla/xla_npu/dependency/tool/function_loader.h"

#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
namespace dep {

FunctionLoader::FunctionLoader(const std::string& file_name,
                               const std::string& file_name_suffix, int flags) {
  this->fileName = file_name + file_name_suffix;
  this->flags = flags;
}

FunctionLoader::~FunctionLoader() {
  if (this->handle != nullptr) {
    dlclose(this->handle);
  }
}

void FunctionLoader::Set(const std::string& name) {
  this->registry[name] = nullptr;
}

void* FunctionLoader::Get(const std::string& name) {
  if (this->handle == nullptr) {
    auto handle = dlopen(this->fileName.c_str(), this->flags);
    if (handle == nullptr) {
      const std::string& err_msg = absl::StrFormat(
          "Open lib '%s' failed, reason: %s", this->fileName, dlerror());
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      throw std::runtime_error(err_msg);
    }
    this->handle = handle;
  }

  auto itr = registry.find(name);
  if (itr == registry.end()) {
    const std::string& err_msg =
        absl::StrFormat("Lib or func '%s' is not registered.", name);
    NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
    throw std::runtime_error(err_msg);
  }

  if (itr->second != nullptr) {
    return itr->second;
  }

  auto func = dlsym(this->handle, name.c_str());
  if (func == nullptr) {
    const std::string& err_msg =
        absl::StrFormat("Can't find func %s in %s.", name, this->fileName);
    NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
    throw std::runtime_error(err_msg);
  }

  this->registry[name] = func;
  return func;
}

FunctionRegister* FunctionRegister::GetInstance() {
  static FunctionRegister instance;
  return &instance;
}

void FunctionRegister::Register(const std::string& soName,
                                std::unique_ptr<FunctionLoader>& libPtr) {
  std::lock_guard<std::mutex> lock(mu_);
  registry.emplace(soName, std::move(libPtr));
  NPU_LOG_FORMAT_PRINT(DEBUG, "Register lib name '%s'", soName);
}

void FunctionRegister::Register(const std::string& soName,
                                const std::string& funcName) {
  auto itr = registry.find(soName);
  if (itr == registry.end()) {
    throw std::runtime_error(soName + " library should register first.");
  }
  itr->second->Set(funcName);
  NPU_LOG_FORMAT_PRINT(DEBUG, "Register lib function '%s' in lib name '%s'",
                       funcName, soName);
}

void* FunctionRegister::Get(const std::string& soName,
                            const std::string& funcName) {
  auto itr = registry.find(soName);
  if (itr != registry.end()) {
    return itr->second->Get(funcName);
  }

  throw std::runtime_error("Can't get func " + funcName + " from " + soName);
  return nullptr;
}

FunctionRegisterBuilder::FunctionRegisterBuilder(
    const std::string& name, std::unique_ptr<FunctionLoader>& ptr) noexcept {
  FunctionRegister::GetInstance()->Register(name, ptr);
}

FunctionRegisterBuilder::FunctionRegisterBuilder(
    const std::string& soName, const std::string& funcName) noexcept {
  FunctionRegister::GetInstance()->Register(soName, funcName);
}

}  // namespace dep
}  // namespace xla
