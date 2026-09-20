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

#ifndef XLA_XLA_NPU_DEPENDENCY_TOOL_FUNCTION_LOADER_H_
#define XLA_XLA_NPU_DEPENDENCY_TOOL_FUNCTION_LOADER_H_

#include <dlfcn.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xla {
namespace dep {

/**
  FunctionLoader is used to store function address in the process.
  */
class FunctionLoader {
 public:
  /**
      ctr
      */
  explicit FunctionLoader(const std::string& file_name,
                          const std::string& file_name_suffix = ".so",
                          int flags = RTLD_LAZY);
  /**
      dectr
      */
  ~FunctionLoader();
  /**
      set function name
      Only register the name of lib or func, which will be verified when getting. Unregistered ones are not allowed to be called.
      */
  void Set(const std::string& name);
  /**
      get function address by function name.
      */
  void* Get(const std::string& name);

 private:
  mutable std::mutex mu_;
  std::string fileName;
  int flags;
  void* handle = nullptr;
  mutable std::unordered_map<std::string, void*> registry;
};  // class FunctionLoader

/**
  this class is used to register
  */
class FunctionRegister {
 public:
  /**
      Singleton
      */
  static FunctionRegister* GetInstance();
  /**
      this API is used to store FunctionLoader class
      */
  void Register(const std::string& name, std::unique_ptr<FunctionLoader>& ptr);
  /**
      this API is used to associate library name and function name.
      */
  void Register(const std::string& name, const std::string& funcName);
  /**
      this API is used to get the function address by library and function name.
      */
  void* Get(const std::string& soName, const std::string& funcName);

 private:
  FunctionRegister() = default;
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, std::unique_ptr<FunctionLoader>>
      registry;
};  // class FunctionRegister

/**
  FunctionRegisterBuilder is the helper of FunctionRegister.
  */
class FunctionRegisterBuilder {
 public:
  /**
      ctr
      */
  FunctionRegisterBuilder(const std::string& soName,
                          std::unique_ptr<FunctionLoader>& libPtr) noexcept;
  /**
      ctr
      */
  FunctionRegisterBuilder(const std::string& soName,
                          const std::string& funcName) noexcept;
};  // class FunctionRegisterBuilder

#define REGISTER_LIBRARY(soName, ...)                                   \
  auto library_##soName = std::unique_ptr<::xla::dep::FunctionLoader>(  \
      new ::xla::dep::FunctionLoader(#soName, ##__VA_ARGS__));          \
  static ::xla::dep::FunctionRegisterBuilder register_library_##soName( \
      #soName, library_##soName);

#define REGISTER_FUNCTION(soName, funcName)                                \
  static ::xla::dep::FunctionRegisterBuilder register_function_##funcName( \
      #soName, #funcName);

// Error value acquisition template: returns -1 by default, can be specialized to adapt to different types
template <typename T>
constexpr T GetDefaultError() {
  // By default, -1 is returned for integer types, and the default constructed value is returned for other types.
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(-1);
  } else {
    return T{};
  }
}

// Specialization of void type
template <>
inline void GetDefaultError<void>() {
  // void type does not return any value
}

// Remove extra parentheses outside the parameter list in the parameter list
#define _REMOVE_PARENS(...) __VA_ARGS__
#define STRIP_PARENS(x) _REMOVE_PARENS x

// Declare the function to be registered
// Parameter description:
// soName: library name (such as libopapi)
// ret: return value type (such as aclnnStatus)
// name: function name (such as AclnnExpandGetWorkspaceSize)
// args_type: type + parameter name list (such as const aclTensor* self,...)
// args_name: only parameter name list (such as self, size,...)
#define DECLARE_ACL_FUNC(soName, ret, name, args_type, args_name)            \
  inline ret name args_type {                                                \
    using FuncPtr = ret(*) args_type;                                        \
    static FuncPtr cached_func = nullptr;                                    \
    if (!cached_func) {                                                      \
      cached_func = reinterpret_cast<FuncPtr>(                               \
          ::xla::dep::FunctionRegister::GetInstance()->Get(#soName, #name)); \
    }                                                                        \
    if (cached_func) {                                                       \
      return cached_func(STRIP_PARENS(args_name));                           \
    } else {                                                                 \
      return ::xla::dep::GetDefaultError<ret>();                             \
    }                                                                        \
  }

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_TOOL_FUNCTION_LOADER_H_
