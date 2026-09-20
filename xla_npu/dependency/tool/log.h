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

#ifndef XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_
#define XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_

#include "absl/strings/str_format.h"

#define NPU_LOG_PRINT(level, log_str)                  \
  ::xla::dep::Log::Print(__FILE__, __LINE__, __func__, \
                         ::xla::dep::Log::Level::level, log_str)

#define NPU_LOG_FORMAT_PRINT(level, format, ...)       \
  ::xla::dep::Log::Print(__FILE__, __LINE__, __func__, \
                         ::xla::dep::Log::Level::level, format, ##__VA_ARGS__)

// Encapsulates a log printing macro that returns absl::Status status
#define CHECK_COND_AND_PRINT_NPU_LOG(err_cond_expr, err_format, ...)          \
  do {                                                                        \
    if ((err_cond_expr)) {                                                    \
      const std::string err_msg = absl::StrFormat(err_format, ##__VA_ARGS__); \
      NPU_LOG_PRINT(ERROR, err_msg);                                          \
      return absl::InternalError(err_msg);                                    \
    }                                                                         \
  } while (0)

// Tips:
// You can use the TF_RETURN_IF_ERROR macro in "XLA/tsl/platform/errors.h" to quickly judge and return the absl::Status results.

namespace xla {
namespace dep {

// NPU log printing control class
class Log {
 public:
  enum class Level { DEBUG, INFO, WARN, ERROR };

  static void Print(const char* file_path, int line, const char* func,
                    Level log_level, const std::string& log_str);

  template <typename... Args>
  static void Print(const char* file_path, int line, const char* func,
                    Level log_level, absl::FormatSpec<Args...> format,
                    Args&&... args) {
    const std::string& log_str =
        absl::StrFormat(format, std::forward<Args>(args)...);
    Print(file_path, line, func, log_level, log_str);
  }
};

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_TOOL_LOG_H_