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

#ifndef AIR_UTILS_UTILS_H_
#define AIR_UTILS_UTILS_H_

#include <cstdint>
#include <iostream>
#include <sstream>

namespace mlir::air {

using int64 = std::int64_t;
using uint64 = std::uint64_t;

static void log(const std::string& str) { std::cout << str << std::endl; }

}  // namespace mlir::air

#endif  // AIR_UTILS_UTILS_H_
