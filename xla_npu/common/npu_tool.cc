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

#include "xla/xla_npu/common/npu_tool.h"

#include <string>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace xla {

// Generate unique folder names
std::string NpuTool::generate_unique_folder_name_absl() {
  // 1. Generate second-level timestamps (Abseil cross-platform time tool)
  absl::Time now = absl::Now();
  // Format: YYYYMMDDHHMMSSmmm (year-month-day-hour-minute-second)
  std::string timestamp =
      absl::FormatTime("%Y%m%d%H%M%S", now, absl::LocalTimeZone());
  // Remove the decimal point after milliseconds (%3f will output 123.456, take the first 3 digits)
  timestamp = timestamp.substr(0, timestamp.find('.'));

  // 2. Generate a 6-digit random number (Abseil correct interface: pass it directly to bitgen)
  static absl::BitGen bitgen;  // Global random number generator, initialized only once
  // Call absl::Uniform directly, passing in bitgen + range
  int random_num = absl::Uniform<int>(bitgen, 100000, 999999);  // 6 digit random number

  // 3. Splicing (timestamp_random number), no illegal characters
  return absl::StrCat(timestamp, "_", random_num);
}

// Automatically select the print type based on the Shape's element type
void NpuTool::PrintNpuDeviceMemoryValueByShape(const std::string& title,
                                               const Shape& xla_shape,
                                               const void* device_addr,
                                               int64_t max_print_count) {
  int64_t size = GetXlaShapeSize(xla_shape);
  PrimitiveType element_type = xla_shape.element_type();
  const char* type_name = nullptr;

  switch (element_type) {
    case PRED:
      type_name = "PRED/ACL_BOOL";
      PrintNpuDeviceMemoryValue(title, size, device_addr, max_print_count,
                                true);
      break;
    case F32:
      type_name = "F32/ACL_FLOAT";
      PrintNpuDeviceMemoryValue<float>(title, size, device_addr,
                                       max_print_count);
      break;
    case F64:
      type_name = "F64/ACL_DOUBLE";
      PrintNpuDeviceMemoryValue<double>(title, size, device_addr,
                                        max_print_count);
      break;
    case S32:
      type_name = "S32/ACL_INT32";
      PrintNpuDeviceMemoryValue<int32_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case S64:
      type_name = "S64/ACL_INT64";
      PrintNpuDeviceMemoryValue<int64_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case U32:
      type_name = "U32/ACL_UINT32";
      PrintNpuDeviceMemoryValue<uint32_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case U64:
      type_name = "U64/ACL_UINT64";
      PrintNpuDeviceMemoryValue<uint64_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case F16:
      type_name = "F16/ACL_FLOAT16";
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case BF16:
      type_name = "BF16/ACL_BF16";
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case S16:
      type_name = "S16/ACL_INT16";
      PrintNpuDeviceMemoryValue<int16_t>(title, size, device_addr,
                                         max_print_count);
      break;
    case U16:
      type_name = "U16/ACL_UINT16";
      PrintNpuDeviceMemoryValue<uint16_t>(title, size, device_addr,
                                          max_print_count);
      break;
    case S8:
      type_name = "S8/ACL_INT8";
      PrintNpuDeviceMemoryValue<int8_t>(title, size, device_addr,
                                        max_print_count);
      break;
    case U8:
      type_name = "U8/ACL_UINT8";
      PrintNpuDeviceMemoryValue<uint8_t>(title, size, device_addr,
                                         max_print_count);
      break;
    default:
      NPU_LOG_FORMAT_PRINT(DEBUG, "%s (unsupported type %d, using float)",
                           title.c_str(), element_type);
      PrintNpuDeviceMemoryValue<float>(title, size, device_addr,
                                       max_print_count);
      return;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "%s (%s)", title.c_str(), type_name);
}

}  // namespace xla
