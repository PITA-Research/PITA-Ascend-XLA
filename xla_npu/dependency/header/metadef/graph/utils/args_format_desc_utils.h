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

/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#ifndef METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
#define METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H

#include <string>
#include <vector>

#include "graph/ge_error_codes.h"
#include "graph/op_desc.h"
#include "register/hidden_input_func_registry.h"

namespace ge {
enum class AddrType {
  INPUT = 0,
  OUTPUT,
  WORKSPACE,
  TILING,
  INPUT_DESC,
  OUTPUT_DESC,
  FFTS_ADDR,
  OVERFLOW_ADDR,
  TILING_FFTS,
  HIDDEN_INPUT,
  TILING_CONTEXT,
  OP_TYPE,
  PLACEHOLDER
};

enum class TilingContextSubType {
  TILING_CONTEXT = -1,
  TILING_DATA,
  TILING_KEY,
  BLOCK_DIM
};

struct ArgDesc {
  AddrType addr_type;
  int32_t ir_idx;
  bool folded;
  uint8_t reserved[4];
};

class ArgsFormatDescUtils {
 public:
  // i* -> ir_idx = -1, folded=false
  // For input and output, idx represents the idx defined by ir, -1 represents all inputs and all outputs. At this time, non-dynamic inputs and outputs are expanded by default. Dynamic output requires i1* to represent expansion.
  // For workspace -1 means the number is unknown, and folded is temporarily meaningless.
  // For ffts tail block non-tail block address, idx=0 means non-tail block, idx=1 means tail block
  // For other types, idx and fold are temporarily meaningless
  static void Append(std::vector<ArgDesc> &arg_descs, AddrType type,
                     int32_t ir_idx = -1, bool folded = false);

  static void AppendHiddenInput(std::vector<ArgDesc> &arg_descs,
                                HiddenInputType hidden_type);

  static void AppendTilingContext(
      std::vector<ArgDesc> &arg_descs,
      TilingContextSubType sub_type = TilingContextSubType::TILING_CONTEXT);

  static std::string ToString(const std::vector<ArgDesc> &arg_descs);

  // When a wildcard character like i* is used in a string, the returned argDesc will not be expanded according to the actual number.
  static graphStatus Parse(const std::string &str,
                           std::vector<ArgDesc> &arg_descs);

  static std::string Serialize(const std::vector<ArgDesc> &arg_descs);
};
}  // namespace ge

#endif  // METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
