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

#include "xla/xla_npu/service/transform/npu_hlo_printer_pass.h"

#include <cstdlib>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/common/npu_tool.h"

namespace xla {

absl::StatusOr<bool> NpuHloPrinterPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // A one-line marker is useful when a native compiler crash happens between
  // pass boundaries. Keep it opt-in: normal builds retain the existing
  // DEBUG-only full HLO printing behavior.
  const char* pipeline_markers = std::getenv("XLA_NPU_PIPELINE_MARKERS");
  if (pipeline_markers != nullptr && pipeline_markers[0] == '1' &&
      pipeline_markers[1] == '\0') {
    NPU_LOG_FORMAT_PRINT(INFO, "NPU pipeline pass completed: %s",
                         pass_name_.c_str());
  }

  // Print HLO module using NpuTool::PrintHloModule
  NpuTool::PrintHloModule(module, pass_name_);

  return false;  // This pass never changes the HLO
}

}  // namespace xla
