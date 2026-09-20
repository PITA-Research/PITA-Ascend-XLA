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

#include "xla/xla_npu/service/npu_hlo_schedule.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// Determines the schedule of HLO instructions for a module run on NPU.
// This is a simplified version of GPU scheduler, adapted for NPU.
// Uses generic ScheduleModule function from xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h
absl::StatusOr<NpuScheduleMetadata> NpuScheduleModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& npu_device_info,
    SymbolicExprContext* symbolic_expr_context,
    const NpuAliasInfo* alias_info) {
  tsl::profiler::TraceMe traceme("NpuScheduleModule");

  // Module already has a schedule, do nothing.
  if (module->has_schedule()) {
    VLOG(1) << "Module already has a schedule, do nothing.";
    return NpuScheduleMetadata{0, 0};
  }

  // Create a size function that calculates buffer sizes.
  BufferValue::SizeFunction size_func =
      [pointer_size](const BufferValue& buffer) -> int64_t {
    const Shape& shape = buffer.shape();
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      return static_cast<int64_t>(0);
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };

  // Run the memory scheduler using the generic ScheduleModule function.
  int64_t peak_memory_bytes = 0;
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(
          module, alias_info, std::move(size_func),
          /*execution_threads=*/{HloInstruction::kMainExecutionThread},
          &peak_memory_bytes));

  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  return NpuScheduleMetadata{0, peak_memory_bytes};
}

}  // namespace xla
