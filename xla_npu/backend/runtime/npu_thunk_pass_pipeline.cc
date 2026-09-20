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

#include "xla/xla_npu/backend/runtime/npu_thunk_pass_pipeline.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"

namespace xla {

absl::StatusOr<bool> NpuThunkPassPipeline::Run(
    NpuSequentialThunk* root_thunk, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    NpuThunkPassBufferAllocator& allocator) {
  bool changed = false;
  for (const auto& pass : passes_) {
    VLOG(1) << "Running ThunkPass: " << pass->name();
    TF_ASSIGN_OR_RETURN(bool pass_changed,
                        pass->Run(root_thunk, debug_options, hlo_module,
                                  device_info, allocator));
    changed |= pass_changed;
  }
  return changed;
}

}  // namespace xla
