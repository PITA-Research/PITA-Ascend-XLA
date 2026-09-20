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

#include "xla/xla_npu/service/npu_latency_hiding_scheduler.h"

#include <cstdint>
#include <optional>

#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

HloCostAnalysis::ShapeSizeFunction NpuShapeSizeBytesFunction(
    int64_t pointer_size, std::optional<int64_t> memory_space) {
  return [pointer_size, memory_space](const Shape& shape) -> int64_t {
    // Filter by memory space if specified
    if (memory_space.has_value() && shape.has_layout() &&
        shape.layout().memory_space() != memory_space.value()) {
      return 0;
    }
    int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
    if (shape.IsTuple() || shape.is_static()) {
      return size;
    }
    // Each dynamic dimension size is represented as a S32.
    int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
    return size + metadata_size;
  };
}

}  // namespace xla
