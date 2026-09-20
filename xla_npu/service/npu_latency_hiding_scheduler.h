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

#ifndef XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_
#define XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_

#include <cstdint>
#include <optional>

#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"

namespace xla {

HloCostAnalysis::ShapeSizeFunction NpuShapeSizeBytesFunction(
    int64_t pointer_size, std::optional<int64_t> memory_space);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_LATENCY_HIDING_SCHEDULER_H_
