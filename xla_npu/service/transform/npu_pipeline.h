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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_

#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {

// Function wrapper around the XLA NPU transform pipeline.
// This is the main entry point for all NPU optimization passes.
// Thread pool may be nullptr.
HloPassPipeline NpuPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& npu_device_info,
    SymbolicExprContext* symbolic_expr_context);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_PIPELINE_H_
