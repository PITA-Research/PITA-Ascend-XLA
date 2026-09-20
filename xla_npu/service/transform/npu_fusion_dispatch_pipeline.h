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

#ifndef XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_
#define XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {

// NPU fusion distribution pipeline
// This PASS is responsible for coordinating the execution sequence of NPU fusion optimization. It does the following:
// 1. Run GEMM-specific fusion optimization first
// 2. Then run the custom kernel fusion rewrite
// 3. Then convert the appropriate fusion to a block-level fusion
// 4. Finally perform nested GEMM fusion transformation
// Unlike the GPU's FusionDispatchPipeline, the NPU version:
// - Does not contain Triton specific optimizations
// - Focus on the integration model of CANN/ACLNN backend
// - Contains NPU-specific memory layout optimizations
class NpuFusionDispatchPipeline : public HloModulePass {
 public:
  NpuFusionDispatchPipeline(
      const se::DeviceDescription& device_description,
      SymbolicExprContext* symbolic_expr_context,
      const DebugOptions& debug_options,
      HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
      tsl::thread::ThreadPool* thread_pool);

  ~NpuFusionDispatchPipeline() override = default;

  absl::string_view name() const override {
    return "npu-fusion-dispatch-pipeline";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription device_description_;
  SymbolicExprContext* symbolic_expr_context_;
  const DebugOptions debug_options_;
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function_;
  tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_TRANSFORM_NPU_FUSION_DISPATCH_PIPELINE_H_
