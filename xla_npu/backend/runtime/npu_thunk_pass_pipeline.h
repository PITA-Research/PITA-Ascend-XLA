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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_PASS_PIPELINE_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_PASS_PIPELINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"

namespace xla {

class NpuThunkPassBufferAllocator {
 public:
  virtual ~NpuThunkPassBufferAllocator() = default;

  // Creates a new allocation of the provided size.
  // The ownership of the returned pointer stays with the
  // NpuThunkPassBufferAllocator object.
  virtual absl::StatusOr<BufferAllocation * absl_nonnull> NewEmptyAllocation(
      int64_t size) = 0;
};

class NpuThunkPassInterface {
 public:
  virtual ~NpuThunkPassInterface() = default;

  virtual absl::StatusOr<bool> Run(NpuSequentialThunk* root_thunk,
                                   const DebugOptions& debug_options,
                                   const HloModule* absl_nullable hlo_module,
                                   const se::DeviceDescription& device_info,
                                   NpuThunkPassBufferAllocator& allocator) = 0;

  virtual absl::string_view name() const = 0;
};

class NpuThunkPassPipeline : public NpuThunkPassInterface {
 public:
  explicit NpuThunkPassPipeline(absl::string_view name) : name_(name) {}

  void AddPass(std::unique_ptr<NpuThunkPassInterface> pass) {
    passes_.push_back(std::move(pass));
  }

  absl::string_view name() const override { return name_; }

  // Runs all optimization passes on the given thunk sequence.
  // Returns true if any pass changed the thunk tree.
  absl::StatusOr<bool> Run(NpuSequentialThunk* root_thunk,
                           const DebugOptions& debug_options,
                           const HloModule* absl_nullable hlo_module,
                           const se::DeviceDescription& device_info,
                           NpuThunkPassBufferAllocator& allocator) override;

 private:
  std::string name_;
  std::vector<std::unique_ptr<NpuThunkPassInterface>> passes_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_PASS_PIPELINE_H_
