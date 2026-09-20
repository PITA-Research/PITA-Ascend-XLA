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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CUSTOM_KERNEL_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CUSTOM_KERNEL_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

namespace xla {

// NpuCustomKernelThunk loads and executes kernels defined by a custom kernel
// (which in practice means hand written CUDA C++ kernel), instead of a kernel
// compiled by XLA and loaded from an executable source.
class NpuCustomKernelThunk : public NpuThunk {
 public:
  NpuCustomKernelThunk(NpuThunk::ThunkInfo thunk_info,
                       NpuCustomKernel custom_kernel,
                       const emitters::KernelArguments& kernel_arguments);

  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const NpuCustomKernel& custom_kernel() const { return custom_kernel_; }

  const std::vector<BufferAllocation::Slice>& arguments() const {
    return args_;
  }

  absl::string_view custom_kernel_name() const { return custom_kernel_.name(); }

  const std::vector<bool>& written() const { return written_; }

  BufferUses buffer_uses() const override;

 private:
  // Print kernel startup information
  void PrintKernelLaunchInfo(int device_ordinal, se::Kernel* kernel) const;

  // Print kernel spec details
  void PrintKernelSpec() const;

  // Print parameter shape information
  void PrintParamShapes() const;

  // Print parameter details (before execution)
  void PrintParamInfoBeforeExecute(
      int device_ordinal, const NpuBufferAllocations* buffer_allocations,
      absl::InlinedVector<se::DeviceMemoryBase, 4>& buffer_args);

  // Print parameter details (after execution)
  void PrintParamInfoAfterExecute(
      int device_ordinal, const NpuBufferAllocations* buffer_allocations);

  // Buffer slices passed to the kernel as arguments.
  std::vector<BufferAllocation::Slice> args_;

  // args_[i] is written iff (written_[i] == true).
  std::vector<bool> written_;

  NpuCustomKernel custom_kernel_;

  // Loaded kernels for each `StreamExecutor`.
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CUSTOM_KERNEL_THUNK_H_
