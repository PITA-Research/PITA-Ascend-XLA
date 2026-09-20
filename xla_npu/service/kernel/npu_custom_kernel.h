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

#ifndef XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
#define XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"

// WARNING: This header (and a build target) should have minimal dependencies as
// it's included into all device kernel implementations, and we want to minimize
// the number of (very expensive!) recompilations.

namespace xla {
namespace se = ::stream_executor;  // NOLINT

// This class is used to generate relevant parameters for executing NPU operators.
// Custom kernel is a mechanism for plugging pre-compiled device kernels into
// XLA GPU runtime. Custom kernel defines how to load the kernel on an executor
// and what are the grid size requirements for running it.
// We use this API to hide kernel implementation details from XLA (e.g. we can
// export CUTLASS gemm kernels as custom kernels to XLA), so that XLA can only
// implement generic interfaces, e.g. command buffer implementation for CUDA
// can automatically add a kernel node to a graph for arbitrary custom kernel.
class NpuCustomKernel {
 public:
  NpuCustomKernel(std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr,
                  std::string name, se::BlockDim block_dims,
                  size_t shared_memory_bytes);

  // Explicitly disable copy operations because the pointer in kernel_spec_ points to npu_kernel_config_
  NpuCustomKernel(const NpuCustomKernel&) = delete;
  NpuCustomKernel& operator=(const NpuCustomKernel&) = delete;

  // Customize the move operation to ensure that the pointer to npu_kernel_config_ in kernel_spec_ remains correct after the move
  NpuCustomKernel(NpuCustomKernel&&) noexcept;
  NpuCustomKernel& operator=(NpuCustomKernel&&) noexcept;

  absl::string_view name() const { return name_; }

  const se::KernelLoaderSpec& kernel_spec() const { return kernel_spec_; }

  se::BlockDim block_dims() const {
    return block_dims_;
  }  // NPU only uses the first dimension, which is the x-axis

  se::ThreadDim thread_dims() const {
    return thread_dims_;
  }  // NPU is not used, it is only used as a placeholder to adapt to the XLA framework

  std::optional<se::ClusterDim> cluster_dims() const {
    return cluster_dims_;
  }  // NPU is not used, it is only used as a placeholder to adapt to the XLA framework

  size_t shared_memory_bytes() const { return shared_memory_bytes_; }

  std::string ToString() const;

  NpuKernelConfig* npu_kernel_config_ptr() const {
    return npu_kernel_config_ptr_.get();
  }

 private:
  // Note: npu_kernel_config_ptr_ must be declared before kernel_spec_,
  // Because the initialization of kernel_spec_ depends on npu_kernel_config_ptr_
  std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr_;
  std::string name_;
  se::KernelLoaderSpec kernel_spec_;
  se::BlockDim block_dims_;
  se::ThreadDim thread_dims_;
  std::optional<se::ClusterDim> cluster_dims_ = std::nullopt;
  size_t shared_memory_bytes_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_KERNEL_NPU_CUSTOM_KERNEL_H_
