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

#include "xla/xla_npu/service/kernel/npu_custom_kernel.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"

namespace xla {

NpuCustomKernel::NpuCustomKernel(
    std::unique_ptr<NpuKernelConfig> npu_kernel_config_ptr, std::string name,
    se::BlockDim block_dims, size_t shared_memory_bytes)
    : npu_kernel_config_ptr_(std::move(npu_kernel_config_ptr)),
      name_(std::move(name)),
      // The first pointer parameter ensures life cycle validity through the unique_ptr pointer (synchronized with the NpuCustomKernel life cycle)
      // The execution of the auto fusion operator does not currently depend on the number of operator parameters represented by arity (the penultimate parameter of the CreateInProcessSymbolSpec interface) set here.
      kernel_spec_(std::move(
          stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
              static_cast<void*>(this->npu_kernel_config_ptr()), name, 0,
              nullptr))),
      block_dims_(block_dims),
      shared_memory_bytes_(shared_memory_bytes) {}

// move constructor
NpuCustomKernel::NpuCustomKernel(NpuCustomKernel&& other) noexcept
    : npu_kernel_config_ptr_(std::move(other.npu_kernel_config_ptr_)),
      name_(std::move(other.name_)),
      kernel_spec_(std::move(
          stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
              static_cast<void*>(this->npu_kernel_config_ptr()), name_, 0,
              nullptr))),
      block_dims_(std::move(other.block_dims_)),
      thread_dims_(std::move(other.thread_dims_)),
      cluster_dims_(std::move(other.cluster_dims_)),
      shared_memory_bytes_(other.shared_memory_bytes_) {
  // Reset the state of the source object
  other.block_dims_ = se::BlockDim(0);
  other.thread_dims_ = se::ThreadDim(0, 0, 0);
  other.cluster_dims_ = std::nullopt;
  other.shared_memory_bytes_ = 0;
}

// move assignment operator
NpuCustomKernel& NpuCustomKernel::operator=(NpuCustomKernel&& other) noexcept {
  if (this != &other) {
    npu_kernel_config_ptr_ = std::move(other.npu_kernel_config_ptr_);
    name_ = std::move(other.name_);
    // Re-create kernel_spec_, making sure its symbol pointer points to the correct npu_kernel_config_ptr_
    kernel_spec_ =
        std::move(stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
            static_cast<void*>(npu_kernel_config_ptr_.get()), name_, 0,
            nullptr));
    block_dims_ = std::move(other.block_dims_);
    thread_dims_ = std::move(other.thread_dims_);
    cluster_dims_ = std::move(other.cluster_dims_);
    shared_memory_bytes_ = other.shared_memory_bytes_;
    // Reset the state of the source object
    other.block_dims_ = se::BlockDim(0);
    other.thread_dims_ = se::ThreadDim(0, 0, 0);
    other.cluster_dims_ = std::nullopt;
    other.shared_memory_bytes_ = 0;
  }
  return *this;
}

std::string NpuCustomKernel::ToString() const {
  return absl::StrFormat("%s block_dim: %d shared_memory: %d bytes", name_,
                         block_dims_.x, shared_memory_bytes_);
}

}  // namespace xla
