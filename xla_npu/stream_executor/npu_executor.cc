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

#include "xla/xla_npu/stream_executor/npu_executor.h"

#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/mem.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/common/npu_operation.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/kernel/npu_kernel_config.h"
#include "xla/xla_npu/stream_executor/npu_auto_fuse_kernel.h"
#include "xla/xla_npu/stream_executor/npu_event.h"
#include "xla/xla_npu/stream_executor/npu_stream.h"
#include "xla/xla_npu/stream_executor/npu_stream_factory.h"

namespace stream_executor {

NpuStream* AsNpuStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<NpuStream*>(stream);
}

// Note that JAX uses a new thread (not the main thread) when calling the XLA interface to obtain the client. Therefore, when the main thread uses this device, it needs to be reset.
// device
absl::Status NpuExecutor::Init() {
  TF_RETURN_IF_ERROR(xla::NpuOperation::SetDevice(device_ordinal_));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> NpuExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  CHECK_COND_AND_PRINT_NPU_LOG(!spec.has_in_process_symbol(),
                               "In process symbol when load kernel is empty");
  auto npu_kernel_config_ptr = static_cast<xla::NpuKernelConfig*>(
      spec.in_process_symbol().value().symbol);
  switch (npu_kernel_config_ptr->npu_kernel_type) {
    case xla::NpuKernelConfig::NpuKernelType::FUSION:
      auto kernel = std::make_unique<stream_executor::NpuAutoFuseKernel>(
          npu_kernel_config_ptr->npu_artifact_info.dir,
          npu_kernel_config_ptr->npu_artifact_info.name, this);
      TF_RETURN_IF_ERROR(kernel->Load());
      return std::move(kernel);
  }

  return absl::InternalError(absl::StrFormat(
      "Kernel type %d does not support launch.",
      static_cast<int64_t>(npu_kernel_config_ptr->npu_kernel_type)));
}

bool NpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  tsl::port::MemoryInfo mem_info = tsl::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase NpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);

  // Ascend aclrtMalloc does not support zero-size allocations.
  if (size == 0) {
    return DeviceMemoryBase(nullptr, 0);
  }

  // Ensure that the current thread has executed set device
  xla::NpuOperation::SetDevice(device_ordinal_);

  void* base = nullptr;
  aclError ret =
      ::xla::dep::aclrtMalloc(&base, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Allocate %d byte on npu failed, error code: %d", size, ret);
    return DeviceMemoryBase(nullptr, size);
  }

  // If the address has been released before, it is removed from the set (the address is reused)
  {
    std::lock_guard<std::mutex> lock(freed_addresses_mutex_);
    freed_addresses_.erase(base);
  }

  // Clear allocated memory
  DeviceMemoryBase mem(base, size);
  absl::Status status = SynchronousMemZero(&mem, size);
  if (!status.ok()) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Failed to zero memory at %p, size %d: %s",
                         base, size, status.ToString().c_str());
    ::xla::dep::aclrtFree(base);
    return DeviceMemoryBase(nullptr, size);
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory allocated, addr %p, size %d", base,
                       size);
  return DeviceMemoryBase(base, size);
}

void NpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  // Ensure that the current thread has executed set device
  xla::NpuOperation::SetDevice(device_ordinal_);

  // Check whether the pointer is empty to avoid repeated release
  if (mem->opaque() == nullptr) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory free skipped, addr is nullptr");
    return;
  }

  // Check if it has been released
  {
    std::lock_guard<std::mutex> lock(freed_addresses_mutex_);
    if (freed_addresses_.find(mem->opaque()) != freed_addresses_.end()) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Npu memory free skipped, addr %p already freed",
                           mem->opaque());
      return;
    }
    // Record the released address
    freed_addresses_.insert(mem->opaque());
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Npu memory free, addr: %p, size: %d",
                       mem->opaque(), mem->size());
  ::xla::dep::aclrtFree(mem->opaque());
}

absl::Status NpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
  // Ensure that the current thread has executed set device
  xla::NpuOperation::SetDevice(device_ordinal_);

  aclError ret = ::xla::dep::aclrtMemset(location->opaque(), size, 0, size);
  if (ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "SynchronousMemZero failed, addr %p, size %d, error code: %d",
        location->opaque(), size, ret);
    return absl::InternalError(
        absl::StrFormat("aclrtMemset failed, error code: %d", ret));
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "SynchronousMemZero success, addr %p, size %d",
                       location->opaque(), size);
  return absl::OkStatus();
}

absl::Status NpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  return absl::UnimplementedError(
      "NpuExecutor::SynchronousMemcpy h2d is not supported.");
}

absl::Status NpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  return absl::UnimplementedError(
      "NpuExecutor::SynchronousMemcpy d2h is not supported.");
}

void NpuExecutor::DeallocateStream(Stream* stream) {}

absl::StatusOr<std::unique_ptr<Event>> NpuExecutor::CreateEvent() {
  return std::make_unique<NpuEvent>();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
NpuExecutor::CreateDeviceDescription(int device_ordinal) {
  DeviceDescription desc;

  desc.set_name("npu");
  desc.set_platform_version("Default Version");

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<Stream>> NpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  const NpuStreamFactory* factory = NpuStreamFactory::GetFactory();
  if (factory != nullptr) {
    TF_ASSIGN_OR_RETURN(auto stream, factory->CreateStream(this));
    return std::move(stream);
  }
  return absl::UnavailableError(absl::StrCat("Npu stream factory not init"));
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
NpuExecutor::CreateMemoryAllocator(MemoryType type) {
  if (type == MemoryType::kHost) {
    return std::make_unique<GenericMemoryAllocator>(
        [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          void* ptr = new char[size];
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [](void* location, uint64_t size) {
                delete[] static_cast<char*>(location);
              });
        });
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
NpuExecutor::CreateOrShareConstant(Stream* stream,
                                   absl::Span<const uint8_t> content) {
  std::shared_ptr<DeviceMemoryBase> shared_constant;
  auto new_constant = std::make_unique<DeviceMemoryBase>(
      Allocate(content.size(), /*memory_space=*/0));
  if (new_constant->opaque() == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %d bytes for new constant", content.size()));
  }

  TF_RETURN_IF_ERROR(
      stream->Memcpy(new_constant.get(), content.data(), content.size()));
  absl::Status status = stream->BlockHostUntilDone();
  if (!status.ok()) {
    Deallocate(new_constant.get());
    status.Update(absl::InternalError(absl::StrFormat(
        "Memcpy to device address %p failed", new_constant->opaque())));
    return status;
  }

  shared_constant = std::shared_ptr<DeviceMemoryBase>(
      new_constant.release(), [this](DeviceMemoryBase* p) {
        Deallocate(p);
        delete p;
      });

  return shared_constant;
}

}  // namespace stream_executor
