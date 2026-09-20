/* Copyright 2019 The OpenXLA Authors.
Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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

#include "xla/xla_npu/pjrt/air/npu/tracked_npu_device_buffer.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "raw_buffer.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
// Returns an AsyncValueRef<CpuEvent> that will be ready after all the async
// values in `events` are ready. If errors occurs, one of the errors will be
// propagated through the returned async value.
tsl::AsyncValueRef<NpuEvent> AfterAll(
    absl::Span<const tsl::AsyncValueRef<NpuEvent>> events) {
  if (events.empty()) {
    return tsl::MakeAvailableAsyncValueRef<NpuEvent>();
  }
  if (events.size() == 1) {
    return events.front();
  }

  tsl::CountDownAsyncValueRef<NpuEvent> after_all(events.size());
  for (auto& event : events) {
    event.AndThen([after_all](absl::Status status) mutable {
      after_all.CountDown(std::move(status));
    });
  }

  return std::move(after_all).AsRef();
}
//===----------------------------------------------------------------------===//
// Default CpuDeviceMemory::RawMemory allocator.
//===----------------------------------------------------------------------===//

class AlignedMemory final : public NpuDeviceMemory::RawMemory {
 public:
  AlignedMemory(void* base, size_t size_bytes)
      : base_(base), size_bytes_(size_bytes) {}

  ~AlignedMemory() final {
    NPU_LOG_FORMAT_PRINT(DEBUG, "AlignedMemory::~AlignedMemory size_bytes: %zu",
                         size_bytes_);
    dep::aclrtFree(base_);
  }

  void* base() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};

class AlignedAllocator final : public NpuDeviceMemory::Allocator {
 public:
  absl::StatusOr<std::unique_ptr<NpuDeviceMemory::RawMemory>> Allocate(
      size_t size_bytes, size_t alignment) const final {
    NPU_LOG_FORMAT_PRINT(DEBUG, "AlignedAllocator::Allocate size_bytes: %zu",
                         size_bytes);
    // Ascend aclrtMalloc does not support zero-size allocations.
    if (size_bytes == 0) {
      return std::make_unique<AlignedMemory>(nullptr, 0);
    }
    void* base = nullptr;
    aclError acl_malloc =
        dep::aclrtMalloc(&base, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_malloc == ACL_SUCCESS) {
      return std::make_unique<AlignedMemory>(base, size_bytes);
    }
    return ResourceExhausted("Out of memory allocating %d bytes.", size_bytes);
  }
};

NpuDeviceMemory::Allocator& NpuDeviceMemory::DefaultAllocator() {
  static absl::NoDestructor<AlignedAllocator> allocator;
  return *allocator;
}

std::unique_ptr<NpuDeviceMemory::Allocator>
NpuDeviceMemory::MakeDefaultAllocator() {
  return std::make_unique<AlignedAllocator>();
}

class NpuDeviceMemoryOwned final : public NpuDeviceMemory {
 public:
  explicit NpuDeviceMemoryOwned(std::unique_ptr<RawMemory> mem)
      : mem_(std::move(mem)) {}

  void* untyped_data() const final { return mem_->base(); }
  size_t size_bytes() const final { return mem_->size_bytes(); }

 private:
  std::unique_ptr<RawMemory> mem_;
};

class NpuDeviceMemoryForeign final : public NpuDeviceMemory {
 public:
  NpuDeviceMemoryForeign(void* base, size_t size,
                         absl::AnyInvocable<void() &&> on_delete_callback)
      : base_(base),
        size_bytes_(size),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~NpuDeviceMemoryForeign() final {
    if (on_delete_callback_) {
      NPU_LOG_FORMAT_PRINT(
          DEBUG, "need call delete function for %p, which size is %zu", base_,
          size_bytes_);
      std::move(on_delete_callback_)();
    }
  }

  void* untyped_data() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

class NpuDeviceMemoryConstant final : public NpuDeviceMemory {
 public:
  NpuDeviceMemoryConstant(void* base, size_t size)
      : base_(base), size_bytes_(size) {}

  void* untyped_data() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};
tsl::AsyncValueRef<NpuDeviceMemory> NpuDeviceMemory::CreateDelayedMemory() {
  return tsl::MakeUnconstructedAsyncValueRef<NpuDeviceMemoryOwned>();
}

tsl::AsyncValueRef<NpuDeviceMemory> NpuDeviceMemory::CreateForeignMemory(
    void* base, size_t size, absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeAvailableAsyncValueRef<NpuDeviceMemoryForeign>(
      base, size, std::move(on_delete_callback));
}

tsl::AsyncValueRef<NpuDeviceMemory> NpuDeviceMemory::CreateConstantMemory(
    void* base, size_t size) {
  return tsl::MakeAvailableAsyncValueRef<NpuDeviceMemoryConstant>(base, size);
}

// Allocates owning memory wrapped in an available `AsyncValueRef`.
absl::StatusOr<tsl::AsyncValueRef<NpuDeviceMemory>> NpuDeviceMemory::Allocate(
    size_t size_bytes) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "NpuDeviceMemory::Allocate size_bytes: %zu",
                       size_bytes);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<RawMemory> mem,
                      DefaultAllocator().Allocate(size_bytes, 0));
  return tsl::MakeAvailableAsyncValueRef<NpuDeviceMemoryOwned>(std::move(mem));
}

absl::Status NpuDeviceMemory::AllocateInto(
    size_t size_bytes, tsl::AsyncValuePtr<NpuDeviceMemory> delayed_memory,
    const Allocator& allocator) {
  auto owned_memory = delayed_memory.DynCast<NpuDeviceMemoryOwned>();
  if (!owned_memory) {
    return Internal("Delayed memory is not a CpuDeviceMemoryOwned");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<RawMemory> mem,
                      allocator.Allocate(size_bytes, 0));
  owned_memory.emplace(std::move(mem));
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// TrackedNpuDeviceBuffer.
//===----------------------------------------------------------------------===//

TrackedNpuDeviceBuffer::TrackedNpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
    absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events)
    : TrackedNpuDeviceBuffer(owns_buffers, std::move(buffer),
                             AfterAll(definition_events)) {}

TrackedNpuDeviceBuffer::TrackedNpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
    size_t buffer_size,
    absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events)
    : TrackedNpuDeviceBuffer(owns_buffers, std::move(buffer), buffer_size,
                             AfterAll(definition_events)) {}

TrackedNpuDeviceBuffer::TrackedNpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
    tsl::AsyncValueRef<NpuEvent> definition_event)
    : owns_buffers_(owns_buffers),
      buffer_(std::move(buffer)),
      definition_event_(std::move(definition_event)) {
  DCHECK(definition_event_);
  CHECK(buffer_.IsConcrete());
  buffer_size_ = buffer_->size_bytes();
}

TrackedNpuDeviceBuffer::TrackedNpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
    size_t buffer_size, tsl::AsyncValueRef<NpuEvent> definition_event)
    : owns_buffers_(owns_buffers),
      buffer_(std::move(buffer)),
      buffer_size_(buffer_size),
      definition_event_(std::move(definition_event)) {
  DCHECK(definition_event_);
}

TrackedNpuDeviceBuffer::~TrackedNpuDeviceBuffer() { ReleaseDeviceMemory(); }

size_t TrackedNpuDeviceBuffer::BufferSize() { return buffer_size_; }

void TrackedNpuDeviceBuffer::AddUsageEvents(
    absl::Span<tsl::AsyncValueRef<NpuEvent>> events) {
  // Periodically remove available usage events to prevent memory blowup.
  if (usage_events_.size() >= 1024) {
    int i = 0;
    while (i < usage_events_.size()) {
      auto& event = usage_events_[i];
      if (event.IsAvailable()) {
        using std::swap;
        swap(event, usage_events_.back());
        usage_events_.pop_back();
        continue;
      }
      ++i;
    }
  }
  for (auto& ev : events) {
    usage_events_.push_back(std::move(ev));
  }
}

absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4>
TrackedNpuDeviceBuffer::LockUseAndTransferUsageEvents() {
  return std::move(usage_events_);
}

void TrackedNpuDeviceBuffer::ReleaseDeviceMemory() {
  buffer_ = tsl::AsyncValueRef<NpuDeviceMemory>();
  definition_event_.reset();
  usage_events_.clear();
}

std::vector<tsl::RCReference<tsl::AsyncValue>>
TrackedNpuDeviceBuffer::GetAsyncValueDefinitionEvents() {
  std::vector<tsl::RCReference<tsl::AsyncValue>> result;
  result.push_back(definition_event_.CopyRCRef());
  return result;
}

tsl::RCReference<CommonPjRtRawBuffer> TrackedNpuDeviceBuffer::GetRawBuffer(
    PjRtMemorySpace* memory_space) {
  if (!buffer_) {
    return tsl::RCReference<CommonPjRtRawBuffer>();
  }
  return tsl::MakeRef<NpuRawBuffer>(memory_space, buffer_);
}

void TrackedNpuDeviceBuffer::AddUsageEvent(
    tsl::RCReference<PjRtDeviceEvent> event) {
  if (event) {
    auto cpu_event =
        tensorflow::down_cast<NpuTrackedDeviceEvent*>(event.get())->event();
    AddUsageEvents({&cpu_event, 1});
  }
}

void TrackedNpuDeviceBuffer::Delete(PjRtMemorySpace* memory_space) {
  std::unique_ptr<TrackedNpuDeviceBuffer> device_buffer(this);
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> usage_events =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tsl::AsyncValue*> event_avs;
  event_avs.reserve(usage_events.size() + 1);
  for (auto& event : usage_events) {
    event_avs.push_back(event.GetAsyncValue());
  }

  // We should also wait for the definition event.
  event_avs.push_back(device_buffer->definition_event().GetAsyncValue());

  RunWhenReady(event_avs, [device_buffer = std::move(device_buffer)]() mutable {
    device_buffer.reset();
  });
}

Future<> TrackedNpuDeviceBuffer::GetReadyFuture(PjRtMemorySpace* memory_space) {
  auto [promise, future] = Future<>::MakePromise();
  promise.Set();

  return future;
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
TrackedNpuDeviceBuffer::GetDefinitionEvent(PjRtMemorySpace* memory_space) {
  if (!definition_event_) {
    return absl::InternalError(
        "GetDefinitionEvent only supported on CPU for buffers with "
        "exactly 1 definition event.");
  }
  return tsl::MakeRef<NpuTrackedDeviceEvent>(definition_event_);
}

absl::Status TrackedNpuDeviceBuffer::BlockForOperationsToComplete(
    PjRtMemorySpace* memory_space) {
  // Block the host until all usage events have completed. We do not return
  // the error of a usage event because it does not matter if these usages
  // failed.
  for (const auto& av : usage_events_) {
    BlockUntilReady(av.GetAsyncValue());
  }

  // Fetch the error from the definition event (if an error is present).
  BlockUntilReady(definition_event_.GetAsyncValue());
  if (auto* error = definition_event_.GetErrorIfPresent()) {
    return absl::InternalError(
        absl::StrFormat("Error Execute: %s", error->message()));
  }
  return absl::OkStatus();
}

}  // namespace xla