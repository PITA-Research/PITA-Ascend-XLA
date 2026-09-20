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

#ifndef NPU_TRACKED_NPU_DEVICE_BUFFER_H_
#define NPU_TRACKED_NPU_DEVICE_BUFFER_H_
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "npu_event.h"
namespace xla {

class NpuDeviceMemory {
 public:
  class Allocator;

  virtual ~NpuDeviceMemory() = default;

  NpuDeviceMemory(const NpuDeviceMemory&) = delete;
  NpuDeviceMemory& operator=(const NpuDeviceMemory&) = delete;

  virtual void* untyped_data() const = 0;
  virtual size_t size_bytes() const = 0;

  // Creates an unavailable AsyncValueRef placeholder for a delayed
  // memory allocation (see `AllocateInto` below).
  static tsl::AsyncValueRef<NpuDeviceMemory> CreateDelayedMemory();

  // Creates an available AsyncValueRef to a npuDeviceMemory that wraps foreign
  // memory. Will call on_delete_callback on the last-ref.
  static tsl::AsyncValueRef<NpuDeviceMemory> CreateForeignMemory(
      void* base, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback);

  // Creates an available AsyncValueRef to a npuDeviceMemory that wraps a
  // constant memory with infinite lifetime. No action will be taken on decref.
  static tsl::AsyncValueRef<NpuDeviceMemory> CreateConstantMemory(void* base,
                                                                  size_t size);

  // Allocates owning memory wrapped in an available `AsyncValueRef`.
  static absl::StatusOr<tsl::AsyncValueRef<NpuDeviceMemory>> Allocate(
      size_t size_bytes);

  // Allocates owning memory into the previously created delayed memory
  // placeholder (see `CreateDelayedMemory` above).
  static absl::Status AllocateInto(
      size_t size_bytes, tsl::AsyncValuePtr<NpuDeviceMemory> delayed_memory,
      const Allocator& allocator = DefaultAllocator());

  //===--------------------------------------------------------------------===//
  // Custom raw memory allocation APIs.
  //===--------------------------------------------------------------------===//

  // Default allocator uses aligned allocation and free APIs from tsl.
  static Allocator& DefaultAllocator();

  // Returns a new instance of the default allocator.
  static std::unique_ptr<Allocator> MakeDefaultAllocator();

  // A raw memory allocation that can be used to construct a npuDeviceMemory.
  class RawMemory {
   public:
    virtual ~RawMemory() = default;
    virtual void* base() const = 0;
    virtual size_t size_bytes() const = 0;
  };

  // A raw memory allocator that allocates memory buffers for constructing
  // npuDeviceMemory.
  // This is a virtual interface to allow for different memory allocation
  // strategies, e.g. aligned_alloc, pre-mapped DMA buffers, etc. For example,
  // when XLA:npu is running as a part of host-offloading computation, we want
  // all buffers used by XLA:npu to be pre-mapped with the accelerator device,
  // so that we can issue zero-copy DMA transfers operations if needed.
  class Allocator {
   public:
    virtual ~Allocator() = default;
    virtual absl::StatusOr<std::unique_ptr<RawMemory>> Allocate(
        size_t size_bytes, size_t alignment) const = 0;
  };

 protected:
  NpuDeviceMemory() = default;
};

// A class that represents a npu device buffer: it can be a single memory region
// or multiple memory regions for a tuple buffers. It also tracks the definition
// and usage of the memory to allow for synchronized usage and deletion of npu
// memory. This class is thread-compatible.
class TrackedNpuDeviceBuffer : public AbstractTrackedDeviceBuffer {
 public:
  // For non-tuple, takes a single buffer.
  // For tuple, takes the leaf buffers. Tuple index table created internally.
  // Nested tuple is not supported.

  // Constructor for allocated npu memory, i.e., `buffer` should have concrete
  // states. Definition event is after the list of `definition_events`.
  TrackedNpuDeviceBuffer(
      bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
      absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events);

  // Variant with single definition event.
  TrackedNpuDeviceBuffer(bool owns_buffers,
                         tsl::AsyncValueRef<NpuDeviceMemory> buffer,
                         tsl::AsyncValueRef<NpuEvent> definition_event);

  // Constructor for unallocated npu memory, i.e., `buffer` will have
  // unconstructed states, and we also need to provide `buffer_size` which will
  // be the size of the `buffer` after allocation. Definition event is after the
  // list of `definition_events`. Callers need to ensure npu memory is allocated
  // before the definition event is ready.
  TrackedNpuDeviceBuffer(
      bool owns_buffers, tsl::AsyncValueRef<NpuDeviceMemory> buffer,
      size_t buffer_size,
      absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> definition_events);

  // Variant with single definition event.
  TrackedNpuDeviceBuffer(bool owns_buffers,
                         tsl::AsyncValueRef<NpuDeviceMemory> buffer,
                         size_t buffer_size,
                         tsl::AsyncValueRef<NpuEvent> definition_event);

  TrackedNpuDeviceBuffer(TrackedNpuDeviceBuffer&&) noexcept = default;
  TrackedNpuDeviceBuffer& operator=(TrackedNpuDeviceBuffer&&) noexcept =
      default;

  ~TrackedNpuDeviceBuffer() override;

  const tsl::AsyncValueRef<NpuDeviceMemory>& buffer() { return buffer_; }

  size_t BufferSize();

  const tsl::AsyncValueRef<NpuEvent>& definition_event() const {
    return definition_event_;
  }

  absl::Span<const tsl::AsyncValueRef<NpuEvent>> UsageEvents() const {
    return usage_events_;
  }

  void AddUsageEvents(absl::Span<tsl::AsyncValueRef<NpuEvent>> events);

  // Return the usage events for the buffers. After
  // LockUseAndTransferUsageEvents is called, it is illegal to AddUsageEvent.
  absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4>
  LockUseAndTransferUsageEvents();

  bool owns_buffers() const { return owns_buffers_; }

  std::vector<tsl::RCReference<tsl::AsyncValue>> GetAsyncValueDefinitionEvents()
      override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> GetDefinitionEvent(
      PjRtMemorySpace* memory_space) override;

  tsl::RCReference<CommonPjRtRawBuffer> GetRawBuffer(
      PjRtMemorySpace* memory_space) override;

  void AddUsageEvent(tsl::RCReference<PjRtDeviceEvent> event) override;

  void Delete(PjRtMemorySpace* memory_space) override;

  Future<> GetReadyFuture(PjRtMemorySpace* memory_space) override;

  absl::Status BlockForOperationsToComplete(
      PjRtMemorySpace* memory_space) override;

 private:
  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory();

  void ConfirmDonation() override { ReleaseDeviceMemory(); }

  bool owns_buffers_;

  // If non-tuple, `buffers_` contains 1 buffer; otherwise all leaf buffers.
  tsl::AsyncValueRef<NpuDeviceMemory> buffer_;
  // Should correspond to size of each buffer in `buffers_` when `buffers_` is
  // available.
  size_t buffer_size_;
  // The definition event are associated with npu operations that write to the
  // buffers.
  tsl::AsyncValueRef<NpuEvent> definition_event_;
  // Usage events are associated with npu operations that read from the buffers.
  absl::InlinedVector<tsl::AsyncValueRef<NpuEvent>, 4> usage_events_;
};

}  // namespace xla
#endif  //NPU_TRACKED_NPU_DEVICE_BUFFER_H_