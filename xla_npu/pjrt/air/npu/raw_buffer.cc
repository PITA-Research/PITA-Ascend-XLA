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

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

#include "raw_buffer.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"

namespace xla {
constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB
void NpuTrackedDeviceEventPromise::Set(
    tsl::RCReference<PjRtDeviceEvent> event) {
  auto cpu_event =
      tensorflow::down_cast<NpuTrackedDeviceEvent*>(event.get())->event();
  av_->ForwardTo(std::move(cpu_event));
}

void NpuTrackedDeviceEventPromise::SetReady() {
  av_->ForwardTo(tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

Future<> NpuTrackedDeviceEvent::GetReadyFuture() {
  auto [promise, future] = Future<>::MakePromise();
  event_.AndThen([promise = std::move(promise), event = event_]() mutable {
    if (auto* error = event.GetErrorIfPresent()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });

  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      [callee_method = callee_method_, callee_type = callee_type_]() {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        return FutureHelpers::ProfilingKeys({traceme.GetContextId()});
      },
      /*on_block_end=*/
      [callee_method = callee_method_,
       callee_type = callee_type_](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
}

/*static*/ absl::StatusOr<tsl::RCReference<NpuRawBuffer>>
NpuRawBuffer::Allocate(PjRtMemorySpace* memory_space, size_t size_bytes) {
  TF_ASSIGN_OR_RETURN(auto memory, NpuDeviceMemory::Allocate(size_bytes));
  return tsl::MakeRef<NpuRawBuffer>(memory_space, std::move(memory));
}

/*static*/ absl::StatusOr<tsl::RCReference<NpuRawBuffer>>
NpuRawBuffer::ImportForeignMemory(
    void* data, absl::AnyInvocable<void() &&> on_delete_callback,
    size_t on_device_bytes_count, PjRtMemorySpace* memory_space) {
  return absl::UnimplementedError(
      "NPU does not support importing foreign memory");
}

size_t NpuRawBuffer::GetOnDeviceSizeInBytes() const {
  return buffer_->size_bytes();
}

void* NpuRawBuffer::GetHostPointer() const { return buffer_->untyped_data(); }

absl::Status NpuRawBuffer::ValidateSlice(int64_t offset, int64_t slice_size) {
  size_t buffer_size = GetOnDeviceSizeInBytes();
  if (offset < 0 || offset > buffer_size || buffer_size - offset < slice_size) {
    return InvalidArgument(
        "Invalid slicing of buffer size %lld with "
        "invalid offset %lld, slice size %lld",
        buffer_size, offset, slice_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                                int64_t transfer_size) {
  TF_RETURN_IF_ERROR(ValidateSlice(offset, transfer_size));
  void* dst = static_cast<char*>(buffer_->untyped_data()) + offset;
  auto acl_ret = dep::aclrtMemcpy(dst, transfer_size, src, transfer_size,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_ret != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("AclrtMemcpy H2D failed, error code: ", acl_ret));
  }
  return tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                                int64_t transfer_size) {
  TF_RETURN_IF_ERROR(ValidateSlice(offset, transfer_size));
  const void* src = static_cast<const char*>(buffer_->untyped_data()) + offset;
  auto acl_ret = dep::aclrtMemcpy(dst, transfer_size, src, transfer_size,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_ret != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("AclrtMemcpy D2H failed, error code: ", acl_ret));
  }
  return tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> NpuRawBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const xla::Layout& layout,
    AsyncWorkRunner* async_work_runner) {
  return absl::UnimplementedError("NPU does not support CopyFromLiteral");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::CopyFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, const Shape& shape,
    AsyncWorkRunner* async_work_runner, absl::Mutex* transpose_mu,
    TransposePlanCache* transpose_cache) {
  tsl::AsyncValueRef<NpuDeviceMemory> device_buffer = buffer_;
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  const int bit_width = primitive_util::BitWidth(type);
  // Packed arrays are unpacked on host and packed on device.
  bool is_packed = primitive_util::IsSubByteNonPredType(type);

  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  size_t dst_byte_size = byte_size;
  if (is_packed) {
    byte_size *= 8 / bit_width;
  }
  auto dst_data_ptr = device_buffer->untyped_data();
  if (!has_default_layout || is_packed) {
    // If the input array does not have a major-to-minor layout, transpose it
    // into major-to-minor layout. Currently we choose to always do this
    // synchronously.
    std::shared_ptr<TransposePlan> transpose;
    {
      absl::InlinedVector<int64_t, 4> permutation(dims.size());
      absl::c_iota(permutation, 0);
      TransposePlan::Options options;
      options.elem_size_in_bytes = primitive_util::ByteWidth(type);
      options.dims = dims;
      options.permutation = permutation;
      if (byte_strides) {
        options.input_layout = TransposePlan::Striding{*byte_strides};
      }
      absl::MutexLock lock(*transpose_mu);
      TF_ASSIGN_OR_RETURN(transpose, transpose_cache->GetOrCreate(options));
    }
    if (!is_packed) {
      auto data_transposed = std::make_unique<char[]>(byte_size);
      transpose->Execute(data, data_transposed.get());
      auto acl_ret =
          dep::aclrtMemcpy(dst_data_ptr, byte_size, data_transposed.get(),
                           byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
      if (acl_ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrCat(
            "AclrtMemcpy H2D failed during transpose, error code: ", acl_ret));
      }
    } else {
      // First transpose the unpacked data into a new temporary buffer, then
      // pack the data.
      auto data_transposed = std::make_unique<char[]>(byte_size);
      transpose->Execute(data, data_transposed.get());
      auto data_packed = std::make_unique<char[]>(dst_byte_size);
      absl::Span<const char> src_data_span(data_transposed.get(), byte_size);
      absl::Span<char> dst_data_span(data_packed.get(), dst_byte_size);
      PackIntN(bit_width, src_data_span, dst_data_span);
      auto acl_ret =
          dep::aclrtMemcpy(dst_data_ptr, dst_byte_size, data_packed.get(),
                           dst_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
      if (acl_ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrCat(
            "AclrtMemcpy H2D failed during packed transpose, error code: ",
            acl_ret));
      }
    }
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
      on_done_with_host_buffer = nullptr;
    }
  } else {
    auto acl_ret = dep::aclrtMemcpy(dst_data_ptr, byte_size, data, byte_size,
                                    ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("AclrtMemcpy H2D failed, error code: ", acl_ret));
    }
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
      on_done_with_host_buffer = nullptr;
    }
  }
  return tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<NpuEvent>());
}

void NpuRawBuffer::ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                                    xla::Shape shape) {
  std::cout << "NPU does not support ReadDynamicShape" << std::endl;
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
NpuRawBuffer::MakeAllocationReadyEvent() {
  return absl::UnimplementedError(
      "NPU does not support MakeAllocationReadyEvent");
}

void NpuRawBuffer::CopyToLiteralAsync(
    Promise<> promise, tsl::RCReference<PjRtDeviceEventPromise> device_promise,
    MutableLiteralBase* literal, xla::Shape shape) {
  tsl::profiler::TraceMe traceme("NpuRawBuffer::CopyToLiteralAsync");

  // Get source device memory pointer and size
  void* src_device_ptr = buffer_->untyped_data();
  size_t device_buffer_size = buffer_->size_bytes();

  // Get destination host memory pointer and size
  size_t output_size =
      static_cast<size_t>(ShapeUtil::ByteSizeOf(literal->shape()));
  void* dst_host_ptr = literal->untyped_data();

  aclError acl_ret = dep::aclrtSetDevice(0);
  if (acl_ret != ACL_SUCCESS) {
    auto error = absl::InternalError(absl::StrCat(
        "AclrtSetDevice set device 0 failed, error code: ", acl_ret));
    device_promise->SetError(error);
    promise.Set(error);
    return;
  }

  // Handle packed types (sub-byte types like INT4)
  if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
    // For packed types, we need to unpack after copying
    // First copy packed data from device to a temporary buffer
    auto temp_buffer = std::make_unique<char[]>(device_buffer_size);

    acl_ret =
        dep::aclrtMemcpy(temp_buffer.get(), device_buffer_size, src_device_ptr,
                         device_buffer_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
      auto error = absl::InternalError(
          absl::StrCat("AclrtMemcpy D2H failed in CopyToLiteralAsync for "
                       "packed type, error code: ",
                       acl_ret));
      device_promise->SetError(error);
      promise.Set(error);
      return;
    }

    // Unpack the data
    absl::Span<const char> input_span(temp_buffer.get(), device_buffer_size);
    absl::Span<char> output_span(static_cast<char*>(dst_host_ptr), output_size);
    primitive_util::UnpackIntN(shape.element_type(), input_span, output_span);
  } else {
    // For normal types, direct copy from device to host
    acl_ret = dep::aclrtMemcpy(dst_host_ptr, output_size, src_device_ptr,
                               output_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
      auto error = absl::InternalError(absl::StrCat(
          "AclrtMemcpy D2H failed in CopyToLiteralAsync, error code: ",
          acl_ret));
      device_promise->SetError(error);
      promise.Set(error);
      return;
    }
  }

  // Set success status
  device_promise->Set(tsl::MakeRef<NpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<NpuEvent>()));
  promise.Set(absl::OkStatus());
}

void NpuRawBuffer::CopyTo(
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  std::cout << "NPU does not support CopyTo" << std::endl;
}

}  // namespace xla