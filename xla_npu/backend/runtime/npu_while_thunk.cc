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

#include "xla/xla_npu/backend/runtime/npu_while_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

NpuWhileThunk::NpuWhileThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    const BufferAllocation::Slice& loop_state_buffer_index,
    NpuThunkSequence condition_thunk_sequence,
    NpuThunkSequence body_thunk_sequence,
    std::vector<NpuWhileElementCopy> element_copies,
    std::optional<int64_t> trip_count)
    : NpuThunk(Kind::kWhile, std::move(thunk_info)),
      condition_result_buffer_index_(condition_result_buffer_index),
      loop_state_buffer_index_(loop_state_buffer_index),
      condition_thunk_sequence_(std::move(condition_thunk_sequence)),
      body_thunk_sequence_(std::move(body_thunk_sequence)),
      trip_count_(trip_count),
      element_copies_(std::move(element_copies)) {
  if (!element_copies_.empty()) {
    NPU_LOG_FORMAT_PRINT(DEBUG,
        "NpuWhileThunk: %d element carry copies configured",
        (int)element_copies_.size());
    for (int i = 0; i < (int)element_copies_.size(); ++i) {
      const auto& c = element_copies_[i];
      NPU_LOG_FORMAT_PRINT(DEBUG,
          "  carry_copy[%d]: src=(idx=%d,off=%lld,sz=%lld) -> "
          "dst=(idx=%d,off=%lld,sz=%lld)",
          i, c.src.index(), (long long)c.src.offset(),
          (long long)c.src.size(),
          c.dst.index(), (long long)c.dst.offset(),
          (long long)c.dst.size());
    }
  } else {
    NPU_LOG_FORMAT_PRINT(DEBUG,
        "NpuWhileThunk: NO element carry copies (relying on buffer aliasing)");
  }
}

std::string NpuWhileThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  std::string result = absl::StrCat(indent_str, "While thunk:\n");
  absl::StrAppend(&result, indent_str,
                  "condition (", condition_thunk_sequence_.size(),
                  " sub-thunks):\n");
  for (size_t i = 0; i < condition_thunk_sequence_.size(); ++i) {
    absl::StrAppend(&result, indent_str, "  ", i, ": ",
                    condition_thunk_sequence_[i]->ToString(indent + 2), "\n");
  }
  absl::StrAppend(&result, indent_str,
                  "body (", body_thunk_sequence_.size(), " sub-thunks):\n");
  for (size_t i = 0; i < body_thunk_sequence_.size(); ++i) {
    absl::StrAppend(&result, indent_str, "  ", i, ": ",
                    body_thunk_sequence_[i]->ToString(indent + 2), "\n");
  }
  return result;
}

absl::Status NpuWhileThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : condition_thunk_sequence_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  for (auto& thunk : body_thunk_sequence_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status NpuWhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  int device_ordinal = stream.parent()->device_ordinal();

  // Lambda: After executing the body thunk sequence, perform element-level carry copy.
  // When buffer assignment fails to alias body_root{i} to while_operand{i},
  // Explicit D2D copy is required to copy the body calculation results back to the carry state buffer.
  auto execute_body_and_copy_carry = [&](int64_t iter) -> absl::Status {
    for (const std::unique_ptr<NpuThunk>& thunk : body_thunk_sequence_) {
      tsl::profiler::TraceMe trace(thunk->profile_annotation());
      absl::Status thunk_status = thunk->ExecuteOnStream(params);
      if (!thunk_status.ok()) {
        tsl::errors::AppendToMessage(&thunk_status,
                                     "while executing while body thunk '",
                                     thunk->profile_annotation(),
                                     "' at iteration ", iter);
        return thunk_status;
      }
    }
    // ★ Iteration boundary synchronization: Ensure that all asynchronous ACLNN operations submitted by the body thunk are fully completed.
    //
    // Root cause: NPU's ACLNN operator is submitted to stream asynchronously, but the workspace memory pool is in the same stream
    // share the same address. The next thunk of GetWorkspaceSize may be synchronized via aclrtMemcpy
    // Write the workspace (tiling/parameter data) to overwrite the workspace of the ACLNN operator currently being executed.
    // Stream sorting only guarantees the serial execution of the ACLNN operator, but does not guarantee the host-side GetWorkspaceSize
    // Workspace writes are mutually exclusive with device-side ACLNN execution.
    //
    // The stream must be drained at the end of each iteration to ensure:
    // 1. After the ACLNN operation of body is completed, the next round of GetWorkspaceSize of condition/body can safely access the workspace.
    // 2. The output buffer data written by the body has been placed on the disk, and the complete data can be read in the next round of condition.
    // 3. ACLNN executor/tensor resources can be safely recycled
    TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

    // Element-level carry copy: body_root{i} -> while_operand{i}
    // Skip self-copy (src and dst point to the same buffer location) to avoid unnecessary D2D copy.
    if (!element_copies_.empty()) {
      for (const NpuWhileElementCopy& copy : element_copies_) {
        if (copy.src == copy.dst) {
          VLOG(2) << "[" << device_ordinal
                  << "] Carry copy SKIPPED (self-copy): idx="
                  << copy.src.index() << ",off=" << copy.src.offset();
          continue;
        }
        se::DeviceMemoryBase src_data =
            params.buffer_allocations->GetDeviceAddress(copy.src);
        se::DeviceMemoryBase dst_data =
            params.buffer_allocations->GetDeviceAddress(copy.dst);
        VLOG(2) << "[" << device_ordinal
                << "] Carry copy: src=(idx=" << copy.src.index()
                << ",off=" << copy.src.offset() << ",sz=" << copy.src.size()
                << ") -> dst=(idx=" << copy.dst.index()
                << ",off=" << copy.dst.offset() << ",sz=" << copy.dst.size()
                << ")";
        TF_RETURN_IF_ERROR(
            stream.MemcpyD2D(&dst_data, src_data, src_data.size()));
      }
    }
    return absl::OkStatus();
  };

  // If trip_count is known at compile time, body will be executed directly N times without condition checking.
  if (trip_count_.has_value()) {
    VLOG(1) << "[" << device_ordinal
            << "] Executing NpuWhileThunk for " << *trip_count_
            << " iterations (known trip count)";
    for (int64_t iter = 0; iter < *trip_count_; ++iter) {
      VLOG(2) << "[" << device_ordinal
              << "] Executing iteration # " << iter;
      TF_RETURN_IF_ERROR(execute_body_and_copy_carry(iter));
    }
    return absl::OkStatus();
  }

  // Runtime while loop: execute condition -> read back pred -> execute body (if pred is true)
  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  int64_t cond_result_byte_size =
      condition_result_buffer_index_.size();

  int64_t iter = 0;

  while (true) {
    tsl::profiler::TraceMe trace(
        [&] { return tsl::profiler::TraceMeEncode("While", {{"iteration:", iter}}); });

    VLOG(1) << "[" << device_ordinal
            << "] Executing NpuWhileThunk condition computation; iter=" << iter;
    for (const std::unique_ptr<NpuThunk>& thunk : condition_thunk_sequence_) {
      tsl::profiler::TraceMe trace(thunk->profile_annotation());
      absl::Status thunk_status = thunk->ExecuteOnStream(params);
      if (!thunk_status.ok()) {
        tsl::errors::AppendToMessage(
            &thunk_status, "while executing while condition thunk '",
            thunk->profile_annotation(), "' at iteration ", iter);
        return thunk_status;
      }
    }

    // Synchronize the stream to ensure that the asynchronous ACLNN operation submitted by the condition thunk completes.
    // NpuStream::Memcpy uses synchronous aclrtMemcpy under the hood (without stream sorting),
    // If sync is not performed first, aclrtMemcpy may read the stale condition result of the previous iteration.
    // Causes the loop to run more or less iterations.
    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      return absl::InternalError(absl::StrFormat(
          "Failed to sync after condition computation on stream %p: %s",
          &stream, blocked.message()));
    }

    // Copy the condition result from device back to host (aclrtMemcpy synchronization, safe)
    uint8_t condition_result = 0;
    TF_RETURN_IF_ERROR(
        stream.Memcpy(&condition_result, condition_result_data,
                      std::min(sizeof(condition_result), (size_t)cond_result_byte_size)));

    VLOG(1) << "[" << device_ordinal
            << "] condition_result = " << (int)condition_result
            << " (raw byte), cond_buffer_size=" << cond_result_byte_size;

    if (condition_result == 0) {
      VLOG(1) << "[" << device_ordinal
              << "] Break NpuWhileThunk loop; iter=" << iter;
      break;
    }

    VLOG(1) << "[" << device_ordinal
            << "] Executing NpuWhileThunk body computation; iter=" << iter;
    TF_RETURN_IF_ERROR(execute_body_and_copy_carry(iter));

    ++iter;
  }
  return absl::OkStatus();
}

NpuThunk::BufferUses NpuWhileThunk::buffer_uses() const {
  BufferUses uses;
  for (const std::unique_ptr<NpuThunk>& thunk : condition_thunk_sequence_) {
    BufferUses thunk_uses = thunk->buffer_uses();
    uses.insert(uses.end(), thunk_uses.begin(), thunk_uses.end());
  }
  for (const std::unique_ptr<NpuThunk>& thunk : body_thunk_sequence_) {
    BufferUses thunk_uses = thunk->buffer_uses();
    uses.insert(uses.end(), thunk_uses.begin(), thunk_uses.end());
  }
  return uses;
}

void NpuWhileThunk::ForAllThunks(
    absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<NpuThunk>& thunk : condition_thunk_sequence_) {
    thunk->ForAllThunks(fn);
  }
  for (const std::unique_ptr<NpuThunk>& thunk : body_thunk_sequence_) {
    thunk->ForAllThunks(fn);
  }
}

void NpuWhileThunk::ForAllThunksMutable(
    absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
  for (std::unique_ptr<NpuThunk>& thunk : condition_thunk_sequence_) {
    thunk->ForAllThunksMutable(fn);
  }
  for (std::unique_ptr<NpuThunk>& thunk : body_thunk_sequence_) {
    thunk->ForAllThunksMutable(fn);
  }
}

absl::Status NpuWhileThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
        fn) {
  for (std::unique_ptr<NpuThunk>& thunk : condition_thunk_sequence_) {
    TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  for (std::unique_ptr<NpuThunk>& thunk : body_thunk_sequence_) {
    TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  return absl::OkStatus();
}

}  // namespace xla
