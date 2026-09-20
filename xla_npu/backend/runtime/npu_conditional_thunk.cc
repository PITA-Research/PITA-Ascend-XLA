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

#include "xla/xla_npu/backend/runtime/npu_conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

NpuConditionalThunk::NpuConditionalThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& branch_index_buffer_index,
    std::vector<NpuThunkSequence> branch_thunk_sequences,
    bool branch_index_is_bool)
    : NpuThunk(Kind::kConditional, std::move(thunk_info)),
      branch_index_buffer_index_(branch_index_buffer_index),
      branch_thunks_(std::move(branch_thunk_sequences)),
      branch_index_is_bool_(branch_index_is_bool) {}

std::string NpuConditionalThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  std::string result =
      absl::StrCat(indent_str, "Conditional thunk (",
                   branch_index_is_bool_ ? "predicated" : "indexed", ", ",
                   branch_thunks_.size(), " branches):\n");
  for (size_t branch = 0; branch < branch_thunks_.size(); ++branch) {
    absl::StrAppend(&result, indent_str, "  branch ", branch, " (",
                    branch_thunks_[branch].size(), " sub-thunks):\n");
    for (size_t i = 0; i < branch_thunks_[branch].size(); ++i) {
      absl::StrAppend(&result, indent_str, "    ", i, ": ",
                      branch_thunks_[branch][i]->ToString(indent + 3), "\n");
    }
  }
  return result;
}

absl::Status NpuConditionalThunk::Initialize(const InitializeParams& params) {
  for (auto& branch : branch_thunks_) {
    for (auto& thunk : branch) {
      TF_RETURN_IF_ERROR(thunk->Initialize(params));
    }
  }
  return absl::OkStatus();
}

absl::Status NpuConditionalThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& stream = *params.stream;
  int device_ordinal = stream.parent()->device_ordinal();

  // Read back branch index/predicate from device
  // First sync stream to ensure that the ACLNN asynchronous operation of branch_index has completed.
  // The bottom layer of stream.Memcpy calls synchronous aclrtMemcpy (without stream sorting),
  // If you do not sync first, you may read the stale data before the previous ACLNN operation is completed.
  if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to sync stream %p before memcpy: %s", &stream,
        blocked.message()));
  }

  se::DeviceMemoryBase branch_index_data =
      params.buffer_allocations->GetDeviceAddress(
          branch_index_buffer_index_);

  int32_t branch_index_raw = 0;
  int64_t buf_size = branch_index_buffer_index_.size();

  TF_RETURN_IF_ERROR(stream.Memcpy(
      &branch_index_raw, branch_index_data,
      std::min(sizeof(branch_index_raw), static_cast<size_t>(buf_size))));

  // Parse branch index
  int32_t branch_index;
  if (branch_index_is_bool_) {
    // PRED: true (non-zero) -> branch 0 (true_computation), false (0) -> branch 1 (false_computation)
    bool pred = reinterpret_cast<uint8_t*>(&branch_index_raw)[0] != 0;
    branch_index = pred ? 0 : 1;
    VLOG(1) << "[" << device_ordinal
            << "] NpuConditionalThunk: predicated, pred=" << pred
            << " -> branch " << branch_index;
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "NpuConditionalThunk: pred=%d -> branch %d",
                         (int)pred, branch_index);
  } else {
    // S32: Directly used as branch index, execute the last branch when out of bounds
    branch_index = branch_index_raw;
    if (branch_index < 0 ||
        branch_index >= static_cast<int32_t>(branch_thunks_.size())) {
      VLOG(1) << "[" << device_ordinal
              << "] NpuConditionalThunk: indexed, branch_index=" << branch_index
              << " out of range [0, " << branch_thunks_.size()
              << "), clamping to last branch";
      branch_index = static_cast<int32_t>(branch_thunks_.size()) - 1;
    }
    VLOG(1) << "[" << device_ordinal
            << "] NpuConditionalThunk: indexed, branch_index=" << branch_index;
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "NpuConditionalThunk: indexed, branch_index=%d",
                         branch_index);
  }

  // Execute the thunk sequence of the selected branch
  NpuThunkSequence& selected_branch = branch_thunks_[branch_index];
  for (const std::unique_ptr<NpuThunk>& thunk : selected_branch) {
    tsl::profiler::TraceMe trace(thunk->profile_annotation());
    absl::Status thunk_status = thunk->ExecuteOnStream(params);
    if (!thunk_status.ok()) {
      tsl::errors::AppendToMessage(
          &thunk_status,
          "while executing conditional branch thunk '", thunk->profile_annotation(),
          "' at branch ", branch_index);
      return thunk_status;
    }
  }
  return absl::OkStatus();
}

NpuThunk::BufferUses NpuConditionalThunk::buffer_uses() const {
  BufferUses uses;
  for (const auto& branch : branch_thunks_) {
    for (const std::unique_ptr<NpuThunk>& thunk : branch) {
      BufferUses thunk_uses = thunk->buffer_uses();
      uses.insert(uses.end(), thunk_uses.begin(), thunk_uses.end());
    }
  }
  return uses;
}

void NpuConditionalThunk::ForAllThunks(
    absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
  for (const auto& branch : branch_thunks_) {
    for (const std::unique_ptr<NpuThunk>& thunk : branch) {
      thunk->ForAllThunks(fn);
    }
  }
}

void NpuConditionalThunk::ForAllThunksMutable(
    absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
  for (auto& branch : branch_thunks_) {
    for (std::unique_ptr<NpuThunk>& thunk : branch) {
      thunk->ForAllThunksMutable(fn);
    }
  }
}

absl::Status NpuConditionalThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
        fn) {
  for (auto& branch : branch_thunks_) {
    for (std::unique_ptr<NpuThunk>& thunk : branch) {
      TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
      TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
