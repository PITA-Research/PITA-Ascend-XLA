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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WHILE_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WHILE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

// Description After body is executed, it needs to be copied from the body_root element to the D2D copy of the while_operand element.
// When buffer assignment fails to alias body_root{i} and while_operand{i} to the same buffer,
// An explicit carry copy is required to ensure state transfer between iterations.
struct NpuWhileElementCopy {
  BufferAllocation::Slice src;  // body_root element buffer
  BufferAllocation::Slice dst;  // while_operand element buffer (carry state)
};

// NpuWhileThunk implements runtime execution of the HLO while instruction.
class NpuWhileThunk : public NpuThunk {
 public:
  NpuWhileThunk(ThunkInfo thunk_info,
                const BufferAllocation::Slice& condition_result_buffer_index,
                const BufferAllocation::Slice& loop_state_buffer_index,
                NpuThunkSequence condition_thunk_sequence,
                NpuThunkSequence body_thunk_sequence,
                std::vector<NpuWhileElementCopy> element_copies,
                std::optional<int64_t> trip_count = std::nullopt);
  NpuWhileThunk(const NpuWhileThunk&) = delete;
  NpuWhileThunk& operator=(const NpuWhileThunk&) = delete;

  const NpuThunkSequence& condition_thunk_sequence() const {
    return condition_thunk_sequence_;
  }
  const NpuThunkSequence& body_thunk_sequence() const {
    return body_thunk_sequence_;
  }

  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override;

  void ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const override;
  void ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) override;
  absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
          fn) override;

 private:
  const BufferAllocation::Slice condition_result_buffer_index_;
  const BufferAllocation::Slice loop_state_buffer_index_;  // while operand buffer

  NpuThunkSequence condition_thunk_sequence_;
  NpuThunkSequence body_thunk_sequence_;
  std::optional<int64_t> trip_count_;

  // Element-level carry copy list required after body execution.
  // When the buffers of body_root{i} and while_operand{i} are different,
  // The body calculation result needs to be copied back to the carry state buffer to ensure that the updated value can be read in the next iteration.
  std::vector<NpuWhileElementCopy> element_copies_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_WHILE_THUNK_H_
