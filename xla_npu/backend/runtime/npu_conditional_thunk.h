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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CONDITIONAL_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CONDITIONAL_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

// NpuConditionalThunk implements runtime execution of HLO conditional instructions.
// Two forms are supported:
// - predicated: operand(0) is a PRED scalar. When true, branch 0 is executed, and when false, branch 1 is executed.
// - indexed: operand(0) is an S32 scalar, used as the branch index
class NpuConditionalThunk : public NpuThunk {
 public:
  NpuConditionalThunk(
      ThunkInfo thunk_info,
      const BufferAllocation::Slice& branch_index_buffer_index,
      std::vector<NpuThunkSequence> branch_thunk_sequences,
      bool branch_index_is_bool);
  NpuConditionalThunk(const NpuConditionalThunk&) = delete;
  NpuConditionalThunk& operator=(const NpuConditionalThunk&) = delete;

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
  // The buffer slice of operand(0) is used to read back the branch index/predicate at runtime
  const BufferAllocation::Slice branch_index_buffer_index_;
  // thunk sequence for each branch
  std::vector<NpuThunkSequence> branch_thunks_;
  // true = predicated (PRED), false = indexed (S32)
  bool branch_index_is_bool_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CONDITIONAL_THUNK_H_
