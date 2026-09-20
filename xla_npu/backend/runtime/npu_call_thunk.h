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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CALL_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CALL_THUNK_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

// A thunk constructed from a call instruction that simply calls a thunk
// sequence emitted from the called computation.
class NpuCallThunk : public NpuThunk {
 public:
  NpuCallThunk(ThunkInfo thunk_info, NpuThunkSequence called_sequence)
      : NpuThunk(Kind::kCall, std::move(thunk_info)),
        called_sequence_(std::move(called_sequence)) {}
  NpuCallThunk(const NpuCallThunk&) = delete;
  NpuCallThunk& operator=(const NpuCallThunk&) = delete;

  const NpuThunkSequence& called_sequence() const { return called_sequence_; }
  NpuThunkSequence& called_sequence() { return called_sequence_; }

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
  // The thunk sequence from the called computation.
  NpuThunkSequence called_sequence_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_CALL_THUNK_H_
