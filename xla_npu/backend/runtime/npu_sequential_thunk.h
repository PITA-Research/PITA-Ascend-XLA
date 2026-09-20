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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_SEQUENTIAL_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_SEQUENTIAL_THUNK_H_

#include <memory>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

// A thunk that wraps a list of sub-thunks. Executing this thunk executes all
// the sub-thunks sequentially. This is useful to implement instructions that
// require multiple kernel launches or library calls.
class NpuSequentialThunk : public NpuThunk {
 public:
  NpuSequentialThunk(ThunkInfo thunk_info, NpuThunkSequence thunks);
  NpuSequentialThunk(const NpuSequentialThunk&) = delete;
  NpuSequentialThunk& operator=(const NpuSequentialThunk&) = delete;

  NpuThunkSequence& thunks() { return thunks_; }
  const NpuThunkSequence& thunks() const { return thunks_; }
  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  void ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const override;
  void ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) override;
  absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
          fn) override;

  // Converts a NpuThunk into a NpuSequentialThunk. If the input is already a
  // NpuSequentialThunk, the returned value is the downcasted input.
  // The new thunk, if created, will use a default-initialized ThunkInfo.
  static std::unique_ptr<NpuSequentialThunk> FromThunk(
      std::unique_ptr<NpuThunk> thunk);

 private:
  // The list of sub-thunks.
  NpuThunkSequence thunks_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_SEQUENTIAL_THUNK_H_
