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

#include "xla/xla_npu/backend/runtime/npu_call_thunk.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

std::string NpuCallThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  std::string result = absl::StrCat(indent_str, "Call thunk with ",
                                    called_sequence_.size(), " sub-thunks:\n");
  for (size_t i = 0; i < called_sequence_.size(); ++i) {
    absl::StrAppend(&result, indent_str, "  ", i, ": ",
                    called_sequence_[i]->ToString(indent + 2), "\n");
  }
  return result;
}

absl::Status NpuCallThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : called_sequence_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status NpuCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  for (const std::unique_ptr<NpuThunk>& thunk : called_sequence_) {
    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    if (params.mock_collectives && thunk->IsCollective()) {
      continue;
    }

    VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
            << "Start NpuCallThunk::ExecuteOnStream: "
            << thunk->profile_annotation();
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
    VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
            << "End NpuCallThunk::ExecuteOnStream: "
            << thunk->profile_annotation();
  }
  return absl::OkStatus();
}

NpuThunk::BufferUses NpuCallThunk::buffer_uses() const {
  BufferUses uses;
  for (const std::unique_ptr<NpuThunk>& thunk : called_sequence_) {
    BufferUses thunk_uses = thunk->buffer_uses();
    uses.insert(uses.end(), thunk_uses.begin(), thunk_uses.end());
  }
  return uses;
}

void NpuCallThunk::ForAllThunks(
    absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<NpuThunk>& thunk : called_sequence_) {
    thunk->ForAllThunks(fn);
  }
}

void NpuCallThunk::ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
  for (std::unique_ptr<NpuThunk>& thunk : called_sequence_) {
    thunk->ForAllThunksMutable(fn);
  }
}

absl::Status NpuCallThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
        fn) {
  for (std::unique_ptr<NpuThunk>& thunk : called_sequence_) {
    TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  return absl::OkStatus();
}

}  // namespace xla
