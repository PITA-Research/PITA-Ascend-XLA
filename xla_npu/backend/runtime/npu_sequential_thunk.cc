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

#include "xla/xla_npu/backend/runtime/npu_sequential_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

NpuSequentialThunk::NpuSequentialThunk(ThunkInfo thunk_info,
                                       NpuThunkSequence thunks)
    : NpuThunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string NpuSequentialThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  if (thunks_.empty()) {
    return indent_str + "No thunks.";
  }

  auto thunk_with_longest_kind =
      absl::c_max_element(thunks_, [](const std::unique_ptr<NpuThunk>& a,
                                      const std::unique_ptr<NpuThunk>& b) {
        return NpuThunk::KindToString(a->kind()).length() <
               NpuThunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      NpuThunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
    absl::StrAppend(&result, indent_str);
    absl::StrAppendFormat(&result,
                          "%03d: ", thunk->thunk_info().thunk_id.value());
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = NpuThunk::KindToString(thunk->kind());
    absl::StrAppend(&result, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    absl::StrAppend(&result, thunk->ToString(indent + 1));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

absl::Status NpuSequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status NpuSequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Start execute thunk sequence");

  for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    if (params.mock_collectives && thunk->IsCollective()) {
      continue;
    }

    VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
            << "Start NpuSequentialThunk::ExecuteOnStream: "
            << thunk->profile_annotation();
    absl::Status thunk_status = thunk->ExecuteOnStream(params);
    if (!thunk_status.ok()) {
      tsl::errors::AppendToMessage(
          &thunk_status, "while executing NPU thunk '",
          thunk->profile_annotation(), "'");
      return thunk_status;
    }
    VLOG(1) << "[" << params.stream->parent()->device_ordinal() << "] "
            << "End NpuSequentialThunk::ExecuteOnStream: "
            << thunk->profile_annotation();
  }
  return absl::OkStatus();
}

void NpuSequentialThunk::ForAllThunks(
    absl::FunctionRef<void(const NpuThunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
    thunk->ForAllThunks(fn);
  }
}

void NpuSequentialThunk::ForAllThunksMutable(
    absl::FunctionRef<void(NpuThunk*)> fn) {
  fn(this);
  for (const std::unique_ptr<NpuThunk>& thunk : thunks_) {
    thunk->ForAllThunksMutable(fn);
  }
}

absl::Status NpuSequentialThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
        fn) {
  for (std::unique_ptr<NpuThunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  return absl::OkStatus();
}

std::unique_ptr<NpuSequentialThunk> NpuSequentialThunk::FromThunk(
    std::unique_ptr<NpuThunk> thunk) {
  if (thunk->kind() == NpuThunk::kSequential) {
    return std::unique_ptr<NpuSequentialThunk>(
        static_cast<NpuSequentialThunk*>(thunk.release()));
  }

  std::vector<std::unique_ptr<NpuThunk>> thunks;
  thunks.push_back(std::move(thunk));
  return std::make_unique<NpuSequentialThunk>(NpuThunk::ThunkInfo(),
                                              std::move(thunks));
}

}  // namespace xla
