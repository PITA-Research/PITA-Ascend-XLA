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

#ifndef XLA_XLA_NPU_SERVICE_NPU_EXECUTION_STREAM_ASSIGNMENT_H_
#define XLA_XLA_NPU_SERVICE_NPU_EXECUTION_STREAM_ASSIGNMENT_H_

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"

namespace xla {

struct NpuExecutionStreamAssignmentOptions {
  // The `NpuExecutionStreamAssignment` will round-robin compute thunks across this
  // many `ExecutionStreams`.
  int number_of_compute_execution_streams = 4;
  // The `NpuExecutionStreamAssignment` will round-robin collective thunks
  // across this many `ExecutionStreams`.
  int number_of_collective_execution_streams = 1;
};

// `ExecutionStreamAssignments` represent a mapping from `HloInstructions` to
// `ExecutionStreamIds`. Asynchronous calls (`async-start`, `async-update`, and
// `async-done`) result in the target computations being assigned new
// `ExecutionStreamIds` to support concurrent execution.
class NpuExecutionStreamAssignment {
 public:
  // The `HloModule` must be flat. In other words, there must be a one-to-one
  // mapping between callsites and computations. One way to guarantee this is to
  // pass the module through the `FlattenCallGraph` pass.
  // The NpuExecutionStreamAssignment does not take ownership of the `HloModule`.
  explicit NpuExecutionStreamAssignment(
      const HloModule* module,
      NpuExecutionStreamAssignmentOptions options = {});

  // Returns the `NpuExecutionStreamId` for the given instruction, which *must* be
  // synchronous. Returns an error if the instruction is either not reachable
  // from the module's entrypoint, or is only reachable through embedded calls.
  absl::StatusOr<NpuExecutionStreamId> GetSyncExecutionStreamId(
      const HloInstruction* instruction) const;

  // Returns the source and destination `ExecutionStreamIds` for the given
  // instruction, which *must* be asynchronous. Returns an error if the
  // instruction is either not reachable from the module's entrypoint, or is
  // only reachable through embedded calls.
  struct AsyncExecutionStreamIds {
    // The `NpuExecutionStreamId` for the calling instruction (e.g. the computation
    // that invokes `async-start`).
    NpuExecutionStreamId source_stream_id;
    // The `NpuExecutionStreamId` for the callee computation (e.g. the callee of an
    // `async-start` instruction).
    NpuExecutionStreamId destination_stream_id;
  };
  absl::StatusOr<AsyncExecutionStreamIds> GetAsyncExecutionStreamIds(
      const HloInstruction* instruction) const;

 private:
  // Maps from `HloInstructions` to `ExecutionStreamIds` for synchronous and
  // asynchronous instructions, respectively. All instructions reachable through
  // non-embedded calls must be present.
  absl::flat_hash_map<HloInstruction*, NpuExecutionStreamId> sync_instructions_;
  absl::flat_hash_map<HloInstruction*, AsyncExecutionStreamIds>
      async_instructions_;
};

inline bool operator==(
    const NpuExecutionStreamAssignment::AsyncExecutionStreamIds& first,
    const NpuExecutionStreamAssignment::AsyncExecutionStreamIds& second) {
  return first.source_stream_id == second.source_stream_id &&
         first.destination_stream_id == second.destination_stream_id;
}

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_EXECUTION_STREAM_ASSIGNMENT_H_
