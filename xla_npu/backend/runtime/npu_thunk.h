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

#ifndef XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_
#define XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/util.h"
#include "xla/xla_npu/backend/runtime/npu_thunk_id.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

// Forward declaration
class NpuKernelConfig;

// Execution stream id allows to specify what Gpu stream NpuThunk should be
// using for launching device work (kernels, library calls, etc.). By default
// all thunks use stream #0, which is the default compute stream of an XLA
// executable.
// Stream synchronizations are explicit and represented as WaitForStreams thunk
// in a NpuThunkSequence. When NpuThunkSequence converted to CommandBuffer,
// execution streams mapped to concurrent execution scopes and barriers between
// them.
// IMPORTANT: Async execution semantics and execution stream id
// For async thunks (i.e. thunks corresponding to `all-reduce-start` and
// `all-reduce-done`) execution stream id means NOT a stream where the async
// operation must execute, but a stream that async operation must be
// synchronized with:
//   - Start operation must wait for the completion of all launched work on the
//     execution stream id (usually by adding a stream wait) and after that
//     launch async work on implementation defined extra stream (can be borrowed
//     from a pool)
//   - Corresponding Done operation must synchronize execution stream id with
//     an implementation defined stream that is running async work, again
//     usually by adding a stream wait.
TSL_LIB_GTL_DEFINE_INT_TYPE(NpuExecutionStreamId, uint64_t);

// Unique identifier for async events. The same identifier is expected to be
// shared between a pair of StartThunk and corresponding DoneThunk. It is used
// to collect async regions for a CommandBufferThunk.
TSL_LIB_GTL_DEFINE_INT_TYPE(NpuAsyncEventsUniqueId, uint64_t);

// NpuThunk acts as the bridge between IrEmitter and GpuExecutable. It stores
// the metadata IrEmitter generates for GpuExecutable to invoke an
// HloInstruction.
// NpuThunk provides the Initialize and ExecuteOnStream interface for
// GpuExecutable to initialize and execute the invocation respectively. Its
// subclasses are supposed to override these interfaces to launch a generated
// kernel or call an external library function (such as operations in cuBLAS).
// Thunks have three execution stages:
// (1) Prepare: at this stage NpuThunk can request shared resources required at
// run
//     time, i.e. collective thunks request collective cliques. Executable(s)
//     will coordinate resource acquisition.
// (2) Initialize: at this stage NpuThunk must initialize all internal state
//     required for execution, maybe using resources requested at prepare stage.
// (3) Execute: at this stage NpuThunk must launch "work" on underlying device
//     using given stream, and it's expected that all expensive initialization
//     is completed at earlier stages.
// This is thread-compatible. NpuThunk implementation should expect that it will
// be called concurrently from multiple threads, for different run ids and for
// different devices (stream executors). For partitioned XLA programs the
// expectation is that all local participants execute simultaneously on
// different threads and coordinate resource acquisition via rendezvous.
class NpuThunk {
 public:
  using ExecutionStreamIdMap =
      absl::flat_hash_map<NpuExecutionStreamId, se::Stream*>;

  // When default execution stream id is used, operations launched by a thunk
  // must be synchronized with a stream passed in ExecuteOptions.
  static constexpr auto kDefaultExecutionStreamId = NpuExecutionStreamId(0);

  enum Kind {
    // # go/keep-sorted start
    kAllGather,
    kAllGatherDone,
    kAllGatherStart,
    kAllReduce,
    kAllReduceDone,
    kAllReduceStart,
    kAllToAll,
    kAllToAllDone,
    kAllToAllStart,
    kBuffersDebugChecksum,
    kBuffersDebugFloatCheck,
    kCollectiveBroadcast,
    kCollectiveBroadcastDone,
    kCollectiveBroadcastStart,
    kCollectiveKernel,
    kCollectivePermute,
    kCollectivePermuteDone,
    kCollectivePermuteStart,
    kCall,
    kCommandBuffer,
    kConditional,
    kConvolution,
    kConvolutionReorder,
    kCopy,
    kCopyDone,
    kCuDnn,
    kCubSort,
    kCublasLtMatmul,
    kCustomCall,
    kCustomKernel,
    kDynamicSlice,
    kFft,
    kGemm,
    kGroupDone,
    kGroupStart,
    kHostExecuteDone,
    kHostExecuteStart,
    kHostRecv,
    kHostRecvDone,
    kHostSend,
    kHostSendDone,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNorm,
    kNvshmemAllReduceDone,
    kNvshmemAllReduceStart,
    kNvshmemCollectivePermute,
    kNvshmemCollectivePermuteDone,
    kNvshmemCollectivePermuteStart,
    kNvshmemRecv,
    kNvshmemRecvDone,
    kNvshmemSend,
    kNvshmemSendDone,
    kOutfeed,
    kPartitionId,
    kRaggedAllToAll,
    kRaggedAllToAllDone,
    kRaggedAllToAllStart,
    kRecv,
    kRecvDone,
    kReduceScatter,
    kReduceScatterDone,
    kReduceScatterStart,
    kReplicaId,
    kSelectK,
    kSend,
    kSendDone,
    kSequential,
    kTriangularSolve,
    kWaitForStreams,
    kWhile
    // go/keep-sorted end
  };

  struct ExecutableSource {
    absl::string_view text;            // PTX for NVIDIA backend
    absl::Span<const uint8_t> binary;  // CUBIN for NVIDIA backends
                                       // BinaryMap dnn_compiled_graphs;
  };

  // Metadata associated with a NpuThunk,
  // including profiling and stream execution info.
  struct ThunkInfo {
    ThunkInfo() = default;  // Disable implicit constructors.

    static ThunkInfo WithProfileAnnotation(const HloInstruction* instr,
                                           NpuThunkId thunk_id);

    std::string profile_annotation;

    NpuExecutionStreamId execution_stream_id = kDefaultExecutionStreamId;

    NpuThunkId thunk_id = NpuThunkId{0};

    // Provided to ACLNN for use
    std::string hloInstrDesc;  // Keep HLO instruction description for easy debugging
    std::string op_name;
    HloOpcode op_code;
    std::vector<Shape>
        param_shapes;  // The order of the parameters in the instruction is consistent. The last position indicates the parameter shape.
    std::vector<int64_t> dimensions;  // Used for operators such as concatenate that require dimensions

    // Reduce reducer kind (for kReduce). 结构化地从 reducer computation 的
    // root instruction opcode 推断,避免依赖 hloInstrDesc 的脆弱子串匹配。
    // 关键修复：PRED 上的 and/or reducer 必须走 aclnnAll/aclnnAny (逻辑归约),
    // 否则会落入 aclnnReduceSum 把布尔当数值求和,导致 jnp.all/allclose 翻车。
    enum class ReduceReducerKind {
      kSum,        // add reducer → aclnnReduceSum
      kMax,        // maximum reducer → aclnnAmax
      kMin,        // minimum reducer → aclnnAmin
      kAnd,        // and reducer (PRED) → aclnnAll (逻辑与归约)
      kOr,         // or reducer (PRED) → aclnnAny (逻辑或归约)
      kUnsupported // 未识别或尚未展开的 tuple reducer → 显式报错
    };
    ReduceReducerKind reduce_reducer_kind = ReduceReducerKind::kUnsupported;

    // Scatter reduction kind
    enum class ScatterReductionKind {
      kReplace,    // return updates → aclnnScatterNd
      kAdd,        // return old + updates → aclnnScatter(reduce=1) / aclnnTfScatterAdd
      kMul,        // return old * updates → aclnnScatter(reduce=2)
      kUnsupported // max/min/other → report error
    };
    ScatterReductionKind scatter_reduction_kind = ScatterReductionKind::kUnsupported;

    // ACLNN scatter APIs cover different subsets of HLO scatter semantics.
    enum class ScatterLoweringKind {
      kScatterNd,
      kSingleSliceIndexCopy,
      kStaticWindowCopy,
      kBatchedSingleDimScatter,
      kSameRankSingleDimScatter,
      kMultiDimScatter,  // multi-dim scatter via index linearization
    };
    ScatterLoweringKind scatter_lowering_kind = ScatterLoweringKind::kScatterNd;
    int64_t scatter_dimension = -1;
    int64_t scatter_window_start = -1;
    int64_t scatter_batch_dimension = -1;
    int64_t scatter_indices_batch_dimension = -1;

    // Multi-dim scatter: operand shape and scatter-dim mapping for linearization
    bool scatter_add_normalized = false;
    std::vector<int64_t> scatter_operand_dims;
    std::vector<int64_t> scatter_dims_to_operand;
    std::vector<int64_t> scatter_update_window_dims;
    std::vector<int64_t> scatter_inserted_window_dims;
    int64_t scatter_index_vector_dim = -1;
    int64_t scatter_operand_batch_dim = -1;       // batch dim in operand (-1 = none)
    int64_t scatter_indices_batch_dim = -1;        // batch dim in indices (-1 = none)

    // Sort parameters
    bool sort_stable = false;
    bool sort_descending = false;

    // TopK parameters
    int64_t topk_k = -1;
    int64_t topk_dimension = -1;
    bool topk_largest = true;

    // Pad parameters
    std::vector<int64_t> pad_edge_padding_low;
    std::vector<int64_t> pad_edge_padding_high;
    std::vector<int64_t> pad_interior_padding;

    // Iota parameters
    int64_t iota_dimension = -1;

    // ReduceWindow parameters
    enum class ReduceWindowReducerKind {
      kMax,          // maximum reducer → aclnnMaxPool
      kCumsum,       // add reducer cumsum pattern → aclnnCumsum
      kUnsupported   // sum/min/other → report error
    };
    ReduceWindowReducerKind reduce_window_reducer_kind =
        ReduceWindowReducerKind::kUnsupported;

    int64_t cumsum_dim = -1;  // cumsum aggregation dimension (valid only for kCumsum)

    std::vector<int64_t> window_dimensions;   // kernel sizes per spatial dim
    std::vector<int64_t> window_strides;      // strides per spatial dim
    std::vector<int64_t> window_padding_low;  // padding_low per spatial dim
    std::vector<int64_t> window_padding_high; // padding_high per spatial dim
    std::vector<int64_t> window_dilations;    // window dilations per spatial dim

    // Cholesky parameters
    bool cholesky_is_lower = true;  // lower=true → L factor; lower=false → U factor

    // TriangularSolve parameters
    bool triangular_solve_upper = true;       // !lower
    bool triangular_solve_transpose = false;  // transpose_a == TRANSPOSE
    bool triangular_solve_unitriangular = false;  // unit_diagonal
    bool triangular_solve_left_side = true;   // left_side

    // Tril parameters
    bool is_tril = false;
    int64_t tril_diagonal = 0;

    // aclnnFmaAddcmul parameters. `value` is carried in the custom-call
    // opaque and materialized as an aclScalar by the ACLNN thunk.
    bool is_fma_addcmul = false;
    float fma_addcmul_value = 1.0f;

    // Gather parameters
    enum class GatherLoweringKind {
      kSimpleGatherV2,       // Single dimension gather + collapsed dim -> aclnnGatherV2 direct call
      kGatherV2WithView,     // GatherV2 + output view adaptation shape
      kMultiDimFlattenGather, // Multidimensional gather -> flatten + flat indices + GatherV2
      kUnsupported           // gather semantics not supported yet
    };
    GatherLoweringKind gather_lowering_kind =
        GatherLoweringKind::kUnsupported;
    std::vector<int64_t> gather_offset_dims;
    std::vector<int64_t> gather_collapsed_slice_dims;
    std::vector<int64_t> gather_start_index_map;
    int64_t gather_index_vector_dim = -1;
    std::vector<int64_t> gather_slice_sizes;
    // Batch dims (XLA batched gather). Stored so the runtime host-side
    // reference gather can handle batched gathers that no aclnn lowering
    // supports (e.g. multi-offset-dim batched gather from linalg.solve).
    std::vector<int64_t> gather_operand_batching_dims;
    std::vector<int64_t> gather_start_indices_batching_dims;
    int64_t gather_effective_dim = -1;       // The only non-trivial gathered dim (slice_size < operand_size)
    int64_t gather_effective_index_pos = -1; // The position of dim in start_index_map (used to extract the corresponding column from indices)
    int64_t gather_partial_slice_size = 0;   // Partial slice size (>1 and < dim_size), 0 means no partial slice
  };

  //===--------------------------------------------------------------------===//
  // ResourceRequests
  //===--------------------------------------------------------------------===//

  // Each individual thunk can request various resources required for execution
  // at prepare stage. XLA executable is responsible for allocating them before
  // initializing and executing thunks.
  class ResourceRequestsInterface {
   public:
    virtual ~ResourceRequestsInterface() = default;
  };

  //===--------------------------------------------------------------------===//
  // PrepareParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Prepare. At thunk prepare time we do not launch any
  // work or do any expensive initialization and only pass resource requirements
  // back to executable, i.e. request collective cliques required at run time.
  struct PrepareParams {};

  //===--------------------------------------------------------------------===//
  // InitializeParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to Initialize. At thunk initialization time we do not
  // launch any "work" on device and only initialize thunks for execution, i.e.
  // we pre-load kernels on device and instantiate all command buffers.
  struct InitializeParams {
    se::StreamExecutor* executor = nullptr;
    ExecutableSource src;

    const NpuBufferAllocations* buffer_allocations = nullptr;

    // Main compute stream that will be used, passed via `ExecuteParams` to
    // `ExecuteOnStream`. It can be used to initialize on-device "state" (i.e.
    // various control structures) at command buffer recording time (we use it
    // to initialize NCCL execution plans on device when we trace NCCL
    // operations into command buffers);
    se::Stream* stream = nullptr;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream = nullptr;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context = nullptr;

    // Total local device count.
    int local_device_count = 0;
  };

  //===--------------------------------------------------------------------===//
  // ExecuteParams
  //===--------------------------------------------------------------------===//

  // Parameters passed to ExecuteOnStream. ExecuteOnStream is responsible for
  // launching "work" on device, i.e. it launches kernels, executes command
  // buffers and calls into libraries (cuBLAS, cuDNN etc.).
  struct ExecuteParams {
    // Constructs execute parameters from an executable run options. Return
    // error if run options are misconfigured.
    static ExecuteParams Create(
        const ServiceExecutableRunOptions& run_options,
        const NpuBufferAllocations& buffer_allocations, se::Stream* stream,
        se::Stream* command_buffer_trace_stream,
        ExecutionStreamIdMap additional_compute_streams = {});

    // Constructs execute parameters from an existing parameters but with
    // different buffer allocations.
    static ExecuteParams CloneWithNewAllocations(
        const ExecuteParams& params,
        const NpuBufferAllocations& buffer_allocations);

    const NpuBufferAllocations* buffer_allocations;  // never null

    // Main compute stream on which thunks launch operations.
    se::Stream* stream;

    // Auxiliary stream for tracing command buffers. We use a separate stream to
    // avoid accidental tracing of unrelated activities on a main stream.
    se::Stream* command_buffer_trace_stream;

    // Streams for moving data between host and device.
    se::Stream* device_to_host_stream;
    se::Stream* host_to_device_stream;

    // Send/Recv callbacks passed to XLA from PjRt.
    SendDeviceMemoryFunction* send_device_memory_function;
    RecvDeviceMemoryFunction* recv_device_memory_function;

    // XLA FFI execution context.
    const ffi::ExecutionContext* ffi_execution_context;

    // Additional compute streams on which thunks launch operations.
    ExecutionStreamIdMap additional_compute_streams;

    bool mock_collectives = false;

    int64_t execution_id = 0;

   private:
    friend class CommandBufferThunk;

    ExecuteParams(const NpuBufferAllocations* buffer_allocations,
                  se::Stream* stream, se::Stream* command_buffer_trace_stream,
                  se::Stream* device_to_host_stream,
                  se::Stream* host_to_device_stream,
                  SendDeviceMemoryFunction* send_device_memory_function,
                  RecvDeviceMemoryFunction* recv_device_memory_function,
                  const ffi::ExecutionContext* ffi_execution_context,
                  ExecutionStreamIdMap additional_compute_streams = {},
                  bool mock_collectives = false, int64_t execution_id = 0);
  };

  //===--------------------------------------------------------------------===//

  NpuThunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind), thunk_info_(std::move(thunk_info)) {}
  virtual ~NpuThunk() = default;
  NpuThunk(const NpuThunk&) = delete;
  NpuThunk& operator=(const NpuThunk&) = delete;

  virtual std::string ToString(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  absl::string_view profile_annotation() const {
    return thunk_info_.profile_annotation;
  }
  const ThunkInfo& thunk_info() const { return thunk_info_; }

  // Prepares thunk for execution.
  // This may be called multiple times. Its main purpose is to pass resource
  // requests up to the parent executable so it can acquire them before
  // initialization and execution.
  virtual absl::Status Prepare(const PrepareParams& params,
                               ResourceRequestsInterface& resource_requests) {
    return absl::OkStatus();
  }

  // Initializes thunk for execution.
  // This may be called multiple times. Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  // Precondition: Prepare(initialize_params) has been called.
  virtual absl::Status Initialize(const InitializeParams& params) {
    return absl::OkStatus();
  }

  // Handle all special values (-inf, +inf, nan) in the constant buffer during the Initialize phase
  // These special values are only written when the graph is allocated and will not be generated during the calculation process, so they are processed uniformly during the initialization phase.
  // Parameters: params - initialization parameters, args - buffer slices of thunk
  absl::Status InitializeSpecialValues(
      const InitializeParams& params,
      const std::vector<BufferAllocation::Slice>& args);

  // Print kernel configuration information
  static void PrintKernelConfig(const NpuKernelConfig* kernel_config);

  // Print all computation graphs
  static void PrintComputationGraphs(
      const std::unordered_map<std::string, std::string>& computation_hlo_map,
      const std::string& entry_computation_name);

  // Print the HLO text of a single computation
  static void PrintComputationHlo(const std::string& hlo_text,
                                  const std::string& name, bool is_entry);

  // Unified stream synchronization and printing control functions
  // According to the environment variable ENABLE_SYNC_AFTER_EXECUTE, control whether to synchronize the stream after the operator is executed
  // Returns whether synchronization is executed
  static absl::Status SyncStreamIfEnabled(se::Stream* stream);

  // Unified parameter printing function
  // Print single parameter information (before or after execution)
  static void PrintParamInfo(int param_id, const std::string& param_type,
                             int device_ordinal,
                             const BufferAllocation::Slice& slice,
                             const Shape& shape, void* device_addr);

  // Check if IO value printing is enabled
  static bool IsPrintIOValuesEnabled();

  // Executes thunk on the given stream. This method must be called after
  // Initialize and can be called multiple times over NpuThunk's lifetime.
  // Precondition: Initialize(initialize_params) has been called.
  virtual absl::Status ExecuteOnStream(const ExecuteParams& params) = 0;

  using BufferUses = absl::InlinedVector<BufferUse, 4>;

  // Returns all device buffers used by the thunk.
  // Does not propagate buffers from nested thunks.
  // The order of the buffers in returned vector is consistent across calls.
  virtual BufferUses buffer_uses() const { return {}; }

  static absl::string_view KindToString(NpuThunk::Kind kind);

  NpuExecutionStreamId execution_stream_id() const {
    return thunk_info_.execution_stream_id;
  }
  void set_execution_stream_id(NpuExecutionStreamId execution_stream_id) {
    thunk_info_.execution_stream_id = execution_stream_id;
  }

  static absl::StatusOr<se::Stream*> GetStreamForExecution(
      NpuExecutionStreamId stream_id, const ExecuteParams& params);

  // Returns `true` if this thunk requires inter-GPU communication.
  bool IsCollective() const;

  // Returns any communicators used during execution.
  virtual absl::StatusOr<std::vector<Communicator*>> GetCommunicators(
      const ExecuteParams& params) const {
    return std::vector<Communicator*>();
  }

  // Invokes `fn` with this thunk and all nested thunks.
  virtual void ForAllThunks(absl::FunctionRef<void(const NpuThunk*)> fn) const;

  // Invokes `fn` with this thunk and all nested thunks.
  virtual void ForAllThunksMutable(absl::FunctionRef<void(NpuThunk*)> fn);

  // Recursively replaces all nested thunks with the result of applying `fn` to them.
  // An error will leave the transformation in invalid state.
  // InternalError should be used for status.
  virtual absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<NpuThunk>>(std::unique_ptr<NpuThunk>)>
          fn) {
    return absl::OkStatus();
  }

  void add_control_predecessor(const NpuThunk* control_predecessor) {
    control_predecessors_.push_back(control_predecessor);
  }

  std::vector<const NpuThunk*> control_predecessors() const {
    return control_predecessors_;
  }

  virtual std::optional<NpuAsyncEventsUniqueId> GetAsyncEventsUniqueId() const {
    return std::nullopt;
  }

  virtual bool IsAsyncStart() const { return false; }

  virtual bool IsAsyncDone() const { return false; }

 protected:
 private:
  Kind kind_;
  ThunkInfo thunk_info_;

  // The list of control predecessors of the thunk.
  // NpuThunk needs to maintain the control dependency information because
  // when it is executed by command buffer, and command buffer may execute the
  // sequence in concurrent mode, and we should make sure that it does not
  // violate the control dependency in the original computation.
  std::vector<const NpuThunk*> control_predecessors_;
};

// A sequence of thunks.
using NpuThunkSequence = std::vector<std::unique_ptr<NpuThunk>>;

std::ostream& operator<<(std::ostream& os, NpuThunk::Kind kind);

// Returns if the thunk implements a reduction collective (all-reduce or
// reduce-scatter).
bool IsReductionCollective(NpuThunk::Kind kind);

}  // namespace xla

#endif  // XLA_XLA_NPU_BACKEND_RUNTIME_NPU_THUNK_H_
