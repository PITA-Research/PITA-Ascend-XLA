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

#ifndef NPU_CLIENT_H_
#define NPU_CLIENT_H_

#include <csignal>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "tsl/platform/fingerprint.h"

#include "xla/xla_npu/pjrt/air/pjrt_common.h"
#include "xla/xla_npu/pjrt/air/npu/npu_device.h"
#include "xla/xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h"
#include "xla/xla_npu/pjrt/air/npu/tracked_npu_device_buffer.h"
#include "xla/xla_npu/pjrt/air/npu/ge/graph_data.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"
#include "xla/xla_npu/pjrt/plugin/npu_device_description.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/header/ge/ge_api.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {
static std::mutex finalize_mutex;
static volatile sig_atomic_t is_initialized = 0;

inline const char* DataLayout() {
  static constexpr char kDataLayout[] =
      "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64";
  return kDataLayout;
}

class PjRtNpuClient final : public CommonPjRtClient {
 public:
  PjRtNpuClient(int process_index,
                std::vector<std::unique_ptr<PjRtNpuDevice>> devices);

  ~PjRtNpuClient() override = default;

  using CollectiveLaunchEvent =
      std::pair<tsl::AsyncValueRef<NpuEvent>,
                tsl::CountDownAsyncValueRef<NpuEvent>>;

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  PjRtPlatformId platform_id() const override {
    return tsl::Fingerprint64(NpuName());
  }

  absl::string_view platform_name() const override { return NpuName(); }

  absl::string_view platform_version() const override { return NpuName(); }

  // Gets the pointer to the topology description held by the client.
  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

  AsyncWorkRunner* async_work_runner() const override {
    return async_work_runner_.get();
  }

  tsl::thread::ThreadPool* pjrt_client_thread_pool() const {
    return pjrt_client_thread_pool_.get();
  }

  ge::Session* GetGlobalGeSession() const { return global_ge_session_.get(); }

  // Returns a pair of async events:
  // - async event that signals the completion of the last collective launch
  // - count down event that must be signalled when each rank completes
  //   a collective launch
  CollectiveLaunchEvent GetLastCollectiveLaunchEvent(
      size_t num_addressable_devices) {
    tsl::CountDownAsyncValueRef<NpuEvent> count_down(num_addressable_devices);
    absl::MutexLock lock(mu_);
    auto last_launch = std::move(last_collective_launch_event_);
    last_collective_launch_event_ = count_down.AsRef();
    return std::make_pair(std::move(last_launch), std::move(count_down));
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  // Computes the memory requirements for storing shape on memory_space.
  absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const override;

  // Allocates a raw buffer of a particular size after an optional
  // allocate_after. Backends may support retrying allocation on oom which
  // can be controlled via retry_on_oom.
  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

  // Defines a pjrt buffer from a shape, raw_buffer and definition events.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events,
      bool raw_buffer_is_mutable) override;

  absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                           tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options, GraphData& graph_data,
      const AotCompilationOptions* absl_nullable aot_options = nullptr);

 private:
  friend class PjRtNpuExecutable;
  friend absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient();

  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<PjRtNpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<PjRtGlobalDeviceId, PjRtNpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;
  NpuTopologyDescription topology_;

  mutable absl::Mutex mu_;
  tsl::AsyncValueRef<NpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  // Thread pool for running PjRtClient tasks.
  std::unique_ptr<tsl::thread::ThreadPool> pjrt_client_thread_pool_;
  std::unique_ptr<AsyncWorkRunner> async_work_runner_;

  bool ge_initialized_ = false;
  std::unique_ptr<ge::Session> global_ge_session_ = nullptr;
  int64_t global_graph_id = 0;
};

class PjRtNpuExecutable final : public PjRtLoadedExecutable {
 public:
  PjRtNpuExecutable(
      int num_replicas, int num_partitions,
      std::shared_ptr<DeviceAssignment> device_assignment,
      bool parameter_is_tupled_arguments, CompileOptions compile_options,
      std::unique_ptr<Executable> npu_executable,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, PjRtNpuClient* client,
      std::unique_ptr<HloModule> unoptimized_hlo_module);

  ~PjRtNpuExecutable() override {
    std::lock_guard<std::mutex> lock(finalize_mutex);
    if (is_initialized <= 0) {
      return;
    }
    aclError ge_ret = ge::GEFinalize();
    if (ge_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR, "GE finalize failed, ret: %d", ge_ret);
    } else {
      NPU_LOG_FORMAT_PRINT(DEBUG, "GE finalize success");
    }
    aclError acl_ret = dep::aclFinalize();
    if (acl_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR, "ACL finalize failed, ret: %d", acl_ret);
    } else {
      NPU_LOG_FORMAT_PRINT(DEBUG, "ACL finalize success");
    }
    is_initialized = 0;
  };

  PjRtNpuClient* client() const override { return client_; }

  absl::string_view name() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable name: ");
    return npu_executable_->shared_module()->name();
  }

  int num_replicas() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable num_replicas: ");
    return num_replicas_;
  }

  int num_partitions() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable num_partitions: ");
    return num_partitions_;
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable SizeOfGeneratedCodeInBytes: ");
    return npu_executable_->SizeOfGeneratedCodeInBytes();
  }

  const DeviceAssignment& device_assignment() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "device_assignment: ");
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "npu executable addressable_device_logical_ids: ");
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable addressable_devices: ");
    return addressable_devices_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable GetHloModules: ");
    return std::vector<std::shared_ptr<HloModule>>{
        npu_executable_->shared_module()};
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    NPU_LOG_FORMAT_PRINT(DEBUG, "npu executable GetOutputMemoryKinds: ");
    return Unimplemented("GetOutputMemoryKinds is not supported.");
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<Future<>>>& returned_futures) const override;

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override;

  bool IsDeleted() const override;

  absl::StatusOr<std::string> SerializeExecutable() const override;

  std::shared_ptr<Executable> npu_executable() const { return npu_executable_; }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const {
    return fingerprint_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

 private:
  friend class PjRtNpuClient;

  // absl::Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  // absl::Status CheckBufferCompatibilities(
  //     absl::Span<std::pair<bool, TrackedNpuDeviceBuffer*> const> input_buffers)
  //     const;

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      PjRtNpuClient::CollectiveLaunchEvent last_collective_launch_event,
      bool fill_future, PjRtNpuDevice* device = nullptr) const;

  PjRtNpuClient* client_;

  int num_replicas_;
  int num_partitions_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  bool parameter_is_tupled_arguments_;
  CompileOptions compile_options_;

  std::shared_ptr<Executable> npu_executable_;

  // Caching `result_buffer_indices_` to avoid lookup
  // HLO dataflow analysis data structures in program execution critical path.

  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;

  std::vector<int64_t> input_buffer_sizes_in_bytes_;

  std::vector<int> parameters_that_must_be_donated_;

  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  std::vector<PjRtDevice*> addressable_devices_;

  bool cheap_computation_;

  std::string fingerprint_;

  std::unique_ptr<HloModule> unoptimized_hlo_module_;
};

inline absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtNpuClient() {
  NPU_LOG_FORMAT_PRINT(INFO, "NPU PJRT_CLIENT_CREATE");
  int device_id = 0;
  int process_index = 0;
  std::vector<std::unique_ptr<xla::PjRtNpuDevice>> devices;

  aclError acl_init = dep::aclInit(nullptr);
  if (acl_init != ACL_SUCCESS) {
    return absl::InternalError("ACL init failed.");
  }
  uint32_t device_count = -1;
  aclError acl_device_count = dep::aclrtGetDeviceCount(&device_count);
  if (acl_device_count != ACL_SUCCESS) {
    return absl::InternalError("ACL get device count failed.");
  }
  if (device_count <= 0) {
    return absl::InternalError("Threre is no valid device.");
  }
  aclError acl_set_device = dep::aclrtSetDevice(device_id);
  if (acl_set_device != ACL_SUCCESS) {
    return absl::InternalError("ACL set device failed.");
  }

  devices.push_back(
      std::make_unique<xla::PjRtNpuDevice>(process_index, device_id));
  return std::unique_ptr<PjRtClient>(
      new PjRtNpuClient(process_index, std::move(devices)));
}

}  // namespace xla
#endif  //NPU_CLIENT_H_