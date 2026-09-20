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

#ifndef XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_
#define XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology.h"
#include "xla/xla_npu/pjrt/plugin/npu_topology_description.h"

namespace xla {

class NpuPjrtStreamExecutorClient : public PjRtStreamExecutorClient {
 public:
  NpuPjrtStreamExecutorClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index,
      std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tsl::Allocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options);

  ~NpuPjrtStreamExecutorClient() override = default;

  // Gets the pointer to the topology description held by the client.
  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_description_;
  }

 private:
  // It is only used to provide context information for dump. It is used at the beginning of PjRtStreamExecutorClient::Compile. It has no other use.
  NpuTopologyDescription topology_description_;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_NPU_NPU_PJRT_STREAM_EXECUTOR_CLIENT_H_
