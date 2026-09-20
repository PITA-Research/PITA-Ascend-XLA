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

#include "xla/xla_npu/stream_executor/npu_auto_fuse_kernel.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_npu/backend/runtime/npu_workspace_memory_pool.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/config_loader.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace stream_executor {

static const std::string get_tiling_size_func_name = "GetTilingDataSize";
static const std::string tiling_func_name = "AutofuseTiling";
static const std::string launch_func_name = "AutofuseLaunchV2";

absl::Status NpuAutoFuseKernel::Load() {
  // Register lib
  const std::string& tiling_file_path =
      tsl::io::JoinPath(load_file_base_dir_, GetTilingFileName());
  const std::string& kernel_file_path =
      tsl::io::JoinPath(load_file_base_dir_, GetKernelFileName());
  if (registry.count(GetTilingFileName()) == 0) {
    registry[GetTilingFileName()] =
        std::make_unique<::xla::dep::FunctionLoader>(tiling_file_path, ".so");
  }
  if (registry.count(GetKernelFileName()) == 0) {
    registry[GetKernelFileName()] =
        std::make_unique<::xla::dep::FunctionLoader>(kernel_file_path, ".so");
  }

  // Register func
  registry[GetTilingFileName()]->Set(get_tiling_size_func_name);
  registry[GetTilingFileName()]->Set(tiling_func_name);
  registry[GetKernelFileName()]->Set(launch_func_name);

  // Get func
  GetTilingSizeFunc_ = reinterpret_cast<GetTilingSizeFunc>(
      registry[GetTilingFileName()]->Get(get_tiling_size_func_name));
  CHECK_COND_AND_PRINT_NPU_LOG(GetTilingSizeFunc_ == nullptr,
                               "Get func %s from %s failed",
                               get_tiling_size_func_name, GetTilingFileName());

  TilingFunc_ = reinterpret_cast<TilingFunc>(
      registry[GetTilingFileName()]->Get(tiling_func_name));
  CHECK_COND_AND_PRINT_NPU_LOG(TilingFunc_ == nullptr,
                               "Get func %s from %s failed", tiling_func_name,
                               GetTilingFileName());

  LaunchFunc_ = reinterpret_cast<LaunchFunc>(
      registry[GetKernelFileName()]->Get(launch_func_name));
  CHECK_COND_AND_PRINT_NPU_LOG(LaunchFunc_ == nullptr,
                               "Get func %s from %s failed", launch_func_name,
                               GetKernelFileName());

  return absl::OkStatus();
}

absl::Status NpuAutoFuseKernel::Launch(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, Stream* stream,
    const KernelArgs& args) {
  // thread_dims, cluster_dims not use
  NPU_LOG_FORMAT_PRINT(DEBUG, "NpuAutoFuseKernel::Launch called");

  CHECK_COND_AND_PRINT_NPU_LOG(
      stream == nullptr,
      "Stream need to be created or init before execute npu thunk.");
  auto stream_handle = stream->platform_specific_handle();
  CHECK_COND_AND_PRINT_NPU_LOG(
      stream_handle.stream == nullptr,
      "Acl stream need to be created or init before execute npu thunk.");
  auto acl_stream = stream_handle.stream;
  NPU_LOG_FORMAT_PRINT(DEBUG, "  acl_stream=%p", acl_stream);

  size_t tiling_size = GetTilingSizeFunc_();
  std::vector<uint8_t> tiling_data_holder(tiling_size);
  void* tiling_data = static_cast<void*>(tiling_data_holder.data());

  uint32_t workspace_size = 0;
  uint32_t block_dim;
  uint32_t auto_fuse_ret =
      TilingFunc_(tiling_data, &workspace_size, &block_dim, nullptr);
  CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0,
                               "Tiling failed, error code: %d.", auto_fuse_ret);
  NPU_LOG_FORMAT_PRINT(DEBUG, "  tiling_size=%zu", tiling_size);
  NPU_LOG_FORMAT_PRINT(DEBUG, "  TilingFunc_ workspace_size=%u, block_dim=%u",
                       workspace_size, block_dim);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    auto* pool = xla::NpuWorkspaceMemoryPool::GetInstance();
    CHECK_COND_AND_PRINT_NPU_LOG(pool == nullptr,
                                 "Workspace memory pool not available.");

    auto alloc_result = pool->AllocateForStream(acl_stream, workspace_size);
    CHECK_COND_AND_PRINT_NPU_LOG(!alloc_result.ok(),
                                 "Failed to allocate workspace from pool: %s",
                                 alloc_result.status().ToString().c_str());
    workspace = alloc_result.value();
    NPU_LOG_FORMAT_PRINT(DEBUG, "  workspace allocated from pool: %p, size=%u",
                         workspace, workspace_size);
  } else {
    NPU_LOG_FORMAT_PRINT(DEBUG, "  workspace_size=%u, no workspace allocated",
                         workspace_size);
  }

  // Refer to cuda_kernel and use the pack function passed in KernelLoaderSpec to encapsulate parameters.
  if (auto* args_ptr = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "  args_ptr=%p, number_of_arguments=%zu",
                         args_ptr, args_ptr->number_of_arguments());
    int32_t input_param_num = args_ptr->number_of_arguments() - 1;
    int32_t output_param_num = 1;  // Currently written to 1
    // Create a contiguous array of input parameter pointers, ensuring 8-byte alignment
    std::vector<void*> input_param_addrs(input_param_num);
    for (int32_t i = 0; i < input_param_num; ++i) {
      input_param_addrs[i] = const_cast<void*>(args_ptr->device_memory_ptr(i));
    }
    void* output_param_addr = const_cast<void*>(args_ptr->device_memory_ptr(
        args_ptr->number_of_arguments() - 1));  // The default is to take the last one
    NPU_LOG_FORMAT_PRINT(DEBUG, "  input_param_num=%d, output_param_num=%d",
                         input_param_num, output_param_num);
    NPU_LOG_FORMAT_PRINT(DEBUG, "  input_param_addrs=%p, output_param_addr=%p",
                         input_param_addrs.data(), output_param_addr);
    NPU_LOG_FORMAT_PRINT(DEBUG, "  tiling_data=%p", tiling_data);

    auto_fuse_ret = LaunchFunc_(block_dim, acl_stream, input_param_addrs.data(),
                                input_param_num, &output_param_addr,
                                output_param_num, workspace, tiling_data);
    CHECK_COND_AND_PRINT_NPU_LOG(auto_fuse_ret != 0,
                                 "Launch fusion kernel failed, error code: %d.",
                                 auto_fuse_ret);
  } else {
    NPU_LOG_FORMAT_PRINT(ERROR, "Get args failed.");
    return absl::InternalError("Get args failed.");
  }

  // Control whether to synchronize after operator execution based on environment variables
  if (NPU_CONFIG_GET_BOOL(xla::NpuConfig::Key::ENABLE_SYNC_AFTER_EXECUTE)) {
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    NPU_LOG_FORMAT_PRINT(DEBUG, "  Sync after execute enabled");
  }

  // The workspace memory is managed uniformly by the memory pool and does not need to be released here.

  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "  NpuAutoFuseKernel::Launch completed successfully");
  return absl::OkStatus();
}

}  // namespace stream_executor
