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

#include "xla/xla_npu/client/npu_module.h"

#include "xla/service/computation_placer.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/npu_compiler.h"
#include "xla/xla_npu/service/npu_transfer_manager.h"
#include "xla/xla_npu/stream_executor/npu_platform.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

static void RegisterNpuPlatform() {
  // Used to execute processes.
  // After registering for this platform,
  // If the calling interface (ClientLibrary::GetOrCreateLocalClient) uses this platform,
  // It will trigger the creation and association of LocalService and Backend, where Backend is bound to LocalService.
  // The actual execution logic will be called through the two intermediate classes of LocalService (Backend) in LocalClient.
  // Take a typical execution call chain as an example
  //   'PjRtStreamExecutorClient'->RunAsync
  //     'LocalExecutable'->RunAsync
  //     The LocalExecutable here is generated after compilation of PjRtStreamExecutorClient, which actually encapsulates NpuExecutable.
  //       'NpuExecutable'->ExecuteAsyncOnStreamWrapper
  //         'NpuExecutable'->'ExecuteAsyncOnStream'
  //           'NpuExecutable'->'ExecuteThunks'
  std::unique_ptr<stream_executor::Platform> platform(
      new stream_executor::NpuPlatform);
  stream_executor::PlatformManager::RegisterPlatform(std::move(platform));
  NPU_LOG_FORMAT_PRINT(INFO, "Resister platform for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuCompiler() {
  // Used in the compilation process.
  // In PjRtStreamExecutorClient::CompileInternal, client()->Compile will be called,
  // The actual compilation logic will be called through the two intermediate classes of LocalService (Backend) in LocalClient.
  // LocalClient is the client() above, which is the following call chain
  //   'LocalClient'->Compile
  //     'LocalService'->CompileExecutables
  //       'LocalService'->BuildExecutable
  //         'Backend'->'NpuCompiler'->RunHloPasses
  //         'Backend'->'NpuCompiler'->RunBackend
  Compiler::RegisterCompilerFactory(stream_executor::kNpuPlatformId, []() {
    return std::make_unique<NpuCompiler>(stream_executor::kNpuPlatformId, "",
                                         "");
  });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister compiler for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuTransferManager() {
  // for dynamic data input/output
  TransferManager::RegisterTransferManager(
      stream_executor::kNpuPlatformId,
      []() -> std::unique_ptr<xla::TransferManager> {
        return std::make_unique<NpuTransferManager>();
      });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister transfer manager for '%s'.",
                       NpuConstant::platformName);
}

static void RegisterNpuComputationPlacer() {
  // Same as cpu/gpu, using public classes
  // Used to allocate computing tasks to multiple cards
  ComputationPlacer::RegisterComputationPlacer(
      stream_executor::kNpuPlatformId,
      []() { return std::make_unique<ComputationPlacer>(); });
  NPU_LOG_FORMAT_PRINT(INFO, "Resister computation placer for '%s'.",
                       NpuConstant::platformName);
}

void InitNpuSubModule() {
  NpuConfig::GetInstance().Init();

  RegisterNpuPlatform();
  RegisterNpuCompiler();
  RegisterNpuTransferManager();
  RegisterNpuComputationPlacer();
}

}  // namespace xla