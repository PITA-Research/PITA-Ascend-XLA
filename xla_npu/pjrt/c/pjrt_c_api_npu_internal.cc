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

#include "pjrt_c_api_npu_internal.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"

#if NPU_AUTO_FUSE_BACKEND == 3
#include "xla/xla_npu/pjrt/air/npu/npu_client.h"
#else
#include "xla/xla_npu/client/npu_stream_client.h"
#endif

namespace pjrt {

namespace npu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
#if NPU_AUTO_FUSE_BACKEND == 3
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetPjRtNpuClient());
  args->client = CreateWrapperClient(std::move(client));
  return nullptr;
#else
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetNpuPjrtStreamExecutorClient());
  args->client = CreateWrapperClient(std::move(client));
  return nullptr;
#endif
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_NpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{
      absl::UnimplementedError("Topology not supported for CPU compilation.")};
}

const PJRT_Api* GetNpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&layouts_extension.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&memory_descriptions_extension.base);

  static PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::CreatePhaseCompileExtension(&ffi_extension.base,
                                        /*get_compiler=*/nullptr,
                                        /*destroy_compiler=*/nullptr);

  static const PJRT_Api pjrt_api =
      CreatePjrtApi(PJRT_Client_Create, PJRT_ExecuteContext_Create,
                    PJRT_NpuDeviceTopology_Create, PJRT_Plugin_Initialize_NoOp,
                    &phase_compile_extension.base, PJRT_Plugin_Attributes_Xla);

  return &pjrt_api;
}

}  // namespace npu_plugin

}  // namespace pjrt
