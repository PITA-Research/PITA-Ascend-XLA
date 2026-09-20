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

#include "xla/xla_npu/stream_executor/npu_platform.h"

#include <csignal>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/stream_executor/npu_executor.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace stream_executor {
namespace {

volatile sig_atomic_t g_acl_initialized = 0;

void AclCleanup() {
  if (g_acl_initialized) {
    aclError ret = ::xla::dep::aclFinalize();
    if (ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR, "ACL finalize failed in atexit, ret: %d",
                           ret);
    } else {
      NPU_LOG_FORMAT_PRINT(INFO, "ACL finalize success in atexit");
    }
    g_acl_initialized = 0;
  }
}

}  // namespace

NpuPlatform::NpuPlatform() : name_(xla::NpuConstant::platformName) {}

NpuPlatform::~NpuPlatform() {
  NPU_LOG_FORMAT_PRINT(INFO, "Finalize platform for '%s'.", name_);
  if (g_acl_initialized) {
    auto ret = ::xla::dep::aclFinalize();
    if (ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(ERROR,
                           "Finalize platform for '%s' failed, error code %d",
                           name_, ret);
    } else {
      NPU_LOG_FORMAT_PRINT(INFO, "Finalize platform for '%s' success.", name_);
    }

    g_acl_initialized = 0;
  }
}

Platform::Id NpuPlatform::id() const { return kNpuPlatformId; }

bool NpuPlatform::Initialized() const { return g_acl_initialized; }

absl::Status NpuPlatform::Initialize() {
  const char* acl_config_json_file_path = nullptr;
  aclError acl_ret = ::xla::dep::aclInit(acl_config_json_file_path);
  if (acl_ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Init platform for '%s' failed.", name_);
    return absl::UnavailableError("Acl init failed.");
  }

  g_acl_initialized = 1;
  std::atexit(AclCleanup);

  NPU_LOG_FORMAT_PRINT(INFO, "Init platform for '%s' success.", name_);
  return absl::OkStatus();
}

int NpuPlatform::VisibleDeviceCount() const {
  uint32_t device_count = 0;
  aclError acl_ret = ::xla::dep::aclrtGetDeviceCount(&device_count);
  if (acl_ret != ACL_SUCCESS) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Get device count for '%s' failed.", name_);
    return 0;
  }

  NPU_LOG_FORMAT_PRINT(
      INFO, "Get device count for '%s' success, total device num is %d.", name_,
      device_count);
  return device_count;
}

const std::string& NpuPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
NpuPlatform::DescriptionForDevice(int ordinal) const {
  return NpuExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> NpuPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
NpuPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<NpuExecutor>(this, ordinal);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing StreamExecutor for device ordinal %d: %s", ordinal,
        init_status.ToString().c_str()));
  }

  return std::move(executor);
}

}  // namespace stream_executor
