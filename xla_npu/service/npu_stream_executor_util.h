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

#ifndef XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_
#define XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/layout.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
absl::Mutex& GetNpuMutex(const se::StreamExecutor* stream_exec);
}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_STREAM_EXECUTOR_UTIL_H_
