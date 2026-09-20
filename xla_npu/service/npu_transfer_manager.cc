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

#include "xla/xla_npu/service/npu_transfer_manager.h"

#include <memory>

#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_npu/stream_executor/npu_platform_id.h"

namespace xla {

NpuTransferManager::NpuTransferManager()
    : GenericTransferManager(stream_executor::kNpuPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

absl::Status NpuTransferManager::TransferLiteralToInfeed(
    stream_executor::StreamExecutor* executor, const LiteralSlice& literal) {
  return absl::UnimplementedError(
      "NpuTransferManager::TransferLiteralToInfeed is not supported.");
}

absl::Status NpuTransferManager::TransferLiteralFromOutfeed(
    stream_executor::StreamExecutor* executor,
    MutableBorrowingLiteral literal) {
  return absl::UnimplementedError(
      "NpuTransferManager::TransferLiteralFromOutfeed is not supported.");
}

absl::Status NpuTransferManager::ReadDynamicShapes(
    stream_executor::Stream* stream, const ShapedBuffer* device_buffer,
    Shape* device_shape) {
  return absl::UnimplementedError(
      "NpuTransferManager::ReadDynamicShapes is not supported.");
}

}  // namespace xla
