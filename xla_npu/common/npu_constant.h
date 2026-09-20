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

#ifndef XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_
#define XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_

#include <string>

#include "xla/codegen/emitters/kernel_arguments.h"

namespace xla {

// NPU constant configuration class
class NpuConstant {
 public:
  static const std::string platformName;

  // Minimum alignment for buffers passed as incoming arguments by TensorFlow.
  static constexpr int64_t kEntryParameterAlignBytes = 512;

  // Minimum alignment for buffers allocated by XLA: the temp buffers and the
  // live
  static constexpr int64_t kXlaAllocatedBufferAlignBytes = 512;

  // Minimum alignment for constant buffers.
  static constexpr int64_t kConstantBufferAlignBytes =
      kXlaAllocatedBufferAlignBytes;

  static emitters::KernelArguments::BufferAlignment
  GetDefaultBufferAlignment() {
    emitters::KernelArguments::BufferAlignment buffer_alignment;
    buffer_alignment.entry_parameter_align_bytes = kEntryParameterAlignBytes;
    buffer_alignment.xla_allocated_buffer_align_bytes =
        kXlaAllocatedBufferAlignBytes;
    buffer_alignment.constant_buffer_align_bytes = kConstantBufferAlignBytes;

    return buffer_alignment;
  }

  // The key name of the Computation used to mark the AFIR entry, used when converting the reduce operation
  static const std::string entryComputationKeyName;
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_CONSTANT_H_