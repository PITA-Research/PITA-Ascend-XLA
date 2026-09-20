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

#ifndef XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
#define XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_

#include "xla/xla_npu/stream_executor/npu_stream.h"

namespace stream_executor {

class NpuStreamFactory {
 public:
  virtual ~NpuStreamFactory() = default;
  virtual absl::StatusOr<std::unique_ptr<NpuStream>> CreateStream(
      StreamExecutor* executor) const = 0;
  static void Register(std::unique_ptr<NpuStreamFactory> factory, int priority);
  static const NpuStreamFactory* GetFactory();
};

class NpuStreamDefaultFactory : public NpuStreamFactory {
 public:
  ~NpuStreamDefaultFactory() override = default;
  absl::StatusOr<std::unique_ptr<NpuStream>> CreateStream(
      StreamExecutor* executor) const override {
    auto npuStream = std::make_unique<NpuStream>(executor);
    TF_RETURN_IF_ERROR(npuStream->Init());
    return std::move(npuStream);
  }
};

template <class Factory>
class NpuStreamFactoryRegistrar {
 public:
  explicit NpuStreamFactoryRegistrar(int priority) {
    NpuStreamFactory::Register(std::make_unique<Factory>(), priority);
  }
};

}  // namespace stream_executor

#define REGISTER_NPU_STREAM_FACTORY(factory, priority) \
  INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, __COUNTER__)

#define INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, ctr) \
  static ::stream_executor::NpuStreamFactoryRegistrar<factory>       \
  INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) {             \
    priority                                                         \
  }

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) ___##ctr##__object_

#endif  // XLA_XLA_NPU_STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
