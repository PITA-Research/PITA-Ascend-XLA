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

#include "xla/xla_npu/stream_executor/npu_stream_factory.h"

#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"

namespace {

static absl::Mutex* get_npu_stream_factory_lock() {
  static absl::Mutex npu_stream_factory_lock(absl::kConstInit);
  return &npu_stream_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<stream_executor::NpuStreamFactory> factory;
  int priority = -1;
};

FactoryItem& npu_stream_factory() {
  static FactoryItem* const factory = new FactoryItem();
  return *factory;
}

}  // namespace

namespace stream_executor {

// static
void NpuStreamFactory::Register(std::unique_ptr<NpuStreamFactory> factory,
                                int priority) {
  absl::MutexLock l(*get_npu_stream_factory_lock());
  FactoryItem& factory_item = npu_stream_factory();
  if (factory_item.factory == nullptr) {
    factory_item.factory = std::move(factory);
    factory_item.priority = priority;
    return;
  }
  if (factory_item.priority < priority) {
    factory_item.factory = std::move(factory);
    factory_item.priority = priority;
  }
}

// static
const NpuStreamFactory* NpuStreamFactory::GetFactory() {
  absl::ReaderMutexLock l(*get_npu_stream_factory_lock());
  FactoryItem& factory_item = npu_stream_factory();
  return factory_item.factory.get();
}

}  // namespace stream_executor

REGISTER_NPU_STREAM_FACTORY(stream_executor::NpuStreamDefaultFactory, 100);
