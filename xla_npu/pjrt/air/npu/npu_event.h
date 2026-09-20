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

#ifndef NPU_EVENT_H_
#define NPU_EVENT_H_
namespace xla {

// Typical use case: AsyncValueRef<NpuEvent> is used to indicate the
// completion of a CPU operation, e.g., data transfer or running a program.
struct NpuEvent {
  NpuEvent() = default;
};

}  // namespace xla
#endif  //NPU_EVENT_H_
