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

#ifndef XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
#define XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_

#include "xla/xla_npu/dependency/header/ge/ge_api.h"
#include "xla/xla_npu/dependency/header/metadef/graph/graph.h"

namespace xla {

enum class Placement {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
};

class GraphData {
 public:
  GraphData() {
    graph = nullptr;
    summary = nullptr;
  }
  ~GraphData() = default;
  uint32_t id = 0U;
  ge::GraphPtr graph = nullptr;
  // std::map<ge::AscendString, ge::AscendString> compile_options;
  // std::map<ge::AscendString, ge::AscendString> load_options;
  // std::vector<Placement> input_placements;
  std::vector<ge::Shape> inputs_shapes;
  std::vector<ge::DataType> input_dtypes;
  std::vector<ge::Shape> outputs_shapes;
  std::vector<ge::DataType> output_dtypes;
  std::shared_ptr<ge::CompiledGraphSummary> summary = nullptr;
};
}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
