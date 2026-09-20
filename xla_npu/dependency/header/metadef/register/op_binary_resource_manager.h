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

/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <map>
#include <tuple>
#include <vector>
#include <mutex>
#include "nlohmann/json.hpp"
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"

#ifndef INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_
#define INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_

namespace nnopbase {
// Binary content for the in-memory binary.o file description
struct Binary {
  const uint8_t *content;  // Starting pointer for binary content
  uint32_t len;            // binary length
};

class OpBinaryResourceManager {
 public:
  static OpBinaryResourceManager &GetInstance() {
    static OpBinaryResourceManager manager;
    return manager;
  }

  ~OpBinaryResourceManager() = default;

  void AddOpFuncHandle(const ge::AscendString &opType,
                       const std::vector<void *> &opResourceHandle);
  ge::graphStatus AddBinary(
      const ge::AscendString &opType,
      const std::vector<std::tuple<const uint8_t *, const uint8_t *>>
          &opBinary);
  ge::graphStatus AddRuntimeKB(
      const ge::AscendString &opType,
      const std::vector<std::tuple<const uint8_t *, const uint8_t *>>
          &opRuntimeKb);

  // Get resources
 public:
  // Get description information of all operators
  const std::map<const ge::AscendString, nlohmann::json> &GetAllOpBinaryDesc()
      const;

  // Get description information of an operator
  ge::graphStatus GetOpBinaryDesc(const ge::AscendString &opType,
                                  nlohmann::json &binDesc) const;

  // Find.JSON/.o information based on the JSON file path (existing in operator JSON)
  ge::graphStatus GetOpBinaryDescByPath(
      const ge::AscendString &jsonFilePath,
      std::tuple<nlohmann::json, Binary> &binInfo) const;

  // Find.JSON/.o information based on simplifiedKey (existing in operator JSON)
  ge::graphStatus GetOpBinaryDescByKey(
      const ge::AscendString &simplifiedKey,
      std::tuple<nlohmann::json, Binary> &binInfo) const;

  // Binary knowledge base
  ge::graphStatus GetOpRuntimeKB(const ge::AscendString &opType,
                                 std::vector<ge::AscendString> &kbList) const;

 private:
  OpBinaryResourceManager() = default;  // Singleton, prohibiting external creation of objects
  OpBinaryResourceManager &operator=(const OpBinaryResourceManager &) =
      delete;  // Copying prohibited
  OpBinaryResourceManager &operator=(OpBinaryResourceManager &&) = delete;
  OpBinaryResourceManager(const OpBinaryResourceManager &) = delete;
  OpBinaryResourceManager(OpBinaryResourceManager &&) = delete;

  mutable std::recursive_mutex mutex_;

  // Binary description information opType -> xxx.JSON
  std::map<const ge::AscendString, nlohmann::json> opBinaryDesc_;

  // Binary jsonPath simplifiedKey -> jsonPath, there may be multiple simplifiedKey corresponding to the same jsonPath
  std::map<const ge::AscendString, ge::AscendString> keyToPath_;

  // Binary information jsonPath -> xxx.JSON, xxx.o
  std::map<const ge::AscendString, std::tuple<nlohmann::json, Binary>>
      pathToBinary_;

  // Binary knowledge base opType -> xxx1.JSON, xxx2.JSON
  std::map<const ge::AscendString, std::vector<ge::AscendString>> runtimeKb_;

  // Infershape/op tiling/runtime kb parser and other global variable pointers, registered resources, only need to hold
  std::map<const ge::AscendString, std::vector<void *>> resourceHandle_;
};
}  // namespace nnopbase

#endif  // INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_
