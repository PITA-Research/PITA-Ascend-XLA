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

#include "xla/xla_npu/dependency/header/metadef/graph/operator_factory.h"

namespace ge {
Operator OperatorFactory::CreateOperator(const char* operator_name,
                                         const char* operator_type) {
  return Operator();
}

Operator OperatorFactory::CreateOperator(const std::string& operator_name,
                                         const std::string& operator_type) {
  return Operator();
}

OperatorCreatorRegister::OperatorCreatorRegister(
    const char* operator_type, const OpCreatorV2& op_creator) {}

bool OperatorFactory::IsExistOp(const char* operator_type) { return true; }
}  // namespace ge
