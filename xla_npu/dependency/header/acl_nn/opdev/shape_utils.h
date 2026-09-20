/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_OPDEV_SHAPE_UTILS_H_
#define OP_API_OP_API_COMMON_INC_OPDEV_SHAPE_UTILS_H_

#include <cstdlib>
#include <string>
#include <exe_graph/runtime/shape.h>
#include "common_types.h"
#include "data_type_utils.h"

namespace op {
ge::AscendString ToString(const op::Shape &shape);
ge::AscendString ToString(const op::Strides &strides);
void ToShape(const int64_t *dims, uint64_t dimNum, op::Shape &shape);
void ToShape(const op::ShapeVector &shapeVector, op::Shape &shape);
ShapeVector ToShapeVector(const op::Shape &shape);
void ToContiguousStrides(const op::Shape &shape, op::Strides &strides);
bool CheckBroadcastShape(const op::Shape &self, const op::Shape &other);
bool BroadcastInferShape(const op::Shape &self, const op::Shape &other,
                         op::Shape &broadcastShape);
}  // namespace op

#endif  //OP_API_OP_API_COMMON_INC_OPDEV_SHAPE_UTILS_H_
