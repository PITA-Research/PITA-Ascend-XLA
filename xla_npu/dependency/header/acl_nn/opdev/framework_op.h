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

#ifndef OP_API_OP_API_COMMON_INC_OPDEV_FRAMEWORK_OP_H
#define OP_API_OP_API_COMMON_INC_OPDEV_FRAMEWORK_OP_H
#include "aclnn/acl_meta.h"

namespace op {
const aclTensor *CopyToNpu(const aclTensor *src, aclOpExecutor *executor);
const aclTensor *CopyToNpuSync(const aclTensor *src, aclOpExecutor *executor);
aclnnStatus CopyNpuToNpu(const aclTensor *src, const aclTensor *dst,
                         aclOpExecutor *executor);
}  // namespace op
#endif  // OP_API_OP_API_COMMON_INC_OPDEV_FRAMEWORK_OP_H
