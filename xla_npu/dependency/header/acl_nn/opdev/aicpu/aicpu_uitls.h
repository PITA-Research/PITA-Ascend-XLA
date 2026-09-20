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

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 */

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_AICPU_UTILS_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_AICPU_UTILS_H_

#include <mutex>
#include <time.h>
#include "aclnn/aclnn_base.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace op {
namespace internal {
#define AICPU_ASSERT(exp)                                 \
  do {                                                    \
    if (!(exp)) {                                         \
      OP_LOGE(ACLNN_ERR_INNER, "Assert %s failed", #exp); \
      return;                                             \
    }                                                     \
  } while (false)

#define AICPU_ASSERT_RETVAL(exp, ret)         \
  do {                                        \
    if (!(exp)) {                             \
      OP_LOGE(ret, "Assert %s failed", #exp); \
      return (ret);                           \
    }                                         \
  } while (false)

#define AICPU_ASSERT_OK_RETVAL(v) AICPU_ASSERT_RETVAL(((v) == OK), (v))
#define AICPU_ASSERT_RTOK_RETVAL(v) \
  AICPU_ASSERT_RETVAL(((v) == 0), (ACLNN_ERR_RUNTIME_ERROR))
#define AICPU_ASSERT_NOTNULL_RETVAL(v) \
  AICPU_ASSERT_RETVAL(((v) != nullptr), (ACLNN_ERR_PARAM_NULLPTR))
#define AICPU_ASSERT_TRUE_RETVAL(v) \
  AICPU_ASSERT_RETVAL((v), (ACLNN_ERR_PARAM_INVALID))
#define AICPU_ASSERT_GE_SUCCESS(v) \
  AICPU_ASSERT_RETVAL(((v) == ge::GRAPH_SUCCESS), (ACLNN_ERR_INNER))

#define AICPU_ASSERT_OK(v) AICPU_ASSERT(((v) == OK))
#define AICPU_ASSERT_RTOK(v) AICPU_ASSERT(((v) == 0))
#define AICPU_ASSERT_NOTNULL(v) AICPU_ASSERT(((v) != nullptr))
#define AICPU_ASSERT_TRUE(v) AICPU_ASSERT((v))

static constexpr size_t kFindTaskStart = 0U;
static constexpr size_t kFindTaskEnd = 1U;
static constexpr size_t kUpdateShapeStart = 2U;
static constexpr size_t kUpdateShapeEnd = 3U;
static constexpr size_t kShapeH2DEnd = 4U;
static constexpr size_t kUpdateArgsStart = 5U;
static constexpr size_t kUpdateArgsEnd = 6U;
static constexpr size_t kLaunchEnd = 7U;
static constexpr size_t kShapeD2hCopyEnd = 8U;
static constexpr size_t kUpdateOutputShapeEnd = 9U;
static constexpr size_t kAicpuTimeStampNum = kUpdateOutputShapeEnd + 1U;

struct AicpuTimeStamp {
  struct timespec tp[kAicpuTimeStampNum];
  bool isEnable;
};
extern AicpuTimeStamp gAicpuTimeStamp;

static std::mutex gTimeStampMutex;
static inline void RecordAicpuTime(const size_t index) {
  if (gAicpuTimeStamp.isEnable && index < kAicpuTimeStampNum) {
    const std::lock_guard<std::mutex> lk(gTimeStampMutex);
    clock_gettime(CLOCK_MONOTONIC, &(gAicpuTimeStamp.tp[index]));
  }
}
}  // namespace internal
}  // namespace op
#endif  // OP_API_COMMON_INC_OPDEV_AICPU_AICPU_UTILS_H_