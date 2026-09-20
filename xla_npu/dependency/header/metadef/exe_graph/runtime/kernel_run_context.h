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

#ifndef METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*FreeCallback)(void *);

/**
 * The underlying data structure of Chain. Do not directly reference and operate this data structure.
 */
typedef struct {
  union {
    void *pointer;
    unsigned char inplace[sizeof(void *)];
  } data;
  FreeCallback deleter;
} AsyncAnyValue;

/**
 * The underlying data structure of KernelContext, do not directly reference and operate this data structure
 */
typedef struct {
  size_t input_size;
  size_t output_size;
  const void *compute_node_info;
  const void *kernel_extend_info;
  AsyncAnyValue **output_start;  // todo delete this
  AsyncAnyValue *values[1];
} KernelRunContext;

#ifdef __cplusplus
}
#endif

#endif  // METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_
