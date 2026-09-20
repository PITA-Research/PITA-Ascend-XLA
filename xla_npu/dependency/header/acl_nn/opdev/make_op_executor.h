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

#ifndef __MAKE_OP_EXECUTOR_H__
#define __MAKE_OP_EXECUTOR_H__
#include <set>
#include "opdev/platform.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_executor.h"

using namespace std;
#define CREATE_EXECUTOR() UniqueExecutor(__func__)

// todo: consider reusing OpArgContext
#define INFER_SHAPE(KERNEL_NAME, op_args...)                               \
  ({                                                                       \
    aclnnStatus inferShapeRet;                                             \
    do {                                                                   \
      op::OpArgContext *opArgCtx = GetOpArgContext(op_args);               \
      if (opArgCtx == nullptr) {                                           \
        inferShapeRet = ACLNN_ERR_PARAM_NULLPTR;                           \
      } else {                                                             \
        inferShapeRet = InferShape(KERNEL_NAME##OpTypeId(),                \
                                   *opArgCtx->GetOpArg(op::OP_INPUT_ARG),  \
                                   *opArgCtx->GetOpArg(op::OP_OUTPUT_ARG), \
                                   *opArgCtx->GetOpArg(op::OP_ATTR_ARG));  \
        op::DestroyOpArgContext(opArgCtx);                                 \
      }                                                                    \
    } while (0);                                                           \
    inferShapeRet;                                                         \
  })

// WARNING: args are rvalue and will be moved. but args are used twice in the macro,
//          so DO NOT put rvalue in args as the rvalue in args will be empty in second usage.
// TODO: check GET_WORKSPACE error
#define ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...)          \
  ({                                                                  \
    aclnnStatus addToLaunchRet;                                       \
    do {                                                              \
      op::OpArgContext *opArgCtx = GetOpArgContext(op_args);          \
      addToLaunchRet = CreatAiCoreKernelLauncher(                     \
          #KERNEL_NAME, KERNEL_NAME##OpTypeId(), executor, opArgCtx); \
    } while (0);                                                      \
    addToLaunchRet;                                                   \
  })

#define ADD_TO_LAUNCHER_LIST_DSA(KERNEL_NAME, op_args...)              \
  do {                                                                 \
    op::OpArgContext *opArgCtx = GetOpArgContext(op_args);             \
    CreatDSAKernelLauncher(#KERNEL_NAME, KERNEL_NAME##OpTypeId(),      \
                           KERNEL_NAME##TaskType, executor, opArgCtx); \
  } while (0)

#endif
