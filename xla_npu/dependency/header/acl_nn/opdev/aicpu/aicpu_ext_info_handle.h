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

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_AICPU_EXT_INFO_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_AICPU_EXT_INFO_H_

#include "aicpu_uitls.h"
#include "cce/aicpu_engine_struct.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "aclnn/aclnn_base.h"
#include "graph/types.h"
#include "opdev/fast_vector.h"

namespace op {
namespace internal {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AicpuSessionInfo = SessionInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(const std::string &nodeName, const uint32_t inputNum,
                      const uint32_t outputNum,
                      const ge::UnknowShapeOpType unknownType)
      : nodeName_(nodeName),
        inputNum_(inputNum),
        outputNum_(outputNum),
        unknownType_(unknownType) {}

  ~AicpuExtInfoHandler() = default;

  aclnnStatus GenTfExtBuffer(const FVector<const aclTensor *> &inputs,
                             const FVector<aclTensor *> &outputs,
                             std::string &taskExtInfo) const;
  aclnnStatus GenCCExtBuffer(const FVector<const aclTensor *> &inputs,
                             const FVector<aclTensor *> &outputs,
                             std::string &taskExtInfo) const;

  aclnnStatus Parse(const std::string &extInfo, uint8_t *hostAddr);

  aclnnStatus UpdateInputShape(const uint32_t inputIndex,
                               const gert::Shape &inputShape);
  aclnnStatus UpdateOutputShape(const uint32_t outputIndex,
                                const gert::Shape &outputShape);

  aclnnStatus GetOutputShapeAndType(const uint32_t outputIndex,
                                    gert::Shape &shape,
                                    ge::DataType &dataType) const;
  aclnnStatus CopyH2D(const aclrtStream stream, const aclOpExecutor *executor,
                      const uint64_t deviceExtMemSize,
                      uint64_t &deviceCacheOffset);
  aclnnStatus CopyOutputShapeD2H();
  aclnnStatus GetExtInfoDeviceBuffer(const aclOpExecutor *executor,
                                     const uint64_t deviceExtMemSize,
                                     uint64_t &deviceCacheOffset);
  static uint64_t GenerateKernelId();
  aclnnStatus UpdateOutputShapeFromExtInfo(const FVector<aclTensor *> &outputs,
                                           aclrtStream stream);
  aclnnStatus UpdateInputAndOutputShape(
      const FVector<const aclTensor *> &inputs,
      const FVector<aclTensor *> &outputs, aclrtStream stream,
      const aclOpExecutor *executor, const uint64_t deviceExtMemSize,
      uint64_t &deviceCacheOffset);
  aclnnStatus UpdateKernelId();

  void SetSpace(void *space) { space_ = space; }

  void *deviceExtInfo_ = nullptr;
  uint64_t deviceCacheOffset_ = 0U;

 private:
  aclnnStatus ParseExtShapeType(AicpuExtInfo &aicpuExtInfo) const;
  aclnnStatus ParseExtInputShape(AicpuExtInfo &aicpuExtInfo);
  aclnnStatus ParseExtOutputShape(AicpuExtInfo &aicpuExtInfo);

  aclnnStatus AppendExtOpName(std::string &taskExtInfo) const;
  aclnnStatus AppendExtShapeType(std::string &taskExtInfo) const;
  aclnnStatus AppendExtBitMap(std::string &taskExtInfo) const;
  aclnnStatus AppendExtInfoShape(
      const FVector<const aclTensor *> &tensors,
      const aicpu::FWKAdapter::FWKTaskExtInfoType type,
      std::string &taskExtInfo, bool isTf = false) const;
  aclnnStatus AppendSessionInfo(std::string &taskExtInfo) const;

  static aclnnStatus UpdateShape(const gert::Shape &shape,
                                 AicpuShapeAndType *const shapeAndType);

  static void GetShapeAndType(const AicpuShapeAndType &shapeAndType,
                              gert::Shape &shape, ge::DataType &dataType);

  // base info
  const std::string nodeName_;
  const uint32_t inputNum_;
  const uint32_t outputNum_;
  const ge::UnknowShapeOpType unknownType_;

  void *space_ = nullptr;
  ;
  // host and device info
  uint8_t *extInfo_ = nullptr;
  size_t extInfoLen_ = 0U;

  std::vector<AicpuShapeAndType *> inputShapeAndType_;
  std::vector<AicpuShapeAndType *> outputShapeAndType_;

  // for 3th op update output
  std::vector<AicpuShapeAndType> outputShape_;
  size_t outputShapeOffset_ = 0U;
  size_t outputShapeLen_ = 0U;
  void *workspace_ = nullptr;
  ;
};
}  // namespace internal
}  // namespace op
#endif  // OP_API_COMMON_INC_OPDEV_AICPU_AICPU_EXT_INFO_H_
