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

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_AICPU_ARGS_HANDLER_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_AICPU_ARGS_HANDLER_H_

#include <memory>
#include <string>
#include "aicpu_ext_info_handle.h"
#include "opdev/op_executor.h"
#include "runtime/kernel.h"
#include "aicpu_any_value.h"

namespace op {
namespace internal {
using AicpuAttrs = std::map<std::string, GeAttrValue>;

class AicpuArgsHandler {
 public:
  AicpuArgsHandler(const std::string &opType, const std::string &nodeName,
                   const uint32_t ioNum, const bool needDeviceExt)
      : opType_(opType),
        nodeName_(nodeName),
        ioNum_(ioNum),
        needDeviceExt_(needDeviceExt),
        args_({}) {}

  uint8_t *GetIoAddr() const { return hostBuffer_.get() + ioAddrOffset_; }

  uint8_t *GetExtInfoAddr() const { return hostBuffer_.get() + extInfoOffset_; }

  uint8_t *GetArgs() const { return hostBuffer_.get(); }

  const rtAicpuArgsEx_t &GetArgsEx() const { return args_; }

  const std::string &GetNodeName() const { return nodeName_; }

  uint32_t GetIoNum() const { return ioNum_; }

  size_t GetHostInputSize() const { return hostInputSize_; }

  size_t GetInputAddrAlignBytes() const { return alignBytes_; }

  void SetSpace(void *space) { space_ = space; }

  aclnnStatus MallocMem();

  void ResetHostInputInfo();

  // alignSize is for op in device, actual alloced srcSize may smaller than alignSize.
  aclnnStatus AddHostInput(const size_t idx, void *data, const size_t srcSize,
                           const size_t alignSize);
  aclnnStatus UpdateIoAddr(const FVector<const aclTensor *> &inputs,
                           const FVector<aclTensor *> &outputs,
                           const aclrtStream stream, aclOpExecutor *executor,
                           const uint64_t deviceExtMemSize,
                           const uint64_t deviceCacheOffset);
  ~AicpuArgsHandler() = default;
  virtual void UpdateDeviceExtInfoAddr(void *deviceExtInfoAddr) = 0;

 protected:
  aclnnStatus SetLaunchArgs(const size_t argSize);
  void GetDeviceCacheAddr(void *&deviceAddr, aclOpExecutor *executor,
                          const uint64_t deviceCacheOffset);
  const std::string opType_;
  const std::string nodeName_;
  uint32_t ioNum_;
  bool needDeviceExt_;

  void *space_ = nullptr;
  // for rtKernelLaunch
  rtAicpuArgsEx_t args_;
  std::vector<rtHostInputInfo_t> hostInputInfo_;

  // args
  std::unique_ptr<uint8_t[]> hostBuffer_;

  // for big host input
  void *deviceCache_ = nullptr;
  size_t deviceCacheSize_ = 0U;

  // offset
  size_t ioAddrOffset_ = 0U;
  size_t extInfoOffset_ = 0U;
  size_t hostMemOffset_ = 0U;
  size_t soNameOffset_ = 0U;
  size_t kernelNameOffset_ = 0U;
  size_t taskInfoOffset_ = 0;

  // size
  size_t extInfoSize_ = 0U;
  size_t bufferSize_ = 0U;
  size_t hostInputSize_ = 0U;
  size_t alignBytes_ = 4U;
};

/* Self-developed hostBuffer arrangement
 * |args, including AicpuParamHead|
 *  |ioAddr|
 *  |kernelName|
 *  |soName|
 *  |extInfo|
 *  |hostInput|
 */
class AicpuCCArgsHandler : public AicpuArgsHandler {
 public:
  AicpuCCArgsHandler(const std::string &opType, const std::string &nodeName,
                     const uint32_t ioNum, const bool needDeviceExt)
      : AicpuArgsHandler(opType, nodeName, ioNum, needDeviceExt) {}

  aclnnStatus GenCCArgs(const FVector<const aclTensor *> &inputs,
                        const FVector<aclTensor *> &outputs,
                        const AicpuAttrs &attrs, std::string &taskInfo) const;
  aclnnStatus BuildCCArgs(const std::string &argData,
                          const std::string &kernelName,
                          const std::string &soName, const size_t extInfoSize);
  void UpdateDeviceExtInfoAddr(void *deviceExtInfoAddr) override;

 private:
  aclnnStatus SetHostArgs(const std::string &argData, const size_t extInfoSize);
  aclnnStatus SetOffsetArgs();
};

/* STR_FWK_OP_KERNEL consists of
 *  |workspaceBaseAddr|
 *  |extInfoAddr|
 *  |extInfoLen|
 *  |inputOutputAddr|
 *  |kernelID|
 *  |sessionID|
 */

/* hostBuffer arrangement
 *  |STR_FWK_OP_KERNEL|
 *  |taskInfo|
 *  |ioAddr|
 *  |extInfo|
 *  |hostInput|
 * |soName| placeholder
 * |kernelName| placeholder
 */
class AicpuTfArgsHandler : public AicpuArgsHandler {
 public:
  AicpuTfArgsHandler(const std::string &opType, const std::string &nodeName,
                     const uint32_t ioNum, const bool needDeviceExt)
      : AicpuArgsHandler(opType, nodeName, ioNum, needDeviceExt) {}

  aclnnStatus GenTfArgs(const FVector<const aclTensor *> &inputs,
                        const FVector<aclTensor *> &outputs,
                        const AicpuAttrs &attrs, STR_FWK_OP_KERNEL &fwkOpKernel,
                        std::string &taskInfo) const;
  aclnnStatus BuildTfArgs(STR_FWK_OP_KERNEL &fwkOpKernel,
                          const std::string &taskInfo,
                          const size_t extInfoSize);
  void UpdateDeviceExtInfoAddr(void *deviceExtInfoAddr) override;

 private:
  aclnnStatus SetOffsetArgs();
};
}  // namespace internal
}  // namespace op
#endif  // OP_API_COMMON_INC_OPDEV_AICPU_AICPU_ARGS_HANDLER_H_
