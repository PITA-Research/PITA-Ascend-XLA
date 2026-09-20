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

#ifndef OP_API_OP_API_COMMON_INC_OPDEV_PLATFORM_H
#define OP_API_OP_API_COMMON_INC_OPDEV_PLATFORM_H

#include <string>
#include "aclnn/aclnn_base.h"
#include "graph/ascend_string.h"
#include "platform/platform_info.h"

namespace op {

enum class SocVersion {
  ASCEND910 = 0,
  ASCEND910B,
  ASCEND910_93,
  ASCEND910_95,
  ASCEND910E,
  ASCEND310,
  ASCEND310P,
  ASCEND310B,
  ASCEND310C,
  ASCEND610LITE,
  RESERVED_VERSION = 99999
};

enum class SocSpec { INST_MMAD = 0, RESERVED_SPEC = 99999 };

enum class SocSpecAbility {
  INST_MMAD_F162F16 = 0,
  INST_MMAD_F162F32,
  INST_MMAD_H322F32,
  INST_MMAD_F322F32,
  INST_MMAD_U32U8U8,
  INST_MMAD_S32S8S8,
  INST_MMAD_S32U8S8,
  INST_MMAD_F16F16F16,
  INST_MMAD_F32F16F16,
  INST_MMAD_F16F16U2,
  INST_MMAD_U8,
  INST_MMAD_S8,
  INST_MMAD_U8S8,
  INST_MMAD_F16U2,
};

class PlatformInfoImpl;

class PlatformThreadLocalCtx;

class PlatformInfo {
  friend const PlatformInfo &GetCurrentPlatformInfo();

  friend class PlatformThreadLocalCtx;

 public:
  PlatformInfo(){};

  PlatformInfo(int32_t deviceId) : deviceId_(deviceId){};

  SocVersion GetSocVersion() const;

  const std::string &GetSocLongVersion() const;

  int32_t GetDeviceId() const;

  bool CheckSupport(SocSpec socSpec, SocSpecAbility ability) const;

  int64_t GetBlockSize() const;

  uint32_t GetCubeCoreNum() const;

  uint32_t GetVectorCoreNum() const;

  bool Valid() const;

  bool GetFftsPlusMode() const;

  fe::PlatFormInfos *GetPlatformInfos() const;

 private:
  PlatformInfo &operator=(const PlatformInfo &other) = delete;

  PlatformInfo &operator=(const PlatformInfo &&other) = delete;

  PlatformInfo(const PlatformInfo &other) = delete;

  PlatformInfo(const PlatformInfo &&other) = delete;

  void SetPlatformImpl(PlatformInfoImpl *impl);

  bool valid_ = false;
  int32_t deviceId_{-1};
  PlatformInfoImpl *impl_ = nullptr;

  ~PlatformInfo();
};

const PlatformInfo &GetCurrentPlatformInfo();

ge::AscendString ToString(SocVersion socVersion);

}  // namespace op

#endif  // OP_API_OP_API_COMMON_INC_OPDEV_PLATFORM_H
