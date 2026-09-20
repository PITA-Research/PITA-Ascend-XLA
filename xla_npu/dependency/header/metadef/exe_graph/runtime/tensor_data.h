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

#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "xla/xla_npu/dependency/header/metadef/graph/ge_error_codes.h"

namespace gert {
using TensorAddress = void *;
using ConstTensorAddressPtr = const void *;

enum TensorPlacement : int32_t {
  kOnDeviceHbm,  ///<Tensor is located in HBM memory on Device
  kOnHost,       ///< Tensor is located in Host
  kFollowing,    ///< Tensor is located in Host, and the data follows the structure
  kOnDeviceP2p,  ///<Tensor is located in the P2p memory on the Device
  kTensorPlacementEnd
};

class TensorPlacementUtils {
 public:
  static bool IsOnDevice(TensorPlacement placement) {
    return (placement == kOnDeviceHbm) || (placement == kOnDeviceP2p);
  }
  static bool IsOnHost(TensorPlacement placement) {
    return (placement == kOnHost) || (placement == kFollowing);
  }
  static bool IsOnHostFollowing(TensorPlacement placement) {
    return (placement == kFollowing);
  }
  static bool IsOnHostNotFollowing(TensorPlacement placement) {
    return (placement == kOnHost);
  }
  static bool IsOnDeviceHbm(TensorPlacement placement) {
    return (placement == kOnDeviceHbm);
  }
  static bool IsOnDeviceP2p(TensorPlacement placement) {
    return (placement == kOnDeviceP2p);
  }
};

enum TensorOperateType : int32_t {
  kGetTensorAddress,  ///< Get the address of Tensor
  kFreeTensor,        ///< Release Tensor
  kPlusShareCount,    ///< Share Tensor
  kTensorOperateType
};

/**
 * Tensor management functions
 */
using TensorAddrManager = ge::graphStatus (*)(TensorAddress addr,
                                              TensorOperateType operate_type,
                                              void **out);

class TensorData {
 public:
  /**
   * Construct a TensorData
   * @param addr tensor address
   * @param manager The management function of tensor data. If manager is empty, addr is considered to be the data address of tensor, and this data does not need to be released.
   */
  // memse function misra alarm shielding
  explicit TensorData(TensorAddress addr = nullptr,
                      const TensorAddrManager manager = nullptr)
      : addr_(addr),
        manager_(manager),
        size_(0U),
        placement_(kTensorPlacementEnd),
        reserved_0_(0U) {
    (void)memset(reserved_1_, 0, sizeof(reserved_1_));
  }
  explicit TensorData(TensorAddress addr, const TensorAddrManager manager,
                      size_t size, TensorPlacement placement)
      : addr_(addr),
        manager_(manager),
        size_(size),
        placement_(placement),
        reserved_0_(0U) {
    (void)memset(reserved_1_, 0, sizeof(reserved_1_));
  }
  TensorData(const TensorData &) = delete;
  TensorData(TensorData &&other) noexcept
      : addr_(other.addr_),
        manager_(other.manager_),
        size_(other.size_),
        placement_(other.placement_) {
    other.addr_ = nullptr;
    other.manager_ = nullptr;
    other.size_ = 0U;
    other.placement_ = kTensorPlacementEnd;
    reserved_0_ = other.reserved_0_;
    (void)memcpy(reserved_1_, other.reserved_1_, sizeof(reserved_1_));
  }
  TensorData &operator=(const TensorData &other) = delete;
  TensorData &operator=(TensorData &&other) noexcept {
    if (this != &other) {
      static_cast<void>(Free());
      addr_ = other.addr_;
      manager_ = other.manager_;
      size_ = other.size_;
      placement_ = other.placement_;
      other.addr_ = nullptr;
      other.manager_ = nullptr;
      other.size_ = 0U;
      other.placement_ = kTensorPlacementEnd;
      reserved_0_ = other.reserved_0_;
      (void)memcpy(reserved_1_, other.reserved_1_, sizeof(reserved_1_));
    }
    return *this;
  }
  ~TensorData() noexcept { static_cast<void>(Free()); }

  /**
   * Get tensor address
   * @return tensor address
   */
  TensorAddress GetAddr() const {
    if (manager_ == nullptr || addr_ == nullptr) {
      return addr_;
    }
    TensorAddress addr;
    if (manager_(addr_, kGetTensorAddress, &addr) != ge::GRAPH_SUCCESS) {
      return nullptr;
    } else {
      return addr;
    }
  }

  /**
  * Get the memory size of tensor
  * @return the memory size occupied by tensor
  */
  size_t GetSize() const { return size_; }
  /**
   * Set the memory size of tensor
   * @param tensor memory size
   */
  void SetSize(const size_t size) { size_ = size; }

  /**
  * Get tensor's placement
  * @return tensor placement
  */
  TensorPlacement GetPlacement() const { return placement_; }
  /**
   * Set tensor placement
   * @param tensor placement
   */
  void SetPlacement(const TensorPlacement placement) { placement_ = placement; }
  /**
   * Release tensor
   * @return Returns ge::GRAPH_SUCCESS on success
   */
  ge::graphStatus Free() {
    if (manager_ == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = manager_(addr_, kFreeTensor, nullptr);
    if (ret == ge::GRAPH_SUCCESS) {
      addr_ = nullptr;
      manager_ = nullptr;
    }
    return ret;
  }
  /**
   * Set tensor address
   * @param addr tensor address
   * @param manager tensor management function
   */
  ge::graphStatus SetAddr(const ConstTensorAddressPtr addr,
                          TensorAddrManager manager) {
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    addr_ = const_cast<TensorAddress>(addr);
    manager_ = manager;
    return ge::GRAPH_SUCCESS;
  }

  bool IsSharedWith(const TensorData &other) const {
    return (addr_ == other.addr_ && manager_ == other.manager_ &&
            size_ == other.size_ && placement_ == other.placement_);
  }

  ge::graphStatus ShareFrom(const TensorData &other) {
    if (IsSharedWith(other)) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }

    addr_ = other.addr_;
    manager_ = other.manager_;
    size_ = other.size_;
    placement_ = other.placement_;
    if (manager_ != nullptr) {
      return manager_(addr_, kPlusShareCount, nullptr);
    } else {
      return ge::GRAPH_SUCCESS;
    }
  }

  /**
   * Releases ownership of TensorAddress. After calling this interface, this object no longer manages TensorAddress.
   * And TensorAddress has not been released, so the caller is responsible for releasing TensorAddress through the manager
   * @param manager parameter, a function used to manage the returned `TensorAddress`. If this object has no ownership, then manager is set to nullptr
   * @return the `TensorAddress` pointer held by this object. If this object does not hold any pointer, nullptr is returned.
   */
  TensorAddress Release(TensorAddrManager &manager) {
    auto tmp_addr = addr_;
    manager = manager_;

    addr_ = nullptr;
    manager_ = nullptr;
    size_ = 0U;
    placement_ = kTensorPlacementEnd;

    return tmp_addr;
  }

 private:
  TensorAddress addr_;
  TensorAddrManager manager_;
  size_t size_;
  TensorPlacement placement_;
  uint32_t reserved_0_;  // Reserved field, 8-byte aligned for TensorPlacement
  uint8_t reserved_1_
      [40];  // Reserved field, 32+8, do not directly use when only 8-byte left

  friend class TensorUtils;
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
