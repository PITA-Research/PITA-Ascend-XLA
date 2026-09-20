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

#ifndef __POOL_ALLOCATOR_H__
#define __POOL_ALLOCATOR_H__

#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>

#include "opdev/op_log.h"

namespace op {
namespace internal {

void* MallocPtr(size_t size);
void FreePtr(void* block);

template <class T>
struct PoolAllocator {
  using value_type = T;
  using size_t = std::size_t;

  PoolAllocator() = default;

  template <class U>
  constexpr PoolAllocator(const PoolAllocator<U>&) noexcept {}

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      return nullptr;
    }

    T* p = static_cast<T*>(MallocPtr(n * sizeof(T)));
    if (p != nullptr) {
      return p;
    }

    return nullptr;
  }

  void deallocate(T* p, [[maybe_unused]] size_t n) noexcept { FreePtr(p); }

  template <typename _Up, typename... _Args>
  void construct(_Up* __p, _Args&&... __args) {
    ::new ((void*)__p) _Up(std::forward<_Args>(__args)...);
  }

  template <typename _Up>
  void destroy(_Up* __p) {
    __p->~_Up();
  }
};

template <class T, class U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) {
  return false;
}

}  // namespace internal
}  // namespace op

#endif
