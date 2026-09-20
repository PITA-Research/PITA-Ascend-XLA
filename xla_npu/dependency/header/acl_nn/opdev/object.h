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

#ifndef OP_API_OBJECT_HEAD_H
#define OP_API_OBJECT_HEAD_H

#include <new>
#include <cstddef>

using std::size_t;

namespace op {

class Object {
 public:
  Object() = default;
  virtual ~Object() = default;

 public:
  void *operator new(size_t size) throw();
  void *operator new[](size_t size) throw();

  void *operator new(size_t size,
                     [[maybe_unused]] const std::nothrow_t &tag) throw();
  void *operator new[](size_t size,
                       [[maybe_unused]] const std::nothrow_t &tag) throw();

  void operator delete(void *addr);
  void operator delete[](void *addr);
};

}  // namespace op

#endif
