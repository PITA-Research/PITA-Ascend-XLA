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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
#include <type_traits>
#include "kernel_run_context.h"
namespace gert {
class Chain {
 public:
  using Deleter = void (*)(void *);
  /**
   * Get the pointer to the data saved in the Chain
   * @tparam T data type
   * @return pointer to data
   */
  template <typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)),
                                                int>::type = 0>
  const T *GetPointer() const {
    return reinterpret_cast<const T *>(any_value_.data.inplace);
  }
  /**
   * Get the pointer to the data saved in the Chain
   * @tparam T data type
   * @return pointer to data
   */
  template <typename T, typename std::enable_if<(sizeof(T) > sizeof(void *)),
                                                int>::type = 0>
  const T *GetPointer() const {
    return reinterpret_cast<const T *>(any_value_.data.pointer);
  }
  /**
   * Get the pointer to the data saved in the Chain
   * @tparam T data type
   * @return pointer to data
   */
  template <typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)),
                                                int>::type = 0>
  auto GetPointer() -> T * {
    return reinterpret_cast<T *>(any_value_.data.inplace);
  }
  /**
   * Get the pointer to the data saved in the Chain
   * @tparam T data type
   * @return pointer to data
   */
  template <typename T, typename std::enable_if<(sizeof(T) > sizeof(void *)),
                                                int>::type = 0>
  auto GetPointer() -> T * {
    return reinterpret_cast<T *>(any_value_.data.pointer);
  }
  /**
   * Get the value of the data saved in the Chain
   * @tparam T data type
   * @return a reference to the value of the data
   */
  template <typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)),
                                                int>::type = 0>
  const T &GetValue() const {
    return *reinterpret_cast<const T *>(any_value_.data.inplace);
  }
  /**
   * Get the value of the data saved in the Chain
   * @tparam T data type
   * @return a reference to the value of the data
   */
  template <typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)),
                                                int>::type = 0>
  auto GetValue() -> T & {
    return *reinterpret_cast<T *>(any_value_.data.inplace);
  }
  /**
   * Set data into Chain. When setting the data pointer, it will try to call delete to delete the data originally saved in the Chain.
   * @param data pointer to data
   * @param deleter is the interface for releasing data. The meaning of null pointer is that it does not need to be released.
   */
  void Set(void *const data, const Chain::Deleter deleter) {
    FreeResource();
    any_value_.data.pointer = data;
    any_value_.deleter = deleter;
  }
  /**
   * Set data into Chain. When setting the data pointer, it will try to call delete to delete the data originally saved in the Chain.
   * @tparam T data type
   * @param data pointer to data
   */
  template <typename T,
            typename std::enable_if<(!std::is_array<T>::value), int>::type = 0>
  void SetWithDefaultDeleter(T *data) {
    Set(data, reinterpret_cast<FreeCallback>(DefaultDeleter<T>));
  }
  /**
   * Set data into Chain. When setting the data pointer, it will try to call delete to delete the data originally saved in the Chain.
   * @tparam T data type
   * @param data pointer to data
   */
  template <typename T, typename PureT = typename std::remove_extent<T>::type,
            typename std::enable_if<std::is_array<T>::value, int>::type = 0>
  void SetWithDefaultDeleter(PureT *data) {
    Set(data, reinterpret_cast<FreeCallback>(DefaultArrayDeleter<PureT>));
  }
  /**
   * Determine whether the data saved in the current Chain has a deleter
   * @return true means it contains deleter
   */
  bool HasDeleter() const { return any_value_.deleter != nullptr; }

 private:
  template <typename T>
  static void DefaultArrayDeleter(T *data) {
    delete[] data;
  }

  template <typename T>
  static void DefaultDeleter(T *data) {
    delete data;
  }

  void FreeResource() {
    if (any_value_.deleter != nullptr) {
      any_value_.deleter(any_value_.data.pointer);
    }
  }

  AsyncAnyValue any_value_;
};
static_assert(std::is_standard_layout<Chain>::value,
              "The class Chain must be a POD");

class KernelContext {
 public:
  /**
   * Get the input number of kernel
   * @return the number of inputs to the kernel
   */
  size_t GetInputNum() const { return context_.input_size; }
  /**
   * Get the output number of kernel
   * @return the number of kernel outputs
   */
  size_t GetOutputNum() const { return context_.output_size; }
  /**
   * Get the input Chain pointer
   * @param i kernel input index
   * @return pointer to input Chain
   */
  const Chain *GetInput(const size_t i) const {
    if (i >= context_.input_size) {
      return nullptr;
    }
    return reinterpret_cast<const Chain *>(context_.values[i]);
  }
  /**
   * Get the input Chain pointer
   * @param i kernel input index
   * @return pointer to input Chain
   */
  Chain *MutableInput(const size_t i) const {
    if (i >= context_.input_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.values[i]);
  }
  /**
   * Get the output Chain pointer
   * @param i kernel output index
   * @return pointer of output Chain
   */
  Chain *GetOutput(const size_t i) {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.values[context_.input_size + i]);
  }
  /**
   * Get the output Chain pointer
   * @param i kernel output index
   * @return pointer of output Chain
   */
  const Chain *GetOutput(const size_t i) const {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<const Chain *>(
        context_.values[context_.input_size + i]);
  }
  Chain *GetOutput2(const size_t i) {
    if (i >= context_.output_size) {
      return nullptr;
    }
    return reinterpret_cast<Chain *>(context_.output_start[i]);
  }
  /**
   * Get the value of the input data. This function first gets the input Chain, and then gets the value from the input Chain.
   * @tparam T value type
   * @param i kernel input index
   * @return input value
   */
  template <typename T>
  const T GetInputValue(size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return {};
    }
    return av->GetValue<T>();
  }
  /**
   * Get the pointer of the input data. This function first gets the input Chain, and then gets the pointer from the input Chain.
   * @tparam T value type
   * @param i kernel input index
   * @return pointer to input data
   */
  template <typename T>
  const T *GetInputPointer(size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * Get the pointer of the input data. This function first gets the input Chain, and then gets the pointer from the input Chain.
   * @tparam T value type
   * @param i kernel input index
   * @return pointer to input data
   */
  template <typename T>
  auto MutableInputPointer(size_t i) const -> T * {
    const auto av = MutableInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * Get the pointer of the input string. This function first gets the input Chain, and then gets the pointer from the input Chain.
   *
   * Just customize a template for todo
   * @param i kernel input index
   * @return pointer to string
   */
  const char *GetInputStrPointer(const size_t i) const {
    const auto av = GetInput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<const char *>();
  }
  /**
   * Get pointer to compute node information
   * @return pointer to calculate node information
   */
  const void *GetComputeNodeExtend() const {
    return context_.compute_node_info;
  }
  /**
   * Get pointer to kernel extension information
   * @return
   */
  const void *GetKernelExtend() const { return context_.kernel_extend_info; }
  /**
   * Get the pointer of the output data. This function first gets the output Chain, and then gets the pointer from the Chain.
   * @tparam T data type
   * @param i kernel output index
   * @return pointer to output data
   */
  template <typename T>
  auto GetOutputPointer(size_t i) -> T * {
    const auto av = GetOutput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * Get the pointer of the output data. This function first gets the output Chain, and then gets the pointer from the Chain.
   * @tparam T data type
   * @param i kernel output index
   * @return pointer to output data
   */
  template <typename T>
  const T *GetOutputPointer(size_t i) const {
    const auto av = GetOutput(i);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetPointer<T>();
  }
  /**
   * Get the underlying context structure. Non-framework code should not directly operate this structure.
   * @return pointer to context structure
   */
  KernelRunContext *GetContext() { return &context_; }
  /**
   * Get the underlying context structure. Non-framework code should not directly operate this structure.
   * @return pointer to context structure
   */
  const KernelRunContext *GetContext() const { return &context_; }
  /**
   * Determine whether a piece of data will be stored inline based on the length of the data. The so-called inline storage means that the data does not need to be allocated memory separately when it is saved in the context.
   * @param size length of data
   * @return true means it will be stored inline
   */
  static bool IsInlineSize(const size_t size) { return size <= sizeof(void *); }

 private:
  KernelRunContext context_;
};
static_assert(std::is_standard_layout<KernelContext>::value,
              "The class KernelContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_KERNEL_CONTEXT_H_
