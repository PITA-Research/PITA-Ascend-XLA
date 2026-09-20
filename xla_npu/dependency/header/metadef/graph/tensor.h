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

#ifndef INC_EXTERNAL_GRAPH_TENSOR_H_
#define INC_EXTERNAL_GRAPH_TENSOR_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "./ge_error_codes.h"
#include "./types.h"
#include "ascend_string.h"

namespace ge {
class ShapeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Shape {
 public:
  Shape();
  ~Shape() = default;
  explicit Shape(const std::vector<int64_t> &dims);

  /**
   * `GetDimNum()` identifies the number of valid dims, which is not equivalent to `GetDims().size()`. The caller can choose as needed.
   * For example, if dim is [-2] and the dimension is unknown:
   * GetDimNum() will return 0;
   * GetDims().size() will return the number of dims, which is 1;
   * In addition, if you need to determine whether it is a scalar, it is recommended to use the interface `GetDims.size() == 0U` to determine
   * @return
   */
  size_t GetDimNum() const;
  // If the idx is invalid, return 0
  int64_t GetDim(size_t idx) const;
  graphStatus SetDim(size_t idx, int64_t value);
  /**
   * `GetDims` identifies the number of dims, which is not equivalent to `GetDimNum()`. The caller can choose according to needs.
   * For example, if dim is [-2] and the dimension is unknown:
   * GetDimNum() will return 0;
   * GetDims().size() will return the number of dims, which is 1;
   * In addition, if you need to determine whether it is a scalar, it is recommended to use the interface `GetDims.size() == 0U` to determine
   * @return
   */
  std::vector<int64_t> GetDims() const;
  /**
   * Get the product of the dim values of each dimension of the shape
   * @return
   * If the dim value contains -1 or -2, then size directly returns -1, which means unknown shape
   * If the dim value contains 0, then size returns 0 directly, which means an empty tensor.
   * If the number of dim values is 0, then size directly returns 0, which means a scalar
   * If the product of dim values produces an overflow of int64, then size directly returns 0, which means that the product overflows.
   */
  int64_t GetShapeSize() const;

 private:
  std::shared_ptr<ShapeImpl> impl_;
};

class TensorDescImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorDesc {
 public:
  TensorDesc();
  ~TensorDesc() = default;
  explicit TensorDesc(Shape shape, Format format = FORMAT_ND,
                      DataType dt = DT_FLOAT);
  // Copy
  TensorDesc(const TensorDesc &desc);
  // Move
  TensorDesc(TensorDesc &&desc);
  // Copy
  TensorDesc &operator=(const TensorDesc &desc);
  // Move
  TensorDesc &operator=(TensorDesc &&desc);

  void Update(const Shape &shape, Format format = FORMAT_ND,
              DataType dt = DT_FLOAT);
  Shape GetShape() const;
  void SetShape(const Shape &shape);
  // set shape with -2, it stand for unknown shape
  graphStatus SetUnknownDimNumShape();
  // for unknown shape
  graphStatus SetShapeRange(
      const std::vector<std::pair<int64_t, int64_t>> &range);
  graphStatus GetShapeRange(
      std::vector<std::pair<int64_t, int64_t>> &range) const;

  Format GetFormat() const;
  void SetFormat(Format format);

  Shape GetOriginShape() const;
  void SetOriginShape(const Shape &origin_shape);

  Format GetOriginFormat() const;
  void SetOriginFormat(Format origin_format);

  DataType GetDataType() const;
  void SetDataType(DataType dt);

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &))
  std::string GetName() const;
  graphStatus GetName(AscendString &name);
  graphStatus GetName(AscendString &name) const;

  ATTRIBUTED_DEPRECATED(void SetName(const char_t *))
  void SetName(const std::string &name);
  void SetName(const char_t *name);

  // Attr acess
  void SetSize(int64_t size);
  int64_t GetSize() const;

  int64_t GetRealDimCnt() const;
  void SetRealDimCnt(const int64_t real_dim_cnt);

  void SetPlacement(Placement placement);
  Placement GetPlacement() const;

  void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer,
                    const size_t &const_data_len);
  bool GetConstData(uint8_t **const_data_buffer, size_t &const_data_len) const;
  /*
  * The complementary dimension is similar to the ExpandDims operator. On the basis of the original shape, one or more dimensions are added. For example, the original shape[2,2] has two axes, then the shape after adding two dimensions between the two axes is [2,1,1,2].
  * The 0th and 3rd axes of the shape after complementation are called the original axes, and the 1st and 2nd axes are called the complementation axes.
  *
  * The complementary dimension rules are described by 1 and 0. 1 represents the current axis as the complementary dimension axis, 0 represents the current axis as the original axis, and represents the source of each axis of the current shape from left to right, for example:
  * | Dimension complement rules | Shape before dimension complement | Shape after dimension complement |
  * | -------- | ----------- | ------------------------------------------------------------ |
  * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
  * | 100      | [2, 3]      | [1, 2, 3]                                                    |
  * | 1000 | [2, 3] | The dimension complementation rule does not match the shape before dimensionality complementation. The rule specifies that the original axis has 3 axes, but the original shape only has 2 axes. An error is reported when the dimensionality complementation is performed. |
  *
  */
  void SetExpandDimsRule(const AscendString &expand_dims_rule);
  graphStatus GetExpandDimsRule(AscendString &expand_dims_rule) const;

 private:
  std::shared_ptr<TensorDescImpl> impl;
  friend class TensorAdapter;
};

class TensorImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Tensor {
 public:
  using DeleteFunc = std::function<void(uint8_t *)>;
  Tensor();
  ~Tensor() = default;
  explicit Tensor(const TensorDesc &tensor_desc);
  Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data);
  Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size);
  Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data);

  TensorDesc GetTensorDesc() const;
  graphStatus SetTensorDesc(const TensorDesc &tensor_desc);

  const uint8_t *GetData() const;
  uint8_t *GetData();
  size_t GetSize() const;
  std::unique_ptr<uint8_t[], Tensor::DeleteFunc> ResetData();

  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const uint8_t *data, size_t size);
  ATTRIBUTED_DEPRECATED(graphStatus SetData(const char_t *data))
  graphStatus SetData(const std::string &data);
  graphStatus SetData(const char_t *data);
  ATTRIBUTED_DEPRECATED(graphStatus SetData(const std::vector<AscendString> &))
  graphStatus SetData(const std::vector<std::string> &data);
  graphStatus SetData(const std::vector<AscendString> &datas);
  graphStatus SetData(uint8_t *data, size_t size,
                      const Tensor::DeleteFunc &deleter_func);
  graphStatus IsValid();

  graphStatus SetOriginShapeDimNum(const size_t dim_num);
  size_t GetOriginShapeDimNum() const;

  graphStatus SetOriginShapeDim(const size_t idx, const int64_t dim_value);
  int64_t GetOriginShapeDim(const size_t idx) const;

  graphStatus SetOriginFormat(const ge::Format &format);
  ge::Format GetOriginFormat() const;

  graphStatus SetShapeDimNum(const size_t dim_num);
  size_t GetShapeDimNum() const;

  graphStatus SetShapeDim(const size_t idx, const int64_t dim_value);
  int64_t GetShapeDim(const size_t idx) const;

  graphStatus SetFormat(const ge::Format &format);
  ge::Format GetFormat() const;

  graphStatus SetDataType(const ge::DataType &dtype);
  ge::DataType GetDataType() const;

  graphStatus SetPlacement(const ge::Placement &placement);
  ge::Placement GetPlacement() const;

  /*
  * The complementary dimension is similar to the ExpandDims operator. On the basis of the original shape, one or more dimensions are added. For example, the original shape[2,2] has two axes, then the shape after adding two dimensions between the two axes is [2,1,1,2].
  * The 0th and 3rd axes of the shape after complementation are called the original axes, and the 1st and 2nd axes are called the complementation axes.
  *
  * The complementary dimension rules are described by 1 and 0. 1 represents the current axis as the complementary dimension axis, 0 represents the current axis as the original axis, and represents the source of each axis of the current shape from left to right, for example:
  * | Dimension complement rules | Shape before dimension complement | Shape after dimension complement |
  * | -------- | ----------- | ------------------------------------------------------------ |
  * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
  * | 100      | [2, 3]      | [1, 2, 3]                                                    |
  * | 1000 | [2, 3] | The dimension complementation rule does not match the shape before dimensionality complementation. The rule specifies that the original axis has 3 axes, but the original shape only has 2 axes. An error is reported when the dimensionality complementation is performed. |
  *
  */
  graphStatus SetExpandDimsRule(const AscendString &expand_dims_rule);
  graphStatus GetExpandDimsRule(AscendString &expand_dims_rule) const;

  // The difference between the high-performance interface and the SetData interface is to avoid repeating make_shared. At this time, the user needs to ensure that the memory of the tensor is only used by the current tensor and has exclusive ownership.
  graphStatus ResetData(uint8_t *data, size_t size,
                        const Tensor::DeleteFunc &deleter_func);

  Tensor Clone() const;

 private:
  std::shared_ptr<TensorImpl> impl;
  friend class TensorAdapter;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_TENSOR_H_
