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

#include "./convert_type.h"

#include "./utils.h"
#include "xla/xla_npu/dependency/tool/log.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project

namespace mlir::air {

using mlir::Builder;
using mlir::ShapedType;
using mlir::Type;

Status ConvertScalarTypeToDataType(Type type, DataType* dtype) {
  if (type.isF16()) {
    *dtype = DT_FLOAT16;
    return Status::OK();
  } else if (type.isBF16()) {
    *dtype = DT_BF16;
    return Status::OK();
  } else if (type.isF32()) {
    *dtype = DT_FLOAT;
    return Status::OK();
  } else if (type.isF64()) {
    *dtype = DT_DOUBLE;
    return Status::OK();
  } else if (auto i_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (i_type.getWidth()) {
      case 1:
        *dtype = DT_BOOL;
        return Status::OK();
      case 8:
        *dtype = i_type.isUnsigned() ? DT_UINT8 : DT_INT8;
        return Status::OK();
      case 16:
        *dtype = i_type.isUnsigned() ? DT_UINT16 : DT_INT16;
        return Status::OK();
      case 32:
        *dtype = i_type.isUnsigned() ? DT_UINT32 : DT_INT32;
        return Status::OK();
      case 64:
        *dtype = i_type.isUnsigned() ? DT_UINT64 : DT_INT64;
        return Status::OK();
      default:
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Converting %s to DataType. [IntegerType_NotFound]",
            debugString(type).c_str());
        return Status::OK();
    }
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    auto e_type = complex_type.getElementType();
    if (e_type.isF32()) {
      *dtype = DT_COMPLEX64;
      return Status::OK();
    } else if (e_type.isF64()) {
      *dtype = DT_COMPLEX128;
      return Status::OK();
    }
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "Converting %s to DataType. [ComplextType_NotFound]",
                         debugString(type).c_str());
    return Status::OK();
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "Converting %s to DataType. [NotFound]",
                       debugString(type).c_str());
  return Status::OK();
}

Status ConvertToDataType(Type type, DataType* dtype) {
  if (auto stype = mlir::dyn_cast<ShapedType>(type)) {
    ConvertScalarTypeToDataType(stype.getElementType(), dtype);
  } else {
    ConvertScalarTypeToDataType(type, dtype);
  }
  return Status::OK();
}

Status ConvertToShape(mlir::ShapedType type, Shape* shape) {
  std::vector<int64_t> dims;
  if (type.hasRank()) {
    std::ostringstream oss;
    oss << "[";
    for (int64_t dim : type.getShape()) {
      dims.push_back(dim);
      oss << dim << ",";
    }
    oss << "]";
    NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertToShape: dims %s", oss.str().c_str());
    *shape = ge::Shape(dims);
  } else {
    *shape = ge::Shape();
  }

  return Status::OK();
}

}  // namespace mlir::air
