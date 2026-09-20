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

#ifndef PJRT_NPU_NPU_TOOL_H_
#define PJRT_NPU_NPU_TOOL_H_

#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

class NpuTool {
public:
    static std::string GetXlaTypeName(const PrimitiveType& type) {
        switch (type) {
            case PrimitiveType::PRED: return "PRED";
            case PrimitiveType::S1: return "S1";
            case PrimitiveType::S2: return "S2";
            case PrimitiveType::S4: return "S4";
            case PrimitiveType::S8: return "S8";
            case PrimitiveType::S16: return "S16";
            case PrimitiveType::S32: return "S32";
            case PrimitiveType::S64: return "S64";
            case PrimitiveType::U1: return "U1";
            case PrimitiveType::U2: return "U2";
            case PrimitiveType::U4: return "U4";
            case PrimitiveType::U8: return "U8";
            case PrimitiveType::U16: return "U16";
            case PrimitiveType::U32: return "U32";
            case PrimitiveType::U64: return "U64";
            case PrimitiveType::F16: return "F16";
            case PrimitiveType::F32: return "F32";
            case PrimitiveType::F64: return "F64";
            case PrimitiveType::BF16: return "BF16";
            case PrimitiveType::C64: return "C64";
            case PrimitiveType::C128: return "C128";
            default: return "UNKNOWN";
        }
    }

    static std::string GetGeTypeName(const ge::DataType& type) {
        switch (type) {
            case ge::DataType::DT_BOOL: return "DT_BOOL";
            case ge::DataType::DT_INT4: return "DT_INT4";
            case ge::DataType::DT_INT2: return "DT_INT2";
            case ge::DataType::DT_INT8: return "DT_INT8";
            case ge::DataType::DT_INT16: return "DT_INT16";
            case ge::DataType::DT_INT32: return "DT_INT32";
            case ge::DataType::DT_INT64: return "DT_INT64";
            case ge::DataType::DT_UINT1: return "DT_UINT1";
            case ge::DataType::DT_UINT2: return "DT_UINT2";
            case ge::DataType::DT_UINT8: return "DT_UINT8";
            case ge::DataType::DT_UINT16: return "DT_UINT16";
            case ge::DataType::DT_UINT32: return "DT_UINT32";
            case ge::DataType::DT_UINT64: return "DT_UINT64";
            case ge::DataType::DT_FLOAT16: return "DT_FLOAT16";
            case ge::DataType::DT_FLOAT: return "DT_FLOAT";
            case ge::DataType::DT_DOUBLE: return "DT_DOUBLE";
            case ge::DataType::DT_BF16: return "DT_BF16";
            case ge::DataType::DT_COMPLEX64: return "DT_COMPLEX64";
            case ge::DataType::DT_COMPLEX128: return "DT_COMPLEX128";
            default: return "DT_UNKNOWN";
        }
    }
    static std::string GetShapeString(const ge::Shape &shape) {
        std::stringstream ss;
        auto dims = shape.GetDims();
        if (dims.empty()) {
            return "empty";
        }
        ss << "[";
        size_t index = 0U;
        for (; index < (dims.size() - 1U); index++) {
        ss << dims[index] << ", ";
        }
        ss << dims[index] << "]";
        
        return ss.str(); 
    }

    static ge::DataType ConvertXlaType2GeType(const PrimitiveType& type)
    {
        ge::DataType geType;
        switch (type) {
            case PrimitiveType::PRED:
                geType = ge::DataType::DT_BOOL;
                break;
            case PrimitiveType::S1:
                geType = ge::DataType::DT_INT4;
                break;
            case PrimitiveType::S2:
                geType = ge::DataType::DT_INT2;
                break;
            case PrimitiveType::S4:
                geType = ge::DataType::DT_INT2;
                break;
            case PrimitiveType::S8:
                geType = ge::DataType::DT_INT8;
                break;
            case PrimitiveType::S16:
                geType = ge::DataType::DT_INT16;
                break;
            case PrimitiveType::S32:
                geType = ge::DataType::DT_INT32;
                break;
            case PrimitiveType::S64:
                geType = ge::DataType::DT_INT64;
                break;
            case PrimitiveType::U1:
                geType = ge::DataType::DT_UINT1;
                break;
            case PrimitiveType::U2:
                geType = ge::DataType::DT_UINT2;
                break;
            case PrimitiveType::U4:
                geType = ge::DataType::DT_UINT2;
                break;
            case PrimitiveType::U8:
                geType = ge::DataType::DT_UINT8;
                break;
            case PrimitiveType::U16:
                geType = ge::DataType::DT_UINT16;
                break;
            case PrimitiveType::U32:
                geType = ge::DataType::DT_UINT32;
                break;
            case PrimitiveType::U64:
                geType = ge::DataType::DT_UINT64;
                break;
            case PrimitiveType::F16:
                geType = ge::DataType::DT_FLOAT16;
                break;
            case PrimitiveType::F32:
                geType = ge::DataType::DT_FLOAT;
                break;
            case PrimitiveType::F64:
                geType = ge::DataType::DT_DOUBLE;
                break;
            case PrimitiveType::BF16:
                geType = ge::DataType::DT_BF16;
                break;
            case PrimitiveType::C64:
                geType = ge::DataType::DT_COMPLEX64;
                break;
            case PrimitiveType::C128:
                geType = ge::DataType::DT_COMPLEX128;
                break;
            default:
                geType = ge::DataType::DT_FLOAT;
                break;
        }
         NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertXlaType2GeType: XLA type %d(%s) -> GE type %d(%s)",
                              static_cast<int>(type), GetXlaTypeName(type).c_str(),
                              static_cast<int>(geType), GetGeTypeName(geType).c_str());
         return geType;
    }

    static PrimitiveType ConvertGeType2XlaType(const ge::DataType& type)
    {
        PrimitiveType xlaType;
        switch (type) {
            case ge::DataType::DT_BOOL:
                xlaType = PrimitiveType::PRED;
                break;
            case ge::DataType::DT_INT4:
                xlaType = PrimitiveType::S1;
                break;
            case ge::DataType::DT_INT2:
                xlaType = PrimitiveType::S2;
                break;
            case ge::DataType::DT_INT8:
                xlaType = PrimitiveType::S8;
                break;
            case ge::DataType::DT_INT16:
                xlaType = PrimitiveType::S16;
                break;
            case ge::DataType::DT_INT32:
                xlaType = PrimitiveType::S32;
                break;
            case ge::DataType::DT_INT64:
                xlaType = PrimitiveType::S64;
                break;
            case ge::DataType::DT_UINT1:
                xlaType = PrimitiveType::U1;
                break;
            case ge::DataType::DT_UINT2:
                xlaType = PrimitiveType::U2;
                break;
            case ge::DataType::DT_UINT8:
                xlaType = PrimitiveType::U8;
                break;
            case ge::DataType::DT_UINT16:
                xlaType = PrimitiveType::U16;
                break;
            case ge::DataType::DT_UINT32:
                xlaType = PrimitiveType::U32;
                break;
            case ge::DataType::DT_UINT64:
                xlaType = PrimitiveType::U64;
                break;
            case ge::DataType::DT_FLOAT16:
                xlaType = PrimitiveType::F16;
                break;
            case ge::DataType::DT_FLOAT:
                xlaType = PrimitiveType::F32;
                break;
            case ge::DataType::DT_DOUBLE:
                xlaType = PrimitiveType::F64;
                break;
            case ge::DataType::DT_BF16:
                xlaType = PrimitiveType::BF16;
                break;
            case ge::DataType::DT_COMPLEX64:
                xlaType = PrimitiveType::C64;
                break;
            case ge::DataType::DT_COMPLEX128:
                xlaType = PrimitiveType::C128;
                break;
            default:
                xlaType = PrimitiveType::F32;
                break;
        }
         NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertGeType2XlaType: GE type %d(%s) -> XLA type %d(%s)",
                              static_cast<int>(type), GetGeTypeName(type).c_str(),
                              static_cast<int>(xlaType), GetXlaTypeName(xlaType).c_str());
         return xlaType;
    }

  static ge::Shape ConvertXlaShape2GeShape(const Shape& xla_data_shape) {
    std::vector<int64_t> ge_shape_vector;
    for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
      const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
      ge_shape_vector.emplace_back(xla_data_shape_dim);
    }

    ge::Shape ge_data_shape(ge_shape_vector);
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "convert xla shape to ge shape, input xla shape is "
                         "%s, output ge shape is %s",
                         xla_data_shape.ToString().c_str(),
                         GetShapeString(ge_data_shape).c_str());

    return ge_data_shape;
  }

  static Shape ConvertGeShape2XlaShape(const ge::Shape& ge_data_shape,
                                       const ge::DataType& ge_data_type) {
    const auto& xla_data_dims = ge_data_shape.GetDims();
    const auto& xla_data_type = ConvertGeType2XlaType(ge_data_type);
    const Shape& xla_data_shape =
        ShapeUtil::MakeShape(xla_data_type, xla_data_dims);

    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "convert ge shape to xla shape, input ge shape is %s "
                         "which type is %d, output xla shape is %s",
                         GetShapeString(ge_data_shape).c_str(),
                         static_cast<int>(xla_data_type),
                         xla_data_shape.ToString().c_str());

    return xla_data_shape;
  }
};

}  // namespace xla
#endif  //PJRT_NPU_PJRT_COMMON_H_
