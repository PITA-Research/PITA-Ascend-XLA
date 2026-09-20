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

#ifndef XLA_XLA_NPU_COMMON_NPU_TOOL_H_
#define XLA_XLA_NPU_COMMON_NPU_TOOL_H_

#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

// NPU common tool function classes
class NpuTool {
 public:
  static HloPrintOptions GetDetailedPrintOptions() {
    HloPrintOptions opts = HloPrintOptions::Default();

    opts
        // Print large constants (off by default, large constants will be truncated)
        .set_print_large_constants(true)
        // Print the complete contents of the subcomputation (default only prints the subcomputation name) HloPrintOptions::PrintSubcomputationMode::kFullBodies
        .set_print_subcomputation_mode(
            HloPrintOptions::PrintSubcomputationMode::kNameOnly)
        // print original value
        .set_print_original_value(true)
        // Print metadata (such as location, comments, etc.) -- do not print
        .set_print_metadata(false)
        // Only print op names in metadata
        .set_print_metadata_only_op_name(true)
        // Print backend configuration (such as GPU/CPU specific optimization configuration)
        .set_print_backend_config(true)
        // JSON output of sorted backend configuration (easier to read)
        .set_sort_backend_config(true)
        // Print the shape of the operand (off by default)
        .set_print_operand_shape(true)
        // Print index comments for operands (annotated every N operands, 0 means all annotations)
        .set_print_operand_index_annotation_interval(0)
        // Print program shape (input and output overall shape)
        .set_print_program_shape(true)
        // Print % prefix (enabled by default, keeping semantics intact)
        .set_print_percent(true)
        // Print control dependencies (enabled by default, retain execution order information)
        .set_print_control_dependencies(true)
        // Print infeed/outfeed configuration
        .set_print_infeed_outfeed_config(true)
        // Do not compress operands (off by default, ensures all operands are shown)
        .set_compact_operands(false)
        // Print operand names (enabled by default)
        .set_print_operand_names(true)
        // Print result shape (on by default)
        .set_print_result_shape(true)
        // Print additional attributes (such as fusion type, replica group, etc.)
        .set_print_extra_attributes(true)
        // Print complete replica group list (avoid compression)
        .set_print_full_replica_group_list(true)
        // Print parameter number and channel ID
        .set_print_parameter_number(true)
        .set_print_channel_id(true);

    return opts;
  }

  static void PrintHloModule(const HloModule* hlo_module,
                             const std::string& title) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    NPU_LOG_FORMAT_PRINT(DEBUG, "**** %s: ****", title);
    HloPrintOptions hloPrintOptions = GetDetailedPrintOptions();
    NPU_LOG_FORMAT_PRINT(DEBUG, hlo_module->ToString(hloPrintOptions));
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
  }

  static void PrintModuleOp(mlir::ModuleOp module, const std::string& title) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
    std::string result;
    llvm::raw_string_ostream os(result);
    module.print(os);
    NPU_LOG_FORMAT_PRINT(DEBUG, "**** %s: ****", title);
    NPU_LOG_FORMAT_PRINT(DEBUG, os.str());
    NPU_LOG_FORMAT_PRINT(DEBUG, "========================================");
  }

  // Specialized version: handling bool types (data() of std::vector<bool> does not return a pointer)
  static void PrintNpuDeviceMemoryValue(const std::string& title,
                                        const int64_t& size,
                                        const void* deviceAddr,
                                        int64_t max_print_count,
                                        bool /*is_bool*/) {
    NPU_LOG_FORMAT_PRINT(DEBUG, title);

    // Get the configured maximum number of prints
    if (max_print_count <= 0) {
      max_print_count =
          NPU_CONFIG_GET_INT(xla::NpuConfig::Key::XLA_NPU_MAX_PRINT_COUNT);
    }

    int64_t print_count = std::min(size, max_print_count);
    // Use uint8_t to receive bool data
    std::vector<uint8_t> resultData(print_count, 0);
    aclError acl_ret = dep::aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(uint8_t), deviceAddr,
        print_count * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "    Copy result from device to host failed. "
                           "ERROR: %d",
                           acl_ret);
      return;
    }

    for (int64_t i = 0; i < print_count; i++) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Value[%ld] on device is: %d", i,
                           static_cast<int>(resultData[i]));
    }

    if (size > max_print_count) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "    ...(total %ld elements, showing first %ld)",
                           size, max_print_count);
    }
  }

  template <typename printDataType>
  static void PrintNpuDeviceMemoryValue(const std::string& title,
                                        const int64_t& size,
                                        const void* deviceAddr,
                                        int64_t max_print_count = -1) {
    NPU_LOG_FORMAT_PRINT(DEBUG, title);

    // Get the configured maximum number of prints
    if (max_print_count <= 0) {
      max_print_count =
          NPU_CONFIG_GET_INT(xla::NpuConfig::Key::XLA_NPU_MAX_PRINT_COUNT);
    }

    int64_t print_count = std::min(size, max_print_count);
    std::vector<printDataType> resultData(print_count, 0);
    aclError acl_ret = dep::aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(printDataType),
        deviceAddr, print_count * sizeof(printDataType),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "    Copy result from device to host failed. "
                           "ERROR: %d",
                           acl_ret);
      return;
    }

    for (int64_t i = 0; i < print_count; i++) {
      NPU_LOG_FORMAT_PRINT(DEBUG, "    Value[%ld] on device is: %f", i,
                           static_cast<double>(resultData[i]));
    }

    if (size > max_print_count) {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "    ...(total %ld elements, showing first %ld)",
                           size, max_print_count);
    }
  }

  // Interface to automatically select printing type based on Shape element type
  static void PrintNpuDeviceMemoryValueByShape(const std::string& title,
                                               const Shape& xla_shape,
                                               const void* device_addr,
                                               int64_t max_print_count = -1);

  // Calculate the number of elements represented by XLA shape
  static int64_t GetXlaShapeSize(const Shape& xla_data_shape) {
    int64_t size = 1;
    for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
      const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
      size *= xla_data_shape_dim;
    }

    return size;
  }

  static std::string GetShapeString(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
      return "empty";
    }

    std::stringstream ss;
    ss << "[";
    for (const auto& shape_dim : shape) {
      ss << shape_dim << ", ";
    }
    ss << "]";

    return ss.str();
  }

  static std::vector<int64_t> ConvertXlaShape2VectorShape(
      const Shape& xla_data_shape) {
    std::vector<int64_t> vector_shape;
    for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
      const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
      vector_shape.emplace_back(xla_data_shape_dim);
    }

    return vector_shape;
  }

  static std::string GetXlaShapeString(const Shape& xla_data_shape) {
    if (xla_data_shape.IsTuple()) {
      std::stringstream ss;
      ss << "(";
      const auto& tuple_shapes = xla_data_shape.tuple_shapes();
      for (size_t i = 0; i < tuple_shapes.size(); ++i) {
        ss << GetXlaShapeString(tuple_shapes[i]);
        if (i < tuple_shapes.size() - 1) {
          ss << ", ";
        }
      }
      ss << ")";
      return ss.str();
    } else {
      const auto& vector_shape = ConvertXlaShape2VectorShape(xla_data_shape);
      return GetShapeString(vector_shape);
    }
  }

  static void PrintXlaShape(const Shape& xla_data_shape,
                            std::string desc = "Xla tensor shape: ") {
    const auto& shape_str = GetXlaShapeString(xla_data_shape);
    NPU_LOG_FORMAT_PRINT(DEBUG, "%s%s", desc, shape_str);
  }

  static aclDataType ConvertXlaType2AclDataType(const PrimitiveType& type) {
    switch (type) {
      case PrimitiveType::PRED:
        return aclDataType::ACL_BOOL;
      case PrimitiveType::F32:
        return aclDataType::ACL_FLOAT;
      case PrimitiveType::F64:
        return aclDataType::ACL_DOUBLE;
      case PrimitiveType::S32:
        return aclDataType::ACL_INT32;
      case PrimitiveType::S64:
        return aclDataType::ACL_INT64;
      case PrimitiveType::S16:
        return aclDataType::ACL_INT16;
      case PrimitiveType::S8:
        return aclDataType::ACL_INT8;
      case PrimitiveType::U8:
        return aclDataType::ACL_UINT8;
      case PrimitiveType::U16:
        return aclDataType::ACL_INT16;
      case PrimitiveType::U32:
        return aclDataType::ACL_INT32;
      case PrimitiveType::U64:
        return aclDataType::ACL_INT64;
      case PrimitiveType::F16:
        return aclDataType::ACL_FLOAT16;
      case PrimitiveType::BF16:
        return aclDataType::ACL_BF16;
      default:
        return aclDataType::ACL_FLOAT;
    };
  }

  static aclTensor* CreateAclTensorPtr(const Shape& xla_data_shape,
                                       const PrimitiveType& xla_data_type,
                                       void* device_addr) {
    const auto& data_shape = ConvertXlaShape2VectorShape(xla_data_shape);
    const auto& data_type = ConvertXlaType2AclDataType(xla_data_type);

    // Always use standard C order strides (row-major order)
    // This is the safest approach because memory allocated by XLA is usually contiguous
    // Calculating strides using XLA's layout will result in a mismatch with the actual memory layout
    std::vector<int64_t> strides(data_shape.size(), 1);
    for (int64_t i = data_shape.size() - 2; i >= 0; i--) {
      strides[i] = data_shape[i + 1] * strides[i + 1];
    }

    // Debug: print strides
    std::string strides_str = "[";
    for (size_t i = 0; i < strides.size(); ++i) {
      if (i > 0) strides_str += ", ";
      strides_str += std::to_string(strides[i]);
    }
    strides_str += "]";
    NPU_LOG_FORMAT_PRINT(INFO, "CreateAclTensorPtr: shape=%s, strides=%s",
                         GetXlaShapeString(xla_data_shape).c_str(),
                         strides_str.c_str());

    // Call aclCreateTensor interface to create aclTensor
    auto tensor =
        dep::aclCreateTensor(data_shape.data(), data_shape.size(), data_type,
                             strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                             data_shape.data(), data_shape.size(), device_addr);

    return tensor;
  }

  // Create an aclTensor using an explicitly specified aclDataType instead of mapping from an XLA type
  // Used for scenarios that require unsigned tensor descriptors (such as aclnnRightShift's logical right shift)
  static aclTensor* CreateAclTensorPtrWithDtype(const Shape& xla_data_shape,
                                                  aclDataType dtype,
                                                  void* device_addr) {
    const auto& data_shape = ConvertXlaShape2VectorShape(xla_data_shape);
    std::vector<int64_t> strides(data_shape.size(), 1);
    for (int64_t i = data_shape.size() - 2; i >= 0; i--) {
      strides[i] = data_shape[i + 1] * strides[i + 1];
    }
    return dep::aclCreateTensor(data_shape.data(), data_shape.size(), dtype,
                                strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                data_shape.data(), data_shape.size(),
                                device_addr);
  }

  static aclTensor* CreateAclTensorPtr(const Shape& xla_data_shape,
                                       const PrimitiveType& xla_data_type,
                                       void* device_addr,
                                       aclFormat format) {
    const auto& data_shape = ConvertXlaShape2VectorShape(xla_data_shape);
    const auto& data_type = ConvertXlaType2AclDataType(xla_data_type);

    std::vector<int64_t> strides(data_shape.size(), 1);
    for (int64_t i = data_shape.size() - 2; i >= 0; i--) {
      strides[i] = data_shape[i + 1] * strides[i + 1];
    }

    auto tensor =
        dep::aclCreateTensor(data_shape.data(), data_shape.size(), data_type,
                             strides.data(), 0, format,
                             data_shape.data(), data_shape.size(), device_addr);

    return tensor;
  }

  // Generate unique folder names (Abseil implementation)
  static std::string generate_unique_folder_name_absl();

  // Parsing the dimension parameters of the reduce operation
  // Format: "dims=[dim1,dim2,...],keepDims=true/false"
  static void ParseReduceDims(const std::string& desc,
                              std::vector<int64_t>& dims, bool& keepDims) {
    dims.clear();
    keepDims = false;

    size_t dims_pos = desc.find("dims=[");
    if (dims_pos == std::string::npos) {
      return;
    }

    size_t dims_end = desc.find("]", dims_pos);
    if (dims_end == std::string::npos) {
      return;
    }

    std::string dims_str = desc.substr(dims_pos + 6, dims_end - dims_pos - 6);
    std::stringstream ss(dims_str);
    std::string dim_str;
    while (std::getline(ss, dim_str, ',')) {
      dims.push_back(std::stoll(dim_str));
    }

    size_t keep_pos = desc.find("keepDims=");
    if (keep_pos != std::string::npos) {
      keepDims = (desc.substr(keep_pos + 9, 5) == "true");
    }
  }
};

}  // namespace xla

#endif  // XLA_XLA_NPU_COMMON_NPU_TOOL_H_