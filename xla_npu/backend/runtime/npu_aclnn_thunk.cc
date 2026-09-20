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

#include "xla/xla_npu/backend/runtime/npu_aclnn_thunk.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/cleanup/cleanup.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_npu/backend/runtime/npu_thunk.h"
#include "xla/xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.h"
#include "xla/xla_npu/backend/runtime/npu_workspace_memory_pool.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_tool.h"
#include "xla/xla_npu/dependency/interface/acl_custom_op_interface.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_interface.h"
#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"
#include "xla/xla_npu/dependency/tool/config_loader.h"
#include "xla/xla_npu/service/npu_buffer_allocations.h"

namespace xla {

namespace {

// Temporary device buffers are slices owned by the stream-ordered arena.
// Individual retirement is intentionally a no-op; slices become reusable only
// after the stream has been drained and the arena is reset.
void RetireTemporaryDeviceMemory(void* ptr) { (void)ptr; }

void DestroyAclResources(
    NpuStreamOrderedDeviceMemoryPool::RetiredAclResources& resources) {
  std::unordered_set<aclTensor*> destroyed_tensors;
  std::unordered_set<aclScalar*> destroyed_scalars;
  std::unordered_set<aclIntArray*> destroyed_int_arrays;
  for (aclTensor* tensor : resources.tensors) {
    if (tensor != nullptr && destroyed_tensors.insert(tensor).second) {
      dep::aclDestroyTensor(tensor);
    }
  }
  for (aclScalar* scalar : resources.scalars) {
    if (scalar != nullptr && destroyed_scalars.insert(scalar).second) {
      dep::aclDestroyScalar(scalar);
    }
  }
  for (aclIntArray* array : resources.int_arrays) {
    if (array != nullptr && destroyed_int_arrays.insert(array).second) {
      dep::aclDestroyIntArray(array);
    }
  }
}

// Parse a dimension set of the form "lhs_batch_dims={0,1}" from hloInstrDesc
std::vector<int64_t> ParseDotDimSet(const std::string& desc,
                                    const std::string& key) {
  std::vector<int64_t> result;
  auto pos = desc.find(key + "={");
  if (pos == std::string::npos) return result;
  pos += key.size() + 2;  // skip "={"
  auto end = desc.find('}', pos);
  if (end == std::string::npos) return result;
  std::string content = desc.substr(pos, end - pos);
  if (content.empty()) return result;
  std::istringstream iss(content);
  std::string token;
  while (std::getline(iss, token, ',')) {
    result.push_back(std::stoll(token));
  }
  return result;
}

// Create an aclTensor of the specified shape (consecutive C-order strides), pointing to the given device address
aclTensor* CreateMatmulAclTensor(const std::vector<int64_t>& shape,
                                 aclDataType data_type, void* device_addr) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return dep::aclCreateTensor(shape.data(), shape.size(), data_type,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), device_addr);
}

// 创建 matmul operand aclTensor，strides 从 operand 的物理 layout 推导。
// 修复: dot 的 operand 若为非 C-order（如折叠 transpose 得到的 column-major 常量
// {0,1}），CreateMatmulAclTensor 硬编码 C-order strides 会让 aclnnMatmul 用错误
// stride 寻址 => 读到打乱的矩阵（static fori/scan 失配根因，s0@K 指纹）。
// 这里按 operand 实际 minor_to_major 计算 strides。仅当 mm_shape 与 operand
// 维度一一对应（无 batch 合并；若 normalizer 重排过维度，operand 已是其 C-order
// transpose 输出，此分支同样给出正确 C-order strides）时启用；否则回退 C-order。
aclTensor* CreateMatmulAclTensorFromShape(
    const std::vector<int64_t>& mm_shape,
    const Shape& operand_shape,
    aclDataType data_type,
    void* device_addr,
    bool trust_operand_layout) {
  auto MakeCOrderStrides = [](const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
  };

  std::vector<int64_t> strides;

  bool use_layout = trust_operand_layout && operand_shape.IsArray() &&
                    operand_shape.has_layout() &&
                    operand_shape.dimensions_size() ==
                        static_cast<int64_t>(mm_shape.size());

  if (use_layout) {
    const int64_t rank = static_cast<int64_t>(mm_shape.size());
    const auto& layout = operand_shape.layout();
    const auto& m2n = layout.minor_to_major();

    if (layout.minor_to_major_size() != rank) {
      use_layout = false;
    }

    if (use_layout && layout.tiles_size() != 0) {
      use_layout = false;
    }

    if (use_layout) {
      for (int64_t i = 0; i < rank; ++i) {
        if (operand_shape.dimensions(i) != mm_shape[i]) {
          use_layout = false;
          break;
        }
      }
    }

    if (use_layout) {
      strides.assign(mm_shape.size(), 1);
      std::vector<bool> seen(mm_shape.size(), false);

      int64_t acc = 1;
      for (int64_t dim : m2n) {
        if (dim < 0 || dim >= rank || seen[dim]) {
          use_layout = false;
          break;
        }

        seen[dim] = true;
        strides[dim] = acc;
        acc *= operand_shape.dimensions(dim);
      }

      if (use_layout) {
        for (int64_t i = 0; i < rank; ++i) {
          if (!seen[i]) {
            use_layout = false;
            break;
          }
        }
      }
    }
  }

  if (!use_layout) {
    strides = MakeCOrderStrides(mm_shape);
  }

  return dep::aclCreateTensor(mm_shape.data(), mm_shape.size(), data_type,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              mm_shape.data(), mm_shape.size(), device_addr);
}

constexpr int64_t kMaxLeftOrderDotK = 64;

bool IsDefaultCOrderLayout(const Shape& shape) {
  if (!shape.IsArray() || !shape.has_layout()) return false;

  const int64_t rank = shape.dimensions_size();
  const auto& m2n = shape.layout().minor_to_major();

  if (m2n.size() != rank) return false;

  for (int64_t i = 0; i < rank; ++i) {
    if (m2n[i] != rank - 1 - i) return false;
  }

  return true;
}

bool LeadingDimsEqual(const Shape& lhs, const Shape& rhs) {
  const int64_t lhs_rank = lhs.dimensions_size();
  const int64_t rhs_rank = rhs.dimensions_size();
  if (!lhs.IsArray() || !rhs.IsArray() || lhs_rank != rhs_rank) {
    return false;
  }
  for (int64_t i = 0; i + 1 < lhs_rank; ++i) {
    if (lhs.dimensions(i) != rhs.dimensions(i)) {
      return false;
    }
  }
  return true;
}

bool OutputMatchesLeadingDims(const Shape& input_shape,
                              const Shape& out_shape) {
  const int64_t input_rank = input_shape.dimensions_size();
  if (!out_shape.IsArray() ||
      out_shape.dimensions_size() != input_rank - 1) {
    return false;
  }
  for (int64_t i = 0; i + 1 < input_rank; ++i) {
    if (out_shape.dimensions(i) != input_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

bool DimsArePrefixBatch(const std::vector<int64_t>& lhs_batch,
                        const std::vector<int64_t>& rhs_batch,
                        int64_t rank) {
  if (lhs_batch.size() != static_cast<size_t>(rank - 1) ||
      rhs_batch.size() != static_cast<size_t>(rank - 1)) {
    return false;
  }
  for (int64_t dim = 0; dim + 1 < rank; ++dim) {
    if (lhs_batch[dim] != dim || rhs_batch[dim] != dim) {
      return false;
    }
  }
  return true;
}

bool IsLeftOrderNoNeumaierDotSupported(
    const Shape& lhs_shape, const Shape& rhs_shape, const Shape& out_shape,
    const std::vector<int64_t>& lhs_batch,
    const std::vector<int64_t>& rhs_batch,
    const std::vector<int64_t>& lhs_contract,
    const std::vector<int64_t>& rhs_contract) {
  if (lhs_shape.element_type() != F32 || rhs_shape.element_type() != F32 ||
      out_shape.element_type() != F32 || !lhs_shape.IsArray() ||
      !rhs_shape.IsArray() || !out_shape.IsArray()) {
    return false;
  }

  const int64_t lhs_rank = lhs_shape.dimensions_size();
  const int64_t rhs_rank = rhs_shape.dimensions_size();
  if (lhs_rank < 1 || rhs_rank < 1 || lhs_contract.size() != 1 ||
      rhs_contract.size() != 1 || lhs_contract[0] != lhs_rank - 1 ||
      rhs_contract[0] != rhs_rank - 1) {
    return false;
  }

  const int64_t dot_len = lhs_shape.dimensions(lhs_rank - 1);
  if (dot_len <= 0 || dot_len > kMaxLeftOrderDotK ||
      rhs_shape.dimensions(rhs_rank - 1) != dot_len) {
    return false;
  }

  if (!IsDefaultCOrderLayout(lhs_shape) || !IsDefaultCOrderLayout(rhs_shape) ||
      !IsDefaultCOrderLayout(out_shape)) {
    return false;
  }

  const bool lhs_vector = lhs_rank == 1;
  const bool rhs_vector = rhs_rank == 1;
  const bool has_no_batch_dims = lhs_batch.empty() && rhs_batch.empty();
  if (lhs_vector && rhs_vector) {
    return has_no_batch_dims && ShapeUtil::IsScalar(out_shape);
  }

  if (lhs_vector) {
    return has_no_batch_dims && OutputMatchesLeadingDims(rhs_shape, out_shape);
  }

  if (rhs_vector) {
    return has_no_batch_dims && OutputMatchesLeadingDims(lhs_shape, out_shape);
  }

  return lhs_rank == rhs_rank && LeadingDimsEqual(lhs_shape, rhs_shape) &&
         DimsArePrefixBatch(lhs_batch, rhs_batch, lhs_rank) &&
         OutputMatchesLeadingDims(lhs_shape, out_shape);
}

absl::StatusOr<aclScalar*> CreateScalarFromDeviceBuffer(
    void* device_addr, PrimitiveType primitive_type) {
  const int64_t byte_width = primitive_util::ByteWidth(primitive_type);
  if (byte_width <= 0 || byte_width > 8) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported scalar byte width for pad value: ",
                     byte_width));
  }

  alignas(8) unsigned char host_storage[8] = {};
  aclError memcpy_ret =
      dep::aclrtMemcpy(host_storage, byte_width, device_addr, byte_width,
                       ACL_MEMCPY_DEVICE_TO_HOST);
  if (memcpy_ret != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Failed to copy pad scalar from device, acl error: ",
                     memcpy_ret));
  }

  aclScalar* scalar = dep::aclCreateScalar(
      host_storage, NpuTool::ConvertXlaType2AclDataType(primitive_type));
  if (scalar == nullptr) {
    return absl::InternalError("Failed to create aclScalar from pad value");
  }
  return scalar;
}

}  // namespace

static bool CheckInputOuputParamsSize(HloOpcode op_code,
                                      const std::vector<Shape>& param_shapes) {
  switch (op_code) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kTanh:
    case HloOpcode::kReshape:
    case HloOpcode::kBitcastConvert:  // bit reinterpret, same as reshape (1 in + 1 out)
    case HloOpcode::kConvert:  // type conversion, same as reshape (1 in + 1 out)
    case HloOpcode::kNegate:
    case HloOpcode::kExp:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kCos:
    case HloOpcode::kIsFinite:
    case HloOpcode::kAbs:
    case HloOpcode::kSign:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kSqrt:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kCholesky:  // Cholesky: 1 input + 1 output = 2 params
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Param shape size %d does not meet target value 2",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kIota:  // iota has no inputs, only 1 output shape
      if (param_shapes.size() != 1) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Param shape size %d does not meet target value 1",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kMultiply:
    case HloOpcode::kDot:
    case HloOpcode::kDivide:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kTriangularSolve:  // 2 inputs (A, b) + 1 output = 3 params
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Param shape size %d does not meet target value 3",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kConcatenate:
      // Concatenate is a variable-length parameter operator, requiring at least 2 inputs + 1 output = at least 3 parameters
      if (param_shapes.size() < 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "Concatenate param shape size %d is less than 3",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kReduce:
      // Variadic Reduce: N inputs + N initial values + N outputs = 3*N parameters
      if (param_shapes.size() < 3 || param_shapes.size() % 3 != 0) {
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "Reduce param shape size %d is not a multiple of 3 (expected 3*N)",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kReduceWindow:
      // ReduceWindow operation: input + initial value + output = 3 parameters
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "ReduceWindow param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kPad:
      // Pad 操作：operand + pad_value + output = 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Pad param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kTranspose:
      // Transpose operation: input + output = 2 parameters
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Transpose param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kSlice:
      // Slice operation: input + output = 2 parameters
      // Slice is a static slice, the parameters are determined at compile time, and no additional start_indices parameters are required.
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Slice param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kDynamicSlice:
      // DynamicSlice operation: input + start_indices... + output = at least 3 parameters
      // start_indices count = param_shapes.size() - 2 (one for each slice dimension)
      if (param_shapes.size() < 3) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "DynamicSlice param shape size %d is less than 3 "
                             "(input + start_indices + output)",
                             param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kDynamicUpdateSlice:
      // DynamicUpdateSlice 操作：operand + update + start_indices... + 输出
      // 至少 4 个参数（operand + update + 1 start_index + output）
      // start_indices 数量 = param_shapes.size() - 3（每个维度一个）
      if (param_shapes.size() < 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "DynamicUpdateSlice param shape size %d is less than 4 "
            "(operand + update + start_indices + output)",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kGather:
      // Gather operation: input + index + output = 3 parameters
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Gather param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kCompare:
      // Compare operation: input 1 + input 2 + output = 3 parameters
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Compare param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kSelect:
      // Select (Where) operation: condition + self + other + output = 4 parameters
      if (param_shapes.size() != 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Select param shape size %d does not meet target value 4",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kNot:
      // BitwiseNot operation: 1 input + 1 output = 2 parameters
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "%s param shape size %d does not meet target value 2",
            HloOpcodeString(op_code), param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      // BitwiseAnd/BitwiseOr/BitwiseXor operation: input1 + input2 + output = 3 parameters
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "%s param shape size %d does not meet target value 3",
            HloOpcodeString(op_code), param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kClamp:
      // Clamp 操作：min + operand + max + output = 4 个参数
      if (param_shapes.size() != 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Clamp param shape size %d does not meet target value 4",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kScatter:
      // Scatter operation: data + indices + updates + output = 4 parameters
      if (param_shapes.size() != 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Scatter param shape size %d does not meet target value 4",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kCopy:
      // Copy operation: input + output = 2 parameters (same as Reshape)
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Copy param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kSort:
      // Sort operation: supports single input and single output (jnp.sort, 2 params) and
      // Double input and double output (jnp.argsort, 4 params: values_in + iota_in +
      // values_out + indices_out）
      if (param_shapes.size() != 2 && param_shapes.size() != 4) {
        NPU_LOG_FORMAT_PRINT(
            ERROR,
            "Sort supports single-operand (2 params) or dual-operand (4 "
            "params), got %d",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kTopK:
      // TopK 操作：1 输入 + 2 个 tuple 输出（values, indices）= 3 个参数
      if (param_shapes.size() != 3) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "TopK param shape size %d does not meet target value 3",
            param_shapes.size());
        return false;
      }
      break;
    case HloOpcode::kReverse:
      // Reverse operation: 1 input + 1 output = 2 parameters
      if (param_shapes.size() != 2) {
        NPU_LOG_FORMAT_PRINT(
            ERROR, "Reverse param shape size %d does not meet target value 2",
            param_shapes.size());
        return false;
      }
      break;
    default:
      NPU_LOG_FORMAT_PRINT(ERROR, "Op code(%v) does not support check now",
                           HloOpcodeString(op_code));
  }

  return true;
}

static bool IsAclnnCastType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::PRED:
    case PrimitiveType::F16:
    case PrimitiveType::F32:
    case PrimitiveType::F64:
    case PrimitiveType::S8:
    case PrimitiveType::U8:
    case PrimitiveType::S16:
    case PrimitiveType::U16:
    case PrimitiveType::S32:
    case PrimitiveType::U32:
    case PrimitiveType::S64:
    case PrimitiveType::U64:
      return true;
    default:
      return false;
  }
}

// Validates the shapes for the single-slice IndexCopy scatter lowering:
// the update tensor is the operand slice with the scatter dimension removed.
static bool CanUseSingleSliceIndexCopy(const std::vector<Shape>& param_shapes,
                                       int64_t scatter_dimension) {
  if (param_shapes.size() != 4 || scatter_dimension < 0) {
    return false;
  }

  const Shape& data_shape = param_shapes[0];
  const Shape& index_shape = param_shapes[1];
  const Shape& update_shape = param_shapes[2];
  if (!data_shape.IsArray() || !index_shape.IsArray() ||
      !update_shape.IsArray() ||
      scatter_dimension >= data_shape.dimensions_size() ||
      index_shape.dimensions_size() > 1 ||
      ShapeUtil::ElementsIn(index_shape) != 1 ||
      update_shape.dimensions_size() + 1 != data_shape.dimensions_size()) {
    return false;
  }

  int64_t update_dim = 0;
  for (int64_t data_dim = 0; data_dim < data_shape.dimensions_size();
       ++data_dim) {
    if (data_dim == scatter_dimension) {
      continue;
    }
    if (data_shape.dimensions(data_dim) !=
        update_shape.dimensions(update_dim++)) {
      return false;
    }
  }
  return true;
}

static std::vector<int64_t> BuildSingleSliceSourceShape(
    const Shape& data_shape, int64_t scatter_dimension) {
  std::vector<int64_t> source_shape(data_shape.dimensions().begin(),
                                    data_shape.dimensions().end());
  source_shape[scatter_dimension] = 1;
  return source_shape;
}

static bool CanUseStaticWindowCopy(const std::vector<Shape>& param_shapes,
                                   int64_t scatter_dimension,
                                   int64_t scatter_window_start) {
  if (param_shapes.size() != 4 || scatter_dimension < 0 ||
      scatter_window_start < 0) {
    return false;
  }

  const Shape& data_shape = param_shapes[0];
  const Shape& index_shape = param_shapes[1];
  const Shape& update_shape = param_shapes[2];
  if (!data_shape.IsArray() || !index_shape.IsArray() ||
      !update_shape.IsArray() ||
      scatter_dimension >= data_shape.dimensions_size() ||
      ShapeUtil::ElementsIn(index_shape) != 1 ||
      update_shape.dimensions_size() != data_shape.dimensions_size() ||
      scatter_window_start + update_shape.dimensions(scatter_dimension) >
          data_shape.dimensions(scatter_dimension)) {
    return false;
  }

  for (int64_t dim = 0; dim < data_shape.dimensions_size(); ++dim) {
    if (dim != scatter_dimension &&
        data_shape.dimensions(dim) != update_shape.dimensions(dim)) {
      return false;
    }
  }
  return true;
}

static std::vector<int64_t> BuildRowMajorStrides(
    absl::Span<const int64_t> dimensions) {
  std::vector<int64_t> strides(dimensions.size(), 1);
  for (int64_t dim = static_cast<int64_t>(dimensions.size()) - 2; dim >= 0;
       --dim) {
    strides[dim] = dimensions[dim + 1] * strides[dim + 1];
  }
  return strides;
}

static bool CanUseBatchedSingleDimScatter(
    const std::vector<Shape>& param_shapes, int64_t scatter_dimension,
    int64_t batch_dimension, int64_t indices_batch_dimension) {
  if (param_shapes.size() != 4 || scatter_dimension < 0 ||
      batch_dimension < 0 || indices_batch_dimension < 0) {
    return false;
  }

  const Shape& data_shape = param_shapes[0];
  const Shape& index_shape = param_shapes[1];
  const Shape& update_shape = param_shapes[2];
  if (!data_shape.IsArray() || !index_shape.IsArray() ||
      !update_shape.IsArray() ||
      scatter_dimension >= data_shape.dimensions_size() ||
      batch_dimension >= data_shape.dimensions_size() ||
      indices_batch_dimension >= index_shape.dimensions_size()) {
    return false;
  }

  // The index tensor for batch scatter is (B,..., N,..., 1), and the last dimension is index_vector
  // Must be size-1 (single-dimensional scatter)
  const int64_t index_vector_dim = index_shape.dimensions_size() - 1;
  if (index_shape.dimensions(index_vector_dim) != 1) {
    return false;
  }

  // Batch dimension sizes must match
  if (data_shape.dimensions(batch_dimension) !=
      index_shape.dimensions(indices_batch_dimension)) {
    return false;
  }

  // Calculate the number of scatter points: the product of non-batch and non-index_vector dimensions in index
  int64_t n_scatter = 1;
  for (int64_t i = 0; i < index_shape.dimensions_size(); ++i) {
    if (i != indices_batch_dimension && i != index_vector_dim) {
      n_scatter *= index_shape.dimensions(i);
    }
  }

  // update shape verification: supports two modes
  // Mode A: update.rank == data.rank, update[i]=n_scatter on scatter_dim
  // Mode B: update.rank == data.rank - 1, scatter_dim is removed (single point scatter)
  if (update_shape.dimensions_size() == data_shape.dimensions_size()) {
    for (int64_t i = 0; i < data_shape.dimensions_size(); ++i) {
      if (i == scatter_dimension) {
        if (update_shape.dimensions(i) != n_scatter) return false;
      } else {
        if (update_shape.dimensions(i) != data_shape.dimensions(i)) return false;
      }
    }
  } else if (update_shape.dimensions_size() + 1 ==
             data_shape.dimensions_size() && n_scatter == 1) {
    // update is missing scatter_dim, the remaining dimensions must match data one by one
    for (int64_t di = 0, ui = 0; di < data_shape.dimensions_size(); ++di) {
      if (di == scatter_dimension) continue;
      if (update_shape.dimensions(ui) != data_shape.dimensions(di)) return false;
      ++ui;
    }
  } else {
    return false;
  }
  return true;
}

static bool CanUseSameRankSingleDimScatter(
    const std::vector<Shape>& param_shapes, int64_t scatter_dimension) {
  if (param_shapes.size() != 4 || scatter_dimension < 0) {
    return false;
  }

  const Shape& data_shape = param_shapes[0];
  const Shape& index_shape = param_shapes[1];
  const Shape& update_shape = param_shapes[2];
  // Accepts 1D indices (N,) or 2D indices (N,1)
  bool indices_ok =
      (index_shape.dimensions_size() == 2 &&
       index_shape.dimensions(1) == 1) ||
      index_shape.dimensions_size() == 1;
  if (!data_shape.IsArray() || !index_shape.IsArray() ||
      !update_shape.IsArray() ||
      scatter_dimension >= data_shape.dimensions_size() || !indices_ok ||
      update_shape.dimensions_size() != data_shape.dimensions_size() ||
      update_shape.dimensions(scatter_dimension) !=
          index_shape.dimensions(0)) {
    return false;
  }

  for (int64_t dim = 0; dim < data_shape.dimensions_size(); ++dim) {
    if (dim == scatter_dimension) {
      continue;
    }
    if (data_shape.dimensions(dim) != update_shape.dimensions(dim)) {
      return false;
    }
  }
  return true;
}


NpuAclnnThunk::NpuAclnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                             std::vector<BufferAllocation::Slice> args,
                             std::vector<bool> output_args,
                             std::optional<int64_t> sdpa_dropout_seed)
    : NpuThunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      // output_args_(std::move(output_args)) indicates which memory addresses in args_ will be written to for memory multiplexing.
      // Assuming that the args_ size of add is 3, it is possible that the last two arg memory addresses are the same and both are writable.
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status NpuAclnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();

  // Handle all special values (-inf, +inf, nan) in the constant buffer during the Initialize phase
  // These special values are only written during graph allocation and will not be generated during calculations.
  TF_RETURN_IF_ERROR(NpuThunk::InitializeSpecialValues(params, args_));
  return ret;
}

absl::Status NpuAclnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "==== Launch aclnn kernel with %d param ====", args_.size());
  NPU_LOG_FORMAT_PRINT(DEBUG, "---- Corresponding hlo instruction: %s ----",
                       thunk_info().hloInstrDesc);

  // Initialize should have been called before execution and should not be called again here

  if (params.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Stream need to be created or init before execute npu thunk.");
    return absl::InternalError("ACLNN thunk received a null stream");
  }
  auto stream_handle = params.stream->platform_specific_handle();
  if (stream_handle.stream == nullptr) {
    NPU_LOG_FORMAT_PRINT(
        ERROR, "Stream need to be created or init before execute npu thunk.");
    return absl::InternalError("ACLNN thunk received a null stream handle");
  }
  auto stream = stream_handle.stream;
  auto acl_stream = static_cast<aclrtStream>(stream_handle.stream);
  std::vector<void*> temporary_device_ptrs;
  auto allocate_temporary_device_memory =
      [acl_stream,
       &temporary_device_ptrs](size_t size) -> absl::StatusOr<void*> {
    absl::StatusOr<void*> allocation =
        NpuStreamOrderedDeviceMemoryPool::GetInstance()->AllocateForStream(
            acl_stream, size);
    if (!allocation.ok()) return allocation.status();
    if (*allocation != nullptr) temporary_device_ptrs.push_back(*allocation);
    return *allocation;
  };

  const NpuBufferAllocations* allocations = params.buffer_allocations;
  const auto& param_shapes = thunk_info().param_shapes;
  aclnnStatus ret = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto exec_ret = absl::OkStatus();

  // Keep the default path asynchronous. Workspace and temporary device pools
  // hand each in-flight operation a distinct stream-ordered slice, and host
  // metadata is retained until a successful stream drain.

  // Tril: independent parameter verification (aclnnTril requires 1 input + 1 output, 2 param)
  if (thunk_info().is_tril) {
    if (param_shapes.size() != 2) {
      return absl::InternalError(absl::StrCat(
          "Tril expects 2 param shapes (input, output), got ",
          param_shapes.size()));
    }
  } else if (thunk_info().is_fma_addcmul) {
    if (param_shapes.size() != 4) {
      return absl::InternalError(absl::StrCat(
          "aclnnFmaAddcmul expects 3 input shapes and 1 output shape, got ",
          param_shapes.size()));
    }
  } else {
    if (!CheckInputOuputParamsSize(thunk_info().op_code, param_shapes)) {
      return absl::InternalError(absl::StrCat(
          "ACLNN thunk parameter validation failed for opcode ",
          HloOpcodeString(thunk_info().op_code)));
    }
  }

  // Print input parameters before execution
  if (IsPrintIOValuesEnabled()) {
    int device_ordinal = params.stream->parent()->device_ordinal();
    for (int i = 0; i < args_.size() - 1; ++i) {
      auto addr = allocations->GetDeviceAddress(args_[i]);
      PrintParamInfo(i, "input", device_ordinal, args_[i], param_shapes[i],
                     addr.opaque());
    }
  }

  // Cat owns a dedicated, stable descriptor set below. Avoid even creating
  // transient descriptors after that set has been initialized.
  std::vector<aclTensor*> acl_tensor_ptr_vector;
  if (thunk_info().op_code != HloOpcode::kConcatenate) {
    TF_ASSIGN_OR_RETURN(
        acl_tensor_ptr_vector,
        ExtractAclInputOutTensorPtr(allocations, param_shapes));
  }
  NpuStreamOrderedDeviceMemoryPool::RetiredAclResources acl_resources;
  for (const auto& arg : args_) {
    void* device_ptr = allocations->GetDeviceAddress(arg).opaque();
    if (device_ptr != nullptr) acl_resources.device_ptrs.push_back(device_ptr);
  }

  auto retire_acl_tensor = [&acl_resources](aclTensor* tensor) {
    if (tensor != nullptr) acl_resources.tensors.push_back(tensor);
  };
  auto retire_acl_scalar = [&acl_resources](aclScalar* scalar) {
    if (scalar != nullptr) acl_resources.scalars.push_back(scalar);
  };
  auto retire_acl_int_array = [&acl_resources](aclIntArray* array) {
    if (array != nullptr) acl_resources.int_arrays.push_back(array);
  };

  // On an error path, a complex thunk may already have queued an earlier
  // ACLNN operation. Drain before destroying its metadata. Once the complete
  // thunk succeeds, ownership moves to the graph-level stream generation.
  bool acl_resources_retired = false;
  auto acl_resource_cleanup = absl::MakeCleanup([&acl_tensor_ptr_vector,
                                                  &acl_resources,
                                                  &acl_resources_retired,
                                                  &acl_stream]() {
    if (acl_resources_retired) {
      return;
    }
    aclError sync_ret = dep::aclrtSynchronizeStream(acl_stream);
    if (sync_ret != ACL_SUCCESS) {
      NPU_LOG_FORMAT_PRINT(
          ERROR,
          "Failed to drain ACLNN stream before error-path metadata cleanup: "
          "%d; leaking descriptors to avoid asynchronous use-after-free",
          sync_ret);
      return;
    }
    acl_resources.tensors.insert(acl_resources.tensors.end(),
                                 acl_tensor_ptr_vector.begin(),
                                 acl_tensor_ptr_vector.end());
    acl_tensor_ptr_vector.clear();
    DestroyAclResources(acl_resources);
  });

  // Print allocation_index of input and output, used to diagnose alias problems
  if (args_.size() >= 2) {
    NPU_LOG_FORMAT_PRINT(
        DEBUG,
        "Allocation indices: input=%d, output=%d",
        args_[0].index(),
        args_.back().index());
  }

  if (thunk_info().is_tril) {
    // Select the inplace or non-inplace version based on whether the input and output buffers are shared.
    bool is_inplace = (args_.size() >= 2 && args_[0] == args_.back());
    if (is_inplace) {
      ret = dep::aclnnInplaceTrilGetWorkspaceSize(
          acl_tensor_ptr_vector[0], thunk_info().tril_diagonal,
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrFormat(
            "aclnnInplaceTrilGetWorkspaceSize failed for diagonal=%lld. "
            "Error code: %d",
            (long long)thunk_info().tril_diagonal, ret));
      }
      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnInplaceTril);
    } else {
      ret = dep::aclnnTrilGetWorkspaceSize(
          acl_tensor_ptr_vector[0], thunk_info().tril_diagonal,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrFormat(
            "aclnnTrilGetWorkspaceSize failed for diagonal=%lld. "
            "Error code: %d",
            (long long)thunk_info().tril_diagonal, ret));
      }
      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnTril);
    }
    // Skip switch dispatch and go straight to stream synchronization (goto is standard practice in this giant function)
    goto tril_dispatch_done;
  }

  if (thunk_info().is_fma_addcmul) {
    float value = thunk_info().fma_addcmul_value;
    aclScalar* scalar = dep::aclCreateScalar(&value, ACL_FLOAT);
    if (scalar == nullptr) {
      return absl::InternalError("Failed to create aclScalar for aclnnFmaAddcmul");
    }
    auto scalar_cleanup = absl::MakeCleanup(
        [&] { retire_acl_scalar(scalar); });
    ret = dep::aclnnFmaAddcmulGetWorkspaceSize(
        acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
        acl_tensor_ptr_vector[2], scalar, acl_tensor_ptr_vector.back(),
        &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
      return absl::InternalError(absl::StrFormat(
          "aclnnFmaAddcmulGetWorkspaceSize failed. Error code: %d", ret));
    }
    exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                            dep::aclnnFmaAddcmul);
    goto tril_dispatch_done;
  }

  switch (thunk_info().op_code) {
    case HloOpcode::kBitcastConvert: {
      // BitcastConvert: reinterpret bits without conversion, pure memcpy
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "BitcastConvert: using aclrtMemcpyAsync to copy data");

      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_.back());
      void* input_device_addr = input_addr.opaque();
      void* output_device_addr = output_addr.opaque();
      int64_t input_size =
          NpuTool::GetXlaShapeSize(param_shapes[0]) *
          ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());

      if (input_size == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      ret = dep::aclrtMemcpyAsync(output_device_addr, input_size,
                                  input_device_addr, input_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclrtMemcpyAsync for bitcast-convert failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kReshape: {
      // Reshape operation: copy data from input to output using aclrtMemcpy
      // Even if shapes are different, the underlying data size is the same
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Reshape operation: using aclrtMemcpyAsync to copy data");

      // Get input and output buffer addresses
      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_.back());
      void* input_device_addr = input_addr.opaque();
      void* output_device_addr = output_addr.opaque();

      // Calculate data size in bytes
      int64_t input_size =
          NpuTool::GetXlaShapeSize(param_shapes[0]) *
          ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());

      // Zero-size reshape is a no-op.
      if (input_size == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      // Perform device-to-device memory copy
      ret = dep::aclrtMemcpyAsync(output_device_addr, input_size,
                                  input_device_addr, input_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclrtMemcpyAsync for reshape failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kCopy: {
      // Copy operation: device-to-device memory copy, same as Reshape.
      // Used by scatter-copy-insertion pass to break parameter aliasing.
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Copy operation: using aclrtMemcpyAsync");

      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_.back());
      void* input_device_addr = input_addr.opaque();
      void* output_device_addr = output_addr.opaque();

      int64_t data_size =
          NpuTool::GetXlaShapeSize(param_shapes[0]) *
          ShapeUtil::ByteSizeOfPrimitiveType(param_shapes[0].element_type());

      if (data_size == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      aclError copy_ret =
          dep::aclrtMemcpyAsync(output_device_addr, data_size,
                                input_device_addr, data_size,
                                ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
      if (copy_ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclrtMemcpyAsync for copy failed. Error code: %d", copy_ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kConvert: {
      // Prefer device-side Cast; keep the host fallback for types ACLNN Cast
      // does not cover yet.
      NPU_LOG_FORMAT_PRINT(DEBUG, "Convert operation: type cast");

      const Shape& in_shape = param_shapes[0];
      const Shape& out_shape = param_shapes[1];
      PrimitiveType src_type = in_shape.element_type();
      PrimitiveType dst_type = out_shape.element_type();
      int64_t num_elements = ShapeUtil::ElementsIn(in_shape);
      int64_t src_elem_size = ShapeUtil::ByteSizeOfPrimitiveType(src_type);
      int64_t dst_elem_size = ShapeUtil::ByteSizeOfPrimitiveType(dst_type);

      if (num_elements == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      if (IsAclnnCastType(src_type) && IsAclnnCastType(dst_type)) {
        aclDataType dst_acl_type =
            NpuTool::ConvertXlaType2AclDataType(dst_type);
        ret = dep::aclnnCastGetWorkspaceSize(
            acl_tensor_ptr_vector[0], dst_acl_type, acl_tensor_ptr_vector[1],
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg =
              absl::StrFormat("aclnnCastGetWorkspaceSize failed: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        if (executor == nullptr) {
          const std::string err_msg = "aclnnCast: nullptr executor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        exec_ret =
            ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnCast);
        break;
      }

      // Read input from device
      void* src_host = malloc(num_elements * src_elem_size);
      void* dst_host = malloc(num_elements * dst_elem_size);
      if (!src_host || !dst_host) {
        free(src_host);
        free(dst_host);
        return absl::InternalError("Convert: host malloc failed");
      }

      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      aclError read_ret = dep::aclrtMemcpy(
          src_host, num_elements * src_elem_size,
          input_addr.opaque(), num_elements * src_elem_size,
          ACL_MEMCPY_DEVICE_TO_HOST);
      if (read_ret != ACL_SUCCESS) {
        free(src_host);
        free(dst_host);
        return absl::InternalError(
            absl::StrFormat("Convert D2H read failed: %d", read_ret));
      }

      // Perform element-wise type conversion on host
      for (int64_t i = 0; i < num_elements; ++i) {
        // Read source value as double (widest common type for simple cases)
        double val = 0.0;
        switch (src_type) {
          case PrimitiveType::PRED: val = static_cast<const bool*>(src_host)[i]; break;
          case PrimitiveType::S32: val = static_cast<const int32_t*>(src_host)[i]; break;
          case PrimitiveType::S64: val = static_cast<const int64_t*>(src_host)[i]; break;
          case PrimitiveType::U32: val = static_cast<const uint32_t*>(src_host)[i]; break;
          case PrimitiveType::U64: val = static_cast<const uint64_t*>(src_host)[i]; break;
          case PrimitiveType::F32: val = static_cast<const float*>(src_host)[i]; break;
          case PrimitiveType::F64: val = static_cast<const double*>(src_host)[i]; break;
          case PrimitiveType::S16: val = static_cast<const int16_t*>(src_host)[i]; break;
          case PrimitiveType::U16: val = static_cast<const uint16_t*>(src_host)[i]; break;
          case PrimitiveType::S8:  val = static_cast<const int8_t*>(src_host)[i]; break;
          case PrimitiveType::U8:  val = static_cast<const uint8_t*>(src_host)[i]; break;
          default:
            free(src_host); free(dst_host);
            return absl::InternalError("Convert: unsupported src type");
        }

        // Write to destination buffer
        switch (dst_type) {
          case PrimitiveType::PRED: static_cast<bool*>(dst_host)[i] = static_cast<bool>(val); break;
          case PrimitiveType::S32: static_cast<int32_t*>(dst_host)[i] = static_cast<int32_t>(val); break;
          case PrimitiveType::S64: static_cast<int64_t*>(dst_host)[i] = static_cast<int64_t>(val); break;
          case PrimitiveType::U32: static_cast<uint32_t*>(dst_host)[i] = static_cast<uint32_t>(val); break;
          case PrimitiveType::U64: static_cast<uint64_t*>(dst_host)[i] = static_cast<uint64_t>(val); break;
          case PrimitiveType::F32: static_cast<float*>(dst_host)[i] = static_cast<float>(val); break;
          case PrimitiveType::F64: static_cast<double*>(dst_host)[i] = val; break;
          case PrimitiveType::S16: static_cast<int16_t*>(dst_host)[i] = static_cast<int16_t>(val); break;
          case PrimitiveType::U16: static_cast<uint16_t*>(dst_host)[i] = static_cast<uint16_t>(val); break;
          case PrimitiveType::S8:  static_cast<int8_t*>(dst_host)[i] = static_cast<int8_t>(val); break;
          case PrimitiveType::U8:  static_cast<uint8_t*>(dst_host)[i] = static_cast<uint8_t>(val); break;
          default:
            free(src_host); free(dst_host);
            return absl::InternalError("Convert: unsupported dst type");
        }
      }

      // Write result to device
      auto output_addr = allocations->GetDeviceAddress(args_[1]);
      aclError write_ret = dep::aclrtMemcpy(
          output_addr.opaque(), num_elements * dst_elem_size,
          dst_host, num_elements * dst_elem_size,
          ACL_MEMCPY_HOST_TO_DEVICE);

      free(src_host);
      free(dst_host);

      if (write_ret != ACL_SUCCESS) {
        return absl::InternalError(
            absl::StrFormat("Convert H2D write failed: %d", write_ret));
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kBroadcast: {
      std::vector<int64_t> output_param_size_data =
          NpuTool::ConvertXlaShape2VectorShape(param_shapes.back());
      std::vector<int64_t> input_shape_data =
          NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);

      // Debug: print broadcast shapes
      std::string shape_str = "[";
      for (size_t i = 0; i < output_param_size_data.size(); ++i) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(output_param_size_data[i]);
      }
      shape_str += "]";
      NPU_LOG_FORMAT_PRINT(DEBUG, "Broadcast: input shape %s, target size %s",
                           NpuTool::GetXlaShapeString(param_shapes[0]).c_str(),
                           shape_str.c_str());

      // Check if input and output shapes are the same (identity broadcast)
      if (input_shape_data == output_param_size_data) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Identity broadcast detected, using stream Memcpy "
                             "instead of aclnnExpand");

        // Get input and output buffer addresses
        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        auto output_addr = allocations->GetDeviceAddress(args_.back());

        // Calculate data size
        int64_t data_size = NpuTool::GetXlaShapeSize(param_shapes.back()) *
                            ShapeUtil::ByteSizeOfPrimitiveType(
                                param_shapes.back().element_type());

        // Perform device-to-device memory copy using stream
        exec_ret = params.stream->Memcpy(&output_addr, input_addr, data_size);
        if (!exec_ret.ok()) {
          NPU_LOG_FORMAT_PRINT(
              ERROR, "Stream Memcpy for identity broadcast failed: %s",
              exec_ret.ToString().c_str());
          return exec_ret;
        }

        break;
      }

      // Handle dimension expansion for aclnnExpand
      // aclnnExpand requires input and output to have the same number of dimensions
      // If output has more dimensions, we need to create a view with expanded dimensions
      aclTensor* input_tensor = acl_tensor_ptr_vector[0];
      // RAII guard: 如果维度扩展创建了新 tensor，确保在所有路径下销毁它。
      // 当 input_tensor == acl_tensor_ptr_vector[0]（未扩展）时，guard 不执行销毁，
      // 因为原始 tensor 已由外层 acl_resource_cleanup 管理。
      aclTensor* expanded_tensor_to_cleanup = nullptr;
      auto expanded_tensor_cleanup = absl::MakeCleanup([&]() {
        if (expanded_tensor_to_cleanup != nullptr) {
          retire_acl_tensor(expanded_tensor_to_cleanup);
        }
      });

      if (output_param_size_data.size() > input_shape_data.size()) {
        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Broadcast: need to expand dimensions from %d to %d",
            input_shape_data.size(), output_param_size_data.size());

        // Get dimensions parameter from HLO instruction
        const auto& broadcast_dims = thunk_info().dimensions;

        // Create expanded shape and strides for the view tensor
        std::vector<int64_t> expanded_shape(output_param_size_data.size(), 1);
        std::vector<int64_t> expanded_strides(output_param_size_data.size(), 0);

        // Map input dimensions to output dimensions based on broadcast_dims
        for (size_t i = 0; i < broadcast_dims.size(); ++i) {
          int64_t out_dim = broadcast_dims[i];
          expanded_shape[out_dim] = input_shape_data[i];
        }

        // Calculate strides for the view tensor
        // Dimensions with size 1 should have stride 0 (for broadcasting)
        // Dimensions mapped from input should have proper strides
        int64_t stride = 1;
        for (int i = output_param_size_data.size() - 1; i >= 0; --i) {
          if (expanded_shape[i] == 1) {
            expanded_strides[i] = 0;  // stride 0 for broadcasting dimension
          } else {
            expanded_strides[i] = stride;
            stride *= expanded_shape[i];
          }
        }

        // Debug print
        std::string expanded_shape_str = "[";
        for (size_t i = 0; i < expanded_shape.size(); ++i) {
          if (i > 0) expanded_shape_str += ", ";
          expanded_shape_str += std::to_string(expanded_shape[i]);
        }
        expanded_shape_str += "]";

        std::string expanded_strides_str = "[";
        for (size_t i = 0; i < expanded_strides.size(); ++i) {
          if (i > 0) expanded_strides_str += ", ";
          expanded_strides_str += std::to_string(expanded_strides[i]);
        }
        expanded_strides_str += "]";

        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Broadcast: expanded view shape %s, strides %s",
            expanded_shape_str.c_str(), expanded_strides_str.c_str());

        // Create a view tensor with expanded dimensions
        aclDataType data_type =
            NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        void* device_addr = input_addr.opaque();

        input_tensor = dep::aclCreateTensor(
            expanded_shape.data(), expanded_shape.size(), data_type,
            expanded_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            expanded_shape.data(), expanded_shape.size(), device_addr);
        if (input_tensor == nullptr) {
          const std::string err_msg = "Failed to create expanded view tensor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        // Register to RAII guard to ensure that all paths (including subsequent early returns) can be automatically destroyed
        expanded_tensor_to_cleanup = input_tensor;
      }

      aclIntArray* output_param_size = dep::aclCreateIntArray(
          output_param_size_data.data(), output_param_size_data.size());

      // Call the first section of aclnnExpand interface
      ret = dep::aclnnExpandGetWorkspaceSize(input_tensor, output_param_size,
                                             acl_tensor_ptr_vector.back(),
                                             &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpandGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_int_array(output_param_size);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExpand);

      retire_acl_int_array(output_param_size);
      break;
    }

    case HloOpcode::kAdd: {
      // Optimization: for variance + epsilon scenario in RMSNorm
      // Detect input type, if it is FP16/BF16, try to use FP32 calculation to reduce accuracy loss
      aclDataType input_type =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      // RMSNorm key optimization: use FP32 calculation when FP16/BF16 input
      bool use_fp32_compute =
          (input_type == ACL_FLOAT16 || input_type == ACL_BF16);

      if (use_fp32_compute) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Add: Using FP32 compute for FP16/BF16 input to "
                             "improve precision (RMSNorm optimization)");
      }

      // Use union to store alpha values of different types
      union {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
        aclFloat16 f16;
        uint16_t bf16;
      } alpha_val;

      // If FP32 computation is enabled, create alpha using FP32 type
      aclDataType alpha_type = use_fp32_compute ? ACL_FLOAT : input_type;

      // Set alpha value based on type (uniformly 1 or 1.0)
      switch (alpha_type) {
        case ACL_INT32:
        case ACL_INT16:
        case ACL_INT8:
          alpha_val.i32 = 1;
          break;
        case ACL_INT64:
          alpha_val.i64 = 1;
          break;
        case ACL_UINT8:
          alpha_val.i32 = 1;
          break;
        case ACL_FLOAT:
          alpha_val.f32 = 1.0f;
          break;
        case ACL_DOUBLE:
          alpha_val.f64 = 1.0;
          break;
        case ACL_FLOAT16:
          alpha_val.f16 = 1.0f;
          break;
        case ACL_BF16:
          alpha_val.bf16 = 0x3F80;
          break;
        default:
          alpha_val.f32 = 1.0f;
          alpha_type = ACL_FLOAT;
          break;
      }

      aclScalar* alpha = dep::aclCreateScalar(&alpha_val, alpha_type);

      ret = dep::aclnnAddGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAddGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_scalar(alpha);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAdd);

      retire_acl_scalar(alpha);
      break;
    }

    case HloOpcode::kTanh: {
      ret = dep::aclnnTanhGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnTanhGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnTanh);
      break;
    }

    case HloOpcode::kMultiply: {
      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Multiply: input[0]=%p (shape=%s), input[1]=%p (shape=%s), output=%p",
                           acl_tensor_ptr_vector[0],
                           NpuTool::GetXlaShapeString(param_shapes[0]).c_str(),
                           acl_tensor_ptr_vector[1],
                           NpuTool::GetXlaShapeString(param_shapes[1]).c_str(),
                           acl_tensor_ptr_vector.back());
      
      // XLA inserts explicit broadcast operation before multiply
      // Therefore, the two input shapes of multiply should be the same and can be executed directly.
      ret = dep::aclnnMulGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMulGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMul);
      break;
    }

    case HloOpcode::kDot: {
      // Parse the dot dimension number and construct the standard shape required by aclnnMatmul
      // aclnnMatmul requires: lhs=[...,M,K], rhs=[...,K,N], out=[...,M,N]
      const auto& lhs_shape = param_shapes[0];
      const auto& rhs_shape = param_shapes[1];

      auto lhs_batch = ParseDotDimSet(thunk_info().hloInstrDesc,
                                      "lhs_batch_dims");
      auto rhs_batch = ParseDotDimSet(thunk_info().hloInstrDesc,
                                      "rhs_batch_dims");
      auto lhs_contract = ParseDotDimSet(thunk_info().hloInstrDesc,
                                         "lhs_contracting_dims");
      auto rhs_contract = ParseDotDimSet(thunk_info().hloInstrDesc,
                                         "rhs_contracting_dims");

      // Calculate batch, M(lhs non-batch non-contract), K, N(rhs non-batch non-contract)
      int64_t batch_size = 1, M = 1, K = 1, N = 1;
      for (auto d : lhs_batch) batch_size *= lhs_shape.dimensions(d);
      for (int i = 0; i < lhs_shape.dimensions_size(); ++i) {
        bool is_b = std::find(lhs_batch.begin(), lhs_batch.end(), i) !=
                    lhs_batch.end();
        bool is_c = std::find(lhs_contract.begin(), lhs_contract.end(), i) !=
                    lhs_contract.end();
        if (!is_b && !is_c) M *= lhs_shape.dimensions(i);
      }
      for (auto d : lhs_contract) K *= lhs_shape.dimensions(d);
      for (int i = 0; i < rhs_shape.dimensions_size(); ++i) {
        bool is_b = std::find(rhs_batch.begin(), rhs_batch.end(), i) !=
                    rhs_batch.end();
        bool is_c = std::find(rhs_contract.begin(), rhs_contract.end(), i) !=
                    rhs_contract.end();
        if (!is_b && !is_c) N *= rhs_shape.dimensions(i);
      }

      // Constructs the matmul standard shape
      bool has_batch = !lhs_batch.empty();
      std::vector<int64_t> lhs_mm, rhs_mm, out_mm;
      if (has_batch) {
        lhs_mm = {batch_size, M, K};
        rhs_mm = {batch_size, K, N};
        out_mm = {batch_size, M, N};
      } else {
        lhs_mm = {M, K};
        rhs_mm = {K, N};
        out_mm = {M, N};
      }

      auto data_type =
          NpuTool::ConvertXlaType2AclDataType(lhs_shape.element_type());
      auto lhs_addr = allocations->GetDeviceAddress(args_[0]);
      auto rhs_addr = allocations->GetDeviceAddress(args_[1]);
      auto out_addr = allocations->GetDeviceAddress(args_[2]);

      if (IsLeftOrderNoNeumaierDotSupported(
              lhs_shape, rhs_shape, param_shapes[2], lhs_batch, rhs_batch,
              lhs_contract, rhs_contract)) {
        NPU_LOG_FORMAT_PRINT(
            INFO,
            "Dot uses custom aclnnDotFp32LeftOrderNoNeumaier: lhs=%s, "
            "rhs=%s, out=%s",
            NpuTool::GetXlaShapeString(lhs_shape).c_str(),
            NpuTool::GetXlaShapeString(rhs_shape).c_str(),
            NpuTool::GetXlaShapeString(param_shapes[2]).c_str());

        ret = dep::aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnDotFp32LeftOrderNoNeumaierGetWorkspaceSize failed. "
              "Error code: %d",
              ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnDotFp32LeftOrderNoNeumaier);
        break;
      }

      // 当前后端的 aclnnPermute / reshape 相关路径实际 materialize 出 C-order buffer。
      // 所以默认只信任 default C-order layout。
      // 非 default layout 先不信任，除非后续能确认它来自未物理重排的 constant buffer。
      const bool lhs_trust_layout = IsDefaultCOrderLayout(lhs_shape);
      const bool rhs_trust_layout = IsDefaultCOrderLayout(rhs_shape);
      const bool out_trust_layout = IsDefaultCOrderLayout(param_shapes[2]);

      // 注意：aclnnMatmul 的 aclTensor strides 必须与 operand buffer 的物理 layout
      // 一致。NpuDotContractingDimsNormalizer 把 dot 维度规范化为 [M,K]×[K,N]，并对
      // 重排的 operand 插入 C-order 的 kTranspose（aclnnPermute 物理重排）。
      // 但当 operand 是非 C-order 常量（例如折叠 transpose(constant) 得到的
      // column-major {0,1} 常量）且维度顺序本就标准时，normalizer 不会插 transpose，
      // 此时硬编码 C-order strides 会误读。故按 operand 实际 minor_to_major 推导 strides。
      aclTensor* lhs_tensor = CreateMatmulAclTensorFromShape(
          lhs_mm, lhs_shape, data_type, lhs_addr.opaque(), lhs_trust_layout);

      aclTensor* rhs_tensor = CreateMatmulAclTensorFromShape(
          rhs_mm, rhs_shape, data_type, rhs_addr.opaque(), rhs_trust_layout);

      aclTensor* out_tensor = CreateMatmulAclTensorFromShape(
          out_mm, param_shapes[2], data_type, out_addr.opaque(), out_trust_layout);

      // NPU_LOG_FORMAT_PRINT(INFO,
      //     "Dot matmul: lhs=%s, rhs=%s, out=%s (B=%lld,M=%lld,K=%lld,N=%lld)",
      //     NpuTool::GetShapeString(lhs_mm).c_str(),
      //     NpuTool::GetShapeString(rhs_mm).c_str(),
      //     NpuTool::GetShapeString(out_mm).c_str(),
      //     batch_size, M, K, N);

      int8_t cubeMathType = 0;
      ret = dep::aclnnMatmulGetWorkspaceSize(
          lhs_tensor, rhs_tensor, out_tensor, cubeMathType, &workspaceSize,
          &executor);
      if (ret != ACL_SUCCESS) {
        retire_acl_tensor(lhs_tensor);
        retire_acl_tensor(rhs_tensor);
        retire_acl_tensor(out_tensor);
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMatmulGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMatmul);

      retire_acl_tensor(lhs_tensor);
      retire_acl_tensor(rhs_tensor);
      retire_acl_tensor(out_tensor);
      break;
    }

    case HloOpcode::kDivide: {
      ret = dep::aclnnDivGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnDivGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnDiv);
      break;
    }

    case HloOpcode::kPower: {
      ret = dep::aclnnPowTensorTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnPowTensorTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                             dep::aclnnPowTensorTensor);
      break;
    }

    case HloOpcode::kAtan2: {
      ret = dep::aclnnAtan2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAtan2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAtan2);
      break;
    }

    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic: {
      // aclnnRightShift performs an arithmetic right shift (sign extension) on signed types,
      // Performs a logical right shift (zero extension) on unsigned types.
      // U32/U64 is mapped to INT32/INT64 in XLA to be compatible with other ACLNN APIs,
      // Signed types such as S32/S8 also need to be converted to unsigned descriptors when performing logical right shifts.
      // All logical right shifts must use unsigned tensor descriptors to ensure zero-extension semantics.
      PrimitiveType elem_type = param_shapes[0].element_type();
      bool need_unsigned_cast =
          thunk_info().op_code == HloOpcode::kShiftRightLogical;

      aclTensor* input_tensor = acl_tensor_ptr_vector[0];
      aclTensor* shift_tensor = acl_tensor_ptr_vector[1];
      aclTensor* output_tensor = acl_tensor_ptr_vector.back();

      // RAII guard: If unsigned tensors are created, ensure all paths destroy them.
      // When the need_unsigned_cast branch is not entered, the vector content is empty and guard does not perform any destruction.
      std::vector<aclTensor*> unsigned_tensors_to_cleanup;
      auto unsigned_tensors_cleanup = absl::MakeCleanup([&]() {
        for (auto* t : unsigned_tensors_to_cleanup) {
          if (t != nullptr) {
            retire_acl_tensor(t);
          }
        }
      });

      if (need_unsigned_cast) {
        // Select the corresponding unsigned dtype according to the element bit width to ensure logical right-shift zero expansion
        aclDataType unsigned_dtype;
        switch (elem_type) {
          case PrimitiveType::S64:
          case PrimitiveType::U64:
            unsigned_dtype = ACL_UINT64;
            break;
          case PrimitiveType::S32:
          case PrimitiveType::U32:
            unsigned_dtype = ACL_UINT32;
            break;
          case PrimitiveType::S16:
          case PrimitiveType::U16:
            unsigned_dtype = ACL_UINT16;
            break;
          case PrimitiveType::S8:
          case PrimitiveType::U8:
          default:
            unsigned_dtype = ACL_UINT8;
            break;
        }
        NPU_LOG_FORMAT_PRINT(INFO,
                             "ShiftRightLogical: recreating tensors as unsigned "
                             "(elem_type=%d -> unsigned_dtype=%d)",
                             static_cast<int>(elem_type),
                             static_cast<int>(unsigned_dtype));
        // Directly use the shape in param_shapes and the device address in allocations
        // Create unsigned tensor descriptor
        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        auto shift_addr = allocations->GetDeviceAddress(args_[1]);
        auto out_addr = allocations->GetDeviceAddress(args_.back());

        input_tensor = NpuTool::CreateAclTensorPtrWithDtype(
            param_shapes[0], unsigned_dtype, input_addr.opaque());
        shift_tensor = NpuTool::CreateAclTensorPtrWithDtype(
            param_shapes[1], unsigned_dtype, shift_addr.opaque());
        output_tensor = NpuTool::CreateAclTensorPtrWithDtype(
            param_shapes.back(), unsigned_dtype, out_addr.opaque());

        // Register to RAII guard
        unsigned_tensors_to_cleanup = {input_tensor, shift_tensor, output_tensor};
      }

      ret = dep::aclnnRightShiftGetWorkspaceSize(
          input_tensor, shift_tensor, output_tensor, &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnRightShiftGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnRightShift);

      break;
    }

    case HloOpcode::kRemainder: {
      ret = dep::aclnnRemainderTensorTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnRemainderTensorTensorGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnRemainderTensorTensor);
      break;
    }

    case HloOpcode::kSubtract: {
      aclDataType alpha_type =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());
      union {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
        aclFloat16 f16;
        uint16_t bf16;
      } alpha_val;
      switch (alpha_type) {
        case ACL_INT32:
        case ACL_INT16:
        case ACL_INT8:
        case ACL_UINT8:
          alpha_val.i32 = 1;
          break;
        case ACL_INT64:
          alpha_val.i64 = 1;
          break;
        case ACL_DOUBLE:
          alpha_val.f64 = 1.0;
          break;
        case ACL_FLOAT16:
          alpha_val.f16 = 1.0f;
          break;
        case ACL_BF16:
          alpha_val.bf16 = 0x3F80;
          break;
        case ACL_FLOAT:
        default:
          alpha_val.f32 = 1.0f;
          alpha_type = ACL_FLOAT;
          break;
      }
      aclScalar* alpha = dep::aclCreateScalar(&alpha_val, alpha_type);
      if (alpha == nullptr) {
        return absl::InternalError("Subtract: failed to create alpha scalar");
      }

      ret = dep::aclnnSubGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1], alpha,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSubGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_scalar(alpha);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSub);
      retire_acl_scalar(alpha);
      break;
    }

    case HloOpcode::kNegate: {
      ret = dep::aclnnNegGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnNegGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnNeg);
      break;
    }

    case HloOpcode::kExp: {
      ret = dep::aclnnExpGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnExpGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnExp);
      break;
    }

    case HloOpcode::kRsqrt: {
      // Optimization: For scenarios such as RMSNorm, check the input type
      // Rsqrt may have accuracy loss when inputting FP16, but ACLNN has no special treatment
      aclDataType input_dtype =
          NpuTool::ConvertXlaType2AclDataType(param_shapes[0].element_type());

      if (input_dtype == ACL_FLOAT16 || input_dtype == ACL_BF16) {
        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Rsqrt: FP16/BF16 input detected, using default "
                             "ACLNN implementation");
      }

      ret = dep::aclnnRsqrtGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                            acl_tensor_ptr_vector.back(),
                                            &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnRsqrtGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnRsqrt);
      break;
    }

    case HloOpcode::kCholesky: {
      // aclnnLinalgCholesky: self → GetWorkspaceSize(self, upper, out, ...)
      // HLO CholeskyOptions.lower() → ACLNN upper = !lower
      bool upper = !thunk_info().cholesky_is_lower;
      ret = dep::aclnnLinalgCholeskyGetWorkspaceSize(
          acl_tensor_ptr_vector[0], upper,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnLinalgCholeskyGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                               dep::aclnnLinalgCholesky);
      break;
    }

    case HloOpcode::kTriangularSolve: {
      // ACLNN API: aclnnTriangularSolve(self=b, A, upper, transpose,
      //                                  unitriangular, xOut, mOut, ...)
      // HLO operands: operand(0)=a, operand(1)=b; output=x (same shape as b)
      // mOut is the redundant output required by ACLNN (copy of A after broadcast), not required by XLA
      if (!thunk_info().triangular_solve_left_side) {
        return absl::UnimplementedError(
            "TriangularSolve with left_side=false is not supported");
      }

      // Allocate temporary device memory for mOut (same shape as A)
      const auto& a_shape = param_shapes[0];
      auto data_type =
          NpuTool::ConvertXlaType2AclDataType(a_shape.element_type());
      int64_t a_num_elements = 1;
      std::vector<int64_t> a_dims(a_shape.dimensions().begin(),
                                  a_shape.dimensions().end());
      for (auto d : a_dims) a_num_elements *= d;
      size_t a_bytes = a_num_elements * ShapeUtil::ByteSizeOfPrimitiveType(
                                             a_shape.element_type());

      TF_ASSIGN_OR_RETURN(void* mout_addr,
                          allocate_temporary_device_memory(a_bytes));

      // Create mOut tensor (same shape as A, using temporary buffer)
      std::vector<int64_t> mout_strides(a_dims.size(), 1);
      for (int64_t i = static_cast<int64_t>(a_dims.size()) - 2; i >= 0; i--) {
        mout_strides[i] = a_dims[i + 1] * mout_strides[i + 1];
      }
      aclTensor* mout_tensor = dep::aclCreateTensor(
          a_dims.data(), a_dims.size(), data_type, mout_strides.data(), 0,
          aclFormat::ACL_FORMAT_ND, a_dims.data(), a_dims.size(), mout_addr);

      // ACLNN: self=b[1], A=a[0], xOut=output[2], mOut=temporary
      ret = dep::aclnnTriangularSolveGetWorkspaceSize(
          acl_tensor_ptr_vector[1],    // self = b
          acl_tensor_ptr_vector[0],    // A
          thunk_info().triangular_solve_upper,
          thunk_info().triangular_solve_transpose,
          thunk_info().triangular_solve_unitriangular,
          acl_tensor_ptr_vector[2],    // xOut = output
          mout_tensor,                 // mOut
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        retire_acl_tensor(mout_tensor);
        RetireTemporaryDeviceMemory(mout_addr);
        const std::string err_msg = absl::StrFormat(
            "Call aclnnTriangularSolveGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                               dep::aclnnTriangularSolve);

      retire_acl_tensor(mout_tensor);
      RetireTemporaryDeviceMemory(mout_addr);
      break;
    }

    case HloOpcode::kSin: {
      ret = dep::aclnnSinGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSinGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSin);
      break;
    }

    case HloOpcode::kCos: {
      ret = dep::aclnnCosGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                          acl_tensor_ptr_vector.back(),
                                          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnCosGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnCos);
      break;
    }

    case HloOpcode::kIsFinite: {
      ret = dep::aclnnIsFiniteGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                                acl_tensor_ptr_vector.back(),
                                                &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnIsFiniteGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnIsFinite);
      break;
    }

    case HloOpcode::kAbs: {
      ret = dep::aclnnAbsGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnAbsGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnAbs);
      break;
    }

    case HloOpcode::kSign: {
      ret = dep::aclnnSignGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSignGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSign);
      break;
    }

    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven: {
      ret = dep::aclnnRoundGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                            acl_tensor_ptr_vector.back(),
                                            &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnRoundGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnRound);
      break;
    }

    case HloOpcode::kSqrt: {
      ret = dep::aclnnSqrtGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                            acl_tensor_ptr_vector.back(),
                                            &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSqrtGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSqrt);
      break;
    }

    case HloOpcode::kLog: {
      ret = dep::aclnnLogGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                           acl_tensor_ptr_vector.back(),
                                           &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnLogGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnLog);
      break;
    }

    case HloOpcode::kLog1p: {
      ret = dep::aclnnLog1pGetWorkspaceSize(acl_tensor_ptr_vector[0],
                                             acl_tensor_ptr_vector.back(),
                                             &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnLog1pGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnLog1p);
      break;
    }

    case HloOpcode::kIota: {
      // iota: fill output buffer with incrementing values along iota_dimension.
      // Since CANN does not provide an aclnnIota API, we compute on host and
      // copy to device via aclrtMemcpy.
      NPU_LOG_FORMAT_PRINT(DEBUG, "Iota operation: host-side fill + H2D copy");

      const Shape& out_shape = param_shapes[0];
      int64_t iota_dim = thunk_info().iota_dimension;
      int64_t num_elements = ShapeUtil::ElementsIn(out_shape);
      int64_t elem_size =
          ShapeUtil::ByteSizeOfPrimitiveType(out_shape.element_type());
      int64_t data_size = num_elements * elem_size;

      if (data_size == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      // Compute strides for multi-dimensional indexing
      auto dims = out_shape.dimensions();
      int64_t rank = dims.size();
      std::vector<int64_t> strides(rank);
      int64_t stride = 1;
      for (int64_t i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
      }

      // Use pinned host memory and an explicit async copy on the execution
      // stream so the H2D transfer is ordered with the rest of the thunk.
      void* host_buffer = nullptr;
      aclError host_alloc_ret = dep::aclrtMallocHost(&host_buffer, data_size);
      if (host_alloc_ret != ACL_SUCCESS || host_buffer == nullptr) {
        return absl::InternalError(absl::StrFormat(
            "Failed to allocate host buffer for iota. Error code: %d",
            host_alloc_ret));
      }

      PrimitiveType dtype = out_shape.element_type();
      for (int64_t linear = 0; linear < num_elements; ++linear) {
        // Compute coordinate along iota_dimension
        int64_t remainder = linear;
        int64_t iota_val = 0;
        for (int64_t d = 0; d < rank; ++d) {
          int64_t coord = remainder / strides[d];
          remainder = remainder % strides[d];
          if (d == iota_dim) {
            iota_val = coord;
          }
        }

        // Write iota_val to host buffer with correct type
        switch (dtype) {
          case PrimitiveType::S32: {
            auto* buf = static_cast<int32_t*>(host_buffer);
            buf[linear] = static_cast<int32_t>(iota_val);
            break;
          }
          case PrimitiveType::S64: {
            auto* buf = static_cast<int64_t*>(host_buffer);
            buf[linear] = static_cast<int64_t>(iota_val);
            break;
          }
          case PrimitiveType::U32: {
            auto* buf = static_cast<uint32_t*>(host_buffer);
            buf[linear] = static_cast<uint32_t>(iota_val);
            break;
          }
          case PrimitiveType::U64: {
            auto* buf = static_cast<uint64_t*>(host_buffer);
            buf[linear] = static_cast<uint64_t>(iota_val);
            break;
          }
          case PrimitiveType::F32: {
            auto* buf = static_cast<float*>(host_buffer);
            buf[linear] = static_cast<float>(iota_val);
            break;
          }
          case PrimitiveType::F64: {
            auto* buf = static_cast<double*>(host_buffer);
            buf[linear] = static_cast<double>(iota_val);
            break;
          }
          case PrimitiveType::S16: {
            auto* buf = static_cast<int16_t*>(host_buffer);
            buf[linear] = static_cast<int16_t>(iota_val);
            break;
          }
          case PrimitiveType::U16: {
            auto* buf = static_cast<uint16_t*>(host_buffer);
            buf[linear] = static_cast<uint16_t>(iota_val);
            break;
          }
          case PrimitiveType::S8: {
            auto* buf = static_cast<int8_t*>(host_buffer);
            buf[linear] = static_cast<int8_t>(iota_val);
            break;
          }
          case PrimitiveType::U8: {
            auto* buf = static_cast<uint8_t*>(host_buffer);
            buf[linear] = static_cast<uint8_t>(iota_val);
            break;
          }
          default: {
            dep::aclrtFreeHost(host_buffer);
            return absl::InternalError(
                absl::StrFormat("Iota: unsupported dtype %s",
                                PrimitiveType_Name(dtype)));
          }
        }
      }

      // Copy from host to device
      auto output_addr = allocations->GetDeviceAddress(args_[0]);
      void* output_device_addr = output_addr.opaque();

      aclError copy_ret =
          dep::aclrtMemcpyAsync(output_device_addr, data_size, host_buffer,
                                data_size, ACL_MEMCPY_HOST_TO_DEVICE,
                                acl_stream);
      if (copy_ret == ACL_SUCCESS) {
        // Keep the staging buffer alive until the execution stream has finished
        // consuming the async copy.
        absl::Status sync_status = params.stream->BlockHostUntilDone();
        if (!sync_status.ok()) {
          dep::aclrtFreeHost(host_buffer);
          return sync_status;
        }
      }
      dep::aclrtFreeHost(host_buffer);

      if (copy_ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclrtMemcpyAsync H2D for iota failed. Error code: %d", copy_ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = absl::OkStatus();
      break;
    }

    case HloOpcode::kMaximum: {
      ret = dep::aclnnMaximumGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMaximumGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMaximum);
      break;
    }

    case HloOpcode::kMinimum: {
      ret = dep::aclnnMinimumGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMinimumGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnMinimum);
      break;
    }

    case HloOpcode::kClamp: {
      const Shape& lower_shape = param_shapes[0];
      const Shape& upper_shape = param_shapes[2];

      const bool lower_is_scalar = ShapeUtil::ElementsIn(lower_shape) == 1;
      const bool upper_is_scalar = ShapeUtil::ElementsIn(upper_shape) == 1;

      if (lower_is_scalar && upper_is_scalar) {
        auto lower_addr = allocations->GetDeviceAddress(args_[0]);
        auto upper_addr = allocations->GetDeviceAddress(args_[2]);

        TF_ASSIGN_OR_RETURN(
            aclScalar * lower_scalar,
            CreateScalarFromDeviceBuffer(lower_addr.opaque(),
                                         lower_shape.element_type()));
        auto lower_scalar_cleanup =
            absl::MakeCleanup([&] { retire_acl_scalar(lower_scalar); });

        TF_ASSIGN_OR_RETURN(
            aclScalar * upper_scalar,
            CreateScalarFromDeviceBuffer(upper_addr.opaque(),
                                         upper_shape.element_type()));
        auto upper_scalar_cleanup =
            absl::MakeCleanup([&] { retire_acl_scalar(upper_scalar); });

        ret = dep::aclnnClampGetWorkspaceSize(
            acl_tensor_ptr_vector[1], lower_scalar, upper_scalar,
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnClampGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        exec_ret =
            ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnClamp);
        break;
      }

      ret = dep::aclnnClampTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[1], acl_tensor_ptr_vector[0],
          acl_tensor_ptr_vector[2], acl_tensor_ptr_vector.back(),
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnClampTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnClampTensor);
      break;
    }

    case HloOpcode::kConcatenate: {
      // Get the concatenate dimensions, there should be only one element in the dimensions vector
      if (thunk_info().dimensions.empty()) {
        const std::string err_msg = "Concatenate dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }
      int64_t dim = thunk_info().dimensions[0];

      // aclnnCatGetWorkspaceSize retains its TensorList input. Creating a new
      // list per launch therefore grows CANN's host-side executor metadata.
      // A compiled thunk has fixed Cat metadata, so retain one list/descriptors
      // and a repeatable executor for each stream, then refresh only addresses.
      absl::MutexLock lock(&cat_executor_cache_mutex_);
      auto cache_it = cat_executor_caches_.find(acl_stream);
      if (cache_it == cat_executor_caches_.end()) {
        TF_ASSIGN_OR_RETURN(
            acl_tensor_ptr_vector,
            ExtractAclInputOutTensorPtr(allocations, param_shapes));
        auto cache = std::make_unique<CatExecutorCache>();
        const size_t input_count = acl_tensor_ptr_vector.size() - 1;
        cache->input_tensors.assign(acl_tensor_ptr_vector.begin(),
                                    acl_tensor_ptr_vector.begin() + input_count);
        cache->output_tensor = acl_tensor_ptr_vector.back();
        cache->input_tensor_list = dep::aclCreateTensorList(
            cache->input_tensors.data(), cache->input_tensors.size());
        if (cache->input_tensor_list == nullptr) {
          const std::string err_msg =
              "Failed to create aclTensorList for concatenate";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        ret = dep::aclnnCatGetWorkspaceSize(
            cache->input_tensor_list, dim, cache->output_tensor,
            &cache->workspace_size, &cache->executor);
        if (ret != ACL_SUCCESS) {
          dep::aclDestroyTensorList(cache->input_tensor_list);
          for (size_t i = 0; i < input_count; ++i) {
            acl_tensor_ptr_vector[i] = nullptr;
          }
          const std::string err_msg = absl::StrFormat(
              "Call aclnnCatGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        ret = dep::aclSetAclOpExecutorRepeatable(cache->executor);
        if (ret != ACL_SUCCESS) {
          dep::aclDestroyAclOpExecutor(cache->executor);
          dep::aclDestroyTensorList(cache->input_tensor_list);
          for (size_t i = 0; i < input_count; ++i) {
            acl_tensor_ptr_vector[i] = nullptr;
          }
          const std::string err_msg = absl::StrFormat(
              "Failed to make aclnnCat executor repeatable. Error code: %d",
              ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        // Ownership moves to the cache. TensorList owns input descriptors;
        // the cached output descriptor is owned independently by the cache.
        std::fill(acl_tensor_ptr_vector.begin(), acl_tensor_ptr_vector.end(),
                  nullptr);
        cache_it = cat_executor_caches_.emplace(acl_stream, std::move(cache))
                       .first;
      } else {
        CatExecutorCache& cache = *cache_it->second;
        const size_t input_count = cache.input_tensors.size();
        if (args_.size() != input_count + 1) {
          return absl::InternalError(
              "aclnnCat cached tensor count does not match this launch");
        }
        for (size_t i = 0; i < input_count; ++i) {
          void* addr = allocations->GetDeviceAddress(args_[i]).opaque();
          ret = dep::aclSetDynamicInputTensorAddr(
              cache.executor, /*ir_index=*/0, i, cache.input_tensor_list,
              addr);
          if (ret != ACL_SUCCESS) {
            return absl::InternalError(absl::StrFormat(
                "Failed to update repeatable aclnnCat input %d. Error code: "
                "%d",
                i, ret));
          }
        }
        void* output_addr = allocations->GetDeviceAddress(args_.back()).opaque();
        ret = dep::aclSetOutputTensorAddr(cache.executor, /*index=*/0,
                                          cache.output_tensor, output_addr);
        if (ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Failed to update repeatable aclnnCat output. Error code: %d",
              ret));
        }
      }

      CatExecutorCache& cache = *cache_it->second;
      exec_ret = ExecuteAclnn(acl_stream, cache.workspace_size, cache.executor,
                              dep::aclnnCat);

      break;
    }

    case HloOpcode::kReduce: {
      const auto reducer_kind = thunk_info().reduce_reducer_kind;
      if (param_shapes.size() != 3 ||
          reducer_kind == ThunkInfo::ReduceReducerKind::kUnsupported ||
          ((reducer_kind == ThunkInfo::ReduceReducerKind::kAnd ||
            reducer_kind == ThunkInfo::ReduceReducerKind::kOr) &&
           param_shapes[0].element_type() != PRED)) {
        return absl::UnimplementedError(absl::StrFormat(
            "Unsupported NPU reducer; coupled reductions must be expanded: %s",
            thunk_info().hloInstrDesc));
      }
      // Get dimension information of reduce
      if (thunk_info().dimensions.empty()) {
        const std::string err_msg = "Reduce dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Create dimension array
      aclIntArray* dims = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (dims == nullptr) {
        const std::string err_msg =
            "Failed to create aclIntArray for reduce dimensions";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Print dimension information
      std::string dims_str = "[";
      for (int i = 0; i < thunk_info().dimensions.size(); ++i) {
        if (i > 0) dims_str += ", ";
        dims_str += std::to_string(thunk_info().dimensions[i]);
      }
      dims_str += "]";
      NPU_LOG_FORMAT_PRINT(DEBUG, "Reduce dimensions: %s", dims_str.c_str());

      // Only scalar reductions reach ACLNN: input, init, output. Coupled
      // reductions are expanded in HLO and rejected by the guard above.
      constexpr int num_reduces = 1;

      for (int idx = 0; idx < num_reduces; ++idx) {
        aclTensor* input_tensor = acl_tensor_ptr_vector[idx];
        aclTensor* output_tensor = acl_tensor_ptr_vector[2 * num_reduces + idx];

        // Get input data type
        aclDataType input_dtype = NpuTool::ConvertXlaType2AclDataType(
            param_shapes[idx].element_type());

        // For FP16/BF16 input, use FP32 accumulation to reduce accuracy loss
        aclDataType accumulate_dtype = input_dtype;
        if (input_dtype == ACL_FLOAT16 || input_dtype == ACL_BF16) {
          accumulate_dtype = ACL_FLOAT;
        }

        // PRED 上的 and/or reducer 走逻辑归约(aclnnAll/aclnnAny),输入必须是 BOOL。
        bool use_logical_all =
            (reducer_kind == ThunkInfo::ReduceReducerKind::kAnd) &&
            (input_dtype == ACL_BOOL);
        bool use_logical_any =
            (reducer_kind == ThunkInfo::ReduceReducerKind::kOr) &&
            (input_dtype == ACL_BOOL);

        NPU_LOG_FORMAT_PRINT(
            DEBUG, "Reduce[%d/%d]: reducer_kind=%d, input_dtype=%d, all=%d, any=%d",
            idx + 1, num_reduces, static_cast<int>(reducer_kind),
            static_cast<int>(input_dtype), use_logical_all, use_logical_any);

        if (use_logical_all) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG, "ReduceAll[%d] Input shape: %s, Output shape: %s", idx,
              NpuTool::GetXlaShapeString(param_shapes[idx]).c_str(),
              NpuTool::GetXlaShapeString(param_shapes[2 * num_reduces + idx])
                  .c_str());

          ret = dep::aclnnAllGetWorkspaceSize(input_tensor, dims, false,
                                              output_tensor, &workspaceSize,
                                              &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnAllGetWorkspaceSize failed for reduce[%d]. Error: %d",
                idx, ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_int_array(dims);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnAll);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAll failed: %s",
                                 exec_ret.ToString().c_str());
            retire_acl_int_array(dims);
            return exec_ret;
          }
        } else if (use_logical_any) {
          ret = dep::aclnnAnyGetWorkspaceSize(input_tensor, dims, false,
                                              output_tensor, &workspaceSize,
                                              &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnAnyGetWorkspaceSize failed for reduce[%d]. Error: %d",
                idx, ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_int_array(dims);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnAny);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAny failed: %s",
                                 exec_ret.ToString().c_str());
            retire_acl_int_array(dims);
            return exec_ret;
          }
        } else if (reducer_kind == ThunkInfo::ReduceReducerKind::kMax) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG, "ReduceMax[%d] Input shape: %s, Output shape: %s", idx,
              NpuTool::GetXlaShapeString(param_shapes[idx]).c_str(),
              NpuTool::GetXlaShapeString(param_shapes[2 * num_reduces + idx])
                  .c_str());

          ret = dep::aclnnAmaxGetWorkspaceSize(input_tensor, dims, false,
                                               output_tensor, &workspaceSize,
                                               &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnAmaxGetWorkspaceSize failed for reduce[%d]. Error: %d",
                idx, ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_int_array(dims);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnAmax);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAmax failed: %s",
                                 exec_ret.ToString().c_str());
            retire_acl_int_array(dims);
            return exec_ret;
          }
        } else if (reducer_kind == ThunkInfo::ReduceReducerKind::kMin) {
          NPU_LOG_FORMAT_PRINT(
              DEBUG, "ReduceMin[%d] Input shape: %s, Output shape: %s", idx,
              NpuTool::GetXlaShapeString(param_shapes[idx]).c_str(),
              NpuTool::GetXlaShapeString(param_shapes[2 * num_reduces + idx])
                  .c_str());

          ret = dep::aclnnAminGetWorkspaceSize(input_tensor, dims, false,
                                               output_tensor, &workspaceSize,
                                               &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnAminGetWorkspaceSize failed for reduce[%d]. Error: %d",
                idx, ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_int_array(dims);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnAmin);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "ExecuteAclnn for aclnnAmin failed: %s",
                                 exec_ret.ToString().c_str());
            retire_acl_int_array(dims);
            return exec_ret;
          }
        } else if (reducer_kind == ThunkInfo::ReduceReducerKind::kSum) {
          ret = dep::aclnnReduceSumGetWorkspaceSize(
              input_tensor, dims, false, accumulate_dtype, output_tensor,
              &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnReduceSumGetWorkspaceSize failed for reduce[%d]. Error: "
                "%d",
                idx, ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_int_array(dims);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnReduceSum);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR,
                                 "ExecuteAclnn for aclnnReduceSum failed: %s",
                                 exec_ret.ToString().c_str());
            retire_acl_int_array(dims);
            return exec_ret;
          }
        }
      }

      // Clean dimension array
      retire_acl_int_array(dims);
      break;
    }

    case HloOpcode::kReduceWindow: {
      if (thunk_info().window_dimensions.empty()) {
        const std::string err_msg = "ReduceWindow window_dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Cumsum path: prefix and pattern of add reducer
      if (thunk_info().reduce_window_reducer_kind ==
          NpuThunk::ThunkInfo::ReduceWindowReducerKind::kCumsum) {
        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        auto output_addr = allocations->GetDeviceAddress(args_[2]);
        void* input_device_ptr = input_addr.opaque();
        void* output_device_ptr = output_addr.opaque();

        const Shape& input_xla_shape = param_shapes[0];
        const Shape& output_xla_shape = param_shapes[2];

        std::vector<int64_t> input_dims =
            NpuTool::ConvertXlaShape2VectorShape(input_xla_shape);
        std::vector<int64_t> output_dims =
            NpuTool::ConvertXlaShape2VectorShape(output_xla_shape);

        aclDataType acl_dtype =
            NpuTool::ConvertXlaType2AclDataType(input_xla_shape.element_type());

        aclTensor* input_tensor = dep::aclCreateTensor(
            input_dims.data(), input_dims.size(), acl_dtype,
            nullptr, 0, ACL_FORMAT_ND,
            input_dims.data(), input_dims.size(), input_device_ptr);
        aclTensor* output_tensor = dep::aclCreateTensor(
            output_dims.data(), output_dims.size(), acl_dtype,
            nullptr, 0, ACL_FORMAT_ND,
            output_dims.data(), output_dims.size(), output_device_ptr);

        if (input_tensor == nullptr || output_tensor == nullptr) {
          if (input_tensor) retire_acl_tensor(input_tensor);
          if (output_tensor) retire_acl_tensor(output_tensor);
          const std::string err_msg =
              "Failed to create tensors for aclnnCumsum";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        ret = dep::aclnnCumsumGetWorkspaceSize(
            input_tensor, thunk_info().cumsum_dim, acl_dtype, output_tensor,
            &workspaceSize, &executor);

        retire_acl_tensor(input_tensor);
        retire_acl_tensor(output_tensor);

        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "Call aclnnCumsumGetWorkspaceSize failed. Error code: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnCumsum);
        if (!exec_ret.ok()) {
          NPU_LOG_FORMAT_PRINT(ERROR,
                               "ExecuteAclnn for aclnnCumsum failed: %s",
                               exec_ret.ToString().c_str());
          return exec_ret;
        }
        break;
      }

      // Max pool path (original logic)

      const int num_dims = thunk_info().window_dimensions.size();

      // Filter out the spatial dimensions that actually participate in pooling
      // CANN MaxPool only accepts spatial dimensions (excluding batch/channel and other non-pooling dimensions)
      // Skip dimensions with window_size=1, stride=1, pad=0, dilation=1
      std::vector<int64_t> spatial_window_dims;
      std::vector<int64_t> spatial_strides;
      std::vector<int64_t> spatial_pads_flat;   // interleaved low, high
      std::vector<int64_t> spatial_dilations;

      for (int i = 0; i < num_dims; ++i) {
        bool is_trivial = (thunk_info().window_dimensions[i] == 1 &&
                           thunk_info().window_strides[i] == 1 &&
                           thunk_info().window_padding_low[i] == 0 &&
                           thunk_info().window_padding_high[i] == 0 &&
                           thunk_info().window_dilations[i] == 1);
        if (!is_trivial) {
          spatial_window_dims.push_back(thunk_info().window_dimensions[i]);
          spatial_strides.push_back(thunk_info().window_strides[i]);
          spatial_pads_flat.push_back(thunk_info().window_padding_low[i]);
          spatial_pads_flat.push_back(thunk_info().window_padding_high[i]);
          spatial_dilations.push_back(thunk_info().window_dilations[i]);
        }
      }

      if (spatial_window_dims.empty()) {
        const std::string err_msg =
            "ReduceWindow: no spatial pooling dimensions found "
            "(all window dimensions are trivial)";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      const int num_spatial_dims = spatial_window_dims.size();

      // Create kernelShape aclIntArray (spatial dimensions only)
      aclIntArray* kernel_shape = dep::aclCreateIntArray(
          spatial_window_dims.data(), num_spatial_dims);
      if (kernel_shape == nullptr) {
        return absl::InternalError(
            "Failed to create kernel_shape array for ReduceWindow");
      }

      // Create strides aclIntArray
      aclIntArray* strides = dep::aclCreateIntArray(
          spatial_strides.data(), num_spatial_dims);
      if (strides == nullptr) {
        retire_acl_int_array(kernel_shape);
        return absl::InternalError(
            "Failed to create strides array for ReduceWindow");
      }

      // Create pads aclIntArray (interleaved: low_0, high_0,...)
      aclIntArray* pads = dep::aclCreateIntArray(
          spatial_pads_flat.data(), spatial_pads_flat.size());
      if (pads == nullptr) {
        retire_acl_int_array(kernel_shape);
        retire_acl_int_array(strides);
        return absl::InternalError(
            "Failed to create pads array for ReduceWindow");
      }

      // Create dilations aclIntArray
      aclIntArray* dilations = dep::aclCreateIntArray(
          spatial_dilations.data(), num_spatial_dims);
      if (dilations == nullptr) {
        retire_acl_int_array(kernel_shape);
        retire_acl_int_array(strides);
        retire_acl_int_array(pads);
        return absl::InternalError(
            "Failed to create dilations array for ReduceWindow");
      }

      // Select CANN API based on reducer type
      bool is_max_pool = (thunk_info().reduce_window_reducer_kind ==
                          NpuThunk::ThunkInfo::ReduceWindowReducerKind::kMax);

      if (!is_max_pool) {
        retire_acl_int_array(kernel_shape);
        retire_acl_int_array(strides);
        retire_acl_int_array(pads);
        retire_acl_int_array(dilations);
        const std::string err_msg = absl::StrFormat(
            "ReduceWindow: unsupported reducer type "
            "(only max/cumsum supported). "
            "hloInstrDesc: %s",
            thunk_info().hloInstrDesc);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // autoPad=0 means use explicit padding
      const int64_t auto_pad = 0;
      const int64_t ceil_mode = 0;

      // aclnnMaxPool requires ACL_FORMAT_NCHW format
      // Get input and output device addresses
      auto input_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_[2]);
      void* input_device_ptr = input_addr.opaque();
      void* output_device_ptr = output_addr.opaque();

      // Get XLA shape
      const Shape& input_xla_shape = param_shapes[0];
      const Shape& output_xla_shape = param_shapes[2];

      // Convert to CANN shape vector
      std::vector<int64_t> input_dims =
          NpuTool::ConvertXlaShape2VectorShape(input_xla_shape);
      std::vector<int64_t> output_dims =
          NpuTool::ConvertXlaShape2VectorShape(output_xla_shape);

      aclDataType acl_dtype =
          NpuTool::ConvertXlaType2AclDataType(input_xla_shape.element_type());

      // Create tensor in NCHW format (stride=nullptr compatible with CANN pooling API)
      aclTensor* nchw_input = dep::aclCreateTensor(
          input_dims.data(), input_dims.size(), acl_dtype,
          nullptr, 0, ACL_FORMAT_NCHW,
          input_dims.data(), input_dims.size(), input_device_ptr);
      aclTensor* nchw_output = dep::aclCreateTensor(
          output_dims.data(), output_dims.size(), acl_dtype,
          nullptr, 0, ACL_FORMAT_NCHW,
          output_dims.data(), output_dims.size(), output_device_ptr);

      if (nchw_input == nullptr || nchw_output == nullptr) {
        if (nchw_input) retire_acl_tensor(nchw_input);
        if (nchw_output) retire_acl_tensor(nchw_output);
        retire_acl_int_array(kernel_shape);
        retire_acl_int_array(strides);
        retire_acl_int_array(pads);
        retire_acl_int_array(dilations);
        return absl::InternalError(
            "Failed to create NCHW tensors for aclnnMaxPool");
      }

      ret = dep::aclnnMaxPoolGetWorkspaceSize(
          nchw_input,
          kernel_shape,
          strides,
          auto_pad,
          pads,
          dilations,
          ceil_mode,
          nchw_output,
          &workspaceSize,
          &executor);

      // Clean NCHW tensor (no longer needed)
      retire_acl_tensor(nchw_input);
      retire_acl_tensor(nchw_output);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnMaxPoolGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_int_array(kernel_shape);
        retire_acl_int_array(strides);
        retire_acl_int_array(pads);
        retire_acl_int_array(dilations);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnMaxPool);

      retire_acl_int_array(kernel_shape);
      retire_acl_int_array(strides);
      retire_acl_int_array(pads);
      retire_acl_int_array(dilations);

      if (!exec_ret.ok()) {
        NPU_LOG_FORMAT_PRINT(ERROR,
                             "ExecuteAclnn for aclnnMaxPool failed: %s",
                             exec_ret.ToString().c_str());
        return exec_ret;
      }
      break;
    }

    case HloOpcode::kTranspose: {
      // Transpose requires the dimensions parameter as the dimension order of permute
      if (thunk_info().dimensions.empty()) {
        const std::string err_msg = "Transpose dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      aclIntArray* perm = dep::aclCreateIntArray(
          thunk_info().dimensions.data(), thunk_info().dimensions.size());
      if (perm == nullptr) {
        const std::string err_msg = "Failed to create perm array for transpose";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      ret = dep::aclnnPermuteGetWorkspaceSize(
          acl_tensor_ptr_vector[0], perm, acl_tensor_ptr_vector.back(),
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnPermuteGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_int_array(perm);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnPermute);
      retire_acl_int_array(perm);
      break;
    }

    case HloOpcode::kSlice: {
      // Slice operation: parsing starts, ends, axes, steps from HLO instruction description
      // Slice is static, parameters are determined at compile time
      const auto& input_shape = param_shapes[0];
      const auto& output_shape = param_shapes.back();

      // Parse slice parameter from hloInstrDesc
      // Format example: "slice={[0:1], [0:14], [0:8], [32:64]}"
      std::vector<int64_t> starts_data;
      std::vector<int64_t> ends_data;
      std::vector<int64_t> axes_data;
      std::vector<int64_t> steps_data;

      const std::string& desc = thunk_info().hloInstrDesc;

      // Try parsing slice parameter from hloInstrDesc
      // Find the location of "slice={"
      size_t slice_pos = desc.find("slice={");
      if (slice_pos != std::string::npos) {
        // Extract slice parameter part
        size_t start_pos = slice_pos + strlen("slice={");
        size_t end_pos = desc.find("}", start_pos);
        if (end_pos != std::string::npos) {
          std::string slice_str = desc.substr(start_pos, end_pos - start_pos);
          NPU_LOG_FORMAT_PRINT(DEBUG, "Parsing slice params from: %s",
                               slice_str.c_str());

          // Parse [start:end] for each dimension
          int dim = 0;
          size_t bracket_pos = 0;
          while ((bracket_pos = slice_str.find("[", bracket_pos)) !=
                 std::string::npos) {
            size_t close_bracket = slice_str.find("]", bracket_pos);
            if (close_bracket == std::string::npos) break;

            std::string range_str = slice_str.substr(
                bracket_pos + 1, close_bracket - bracket_pos - 1);

            // Parse start:end
            size_t colon_pos = range_str.find(":");
            if (colon_pos != std::string::npos) {
              std::string start_str = range_str.substr(0, colon_pos);
              std::string end_str = range_str.substr(colon_pos + 1);

              try {
                int64_t start_val = std::stoll(start_str);
                int64_t end_val = std::stoll(end_str);

                starts_data.push_back(start_val);
                ends_data.push_back(end_val);
                // CANN's aclnnSliceV2 expects axes in logical dimension order
                // The slice parameter of XLA is already in logical dimension order
                axes_data.push_back(dim);
                steps_data.push_back(1);

                NPU_LOG_FORMAT_PRINT(
                     INFO, "Slice logical dim %d: [%ld:%ld]",
                     dim, start_val, end_val);
              } catch (...) {
                NPU_LOG_FORMAT_PRINT(ERROR, "Failed to parse slice range: %s",
                                     range_str.c_str());
              }
            }

            dim++;
            bracket_pos = close_bracket + 1;
          }
        }
      }

      // If parsing fails, use default parameters
      if (starts_data.empty()) {
        NPU_LOG_FORMAT_PRINT(
            WARN,
            "Failed to parse slice params from hloInstrDesc, using default");
        for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
          starts_data.push_back(0);
          ends_data.push_back((i < output_shape.dimensions().size())
                                  ? output_shape.dimensions()[i]
                                  : input_shape.dimensions()[i]);
          // CANN's aclnnSliceV2 expects axes in logical dimension order
          axes_data.push_back(i);
          steps_data.push_back(1);
        }
      }

      NPU_LOG_FORMAT_PRINT(DEBUG, "Slice: input=%s, output=%s",
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str());

      // Create aclIntArray
      aclIntArray* starts =
          dep::aclCreateIntArray(starts_data.data(), starts_data.size());
      aclIntArray* ends =
          dep::aclCreateIntArray(ends_data.data(), ends_data.size());
      aclIntArray* axes =
          dep::aclCreateIntArray(axes_data.data(), axes_data.size());
      aclIntArray* steps =
          dep::aclCreateIntArray(steps_data.data(), steps_data.size());

      if (starts == nullptr || ends == nullptr || axes == nullptr ||
          steps == nullptr) {
        const std::string err_msg = "Failed to create aclIntArray for slice";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        if (starts) retire_acl_int_array(starts);
        if (ends) retire_acl_int_array(ends);
        if (axes) retire_acl_int_array(axes);
        if (steps) retire_acl_int_array(steps);
        return absl::InternalError(err_msg);
      }

      // Call aclnnSliceV2GetWorkspaceSize
      ret = dep::aclnnSliceV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], starts, ends, axes, steps,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);

      // Clean up aclIntArray
      retire_acl_int_array(starts);
      retire_acl_int_array(ends);
      retire_acl_int_array(axes);
      retire_acl_int_array(steps);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSliceV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSliceV2);
      break;
    }

    case HloOpcode::kDynamicSlice: {
      // Order index copies after their producers on acl_stream. CANN completes
      // pageable-host D2H before returning, allowing immediate CPU consumption.
      // Pinned host storage would require waiting for copy completion.
      // DynamicSlice operation: input + start_indices... + output
      // Parsing slice_sizes and start_indices from HLO instructions
      const auto& slice_sizes = thunk_info().dimensions;

      if (slice_sizes.empty()) {
        const std::string err_msg = "DynamicSlice dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Get the shape of the input tensor
      const auto& input_shape = param_shapes[0];
      const auto& output_shape = param_shapes.back();

      // start_indices Number of parameters = Total number of parameters - 2 (input and output)
      size_t num_start_indices = acl_tensor_ptr_vector.size() - 2;

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "DynamicSlice: input_shape=%s, output_shape=%s, "
                           "slice_sizes=%zu, num_start_indices=%zu",
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str(),
                           slice_sizes.size(), num_start_indices);

      // output_shape.dimensions() returns slice sizes in logical dimension order
      // CANN's aclnnSliceV2 also expects axes/starts/ends in logical dimension order
      // So directly use output_shape.dimensions() as slice_sizes_logical
      std::vector<int64_t> slice_sizes_logical(output_shape.dimensions().begin(),
                                                output_shape.dimensions().end());

      NPU_LOG_FORMAT_PRINT(INFO, "DynamicSlice: slice_sizes_from_thunk=[%s], slice_sizes_from_output=[%s]",
                           [&slice_sizes]() {
                             std::string s;
                             for (size_t i = 0; i < slice_sizes.size(); ++i) {
                               if (i > 0) s += ", ";
                               s += std::to_string(slice_sizes[i]);
                             }
                             return s;
                           }().c_str(),
                           [&slice_sizes_logical]() {
                             std::string s;
                             for (size_t i = 0; i < slice_sizes_logical.size(); ++i) {
                               if (i > 0) s += ", ";
                               s += std::to_string(slice_sizes_logical[i]);
                             }
                             return s;
                           }().c_str());

      // Construct starts, ends, axes, steps array
      // axes uses logical dimensions and corresponds to starts/ends one-to-one
      std::vector<int64_t> starts_data;
      std::vector<int64_t> ends_data;
      std::vector<int64_t> axes_data;
      std::vector<int64_t> steps_data;

      // Process each slice dimension (in logical dimension order)
      for (size_t i = 0; i < slice_sizes_logical.size(); ++i) {
        int64_t start_val = 0;

        // If the corresponding start_indices parameter exists, read it from the device memory
        if (i < num_start_indices) {
          // The start_indices parameter starts from args_[1] (args_[0] is the input, args_.back() is the output)
          auto start_idx_addr = allocations->GetDeviceAddress(args_[i + 1]);
          void* start_idx_device_addr = start_idx_addr.opaque();

          // Get the data type and shape of start_indices
          xla::PrimitiveType start_dtype = param_shapes[i + 1].element_type();
          const auto& start_shape = param_shapes[i + 1];

          NPU_LOG_FORMAT_PRINT(
              DEBUG,
              "DynamicSlice: start_indices[%zu] shape=%s, dtype=%s, "
              "device_addr=%p",
              i, NpuTool::GetXlaShapeString(start_shape).c_str(),
              xla::PrimitiveType_Name(start_dtype).c_str(),
              start_idx_device_addr);

          // Read value based on data type
          if (start_dtype == xla::S32) {
            int32_t host_start_val = 0;
            aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
            aclError memcpy_ret =
                dep::aclrtMemcpyAsync(&host_start_val, sizeof(int32_t),
                                     start_idx_device_addr, sizeof(int32_t),
                                     kind, acl_stream);
            if (memcpy_ret == ACL_SUCCESS) {
              start_val = static_cast<int64_t>(host_start_val);
              NPU_LOG_FORMAT_PRINT(
                  DEBUG,
                  "DynamicSlice: read start_indices[%zu] = %d (S32) from "
                  "device",
                  i, host_start_val);
            } else {
              NPU_LOG_FORMAT_PRINT(WARN,
                                   "Failed to read start_indices[%zu] from "
                                   "device, using default 0",
                                   i);
              start_val = 0;
            }
          } else if (start_dtype == xla::S64) {
            int64_t host_start_val = 0;
            aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
            aclError memcpy_ret =
                dep::aclrtMemcpyAsync(&host_start_val, sizeof(int64_t),
                                     start_idx_device_addr, sizeof(int64_t),
                                     kind, acl_stream);
            if (memcpy_ret == ACL_SUCCESS) {
              start_val = host_start_val;
              NPU_LOG_FORMAT_PRINT(
                  DEBUG,
                  "DynamicSlice: read start_indices[%zu] = %ld (S64) from "
                  "device",
                  i, start_val);
            } else {
              NPU_LOG_FORMAT_PRINT(WARN,
                                   "Failed to read start_indices[%zu] from "
                                   "device, using default 0",
                                   i);
              start_val = 0;
            }
          } else {
            NPU_LOG_FORMAT_PRINT(
                WARN, "Unsupported start_indices dtype %s, using default 0",
                xla::PrimitiveType_Name(start_dtype).c_str());
          }
        }

        // Calculate end value: start + slice_size
        int64_t end_val = start_val + slice_sizes_logical[i];

        // Make sure end does not exceed the input dimension size
        if (i < input_shape.dimensions().size()) {
          end_val = std::min(end_val, input_shape.dimensions()[i]);
        }

        starts_data.push_back(start_val);
        ends_data.push_back(end_val);
        // Use logical dimensions as axes
        axes_data.push_back(static_cast<int64_t>(i));
        steps_data.push_back(1);

        NPU_LOG_FORMAT_PRINT(
            INFO, "DynamicSlice logical_dim %zu: start=%ld, end=%ld, axis=%zu, step=1",
            i, start_val, end_val, i);
      }

      // Create aclIntArray
      aclIntArray* starts =
          dep::aclCreateIntArray(starts_data.data(), starts_data.size());
      aclIntArray* ends =
          dep::aclCreateIntArray(ends_data.data(), ends_data.size());
      aclIntArray* axes =
          dep::aclCreateIntArray(axes_data.data(), axes_data.size());
      aclIntArray* steps =
          dep::aclCreateIntArray(steps_data.data(), steps_data.size());

      if (starts == nullptr || ends == nullptr || axes == nullptr ||
          steps == nullptr) {
        const std::string err_msg = "Failed to create aclIntArray for slice";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        if (starts) retire_acl_int_array(starts);
        if (ends) retire_acl_int_array(ends);
        if (axes) retire_acl_int_array(axes);
        if (steps) retire_acl_int_array(steps);
        return absl::InternalError(err_msg);
      }

      // Call aclnnSliceV2GetWorkspaceSize
      ret = dep::aclnnSliceV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], starts, ends, axes, steps,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);

      // Clean up aclIntArray
      retire_acl_int_array(starts);
      retire_acl_int_array(ends);
      retire_acl_int_array(axes);
      retire_acl_int_array(steps);

      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSliceV2GetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSliceV2);
      break;
    }

    case HloOpcode::kDynamicUpdateSlice: {
      // DynamicUpdateSlice: output = operand, then write `update` into the
      // subregion starting at start_indices (clamped so the update fits).
      //   args_[0]      = operand
      //   args_[1]      = update
      //   args_[2..]     = one scalar start_index per operand dimension
      //   args_.back()  = output
      if (acl_tensor_ptr_vector.size() < 4) {
        const std::string err_msg = absl::StrFormat(
            "DynamicUpdateSlice expects >= 4 params, got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      const Shape& operand_shape = param_shapes[0];
      const Shape& update_shape = param_shapes[1];
      if (!operand_shape.IsArray() || !update_shape.IsArray()) {
        const std::string err_msg =
            "DynamicUpdateSlice: operand/update must be arrays";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      const int64_t rank = operand_shape.dimensions_size();
      const size_t num_start_indices = acl_tensor_ptr_vector.size() - 3;
      if (static_cast<size_t>(rank) != num_start_indices ||
          update_shape.dimensions_size() != rank) {
        const std::string err_msg = absl::StrFormat(
            "DynamicUpdateSlice shape mismatch: operand rank=%ld, "
            "update rank=%d, num_start_indices=%zu",
            rank, update_shape.dimensions_size(), num_start_indices);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // 1. Enqueue operand -> output (D2D) on the producer/update stream.
      //    Skipped if they share a buffer
      //    (in-place update), which is also correct.
      auto operand_addr = allocations->GetDeviceAddress(args_[0]);
      auto output_addr = allocations->GetDeviceAddress(args_.back());
      const int64_t operand_size = ShapeUtil::ByteSizeOf(operand_shape);
      if (operand_addr.opaque() != output_addr.opaque() &&
          operand_size > 0) {
        aclError copy_ret =
            dep::aclrtMemcpyAsync(output_addr.opaque(), operand_size,
                                 operand_addr.opaque(), operand_size,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
        if (copy_ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "DynamicUpdateSlice base copy failed: %d", copy_ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
      }

      // 2. Same-stream D2H to pageable host scalars completes before return.
      //    Read dynamic start_indices from device (D2H) and compute the
      //    element offset of the update window. HLO clamps each start to
      //    [0, dim_size - update_size].
      std::vector<int64_t> operand_dims_vec(
          operand_shape.dimensions().begin(), operand_shape.dimensions().end());
      std::vector<int64_t> update_dims_vec(
          update_shape.dimensions().begin(), update_shape.dimensions().end());
      std::vector<int64_t> operand_strides =
          BuildRowMajorStrides(operand_dims_vec);

      int64_t element_offset = 0;
      for (int64_t i = 0; i < rank; ++i) {
        int64_t start_val = 0;
        auto start_addr = allocations->GetDeviceAddress(args_[2 + i]);
        xla::PrimitiveType start_dtype = param_shapes[2 + i].element_type();
        if (start_dtype == xla::S32) {
          int32_t v = 0;
          dep::aclrtMemcpyAsync(&v, sizeof(int32_t), start_addr.opaque(),
                               sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST,
                               acl_stream);
          start_val = static_cast<int64_t>(v);
        } else if (start_dtype == xla::S64) {
          int64_t v = 0;
          dep::aclrtMemcpyAsync(&v, sizeof(int64_t), start_addr.opaque(),
                               sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST,
                               acl_stream);
          start_val = v;
        } else if (start_dtype == xla::U32) {
          uint32_t v = 0;
          dep::aclrtMemcpyAsync(&v, sizeof(uint32_t), start_addr.opaque(),
                               sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST,
                               acl_stream);
          start_val = static_cast<int64_t>(v);
        } else if (start_dtype == xla::U64) {
          uint64_t v = 0;
          dep::aclrtMemcpyAsync(&v, sizeof(uint64_t), start_addr.opaque(),
                               sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST,
                               acl_stream);
          start_val = static_cast<int64_t>(v);
        } else {
          NPU_LOG_FORMAT_PRINT(
              WARN,
              "DynamicUpdateSlice: unsupported start dtype %s, using 0",
              xla::PrimitiveType_Name(start_dtype).c_str());
        }
        int64_t hi = operand_dims_vec[i] - update_dims_vec[i];
        if (hi < 0) hi = 0;
        if (start_val < 0) start_val = 0;
        if (start_val > hi) start_val = hi;
        element_offset += start_val * operand_strides[i];
      }

      // 3. Build a strided view of the output subregion and InplaceCopy the
      //    update into it. The view shares the operand's row-major strides so
      //    it correctly describes a (possibly non-contiguous) sub-block.
      aclDataType dtype = NpuTool::ConvertXlaType2AclDataType(
          operand_shape.element_type());
      aclTensor* output_window = dep::aclCreateTensor(
          update_dims_vec.data(), update_dims_vec.size(), dtype,
          operand_strides.data(), element_offset, ACL_FORMAT_ND,
          operand_dims_vec.data(), operand_dims_vec.size(),
          output_addr.opaque());
      if (output_window == nullptr) {
        const std::string err_msg =
            "DynamicUpdateSlice failed to create output window view";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      ret = dep::aclnnInplaceCopyGetWorkspaceSize(
          output_window, acl_tensor_ptr_vector[1], &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "DynamicUpdateSlice aclnnInplaceCopyGetWorkspaceSize failed: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_tensor(output_window);
        return absl::InternalError(err_msg);
      }
      if (executor == nullptr) {
        const std::string err_msg =
            "DynamicUpdateSlice aclnnInplaceCopy: nullptr executor";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_tensor(output_window);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnInplaceCopy);
      retire_acl_tensor(output_window);
      break;
    }

    case HloOpcode::kGather: {
      // Gather operation: input + index + output = 3 parameters
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Gather operation expects 3 parameters (input, index, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      const auto& input_shape = param_shapes[0];
      const auto& index_shape = param_shapes[1];
      const auto& output_shape = param_shapes[2];

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Gather: lowering_kind=%d, "
                           "input=%s, index=%s, output=%s",
                           static_cast<int>(thunk_info().gather_lowering_kind),
                           NpuTool::GetXlaShapeString(input_shape).c_str(),
                           NpuTool::GetXlaShapeString(index_shape).c_str(),
                           NpuTool::GetXlaShapeString(output_shape).c_str());

      // ============================================================
      // Host-side reference gather. Used for:
      //  (1) gathers no aclnn lowering supports (kUnsupported) — e.g.
      //      batched gathers with multiple offset dims such as the
      //      permutation gather inside _lu_solve; and
      //  (2) multi-dim (kMultiDimFlattenGather) gathers whose gathered
      //      dims are not all unit (slice_size == 1). The device flatten
      //      path mishandles trivial (full-slice) and partial gathered
      //      dims (e.g. linalg.solve's gather.8), so fall back to the
      //      correct reference implementation here.
      // Implements the full XLA gather semantics on the host (correct for
      // any shape and dtype) at the cost of D2H/H2D round-trips.
      // ============================================================
      bool host_gather_needed =
          (thunk_info().gather_lowering_kind ==
           ThunkInfo::GatherLoweringKind::kUnsupported);
      if (thunk_info().gather_lowering_kind ==
          ThunkInfo::GatherLoweringKind::kMultiDimFlattenGather) {
        for (int64_t gd : thunk_info().gather_start_index_map) {
          if (gd >= 0 &&
              gd < (int64_t)thunk_info().gather_slice_sizes.size() &&
              thunk_info().gather_slice_sizes[gd] != 1) {
            host_gather_needed = true;
            break;
          }
        }
      }
      if (host_gather_needed) {
        // Use stream-ordered copies and ordinary (non-pinned) host storage.
        // CANN completes pageable-host MemcpyAsync before returning, so CPU
        // reads and local-buffer destruction are safe without a separate
        // stream drain. Do not substitute aclrtMallocHost storage here.
        const int64_t operand_rank = input_shape.dimensions_size();
        const int64_t index_rank = index_shape.dimensions_size();
        const int64_t output_rank = output_shape.dimensions_size();
        const int64_t ivd = thunk_info().gather_index_vector_dim;

        if (ivd < 0 || ivd >= index_rank) {
          return absl::InternalError(absl::StrFormat(
              "Host gather: unsupported index_vector_dim=%lld "
              "(index_rank=%lld)",
              static_cast<long long>(ivd),
              static_cast<long long>(index_rank)));
        }

        const auto& offset_dims = thunk_info().gather_offset_dims;
        const auto& collapsed = thunk_info().gather_collapsed_slice_dims;
        const auto& sim = thunk_info().gather_start_index_map;
        const auto& slice_sizes = thunk_info().gather_slice_sizes;
        const auto& obd = thunk_info().gather_operand_batching_dims;
        const auto& sibd = thunk_info().gather_start_indices_batching_dims;
        const int64_t K = static_cast<int64_t>(sim.size());

        const size_t elem_bytes =
            ShapeUtil::ByteSizeOfPrimitiveType(input_shape.element_type());
        const size_t idx_elem_bytes =
            ShapeUtil::ByteSizeOfPrimitiveType(index_shape.element_type());
        const bool idx_is_s64 = (index_shape.element_type() == PrimitiveType::S64);

        const int64_t operand_elems = ShapeUtil::ElementsIn(input_shape);
        const int64_t index_elems = ShapeUtil::ElementsIn(index_shape);
        const int64_t output_elems = ShapeUtil::ElementsIn(output_shape);

        std::vector<int64_t> Os(operand_rank), Is(index_rank), Gs(output_rank);
        for (int64_t i = 0; i < operand_rank; ++i) Os[i] = input_shape.dimensions(i);
        for (int64_t i = 0; i < index_rank; ++i) Is[i] = index_shape.dimensions(i);
        for (int64_t i = 0; i < output_rank; ++i) Gs[i] = output_shape.dimensions(i);

        // Row-major strides.
        std::vector<int64_t> o_stride(operand_rank, 1);
        for (int64_t i = operand_rank - 2; i >= 0; --i)
          o_stride[i] = o_stride[i + 1] * Os[i + 1];
        std::vector<int64_t> i_stride(index_rank, 1);
        for (int64_t i = index_rank - 2; i >= 0; --i)
          i_stride[i] = i_stride[i + 1] * Is[i + 1];
        std::vector<int64_t> g_stride(output_rank, 1);
        for (int64_t i = output_rank - 2; i >= 0; --i)
          g_stride[i] = g_stride[i + 1] * Gs[i + 1];

        auto is_in = [](const std::vector<int64_t>& v, int64_t x) -> bool {
          return std::find(v.begin(), v.end(), x) != v.end();
        };

        // Non-offset output dims correspond positionally to index dims
        // excluding index_vector_dim (batch dims included on the index side).
        std::vector<int64_t> non_offset_out;
        for (int64_t d = 0; d < output_rank; ++d)
          if (!is_in(offset_dims, d)) non_offset_out.push_back(d);
        std::vector<int64_t> idx_excl_ivd;
        for (int64_t d = 0; d < index_rank; ++d)
          if (d != ivd) idx_excl_ivd.push_back(d);

        // Window operand dims (non-collapsed, non-batch) map positionally to
        // the sorted offset output dims.
        std::vector<int64_t> sorted_offset = offset_dims;
        std::sort(sorted_offset.begin(), sorted_offset.end());
        std::vector<int64_t> window_op_dims;
        for (int64_t d = 0; d < operand_rank; ++d)
          if (!is_in(collapsed, d) && !is_in(obd, d))
            window_op_dims.push_back(d);

        // operand dim -> offset output dim (for window dims), else -1
        std::vector<int64_t> offset_out_for_op(operand_rank, -1);
        for (size_t j = 0; j < window_op_dims.size() && j < sorted_offset.size(); ++j)
          offset_out_for_op[window_op_dims[j]] = sorted_offset[j];

        // batch operand dim -> output dim
        std::vector<int64_t> batch_out_for_op(operand_rank, -1);
        for (size_t i = 0; i < obd.size() && i < sibd.size(); ++i) {
          int64_t sib = sibd[i];
          for (size_t j = 0; j < idx_excl_ivd.size(); ++j)
            if (idx_excl_ivd[j] == sib && j < non_offset_out.size()) {
              batch_out_for_op[obd[i]] = non_offset_out[j];
              break;
            }
        }

        std::vector<uint8_t> host_operand(operand_elems * elem_bytes);
        std::vector<uint8_t> host_index(index_elems * idx_elem_bytes);
        std::vector<uint8_t> host_output(output_elems * elem_bytes, 0);

        auto op_addr = allocations->GetDeviceAddress(args_[0]);
        auto idx_addr = allocations->GetDeviceAddress(args_[1]);
        auto out_addr = allocations->GetDeviceAddress(args_.back());

        if (operand_elems > 0) {
          aclError r = dep::aclrtMemcpyAsync(host_operand.data(),
                                        operand_elems * elem_bytes,
                                        op_addr.opaque(),
                                        operand_elems * elem_bytes,
                                        ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
          if (r != ACL_SUCCESS)
            return absl::InternalError(absl::StrFormat(
                "Host gather: failed to read operand: %d", r));
        }
        if (index_elems > 0) {
          aclError r = dep::aclrtMemcpyAsync(host_index.data(),
                                        index_elems * idx_elem_bytes,
                                        idx_addr.opaque(),
                                        index_elems * idx_elem_bytes,
                                        ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
          if (r != ACL_SUCCESS)
            return absl::InternalError(absl::StrFormat(
                "Host gather: failed to read indices: %d", r));
        }

        auto read_index = [&](const std::vector<int64_t>& pos) -> int64_t {
          int64_t off = 0;
          for (int64_t d = 0; d < index_rank; ++d) off += pos[d] * i_stride[d];
          if (idx_is_s64)
            return reinterpret_cast<const int64_t*>(host_index.data())[off];
          return reinterpret_cast<const int32_t*>(host_index.data())[off];
        };

        std::vector<int64_t> o(output_rank, 0);
        std::vector<int64_t> ipos(index_rank, 0);
        std::vector<int64_t> op_idx(operand_rank, 0);
        std::vector<int64_t> svals(K, 0);
        for (int64_t linear = 0; linear < output_elems; ++linear) {
          // Decode linear (row-major) -> output multi-index o.
          int64_t rem = linear;
          for (int64_t d = 0; d < output_rank; ++d) {
            o[d] = rem / g_stride[d];
            rem %= g_stride[d];
          }

          // Read the K start-index values from the indices tensor.
          for (int64_t j = 0; j < static_cast<int64_t>(idx_excl_ivd.size()); ++j)
            ipos[idx_excl_ivd[j]] = o[non_offset_out[j]];
          for (int64_t k = 0; k < K; ++k) {
            ipos[ivd] = k;
            svals[k] = read_index(ipos);
          }

          // Base operand index (start), then clamp to valid slice range.
          for (int64_t d = 0; d < operand_rank; ++d) op_idx[d] = 0;
          for (size_t i = 0; i < obd.size(); ++i)
            if (batch_out_for_op[obd[i]] >= 0)
              op_idx[obd[i]] = o[batch_out_for_op[obd[i]]];
          for (int64_t k = 0; k < K; ++k) op_idx[sim[k]] = svals[k];
          for (int64_t d = 0; d < operand_rank; ++d) {
            int64_t hi = Os[d] - slice_sizes[d];
            if (op_idx[d] < 0) op_idx[d] = 0;
            if (op_idx[d] > hi) op_idx[d] = hi;
          }
          // Add window offsets for non-collapsed operand dims.
          for (int64_t d = 0; d < operand_rank; ++d)
            if (offset_out_for_op[d] >= 0)
              op_idx[d] += o[offset_out_for_op[d]];

          int64_t op_off = 0;
          for (int64_t d = 0; d < operand_rank; ++d) op_off += op_idx[d] * o_stride[d];
          std::memcpy(host_output.data() + linear * elem_bytes,
                      host_operand.data() + op_off * elem_bytes, elem_bytes);
        }

        if (output_elems > 0) {
          aclError r = dep::aclrtMemcpyAsync(out_addr.opaque(),
                                        output_elems * elem_bytes,
                                        host_output.data(),
                                        output_elems * elem_bytes,
                                        ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
          if (r != ACL_SUCCESS)
            return absl::InternalError(absl::StrFormat(
                "Host gather: failed to write output: %d", r));
        }
        break;
      }

      // ===================== Multidimensional gather: flatten + flat indices =====================
      if (thunk_info().gather_lowering_kind ==
          ThunkInfo::GatherLoweringKind::kMultiDimFlattenGather) {
        // Strategy: Flatten gathered dims to one dimension, calculate flat indices, then use GatherV2
        //
        // Example: operand [4,2,3,3], start_index_map={2,3}
        //     → flatten dim 2,3 → operand_view [4,2,9]
        //     indices [6,2] → flat_indices [6], flat[k]=idx0*3+idx1
        //     GatherV2(operand_view, dim=2, flat_indices_reshaped[6,1]) → [4,2,6]

        const auto& start_index_map = thunk_info().gather_start_index_map;

        // Sort gathered dims to determine flattening range
        std::vector<int64_t> sorted_dims(start_index_map.begin(),
                                         start_index_map.end());
        std::sort(sorted_dims.begin(), sorted_dims.end());
        int64_t first_gather_dim = sorted_dims.front();
        int64_t flat_dim_size = 1;
        for (auto d : sorted_dims) {
          flat_dim_size *= input_shape.dimensions(d);
        }

        // Calculate the stride of flat index (column-major in gathered dims)
        // flat = sum(indices[:,i] * stride[i])
        // stride[0] = dim_sizes[1]*...*dim_sizes[N-1], stride[1] = dim_sizes[2]*...*dim_sizes[N-1], ...
        std::vector<int64_t> gather_strides(start_index_map.size());
        // Calculate stride in sorted_dims order
        std::vector<int64_t> dim_sizes_sorted;
        for (auto d : sorted_dims) {
          dim_sizes_sorted.push_back(input_shape.dimensions(d));
        }
        for (size_t i = 0; i < dim_sizes_sorted.size(); ++i) {
          gather_strides[i] = 1;
          for (size_t j = i + 1; j < dim_sizes_sorted.size(); ++j) {
            gather_strides[i] *= dim_sizes_sorted[j];
          }
        }
        // Map stride back to the order of start_index_map
        std::vector<int64_t> index_strides(start_index_map.size());
        for (size_t i = 0; i < start_index_map.size(); ++i) {
          auto it = std::lower_bound(sorted_dims.begin(), sorted_dims.end(),
                                     start_index_map[i]);
          size_t sorted_idx = std::distance(sorted_dims.begin(), it);
          index_strides[i] = gather_strides[sorted_idx];
        }

        // D2H: Copy indices to host and calculate flat indices
        auto index_addr = allocations->GetDeviceAddress(args_[1]);
        int64_t index_total = 1;
        for (auto d : index_shape.dimensions()) index_total *= d;
        size_t index_bytes =
            index_total * ShapeUtil::ByteSizeOfPrimitiveType(
                              index_shape.element_type());

        // Pageable storage is required: MemcpyAsync must complete before the
        // CPU reads indices. Pinned storage would require an explicit wait.
        std::vector<int64_t> host_index_storage(
            (index_bytes + sizeof(int64_t) - 1) / sizeof(int64_t));
        void* host_indices = host_index_storage.data();
        auto memcpy_ret = dep::aclrtMemcpyAsync(
            host_indices, index_bytes, index_addr.opaque(), index_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
        if (memcpy_ret != ACL_SUCCESS) {
          return absl::InternalError("Failed to D2H copy gather indices");
        }

        // Calculate flat indices
        // indices layout: batch_dims... × index_vector_dim
        // index_vector_dim = last dim of indices (validated)
        int64_t batch_count = index_total / (int64_t)start_index_map.size();
        auto index_elem_type = index_shape.element_type();

        int64_t partial_slice_size = thunk_info().gather_partial_slice_size;
        int64_t expansion_factor = (partial_slice_size > 0) ? partial_slice_size : 1;

        // When slicing partially, you need to find the flat stride corresponding to dim for index expansion.
        // Expanded formula: expanded[b*S+k] = flat_start[b] + k * stride_of_partial_dim
        int64_t partial_dim_stride = 1;
        if (partial_slice_size > 0) {
          // Find gathered dim where slice_size > 1 and < dim_size
          for (size_t i = 0; i < start_index_map.size(); ++i) {
            int64_t gd = start_index_map[i];
            int64_t ss = input_shape.dimensions(gd);
            // position in sorted_dims
            auto it = std::lower_bound(sorted_dims.begin(), sorted_dims.end(),
                                       gd);
            size_t sorted_idx = std::distance(sorted_dims.begin(), it);
            // The slice_size of this dim in start_index_map
            if (thunk_info().gather_slice_sizes[gd] > 1 &&
                thunk_info().gather_slice_sizes[gd] < ss) {
              partial_dim_stride = gather_strides[sorted_idx];
              break;
            }
          }
        }

        // Process flat indices uniformly as int64
        int64_t expanded_batch_count = batch_count * expansion_factor;
        size_t flat_bytes = (size_t)expanded_batch_count * sizeof(int64_t);
        std::vector<int64_t> host_flat(expanded_batch_count);
        auto* flat_ptr = host_flat.data();
        // Read and calculate flat index based on indices element type
        for (int64_t b = 0; b < batch_count; ++b) {
          int64_t flat_val = 0;
          for (size_t g = 0; g < start_index_map.size(); ++g) {
            int64_t idx_val = 0;
            int64_t offset = b * (int64_t)start_index_map.size() + (int64_t)g;
            if (index_elem_type == S32) {
              idx_val = static_cast<const int32_t*>(host_indices)[offset];
            } else if (index_elem_type == S64) {
              idx_val = static_cast<const int64_t*>(host_indices)[offset];
            } else {
              return absl::InternalError(
                  "Unsupported gather index element type");
            }
            flat_val += idx_val * index_strides[g];
          }
          // Store expanded flat indices
          for (int64_t k = 0; k < expansion_factor; ++k) {
            flat_ptr[b * expansion_factor + k] =
                flat_val + k * partial_dim_stride;
          }
        }

        // H2D: Copy flat indices to device
        TF_ASSIGN_OR_RETURN(void* dev_flat,
                            allocate_temporary_device_memory(flat_bytes));

        memcpy_ret = dep::aclrtMemcpyAsync(
            dev_flat, flat_bytes, host_flat.data(), flat_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        if (memcpy_ret != ACL_SUCCESS) {
          RetireTemporaryDeviceMemory(dev_flat);
          return absl::InternalError("Failed to H2D copy flat indices");
        }

        // Create operand view: flatten gathered dims
        std::vector<int64_t> operand_view_dims;
        for (int64_t i = 0; i < first_gather_dim; ++i) {
          operand_view_dims.push_back(input_shape.dimensions(i));
        }
        operand_view_dims.push_back(flat_dim_size);
        for (int64_t i = (int64_t)input_shape.dimensions().size() - 1;
             i > sorted_dims.back(); --i) {
          // Insert from back to front, then reverse
        }
        for (int64_t i = sorted_dims.back() + 1;
             i < (int64_t)input_shape.dimensions().size(); ++i) {
          operand_view_dims.push_back(input_shape.dimensions(i));
        }

        std::vector<int64_t> operand_view_strides(operand_view_dims.size());
        {
          int64_t s = 1;
          for (int i = operand_view_dims.size() - 1; i >= 0; --i) {
            operand_view_strides[i] = s;
            s *= operand_view_dims[i];
          }
        }

        auto input_addr = allocations->GetDeviceAddress(args_[0]);
        aclDataType input_dtype =
            NpuTool::ConvertXlaType2AclDataType(input_shape.element_type());
        aclTensor* operand_view = dep::aclCreateTensor(
            operand_view_dims.data(), operand_view_dims.size(), input_dtype,
            operand_view_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            operand_view_dims.data(), operand_view_dims.size(),
            input_addr.opaque());
        if (operand_view == nullptr) {
          RetireTemporaryDeviceMemory(dev_flat);
          return absl::InternalError("Failed to create operand view for multi-dim gather");
        }

        // Create flat indices tensor: shape = batch_dims + [expansion_factor]
        // expansion_factor=1 is standard multi-dimensional gather, >1 is partial slicing
        std::vector<int64_t> flat_index_dims;
        for (int64_t i = 0; i < (int64_t)index_shape.dimensions().size(); ++i) {
          if (i != thunk_info().gather_index_vector_dim) {
            flat_index_dims.push_back(index_shape.dimensions(i));
          }
        }
        flat_index_dims.push_back(expansion_factor);
        if (flat_index_dims.empty()) flat_index_dims = {expansion_factor};

        std::vector<int64_t> flat_index_strides(flat_index_dims.size());
        {
          int64_t s = 1;
          for (int i = flat_index_dims.size() - 1; i >= 0; --i) {
            flat_index_strides[i] = s;
            s *= flat_index_dims[i];
          }
        }

        aclTensor* flat_indices_tensor = dep::aclCreateTensor(
            flat_index_dims.data(), flat_index_dims.size(), ACL_INT64,
            flat_index_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            flat_index_dims.data(), flat_index_dims.size(), dev_flat);
        if (flat_indices_tensor == nullptr) {
          retire_acl_tensor(operand_view);
          RetireTemporaryDeviceMemory(dev_flat);
          return absl::InternalError("Failed to create flat indices tensor");
        }

        // GatherV2(operand_view, dim=first_gather_dim, flat_indices, output)
        // Output shape = operand_view[:dim] + flat_index_dims + operand_view[dim+1:]
        std::vector<int64_t> gatherv2_output_dims;
        for (int64_t i = 0; i < first_gather_dim; ++i) {
          gatherv2_output_dims.push_back(operand_view_dims[i]);
        }
        for (auto d : flat_index_dims) {
          gatherv2_output_dims.push_back(d);
        }
        for (int64_t i = first_gather_dim + 1;
             i < (int64_t)operand_view_dims.size(); ++i) {
          gatherv2_output_dims.push_back(operand_view_dims[i]);
        }

        // Output view (if shapes don't match)
        aclTensor* output_tensor = acl_tensor_ptr_vector[2];
        aclTensor* output_view = nullptr;
        bool need_cleanup_output = false;

        bool out_match =
            (gatherv2_output_dims.size() == (size_t)output_shape.dimensions().size());
        if (out_match) {
          for (size_t i = 0; i < gatherv2_output_dims.size(); ++i) {
            if (gatherv2_output_dims[i] != output_shape.dimensions(i)) {
              out_match = false;
              break;
            }
          }
        }

        if (!out_match) {
          std::vector<int64_t> view_strides(gatherv2_output_dims.size());
          int64_t s = 1;
          for (int i = gatherv2_output_dims.size() - 1; i >= 0; --i) {
            view_strides[i] = s;
            s *= gatherv2_output_dims[i];
          }
          aclDataType output_dtype =
              NpuTool::ConvertXlaType2AclDataType(output_shape.element_type());
          auto output_addr = allocations->GetDeviceAddress(args_.back());

          output_view = dep::aclCreateTensor(
              gatherv2_output_dims.data(), gatherv2_output_dims.size(),
              output_dtype, view_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
              gatherv2_output_dims.data(), gatherv2_output_dims.size(),
              output_addr.opaque());
          output_tensor = output_view;
          need_cleanup_output = true;
        }

        ret = dep::aclnnGatherV2GetWorkspaceSize(
            operand_view, first_gather_dim, flat_indices_tensor,
            output_tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          retire_acl_tensor(operand_view);
          retire_acl_tensor(flat_indices_tensor);
          RetireTemporaryDeviceMemory(dev_flat);
          if (need_cleanup_output && output_view) retire_acl_tensor(output_view);
          return absl::InternalError(absl::StrFormat(
              "aclnnGatherV2GetWorkspaceSize (multi-dim) failed: %d", ret));
        }

        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnGatherV2);

        retire_acl_tensor(operand_view);
        retire_acl_tensor(flat_indices_tensor);
        RetireTemporaryDeviceMemory(dev_flat);
        if (need_cleanup_output && output_view) retire_acl_tensor(output_view);

        break;
      }

      // ===================== Single dimension gather: squeeze index_vector_dim =====================
      // Use effective dim (the only non-trivial gathered dim)
      int64_t dim = thunk_info().gather_effective_dim;
      int64_t effective_pos = thunk_info().gather_effective_index_pos;
      int64_t index_vector_size =
          index_shape.dimensions(thunk_info().gather_index_vector_dim);
      int64_t partial_slice_size = thunk_info().gather_partial_slice_size;

      // Extract the index column corresponding to the valid dim, squeeze index_vector_dim
      std::vector<int64_t> squeezed_index_dims;
      for (int64_t i = 0; i < (int64_t)index_shape.dimensions().size(); ++i) {
        if (i != thunk_info().gather_index_vector_dim) {
          squeezed_index_dims.push_back(index_shape.dimensions(i));
        }
      }
      if (squeezed_index_dims.empty()) {
        squeezed_index_dims = {1};
      }

      // ============================================================
      // partial slice gather (slice_size > 1):
      // Expand a single start index to [start, start+1,..., start+slice_size-1]
      // Then GatherV2 takes all elements at once
      // ============================================================
      if (partial_slice_size > 1) {
        // 1) D2H: copy start indices to host
        auto index_addr = allocations->GetDeviceAddress(args_[1]);
        int64_t batch_count = 1;
        for (auto d : squeezed_index_dims) batch_count *= d;
        int64_t total_indices = batch_count * partial_slice_size;

        // indices may have an index_vector_size column, and the effective_pos column needs to be extracted
        size_t index_bytes = batch_count * (size_t)index_vector_size *
                             ShapeUtil::ByteSizeOfPrimitiveType(
                                 index_shape.element_type());

        // Keep both host buffers pageable: the same-stream copies complete
        // before CPU consumption / buffer destruction, without a stream drain.
        std::vector<int64_t> host_index_storage(
            (index_bytes + sizeof(int64_t) - 1) / sizeof(int64_t));
        void* host_indices = host_index_storage.data();
        aclError copy_ret = dep::aclrtMemcpyAsync(
            host_indices, index_bytes, index_addr.opaque(), index_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
        if (copy_ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Partial slice gather: failed to read indices: %d", copy_ret));
        }

        // 2) Expand on the host: generate partial_slice_size consecutive indexes at each batch position
        size_t expanded_bytes = total_indices * sizeof(int64_t);
        std::vector<int64_t> host_expanded(total_indices);
        auto* out_ptr = host_expanded.data();
        auto index_type = index_shape.element_type();
        for (int64_t b = 0; b < batch_count; ++b) {
          // Extract start index
          int64_t offset = b * index_vector_size + effective_pos;
          int64_t start_idx = 0;
          if (index_type == S32) {
            start_idx = static_cast<const int32_t*>(host_indices)[offset];
          } else {
            start_idx = static_cast<const int64_t*>(host_indices)[offset];
          }
          // Expand: [start_idx, start_idx+1,..., start_idx+partial_slice_size-1]
          for (int64_t k = 0; k < partial_slice_size; ++k) {
            out_ptr[b * partial_slice_size + k] = start_idx + k;
          }
        }

        // 3) H2D: Copy the expanded indices
        TF_ASSIGN_OR_RETURN(
            void* dev_expanded,
            allocate_temporary_device_memory(expanded_bytes));
        copy_ret = dep::aclrtMemcpyAsync(
            dev_expanded, expanded_bytes, host_expanded.data(), expanded_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        if (copy_ret != ACL_SUCCESS) {
          RetireTemporaryDeviceMemory(dev_expanded);
          return absl::InternalError(absl::StrFormat(
              "Partial slice gather: failed to write indices: %d", copy_ret));
        }

        // expanded indices shape: batch_dims ++ [partial_slice_size]
        std::vector<int64_t> expanded_dims = squeezed_index_dims;
        expanded_dims.push_back(partial_slice_size);
        std::vector<int64_t> expanded_strides(expanded_dims.size());
        {
          int64_t s = 1;
          for (int i = expanded_dims.size() - 1; i >= 0; --i) {
            expanded_strides[i] = s;
            s *= expanded_dims[i];
          }
        }

        aclTensor* expanded_indices = dep::aclCreateTensor(
            expanded_dims.data(), expanded_dims.size(), ACL_INT64,
            expanded_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            expanded_dims.data(), expanded_dims.size(), dev_expanded);
        if (!expanded_indices) {
          RetireTemporaryDeviceMemory(dev_expanded);
          return absl::InternalError(
              "Failed to create expanded indices tensor for partial slice Gather");
        }

        // GatherV2 output shape: input[:dim] + expanded_dims + input[dim+1:]
        std::vector<int64_t> gatherv2_output_dims;
        for (int64_t i = 0; i < dim; ++i) {
          gatherv2_output_dims.push_back(input_shape.dimensions(i));
        }
        for (auto d : expanded_dims) {
          gatherv2_output_dims.push_back(d);
        }
        for (int64_t i = dim + 1;
             i < (int64_t)input_shape.dimensions().size(); ++i) {
          gatherv2_output_dims.push_back(input_shape.dimensions(i));
        }

        // Output view (if GatherV2 output shape is different from XLA output shape)
        aclTensor* output_tensor = acl_tensor_ptr_vector[2];
        aclTensor* view_tensor = nullptr;
        bool need_cleanup_output = false;

        bool out_match =
            (gatherv2_output_dims.size() ==
             (size_t)output_shape.dimensions().size());
        if (out_match) {
          for (size_t i = 0; i < gatherv2_output_dims.size(); ++i) {
            if (gatherv2_output_dims[i] != output_shape.dimensions(i)) {
              out_match = false;
              break;
            }
          }
        }
        if (!out_match) {
          int64_t gv2_el = 1, xla_el = 1;
          for (auto d : gatherv2_output_dims) gv2_el *= d;
          for (auto d : output_shape.dimensions()) xla_el *= d;
          if (gv2_el != xla_el) {
            retire_acl_tensor(expanded_indices);
            RetireTemporaryDeviceMemory(dev_expanded);
            return absl::InternalError(absl::StrFormat(
                "Partial slice Gather element count mismatch: "
                "GatherV2 expects %lld, XLA output has %lld",
                gv2_el, xla_el));
          }
          std::vector<int64_t> view_strides(gatherv2_output_dims.size());
          int64_t s = 1;
          for (int i = gatherv2_output_dims.size() - 1; i >= 0; --i) {
            view_strides[i] = s;
            s *= gatherv2_output_dims[i];
          }
          aclDataType out_dtype =
              NpuTool::ConvertXlaType2AclDataType(output_shape.element_type());
          auto out_addr = allocations->GetDeviceAddress(args_.back());
          view_tensor = dep::aclCreateTensor(
              gatherv2_output_dims.data(), gatherv2_output_dims.size(),
              out_dtype, view_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
              gatherv2_output_dims.data(), gatherv2_output_dims.size(),
              out_addr.opaque());
          if (!view_tensor) {
            retire_acl_tensor(expanded_indices);
            RetireTemporaryDeviceMemory(dev_expanded);
            return absl::InternalError(
                "Failed to create partial slice output view");
          }
          output_tensor = view_tensor;
          need_cleanup_output = true;
        }

        ret = dep::aclnnGatherV2GetWorkspaceSize(
            acl_tensor_ptr_vector[0], dim, expanded_indices, output_tensor,
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          retire_acl_tensor(expanded_indices);
          RetireTemporaryDeviceMemory(dev_expanded);
          if (need_cleanup_output && view_tensor) retire_acl_tensor(view_tensor);
          return absl::InternalError(absl::StrFormat(
              "aclnnGatherV2GetWorkspaceSize (partial slice) failed: %d", ret));
        }
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnGatherV2);

        retire_acl_tensor(expanded_indices);
        RetireTemporaryDeviceMemory(dev_expanded);
        if (need_cleanup_output && view_tensor) retire_acl_tensor(view_tensor);
        break;
      }

      // ============================================================
      // Standard single-dimensional gather (slice_size == 1)
      // ============================================================
      aclTensor* squeezed_indices = nullptr;
      void* dev_extracted_indices = nullptr;

      if (index_vector_size == 1) {
        // Classic single-dimensional gather: indices has only 1 column, squeeze directly
        std::vector<int64_t> squeezed_index_strides(squeezed_index_dims.size());
        int64_t stride = 1;
        for (int i = squeezed_index_dims.size() - 1; i >= 0; --i) {
          squeezed_index_strides[i] = stride;
          stride *= squeezed_index_dims[i];
        }

        aclDataType index_dtype =
            NpuTool::ConvertXlaType2AclDataType(index_shape.element_type());
        auto index_addr = allocations->GetDeviceAddress(args_[1]);
        squeezed_indices = dep::aclCreateTensor(
            squeezed_index_dims.data(), squeezed_index_dims.size(), index_dtype,
            squeezed_index_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            squeezed_index_dims.data(), squeezed_index_dims.size(),
            index_addr.opaque());
      } else {
        // Multi-column indices: Need to extract the effective_pos column through D2H
        auto index_addr = allocations->GetDeviceAddress(args_[1]);
        int64_t batch_count = 1;
        for (auto d : squeezed_index_dims) batch_count *= d;
        size_t index_bytes = batch_count * (size_t)index_vector_size *
                             ShapeUtil::ByteSizeOfPrimitiveType(
                                 index_shape.element_type());
        size_t extracted_bytes = batch_count * sizeof(int64_t);

        // Pageable buffers make the same-stream copies complete before the
        // CPU reads the indices or releases the extracted column.
        std::vector<int64_t> host_index_storage(
            (index_bytes + sizeof(int64_t) - 1) / sizeof(int64_t));
        void* host_indices = host_index_storage.data();
        aclError copy_ret = dep::aclrtMemcpyAsync(
            host_indices, index_bytes, index_addr.opaque(), index_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
        if (copy_ret != ACL_SUCCESS) {
          return absl::InternalError(absl::StrFormat(
              "Gather index extraction: failed to read indices: %d", copy_ret));
        }

        std::vector<int64_t> host_extracted(batch_count);
        auto* out_ptr = host_extracted.data();
        auto index_type = index_shape.element_type();
        for (int64_t b = 0; b < batch_count; ++b) {
          int64_t offset = b * index_vector_size + effective_pos;
          if (index_type == S32) {
            out_ptr[b] = static_cast<const int32_t*>(host_indices)[offset];
          } else {
            out_ptr[b] = static_cast<const int64_t*>(host_indices)[offset];
          }
        }

        TF_ASSIGN_OR_RETURN(
            dev_extracted_indices,
            allocate_temporary_device_memory(extracted_bytes));
        copy_ret = dep::aclrtMemcpyAsync(
            dev_extracted_indices, extracted_bytes, host_extracted.data(),
            extracted_bytes, ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        if (copy_ret != ACL_SUCCESS) {
          RetireTemporaryDeviceMemory(dev_extracted_indices);
          return absl::InternalError(absl::StrFormat(
              "Gather index extraction: failed to write indices: %d", copy_ret));
        }

        std::vector<int64_t> extracted_strides(squeezed_index_dims.size());
        int64_t s = 1;
        for (int i = squeezed_index_dims.size() - 1; i >= 0; --i) {
          extracted_strides[i] = s;
          s *= squeezed_index_dims[i];
        }
        squeezed_indices = dep::aclCreateTensor(
            squeezed_index_dims.data(), squeezed_index_dims.size(), ACL_INT64,
            extracted_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            squeezed_index_dims.data(), squeezed_index_dims.size(),
            dev_extracted_indices);
      }

      if (squeezed_indices == nullptr) {
        if (dev_extracted_indices) RetireTemporaryDeviceMemory(dev_extracted_indices);
        return absl::InternalError("Failed to create indices tensor for Gather");
      }

      // Calculate the expected output shape of GatherV2: input[:dim] + squeezed_indices + input[dim+1:]
      std::vector<int64_t> gatherv2_output_dims;
      for (int64_t i = 0; i < dim; ++i) {
        gatherv2_output_dims.push_back(input_shape.dimensions(i));
      }
      for (int64_t d : squeezed_index_dims) {
        gatherv2_output_dims.push_back(d);
      }
      for (int64_t i = dim + 1; i < (int64_t)input_shape.dimensions().size(); ++i) {
        gatherv2_output_dims.push_back(input_shape.dimensions(i));
      }

      // Dimension-by-dimension comparison of GatherV2 output shape and XLA output shape
      bool shapes_match =
          (gatherv2_output_dims.size() == (size_t)output_shape.dimensions().size());
      if (shapes_match) {
        for (size_t i = 0; i < gatherv2_output_dims.size(); ++i) {
          if (gatherv2_output_dims[i] != output_shape.dimensions(i)) {
            shapes_match = false;
            break;
          }
        }
      }

      aclTensor* output_tensor = acl_tensor_ptr_vector[2];
      aclTensor* view_tensor = nullptr;
      bool need_cleanup_output = false;

      if (!shapes_match) {
        // The shapes do not match but the number of elements should be the same (guaranteed by compile-time checking)
        // Create a view tensor of the desired shape of GatherV2, pointing to the XLA output buffer
        int64_t gatherv2_elements = 1;
        for (auto d : gatherv2_output_dims) gatherv2_elements *= d;
        int64_t xla_elements = 1;
        for (auto d : output_shape.dimensions()) xla_elements *= d;

        if (gatherv2_elements != xla_elements) {
          retire_acl_tensor(squeezed_indices);
          return absl::InternalError(absl::StrFormat(
              "Gather element count mismatch: GatherV2 expects %lld, "
              "XLA output has %lld",
              gatherv2_elements, xla_elements));
        }

        std::vector<int64_t> view_strides(gatherv2_output_dims.size());
        int64_t s = 1;
        for (int i = gatherv2_output_dims.size() - 1; i >= 0; --i) {
          view_strides[i] = s;
          s *= gatherv2_output_dims[i];
        }

        aclDataType output_dtype =
            NpuTool::ConvertXlaType2AclDataType(output_shape.element_type());
        auto output_addr = allocations->GetDeviceAddress(args_.back());

        view_tensor = dep::aclCreateTensor(
            gatherv2_output_dims.data(), gatherv2_output_dims.size(),
            output_dtype, view_strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            gatherv2_output_dims.data(), gatherv2_output_dims.size(),
            output_addr.opaque());

        if (view_tensor == nullptr) {
          retire_acl_tensor(squeezed_indices);
          const std::string err_msg =
              "Failed to create GatherV2 output view tensor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        output_tensor = view_tensor;
        need_cleanup_output = true;

        NPU_LOG_FORMAT_PRINT(DEBUG,
                             "Gather: created output view, GatherV2 shape size=%d, "
                             "XLA output shape size=%d",
                             (int)gatherv2_output_dims.size(),
                             (int)output_shape.dimensions().size());
      }

      // Call GatherV2(input, dim, squeezed_indices, output)
      ret = dep::aclnnGatherV2GetWorkspaceSize(
          acl_tensor_ptr_vector[0], dim, squeezed_indices, output_tensor,
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclnnGatherV2GetWorkspaceSize failed: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_tensor(squeezed_indices);
        if (dev_extracted_indices) RetireTemporaryDeviceMemory(dev_extracted_indices);
        if (need_cleanup_output && view_tensor) {
          retire_acl_tensor(view_tensor);
        }
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnGatherV2);

      retire_acl_tensor(squeezed_indices);
      if (dev_extracted_indices) RetireTemporaryDeviceMemory(dev_extracted_indices);
      if (need_cleanup_output && view_tensor) {
        retire_acl_tensor(view_tensor);
      }

      break;
    }

    case HloOpcode::kCompare: {
      // Compare operation: input 1 + input 2 + output = 3 parameters
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Compare operation expects 3 parameters (input1, input2, output), "
            "but got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Determine the comparison direction based on hloInstrDesc
      // Format: comparison_direction=EQ/NE/LT/LE/GT/GE
      const std::string& hlo_desc = thunk_info().hloInstrDesc;

      aclnnStatus compare_ret = ACL_SUCCESS;

      if (hlo_desc.find("direction=EQ") != std::string::npos ||
          hlo_desc.find("comparison_direction=EQ") != std::string::npos) {
        // Equal
        compare_ret = dep::aclnnEqTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnEqTensor);
        }
      } else if (hlo_desc.find("direction=NE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=NE") !=
                     std::string::npos) {
        // Not Equal
        compare_ret = dep::aclnnNeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnNeTensor);
        }
      } else if (hlo_desc.find("direction=LT") != std::string::npos ||
                 hlo_desc.find("comparison_direction=LT") !=
                     std::string::npos) {
        // Less Than
        compare_ret = dep::aclnnLtTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnLtTensor);
        }
      } else if (hlo_desc.find("direction=LE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=LE") !=
                     std::string::npos) {
        // Less Equal
        compare_ret = dep::aclnnLeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnLeTensor);
        }
      } else if (hlo_desc.find("direction=GT") != std::string::npos ||
                 hlo_desc.find("comparison_direction=GT") !=
                     std::string::npos) {
        // Greater Than
        compare_ret = dep::aclnnGtTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnGtTensor);
        }
      } else if (hlo_desc.find("direction=GE") != std::string::npos ||
                 hlo_desc.find("comparison_direction=GE") !=
                     std::string::npos) {
        // Greater Equal
        compare_ret = dep::aclnnGeTensorGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
        if (compare_ret == ACL_SUCCESS) {
          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnGeTensor);
        }
      } else {
        const std::string err_msg = absl::StrFormat(
            "Unknown comparison direction in: %s", hlo_desc.c_str());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      if (compare_ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Compare GetWorkspaceSize failed. Error code: %d", compare_ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }
      break;
    }

    case HloOpcode::kSelect: {
      // Select (Where) operation: condition + self + other + output = 4 parameters
      // Selects elements in self or other according to condition
      if (acl_tensor_ptr_vector.size() < 4) {
        const std::string err_msg = absl::StrFormat(
            "Select operation expects 4 parameters (condition, self, other, "
            "output), but got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // acl_tensors[0] = condition
      // acl_tensors[1] = self
      // acl_tensors[2] = other
      // acl_tensors[3] = output
      ret = dep::aclnnSWhereGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector[2], acl_tensor_ptr_vector[3], &workspaceSize,
          &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSWhereGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSWhere);
      break;
    }

    case HloOpcode::kNot: {
      // Not (BitwiseNot/LogicalNot) operation: 1 input + 1 output = 2 parameters
      // Supports logical NOT of BOOL type and bitwise NOT of integer type
      if (acl_tensor_ptr_vector.size() < 2) {
        const std::string err_msg = absl::StrFormat(
            "Not operation expects 2 parameters (input, output), but got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      ret = dep::aclnnBitwiseNotGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector.back(),
          &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnBitwiseNotGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseNot);
      break;
    }

    case HloOpcode::kAnd: {
      // And (BitwiseAnd/LogicalAnd) operation: input 1 + input 2 + output = 3 parameters
      // Supports logical AND of BOOL type and bitwise AND of integer type
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "And operation expects 3 parameters (input1, input2, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // acl_tensors[0] = self
      // acl_tensors[1] = other
      // acl_tensors[2] = output
      ret = dep::aclnnBitwiseAndTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnBitwiseAndTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseAndTensor);
      break;
    }

    case HloOpcode::kOr: {
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Or operation expects 3 parameters (input1, input2, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      ret = dep::aclnnBitwiseOrTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnBitwiseOrTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseOrTensor);
      break;
    }

    case HloOpcode::kXor: {
      if (acl_tensor_ptr_vector.size() < 3) {
        const std::string err_msg = absl::StrFormat(
            "Xor operation expects 3 parameters (input1, input2, output), but "
            "got %d",
            acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      ret = dep::aclnnBitwiseXorTensorGetWorkspaceSize(
          acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "Call aclnnBitwiseXorTensorGetWorkspaceSize failed. Error code: %d",
            ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnBitwiseXorTensor);
      break;
    }

    case HloOpcode::kScatter: {
      // Scatter: data=tensors[0], indices=tensors[1], updates=tensors[2], out=tensors[3]
      if (acl_tensor_ptr_vector.size() < 4) {
        const std::string err_msg = absl::StrFormat(
            "Scatter expects 4 params, got %d", acl_tensor_ptr_vector.size());
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      if (thunk_info().scatter_reduction_kind ==
              NpuThunk::ThunkInfo::ScatterReductionKind::kAdd &&
          !thunk_info().scatter_add_normalized) {
        return absl::UnimplementedError(
            "NPU scatter-add reached runtime without semantic normalization");
      }

      // Layer 2: Degenerate scatter — when indices have zero elements,
      // scatter_dims_to_operand_dims is empty, and the scatter degenerates
      // to a full-window elementwise reduction of data and updates.
      // Correct semantics (NOT identity):
      //   kReplace → output = updates
      //   kAdd is handled by compiler normalization (including empty points).
      //   kMul     → output = data * updates
      if (ShapeUtil::ElementsIn(param_shapes[1]) == 0) {
        auto scatter_kind = thunk_info().scatter_reduction_kind;
        NPU_LOG_FORMAT_PRINT(
            INFO,
            "Degenerate scatter '%s' with zero-element indices (shape=%s): "
            "reduction_kind=%d, applying full-window elementwise update",
            thunk_info().op_name.c_str(),
            ShapeUtil::HumanString(param_shapes[1]).c_str(),
            static_cast<int>(scatter_kind));

        switch (scatter_kind) {
          case NpuThunk::ThunkInfo::ScatterReductionKind::kReplace: {
            // output = updates (D2D copy from args_[2] to args_[3])
            auto update_addr = allocations->GetDeviceAddress(args_[2]);
            auto output_addr = allocations->GetDeviceAddress(args_[3]);
            if (update_addr.opaque() != output_addr.opaque()) {
              const int64_t update_size =
                  ShapeUtil::ByteSizeOf(param_shapes[2]);
              if (update_size > 0) {
                aclError copy_ret = dep::aclrtMemcpy(
                    output_addr.opaque(), update_size,
                    update_addr.opaque(), update_size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
                if (copy_ret != ACL_SUCCESS) {
                  const std::string err_msg = absl::StrFormat(
                      "Degenerate scatter kReplace copy failed: %d", copy_ret);
                  NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                  return absl::InternalError(err_msg);
                }
              }
            }
            return absl::OkStatus();
          }
          case NpuThunk::ThunkInfo::ScatterReductionKind::kAdd:
            return absl::UnimplementedError(
                "Empty-index scatter-add must be eliminated or normalized");
          case NpuThunk::ThunkInfo::ScatterReductionKind::kMul: {
            // output = data * updates
            aclnnStatus ret = dep::aclnnMulGetWorkspaceSize(
                acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[2],
                acl_tensor_ptr_vector[3], &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
              const std::string err_msg = absl::StrFormat(
                  "Degenerate scatter kMul GetWorkspaceSize failed: %d", ret);
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
            exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                    dep::aclnnMul);
            if (!exec_ret.ok()) {
              const std::string err_msg = absl::StrFormat(
                  "Degenerate scatter kMul failed: %s",
                  std::string(exec_ret.message()).c_str());
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
            return absl::OkStatus();
          }
          default: {
            const std::string err_msg = absl::StrFormat(
                "Degenerate scatter '%s': unsupported reduction kind %d",
                thunk_info().op_name.c_str(),
                static_cast<int>(scatter_kind));
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }
        }
      }

      // Null-check tensors
      for (int i = 0; i < 4; ++i) {
        if (acl_tensor_ptr_vector[i] == nullptr) {
          const std::string err_msg =
              absl::StrFormat("Scatter acl_tensor[%d] is nullptr", i);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
      }

      // Layer 1 & 3: Null-check device addresses.
      // Layer 1: Zero-size buffers are legitimate — the constant buffer for
      // empty tensors (e.g., s32[0]) may be null because no device memory
      // is needed. The degenerate scatter case above handles this before
      // reaching here; for non-degenerate scatters the check still applies.
      for (int i = 0; i < 4; ++i) {
        auto addr = allocations->GetDeviceAddress(args_[i]);
        if (addr.opaque() == nullptr) {
          // Layer 1: zero-size buffer is valid, skip
          if (addr.size() == 0) {
            NPU_LOG_FORMAT_PRINT(
                WARN,
                "Scatter arg[%d] alloc=%d is zero-size with null data pointer; "
                "this is valid for empty tensors. "
                "If the scatter is non-degenerate, this may indicate a "
                "buffer assignment issue.",
                i, args_[i].index());
            continue;
          }
          // Layer 3: Non-zero-size null buffer is a real error
          const std::string err_msg = absl::StrFormat(
              "Scatter device_addr[%d] is nullptr, alloc=%d, size=%lld; "
              "this is NOT a zero-size buffer — possible buffer assignment "
              "or allocation failure. Check constant globals map and "
              "BufferForAllocation resolution.",
              i, args_[i].index(),
              static_cast<long long>(addr.size()));
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
      }

      if (acl_stream == nullptr) {
        const std::string err_msg = "Scatter acl_stream is nullptr";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      auto scatter_kind = thunk_info().scatter_reduction_kind;

      if (scatter_kind ==
          NpuThunk::ThunkInfo::ScatterReductionKind::kReplace) {
        if (thunk_info().scatter_lowering_kind ==
                          NpuThunk::ThunkInfo::ScatterLoweringKind::kStaticWindowCopy) {
          if (!CanUseStaticWindowCopy(param_shapes,
                                      thunk_info().scatter_dimension,
                                      thunk_info().scatter_window_start)) {
            const std::string err_msg =
                "Static-window scatter copy has incompatible shapes";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          auto data_addr = allocations->GetDeviceAddress(args_[0]);
          auto output_addr = allocations->GetDeviceAddress(args_[3]);
          const int64_t data_size = ShapeUtil::ByteSizeOf(param_shapes[0]);
          // The base value is produced on acl_stream.  A synchronous
          // aclrtMemcpy is not ordered with that custom stream and can read
          // stale data.  Queue the copy on the same stream, immediately
          // before the window update below.
          aclError copy_ret =
              dep::aclrtMemcpyAsync(output_addr.opaque(), data_size,
                                    data_addr.opaque(), data_size,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
          if (copy_ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "Static-window scatter base copy failed: %d", copy_ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          const std::vector<int64_t> data_shape =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);
          const std::vector<int64_t> update_shape =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[2]);
          const std::vector<int64_t> data_strides =
              BuildRowMajorStrides(data_shape);
          const int64_t window_offset =
              thunk_info().scatter_window_start *
              data_strides[thunk_info().scatter_dimension];
          aclDataType data_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  param_shapes[0].element_type());
          aclTensor* output_window = dep::aclCreateTensor(
              update_shape.data(), update_shape.size(), data_dtype,
              data_strides.data(), window_offset, ACL_FORMAT_ND,
              data_shape.data(), data_shape.size(), output_addr.opaque());
          if (output_window == nullptr) {
            const std::string err_msg =
                "Static-window scatter failed to create output view";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          ret = dep::aclnnInplaceCopyGetWorkspaceSize(
              output_window, acl_tensor_ptr_vector[2], &workspaceSize,
              &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnInplaceCopyGetWorkspaceSize for scatter window failed: "
                "%d",
                ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(output_window);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg =
                "aclnnInplaceCopy for scatter window: nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(output_window);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnInplaceCopy);
          retire_acl_tensor(output_window);
          break;
        }

        if (thunk_info().scatter_lowering_kind ==
            NpuThunk::ThunkInfo::ScatterLoweringKind::kSingleSliceIndexCopy) {
          if (!CanUseSingleSliceIndexCopy(
                  param_shapes, thunk_info().scatter_dimension)) {
            const std::string err_msg =
                "Single-slice IndexCopy scatter has incompatible shapes";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          const std::vector<int64_t> source_shape =
              BuildSingleSliceSourceShape(param_shapes[0],
                                          thunk_info().scatter_dimension);
          auto update_addr = allocations->GetDeviceAddress(args_[2]);
          aclDataType update_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  param_shapes[2].element_type());
          aclTensor* source_view = dep::aclCreateTensor(
              source_shape.data(), source_shape.size(), update_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              source_shape.data(), source_shape.size(), update_addr.opaque());
          if (source_view == nullptr) {
            const std::string err_msg =
                "Single-slice IndexCopy scatter failed to create source view";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          ret = dep::aclnnIndexCopyGetWorkspaceSize(
              acl_tensor_ptr_vector[0], thunk_info().scatter_dimension,
              acl_tensor_ptr_vector[1], source_view,
              acl_tensor_ptr_vector[3], &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnIndexCopyGetWorkspaceSize failed: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(source_view);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg = "aclnnIndexCopy: nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(source_view);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnIndexCopy);
          retire_acl_tensor(source_view);
          break;
        }

        if (thunk_info().scatter_lowering_kind ==
            NpuThunk::ThunkInfo::ScatterLoweringKind::
                kSameRankSingleDimScatter) {
          if (!CanUseSameRankSingleDimScatter(
                  param_shapes, thunk_info().scatter_dimension)) {
            const std::string err_msg =
                "Same-rank single-dim scatter has incompatible shapes";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          const std::vector<int64_t> data_shape =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);
          std::vector<int64_t> index_view_shape = data_shape;
          index_view_shape[thunk_info().scatter_dimension] =
              param_shapes[1].dimensions(0);

          auto index_addr = allocations->GetDeviceAddress(args_[1]);
          const aclDataType index_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  param_shapes[1].element_type());

          std::vector<int64_t> index_strides(index_view_shape.size(), 0);
          index_strides[thunk_info().scatter_dimension] = 1;

          const std::vector<int64_t> raw_index_shape =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[1]);
          aclTensor* index_view = dep::aclCreateTensor(
              index_view_shape.data(), index_view_shape.size(), index_dtype,
              index_strides.data(), 0, ACL_FORMAT_ND,
              raw_index_shape.data(), raw_index_shape.size(),
              index_addr.opaque());
          if (index_view == nullptr) {
            const std::string err_msg =
                "Same-rank single-dim scatter failed to create index view";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          ret = dep::aclnnScatterGetWorkspaceSize(
              acl_tensor_ptr_vector[0], thunk_info().scatter_dimension,
              index_view, acl_tensor_ptr_vector[2], /*reduce=*/0,
              acl_tensor_ptr_vector[3], &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnScatterGetWorkspaceSize failed: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg = "aclnnScatter: nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnScatter);
          retire_acl_tensor(index_view);
          break;
        }

        if (thunk_info().scatter_lowering_kind ==
            NpuThunk::ThunkInfo::ScatterLoweringKind::kBatchedSingleDimScatter) {
          if (!CanUseBatchedSingleDimScatter(
                  param_shapes, thunk_info().scatter_dimension,
                  thunk_info().scatter_batch_dimension,
                  thunk_info().scatter_indices_batch_dimension)) {
            const std::string err_msg =
                "Batched single-dim scatter has incompatible shapes";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          const std::vector<int64_t> data_shape_vec =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);
          const std::vector<int64_t> index_shape_vec =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[1]);

          // Calculate the number of scatter points: the product of non-batch and non-index_vector dimensions in index
          const int64_t index_vector_dim = index_shape_vec.size() - 1;
          int64_t n_scatter = 1;
          int64_t n_dim_in_index = -1;
          for (int64_t i = 0;
               i < static_cast<int64_t>(index_shape_vec.size()); ++i) {
            if (i != thunk_info().scatter_indices_batch_dimension &&
                i != index_vector_dim) {
              n_scatter *= index_shape_vec[i];
              n_dim_in_index = i;
            }
          }

          // view shape: same rank as data, scatter_dim replaced by scatter points
          std::vector<int64_t> view_shape = data_shape_vec;
          view_shape[thunk_info().scatter_dimension] = n_scatter;

          // Compute index raw row-major strides
          std::vector<int64_t> raw_index_strides(index_shape_vec.size(), 1);
          for (int64_t i = static_cast<int64_t>(index_shape_vec.size()) - 2;
               i >= 0; --i) {
            raw_index_strides[i] =
                raw_index_strides[i + 1] * index_shape_vec[i + 1];
          }

          // index view strides: batch and scatter_points dimensions map original strides,
          // Other dimensions stride=0 (broadcast)
          std::vector<int64_t> index_strides(view_shape.size(), 0);
          index_strides[thunk_info().scatter_batch_dimension] =
              raw_index_strides[thunk_info().scatter_indices_batch_dimension];
          if (n_dim_in_index >= 0) {
            index_strides[thunk_info().scatter_dimension] =
                raw_index_strides[n_dim_in_index];
          }

          auto index_addr = allocations->GetDeviceAddress(args_[1]);
          const aclDataType index_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  param_shapes[1].element_type());

          aclTensor* index_view = dep::aclCreateTensor(
              view_shape.data(), view_shape.size(), index_dtype,
              index_strides.data(), 0, ACL_FORMAT_ND,
              index_shape_vec.data(), index_shape_vec.size(),
              index_addr.opaque());

          // update tensor: rank may be smaller than data (scatter dim removed single-point scatter)
          // At this time, you need to create a view to complete scatter_dim (size=1) to make rank consistent with data.
          if (index_view == nullptr) {
            const std::string err_msg =
                "Batched single-dim scatter failed to create index view";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          aclTensor* update_tensor = acl_tensor_ptr_vector[2];
          aclTensor* update_view = nullptr;
          bool need_cleanup_update = false;
          const auto& update_shape_ref = param_shapes[2];
          if (update_shape_ref.dimensions_size() + 1 ==
              static_cast<int64_t>(data_shape_vec.size()) && n_scatter == 1) {
            // update.rank == data.rank - 1: insert size-1 dimension at scatter_dim position
            std::vector<int64_t> up_view_shape = data_shape_vec;
            up_view_shape[thunk_info().scatter_dimension] = 1;
            std::vector<int64_t> up_storage_shape(
                update_shape_ref.dimensions().begin(),
                update_shape_ref.dimensions().end());
            std::vector<int64_t> up_view_strides(up_view_shape.size());
            {
              int64_t s = 1;
              for (int i = up_view_shape.size() - 1; i >= 0; --i) {
                up_view_strides[i] = s;
                s *= up_view_shape[i];
              }
            }
            auto update_addr = allocations->GetDeviceAddress(args_[2]);
            aclDataType update_dtype =
                NpuTool::ConvertXlaType2AclDataType(update_shape_ref.element_type());
            update_view = dep::aclCreateTensor(
                up_view_shape.data(), up_view_shape.size(), update_dtype,
                up_view_strides.data(), 0, ACL_FORMAT_ND,
                up_storage_shape.data(), up_storage_shape.size(),
                update_addr.opaque());
            if (update_view == nullptr) {
              retire_acl_tensor(index_view);
              return absl::InternalError(
                  "Batched single-dim scatter failed to create update view");
            }
            update_tensor = update_view;
            need_cleanup_update = true;
          }

          ret = dep::aclnnScatterGetWorkspaceSize(
              acl_tensor_ptr_vector[0], thunk_info().scatter_dimension,
              index_view, update_tensor, /*reduce=*/0,
              acl_tensor_ptr_vector[3], &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnScatterGetWorkspaceSize failed: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            if (need_cleanup_update && update_view) retire_acl_tensor(update_view);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg = "aclnnScatter: nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            if (need_cleanup_update && update_view) retire_acl_tensor(update_view);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnScatter);
          retire_acl_tensor(index_view);
          if (need_cleanup_update && update_view) retire_acl_tensor(update_view);
          break;
        }

        // kMultiDimScatter: linearize multi-dim indices to 1D, then
        // copy data→output and use aclnnScatterNd on a flat 1D view.
        // Supports optional batch dims from vmap.
        if (thunk_info().scatter_lowering_kind ==
            NpuThunk::ThunkInfo::ScatterLoweringKind::kMultiDimScatter) {
          // Order the base copy and index transfers on the execution stream.
          // Host vectors must stay pageable: CANN completes these transfers
          // before CPU linearization / vector destruction without a stream drain.
          const auto& data_xla_shape = param_shapes[0];
          const auto& index_xla_shape = param_shapes[1];
          const auto& update_xla_shape = param_shapes[2];
          const int64_t total_elements = ShapeUtil::ElementsIn(data_xla_shape);
          const aclDataType data_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  data_xla_shape.element_type());
          const aclDataType index_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  index_xla_shape.element_type());
          const aclDataType update_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  update_xla_shape.element_type());
          const size_t elem_bytes =
              ShapeUtil::ByteSizeOfPrimitiveType(data_xla_shape.element_type());
          const size_t index_elem_bytes =
              ShapeUtil::ByteSizeOfPrimitiveType(index_xla_shape.element_type());

          // Step 1: copy data → output
          auto data_addr = allocations->GetDeviceAddress(args_[0]);
          auto output_addr = allocations->GetDeviceAddress(args_[3]);
          const int64_t data_bytes = total_elements * elem_bytes;
          if (data_bytes > 0) {
            aclError copy_ret =
                dep::aclrtMemcpyAsync(output_addr.opaque(), data_bytes,
                                 data_addr.opaque(), data_bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
            if (copy_ret != ACL_SUCCESS) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter base copy failed: %d", copy_ret);
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
          }

          // Step 2: read indices from device and linearize on host
          auto index_addr = allocations->GetDeviceAddress(args_[1]);
          const int64_t index_total = ShapeUtil::ElementsIn(index_xla_shape);
          const int64_t k = thunk_info().scatter_dims_to_operand.size();
          int64_t num_points = index_total / k;

          std::vector<int32_t> host_indices(index_total);
          if (index_total > 0) {
            int64_t idx_bytes = index_total * index_elem_bytes;
            aclError idx_ret = dep::aclrtMemcpyAsync(
                host_indices.data(), idx_bytes,
                index_addr.opaque(), idx_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
            if (idx_ret != ACL_SUCCESS) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter: failed to read indices: %d", idx_ret);
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
          }

          // Compute row-major strides for the operand dimensions
          const auto& operand_dims = thunk_info().scatter_operand_dims;
          const int64_t operand_rank = operand_dims.size();
          std::vector<int64_t> row_major_strides(operand_rank, 1);
          for (int64_t i = operand_rank - 2; i >= 0; --i) {
            row_major_strides[i] =
                row_major_strides[i + 1] * operand_dims[i + 1];
          }

          // Determine batch parameters
          const int64_t operand_batch_dim =
              thunk_info().scatter_operand_batch_dim;
          const int64_t indices_batch_dim =
              thunk_info().scatter_indices_batch_dim;
          const bool has_batch = operand_batch_dim >= 0 && indices_batch_dim >= 0;
          const int64_t batch_size = has_batch
              ? operand_dims[operand_batch_dim]
              : 1;
          // Elements per batch slice (excluding batch dim)
          int64_t per_batch_elements = 1;
          for (int64_t i = 0; i < operand_rank; ++i) {
            if (!has_batch || i != operand_batch_dim) {
              per_batch_elements *= operand_dims[i];
            }
          }

          // Linearize each scatter point
          // When batched, indices layout is [batch, num_points_per_batch, k]
          // or [batch, k] depending on index_vector_dim. We treat batch as
          // another dimension in the linearized flat tensor.
          std::vector<int32_t> linear_indices(num_points);
          const int64_t batch_stride = has_batch ? row_major_strides[operand_batch_dim] : 0;

          for (int64_t p = 0; p < num_points; ++p) {
            int64_t linear = 0;
            // Add batch offset if batched: each batch slice maps to
            // consecutive regions in the flat tensor
            if (has_batch) {
              const int64_t pts_per_batch = num_points / batch_size;
              const int64_t batch_idx = p / pts_per_batch;
              linear += batch_idx * per_batch_elements;
            }
            for (int64_t d = 0; d < k; ++d) {
              int64_t operand_dim =
                  thunk_info().scatter_dims_to_operand[d];
              linear += static_cast<int64_t>(host_indices[p * k + d]) *
                        row_major_strides[operand_dim];
            }
            linear_indices[p] = static_cast<int32_t>(linear);
          }

          const int64_t update_total = ShapeUtil::ElementsIn(update_xla_shape);
          if (update_total > num_points) {
            if (num_points <= 0 || update_total % num_points != 0) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter: update element count %lld is not a "
                  "multiple of scatter point count %lld",
                  static_cast<long long>(update_total),
                  static_cast<long long>(num_points));
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }

            std::vector<int64_t> window_operand_dims;
            window_operand_dims.reserve(operand_rank);
            auto is_inserted_window_dim = [&](int64_t dim) {
              return absl::c_find(thunk_info().scatter_inserted_window_dims,
                                  dim) !=
                     thunk_info().scatter_inserted_window_dims.end();
            };
            for (int64_t dim = 0; dim < operand_rank; ++dim) {
              if (!is_inserted_window_dim(dim)) {
                window_operand_dims.push_back(dim);
              }
            }
            if (window_operand_dims.size() !=
                thunk_info().scatter_update_window_dims.size()) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter: update_window_dims size %zu does not "
                  "match window operand dims size %zu",
                  thunk_info().scatter_update_window_dims.size(),
                  window_operand_dims.size());
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }

            const std::vector<int64_t> update_dims =
                NpuTool::ConvertXlaShape2VectorShape(update_xla_shape);
            const std::vector<int64_t> update_strides =
                BuildRowMajorStrides(update_dims);
            auto is_update_window_dim = [&](int64_t dim) {
              return absl::c_find(thunk_info().scatter_update_window_dims,
                                  dim) !=
                     thunk_info().scatter_update_window_dims.end();
            };

            std::vector<int32_t> expanded_linear_indices(update_total);
            std::vector<int64_t> update_coords(update_dims.size(), 0);
            for (int64_t update_linear = 0; update_linear < update_total;
                 ++update_linear) {
              for (int64_t dim = 0;
                   dim < static_cast<int64_t>(update_dims.size()); ++dim) {
                update_coords[dim] =
                    (update_linear / update_strides[dim]) % update_dims[dim];
              }

              int64_t point_index = 0;
              for (int64_t dim = 0;
                   dim < static_cast<int64_t>(update_dims.size()); ++dim) {
                if (is_update_window_dim(dim)) {
                  continue;
                }
                point_index = point_index * update_dims[dim] + update_coords[dim];
              }
              if (point_index < 0 || point_index >= num_points) {
                const std::string err_msg = absl::StrFormat(
                    "MultiDim scatter: computed point index %lld out of "
                    "range [0, %lld)",
                    static_cast<long long>(point_index),
                    static_cast<long long>(num_points));
                NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
                return absl::InternalError(err_msg);
              }

              int64_t output_linear = linear_indices[point_index];
              for (int64_t i = 0;
                   i < static_cast<int64_t>(
                           thunk_info().scatter_update_window_dims.size());
                   ++i) {
                const int64_t update_dim =
                    thunk_info().scatter_update_window_dims[i];
                output_linear +=
                    update_coords[update_dim] *
                    row_major_strides[window_operand_dims[i]];
              }
              expanded_linear_indices[update_linear] =
                  static_cast<int32_t>(output_linear);
            }

            linear_indices = std::move(expanded_linear_indices);
            num_points = update_total;
          }

          // Step 3: copy linear indices to device
          int64_t linear_idx_bytes = num_points * sizeof(int32_t);
          TF_ASSIGN_OR_RETURN(
              void* device_linear_addr,
              allocate_temporary_device_memory(linear_idx_bytes));
          aclError write_ret = dep::aclrtMemcpyAsync(
              device_linear_addr, linear_idx_bytes,
              linear_indices.data(), linear_idx_bytes,
              ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
          if (write_ret != ACL_SUCCESS) {
            RetireTemporaryDeviceMemory(device_linear_addr);
            const std::string err_msg = absl::StrFormat(
                "MultiDim scatter: failed to write linear indices: %d",
                write_ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          // Step 4: create flat views
          std::vector<int64_t> flat_shape = {total_elements};
          aclTensor* flat_output = dep::aclCreateTensor(
              flat_shape.data(), 1, data_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_shape.data(), 1, output_addr.opaque());
          std::vector<int64_t> flat_index_shape = {num_points, 1};
          aclTensor* flat_indices = dep::aclCreateTensor(
              flat_index_shape.data(), 2, index_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_index_shape.data(), 2, device_linear_addr);

          // For updates: flatten to [num_points] (or [num_points, 1])
          auto update_addr = allocations->GetDeviceAddress(args_[2]);
          std::vector<int64_t> flat_update_shape = {num_points};
          aclTensor* flat_updates = dep::aclCreateTensor(
              flat_update_shape.data(),
              static_cast<int64_t>(flat_update_shape.size()),
              update_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_update_shape.data(),
              static_cast<int64_t>(flat_update_shape.size()),
              update_addr.opaque());

          if (flat_output == nullptr || flat_indices == nullptr ||
              flat_updates == nullptr) {
            if (flat_output) retire_acl_tensor(flat_output);
            if (flat_indices) retire_acl_tensor(flat_indices);
            if (flat_updates) retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            const std::string err_msg =
                "MultiDim scatter: failed to create flat tensor views";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          auto DTypeToString = [](aclDataType dtype) -> std::string {
  switch (dtype) {
    case ACL_FLOAT:
      return "ACL_FLOAT";
    case ACL_FLOAT16:
      return "ACL_FLOAT16";
    case ACL_BOOL:
      return "ACL_BOOL";
    case ACL_INT32:
      return "ACL_INT32";
    case ACL_INT64:
      return "ACL_INT64";
    default:
      return absl::StrFormat("UNKNOWN(%d)", static_cast<int>(dtype));
  }
};

auto DimsToString = [](const std::vector<int64_t>& dims) -> std::string {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) s += ",";
    s += std::to_string(dims[i]);
  }
  s += "]";
  return s;
};

auto NumElements = [](const std::vector<int64_t>& dims) -> int64_t {
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
};

auto IsScatterNdDataDtypeSupported = [](aclDataType dtype) -> bool {
  return dtype == ACL_FLOAT ||
         dtype == ACL_FLOAT16 ||
         dtype == ACL_BOOL;
  // 如果当前平台确认支持 BF16，可以在这里补 ACL_BF16。
};

if (!IsScatterNdDataDtypeSupported(data_dtype) ||
    !IsScatterNdDataDtypeSupported(update_dtype)) {
  // aclnnScatterNd only supports FLOAT/FLOAT16/BOOL for data/updates/out
  // (INT32/INT64 are allowed only for indices). Integer multi-dim scatters
  // arise from linalg.solve's LU pivoting: BatchedGatherScatterNormalizer
  // turns the batched single-dim pivot/permutation scatter into a true
  // multi-dim scatter whose data is s32. Since no aclnn primitive supports
  // integer multi-dim scatter, fall back to a host-assisted scatter for the
  // kReplace reduction:
  //     out[linear_indices[p]] = updates[p]   (last-writer wins)
  // `output` already holds a copy of `data` (Step 1 above), so we only
  // overwrite the touched elements. Indices are already read D2H and
  // linearized, consistent with the host-assisted design of this path.
  NPU_LOG_FORMAT_PRINT(INFO,
      "MultiDim scatter: data/update dtype %s not supported by "
      "aclnnScatterNd; using host-assisted scatter fallback "
      "(num_points=%lld, elem_bytes=%zu, flat_shape=%s).",
      DTypeToString(data_dtype).c_str(),
      static_cast<long long>(num_points), elem_bytes,
      DimsToString(flat_shape).c_str());

  // Read the per-point updates (num_points scalars, update_window_dims is
  // empty for this path) from device to host.
  const int64_t elem_bytes_i64 = static_cast<int64_t>(elem_bytes);
  const int64_t update_bytes = num_points * elem_bytes_i64;
  std::vector<uint8_t> host_updates(update_bytes, 0);
  if (update_bytes > 0) {
    aclError up_ret = dep::aclrtMemcpyAsync(
        host_updates.data(), update_bytes,
        update_addr.opaque(), update_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
    if (up_ret != ACL_SUCCESS) {
      const std::string err_msg = absl::StrFormat(
          "MultiDim scatter host fallback: failed to read updates: %d",
          up_ret);
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      retire_acl_tensor(flat_output);
      retire_acl_tensor(flat_indices);
      retire_acl_tensor(flat_updates);
      RetireTemporaryDeviceMemory(device_linear_addr);
      return absl::InternalError(err_msg);
    }
  }

  // Overwrite each touched output element with its update value.
  uint8_t* output_base = static_cast<uint8_t*>(output_addr.opaque());
  for (int64_t p = 0; p < num_points; ++p) {
    const int64_t off =
        static_cast<int64_t>(linear_indices[p]) * elem_bytes_i64;
    aclError w_ret = dep::aclrtMemcpyAsync(
        output_base + off, elem_bytes,
        host_updates.data() + p * elem_bytes_i64, elem_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
    if (w_ret != ACL_SUCCESS) {
      const std::string err_msg = absl::StrFormat(
          "MultiDim scatter host fallback: failed to write element %lld "
          "(linear_idx=%d): %d",
          static_cast<long long>(p), linear_indices[p], w_ret);
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      retire_acl_tensor(flat_output);
      retire_acl_tensor(flat_indices);
      retire_acl_tensor(flat_updates);
      RetireTemporaryDeviceMemory(device_linear_addr);
      return absl::InternalError(err_msg);
    }
  }

  retire_acl_tensor(flat_output);
  retire_acl_tensor(flat_indices);
  retire_acl_tensor(flat_updates);
  RetireTemporaryDeviceMemory(device_linear_addr);
  break;
}

          ret = dep::aclnnScatterNdGetWorkspaceSize(
              flat_output, flat_indices, flat_updates, flat_output,
              &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "MultiDim scatter (flat) aclnnScatterNd failed: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(flat_output);
            retire_acl_tensor(flat_indices);
            retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg =
                "MultiDim scatter (flat): nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(flat_output);
            retire_acl_tensor(flat_indices);
            retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnScatterNd);
          retire_acl_tensor(flat_output);
          retire_acl_tensor(flat_indices);
          retire_acl_tensor(flat_updates);
          RetireTemporaryDeviceMemory(device_linear_addr);
          break;
        }

        ret = dep::aclnnScatterNdGetWorkspaceSize(
            acl_tensor_ptr_vector[0], acl_tensor_ptr_vector[1],
            acl_tensor_ptr_vector[2], acl_tensor_ptr_vector[3],
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
          const std::string err_msg = absl::StrFormat(
              "aclnnScatterNdGetWorkspaceSize failed: %d", ret);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        if (executor == nullptr) {
          const std::string err_msg = "aclnnScatterNd: nullptr executor";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnScatterNd);

      } else if (scatter_kind ==
                 NpuThunk::ThunkInfo::ScatterReductionKind::kAdd) {
        // All add scatters are normalized by NpuScatterWindowExpander.
        // Refuse an unnormalized graph instead of guessing from ranks and
        // falling back to row-only TfScatterAdd with incompatible semantics.
        if (param_shapes.size() != 4 ||
            param_shapes[0].dimensions_size() != 1 ||
            param_shapes[1].dimensions_size() != 2 ||
            param_shapes[1].dimensions(1) != 1 ||
            param_shapes[1].element_type() != S32 ||
            !CanUseSameRankSingleDimScatter(param_shapes, 0) ||
            thunk_info().scatter_dimension != 0 ||
            !ShapeUtil::Equal(param_shapes[0], param_shapes[3]) ||
            !thunk_info().scatter_add_normalized) {
          return absl::UnimplementedError(
              "NPU scatter-add must be normalized to bounded rank-1 points");
        }
        auto index_addr = allocations->GetDeviceAddress(args_[1]);
        const int64_t count = param_shapes[1].dimensions(0);
        const int64_t stride = 1;
        aclTensor* index_view = dep::aclCreateTensor(
            &count, 1, ACL_INT32, &stride, 0, ACL_FORMAT_ND,
            &count, 1, index_addr.opaque());
        if (index_view == nullptr) {
          return absl::InternalError("Scatter-add: failed to create flat index view");
        }
        ret = dep::aclnnScatterGetWorkspaceSize(
            acl_tensor_ptr_vector[0], /*dim=*/0, index_view,
            acl_tensor_ptr_vector[2], /*reduce=*/1, acl_tensor_ptr_vector[3],
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS || executor == nullptr) {
          retire_acl_tensor(index_view);
          return absl::InternalError(absl::StrFormat(
              "Normalized scatter-add workspace/executor failure: %d", ret));
        }
        exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                dep::aclnnScatter);
        retire_acl_tensor(index_view);

      } else if (scatter_kind ==
                 NpuThunk::ThunkInfo::ScatterReductionKind::kMul) {
        // scatter-mul: aclnnScatter(reduce=2) implements element-wise multiplication update
        // Symmetrical to the scatter-add path, only the reduce parameter differs
        const int64_t sd = thunk_info().scatter_dimension;
        bool use_dim_scatter =
            sd >= 0 &&
            CanUseSameRankSingleDimScatter(param_shapes, sd);

        if (use_dim_scatter) {
          const std::vector<int64_t> data_shape =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[0]);
          std::vector<int64_t> index_view_shape = data_shape;
          index_view_shape[sd] = param_shapes[1].dimensions(0);

          auto index_addr = allocations->GetDeviceAddress(args_[1]);
          const aclDataType index_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  param_shapes[1].element_type());

          std::vector<int64_t> index_strides(index_view_shape.size(), 0);
          index_strides[sd] = 1;
          const std::vector<int64_t> raw_index_shape_mul =
              NpuTool::ConvertXlaShape2VectorShape(param_shapes[1]);
          aclTensor* index_view = dep::aclCreateTensor(
              index_view_shape.data(), index_view_shape.size(), index_dtype,
              index_strides.data(), 0, ACL_FORMAT_ND,
              raw_index_shape_mul.data(), raw_index_shape_mul.size(),
              index_addr.opaque());
          if (index_view == nullptr) {
            const std::string err_msg =
                "Scatter-mul (aclnnScatter): failed to create index view";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          ret = dep::aclnnScatterGetWorkspaceSize(
              acl_tensor_ptr_vector[0], sd,
              index_view, acl_tensor_ptr_vector[2], /*reduce=*/2,
              acl_tensor_ptr_vector[3], &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "aclnnScatterGetWorkspaceSize (scatter-mul) failed: %d", ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg =
                "aclnnScatter (scatter-mul): nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(index_view);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnScatter);
          retire_acl_tensor(index_view);

        } else if (thunk_info().scatter_lowering_kind ==
                   NpuThunk::ThunkInfo::ScatterLoweringKind::
                       kMultiDimScatter) {
          // As in replace, pageable host vectors make same-stream transfers
          // complete before CPU consumption, without a separate stream drain.
          // Multidimensional scatter-mul: symmetrical with scatter-add multidimensional path, reduce=2
          const auto& data_xla_shape = param_shapes[0];
          const auto& index_xla_shape = param_shapes[1];
          const auto& update_xla_shape = param_shapes[2];
          const int64_t total_elements =
              ShapeUtil::ElementsIn(data_xla_shape);
          const aclDataType data_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  data_xla_shape.element_type());
          const aclDataType index_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  index_xla_shape.element_type());
          const aclDataType update_dtype =
              NpuTool::ConvertXlaType2AclDataType(
                  update_xla_shape.element_type());
          const size_t elem_bytes =
              ShapeUtil::ByteSizeOfPrimitiveType(data_xla_shape.element_type());
          const size_t index_elem_bytes =
              ShapeUtil::ByteSizeOfPrimitiveType(index_xla_shape.element_type());

          auto data_addr = allocations->GetDeviceAddress(args_[0]);
          auto output_addr = allocations->GetDeviceAddress(args_[3]);
          const int64_t data_bytes = total_elements * elem_bytes;
          if (data_bytes > 0) {
            aclError copy_ret =
                dep::aclrtMemcpyAsync(output_addr.opaque(), data_bytes,
                                 data_addr.opaque(), data_bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
            if (copy_ret != ACL_SUCCESS) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter-mul base copy failed: %d", copy_ret);
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
          }

          auto index_addr = allocations->GetDeviceAddress(args_[1]);
          const int64_t index_total =
              ShapeUtil::ElementsIn(index_xla_shape);
          const int64_t k = thunk_info().scatter_dims_to_operand.size();
          int64_t num_points = index_total / k;

          std::vector<int32_t> host_indices(index_total);
          if (index_total > 0) {
            int64_t idx_bytes = index_total * index_elem_bytes;
            aclError idx_ret = dep::aclrtMemcpyAsync(
                host_indices.data(), idx_bytes,
                index_addr.opaque(), idx_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
            if (idx_ret != ACL_SUCCESS) {
              const std::string err_msg = absl::StrFormat(
                  "MultiDim scatter-mul: failed to read indices: %d",
                  idx_ret);
              NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
              return absl::InternalError(err_msg);
            }
          }

          const auto& operand_dims = thunk_info().scatter_operand_dims;
          const int64_t operand_rank = operand_dims.size();
          std::vector<int64_t> row_major_strides(operand_rank, 1);
          for (int64_t i = operand_rank - 2; i >= 0; --i) {
            row_major_strides[i] =
                row_major_strides[i + 1] * operand_dims[i + 1];
          }

          const int64_t operand_batch_dim =
              thunk_info().scatter_operand_batch_dim;
          const int64_t indices_batch_dim =
              thunk_info().scatter_indices_batch_dim;
          const bool has_batch =
              operand_batch_dim >= 0 && indices_batch_dim >= 0;
          const int64_t batch_size =
              has_batch ? operand_dims[operand_batch_dim] : 1;
          int64_t per_batch_elements = 1;
          for (int64_t i = 0; i < operand_rank; ++i) {
            if (!has_batch || i != operand_batch_dim) {
              per_batch_elements *= operand_dims[i];
            }
          }

          std::vector<int32_t> linear_indices(num_points);
          for (int64_t p = 0; p < num_points; ++p) {
            int64_t linear = 0;
            if (has_batch) {
              const int64_t pts_per_batch = num_points / batch_size;
              const int64_t batch_idx = p / pts_per_batch;
              linear += batch_idx * per_batch_elements;
            }
            for (int64_t d = 0; d < k; ++d) {
              int64_t operand_dim =
                  thunk_info().scatter_dims_to_operand[d];
              linear += static_cast<int64_t>(host_indices[p * k + d]) *
                        row_major_strides[operand_dim];
            }
            linear_indices[p] = static_cast<int32_t>(linear);
          }

          int64_t linear_idx_bytes = num_points * sizeof(int32_t);
          TF_ASSIGN_OR_RETURN(
              void* device_linear_addr,
              allocate_temporary_device_memory(linear_idx_bytes));
          aclError write_ret = dep::aclrtMemcpyAsync(
              device_linear_addr, linear_idx_bytes,
              linear_indices.data(), linear_idx_bytes,
              ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
          if (write_ret != ACL_SUCCESS) {
            RetireTemporaryDeviceMemory(device_linear_addr);
            const std::string err_msg = absl::StrFormat(
                "MultiDim scatter-mul: failed to write linear indices: %d",
                write_ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          std::vector<int64_t> flat_shape = {total_elements};
          std::vector<int64_t> flat_idx_shape = {num_points};
          std::vector<int64_t> flat_up_shape = {num_points};

          aclTensor* flat_output = dep::aclCreateTensor(
              flat_shape.data(), 1, data_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_shape.data(), 1, output_addr.opaque());
          aclTensor* flat_indices = dep::aclCreateTensor(
              flat_idx_shape.data(), 1, index_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_idx_shape.data(), 1, device_linear_addr);
          auto update_addr = allocations->GetDeviceAddress(args_[2]);
          aclTensor* flat_updates = dep::aclCreateTensor(
              flat_up_shape.data(), 1, update_dtype,
              nullptr, 0, ACL_FORMAT_ND,
              flat_up_shape.data(), 1, update_addr.opaque());

          if (flat_output == nullptr || flat_indices == nullptr ||
              flat_updates == nullptr) {
            if (flat_output) retire_acl_tensor(flat_output);
            if (flat_indices) retire_acl_tensor(flat_indices);
            if (flat_updates) retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            const std::string err_msg =
                "MultiDim scatter-mul: failed to create flat tensor views";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }

          ret = dep::aclnnScatterGetWorkspaceSize(
              flat_output, /*dim=*/0,
              flat_indices, flat_updates, /*reduce=*/2,
              flat_output, &workspaceSize, &executor);
          if (ret != ACL_SUCCESS) {
            const std::string err_msg = absl::StrFormat(
                "MultiDim scatter-mul aclnnScatter(reduce=2) failed: %d",
                ret);
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(flat_output);
            retire_acl_tensor(flat_indices);
            retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            return absl::InternalError(err_msg);
          }
          if (executor == nullptr) {
            const std::string err_msg =
                "MultiDim scatter-mul: nullptr executor";
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            retire_acl_tensor(flat_output);
            retire_acl_tensor(flat_indices);
            retire_acl_tensor(flat_updates);
            RetireTemporaryDeviceMemory(device_linear_addr);
            return absl::InternalError(err_msg);
          }

          exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                                  dep::aclnnScatter);
          retire_acl_tensor(flat_output);
          retire_acl_tensor(flat_indices);
          retire_acl_tensor(flat_updates);
          RetireTemporaryDeviceMemory(device_linear_addr);

        } else {
          const std::string err_msg = absl::StrFormat(
              "Scatter-mul only supports same-rank single-dim or multi-dim "
              "lowering, got scatter_dimension=%ld", sd);
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

      } else {
        const std::string err_msg = absl::StrFormat(
            "Scatter reduction kind unsupported: %d",
            static_cast<int>(scatter_kind));
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      break;
    }

    case HloOpcode::kSort: {
      // Sort operation: call aclnnSort to return the sorted value and index
      // aclnnSort always outputs valuesOut + indicesOut (INT64)
      // - Single input mode (2 params): values_in -> values_out, temporary indices discarded
      // - Dual input INT64 mode (4 params): values_in + iota_in -> values_out + indices_out
      // - Dual input INT32 mode (4 params): Same as above, but requires INT64->INT32 Cast
      if (thunk_info().dimensions.empty()) {
        const std::string err_msg = "Sort dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }
      int64_t dim = thunk_info().dimensions[0];
      bool stable = thunk_info().sort_stable;
      bool descending = thunk_info().sort_descending;

      NPU_LOG_FORMAT_PRINT(DEBUG,
                           "Sort: dim=%ld, stable=%d, descending=%d, params=%d",
                           dim, stable, descending, param_shapes.size());

      const bool is_dual_input = (param_shapes.size() == 4);
      const auto& values_shape = param_shapes[0];
      int64_t num_elements = NpuTool::GetXlaShapeSize(values_shape);
      int64_t indices_size = num_elements * sizeof(int64_t);

      // Determine whether INT64->INT32 Cast is required in dual input mode
      bool needs_cast = false;
      if (is_dual_input) {
        if (param_shapes[3].element_type() != PrimitiveType::S64) {
          if (param_shapes[3].element_type() == PrimitiveType::S32) {
            needs_cast = true;
          } else {
            const std::string err_msg = absl::StrFormat(
                "Dual-input sort requires INT64 or INT32 indices output, got %s",
                PrimitiveType_Name(param_shapes[3].element_type()));
            NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
            return absl::InternalError(err_msg);
          }
        }
      }

      // Prepare aclTensor for valuesOut and indicesOut
      aclTensor* values_out_tensor = nullptr;
      aclTensor* indices_out_tensor = nullptr;
      void* temp_indices_buffer = nullptr;
      bool owns_indices_tensor = false;

      if (is_dual_input && !needs_cast) {
        // Dual input INT64: use output buffer directly
        values_out_tensor = acl_tensor_ptr_vector[2];
        indices_out_tensor = acl_tensor_ptr_vector[3];
      } else {
        // Single input/double input INT32: allocate temporary INT64 indices buffer
        TF_ASSIGN_OR_RETURN(
            void* indices_device_ptr,
            allocate_temporary_device_memory(indices_size));
        temp_indices_buffer = indices_device_ptr;
        owns_indices_tensor = true;

        std::vector<int64_t> input_dims =
            NpuTool::ConvertXlaShape2VectorShape(values_shape);
        indices_out_tensor = dep::aclCreateTensor(
            input_dims.data(), input_dims.size(), ACL_INT64,
            nullptr, 0, ACL_FORMAT_ND,
            input_dims.data(), input_dims.size(), indices_device_ptr);
        if (indices_out_tensor == nullptr) {
          RetireTemporaryDeviceMemory(indices_device_ptr);
          const std::string err_msg =
              "Failed to create indices tensor for sort";
          NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
          return absl::InternalError(err_msg);
        }

        if (is_dual_input) {
          // Double input INT32: valuesOut is written directly to output[0]
          values_out_tensor = acl_tensor_ptr_vector[2];
        } else {
          // Single input: valuesOut uses only output
          values_out_tensor = acl_tensor_ptr_vector.back();
        }
      }

      // Call aclnnSortGetWorkspaceSize
      ret = dep::aclnnSortGetWorkspaceSize(
          acl_tensor_ptr_vector[0], stable, dim, descending,
          values_out_tensor, indices_out_tensor,
          &workspaceSize, &executor);

      if (ret != ACL_SUCCESS) {
        if (owns_indices_tensor && indices_out_tensor != nullptr) {
          retire_acl_tensor(indices_out_tensor);
        }
        if (temp_indices_buffer != nullptr) {
          RetireTemporaryDeviceMemory(temp_indices_buffer);
        }
        const std::string err_msg = absl::StrFormat(
            "Call aclnnSortGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnSort);

      // Dual input INT32 mode: temporary INT64 indices need to be cast to INT32 to the output buffer
      if (exec_ret.ok() && needs_cast) {
        uint64_t cast_workspace_size = 0;
        aclOpExecutor* cast_executor = nullptr;

        ret = dep::aclnnCastGetWorkspaceSize(
            indices_out_tensor, ACL_INT32,
            acl_tensor_ptr_vector[3],
            &cast_workspace_size, &cast_executor);
        if (ret != ACL_SUCCESS) {
          exec_ret = absl::InternalError(absl::StrFormat(
              "aclnnCastGetWorkspaceSize (I64→I32) failed: %d", ret));
        } else {
          exec_ret = ExecuteAclnn(acl_stream, cast_workspace_size,
                                  cast_executor, dep::aclnnCast);
          if (!exec_ret.ok()) {
            NPU_LOG_FORMAT_PRINT(ERROR, "aclnnCast (I64→I32) failed");
          }
        }
      }

      // Clean up temporary resources
      if (owns_indices_tensor) {
        if (indices_out_tensor != nullptr) {
          retire_acl_tensor(indices_out_tensor);
        }
        if (temp_indices_buffer != nullptr) {
          RetireTemporaryDeviceMemory(temp_indices_buffer);
        }
      }
      break;
    }

    case HloOpcode::kTopK: {
      if (thunk_info().topk_k < 0) {
        return absl::InternalError("TopK: invalid k metadata");
      }

      const Shape& input_shape = param_shapes[0];
      const Shape& value_out_shape = param_shapes[1];
      const Shape& index_out_shape = param_shapes[2];
      if (ShapeUtil::ElementsIn(value_out_shape) == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      const int64_t dim = thunk_info().topk_dimension >= 0
                              ? thunk_info().topk_dimension
                              : input_shape.dimensions_size() - 1;
      const bool largest = thunk_info().topk_largest;
      const bool sorted = true;

      aclTensor* values_out_tensor = acl_tensor_ptr_vector[1];
      aclTensor* indices_out_tensor = nullptr;
      void* temp_indices_buffer = nullptr;
      bool owns_indices_tensor = false;

      bool needs_cast = false;
      if (index_out_shape.element_type() == PrimitiveType::S64) {
        indices_out_tensor = acl_tensor_ptr_vector[2];
      } else if (index_out_shape.element_type() == PrimitiveType::S32) {
        needs_cast = true;
      } else {
        return absl::InternalError(absl::StrFormat(
            "TopK requires S32 or S64 indices output, got %s",
            PrimitiveType_Name(index_out_shape.element_type())));
      }

      if (needs_cast) {
        const int64_t indices_size =
            ShapeUtil::ElementsIn(index_out_shape) * sizeof(int64_t);
        TF_ASSIGN_OR_RETURN(
            void* indices_device_ptr,
            allocate_temporary_device_memory(indices_size));
        temp_indices_buffer = indices_device_ptr;
        owns_indices_tensor = true;

        std::vector<int64_t> index_dims =
            NpuTool::ConvertXlaShape2VectorShape(index_out_shape);
        indices_out_tensor = dep::aclCreateTensor(
            index_dims.data(), index_dims.size(), ACL_INT64,
            nullptr, 0, ACL_FORMAT_ND,
            index_dims.data(), index_dims.size(), indices_device_ptr);
        if (indices_out_tensor == nullptr) {
          RetireTemporaryDeviceMemory(indices_device_ptr);
          return absl::InternalError(
              "TopK: failed to create temporary INT64 indices tensor");
        }
      }

      ret = dep::aclnnTopkGetWorkspaceSize(
          acl_tensor_ptr_vector[0], thunk_info().topk_k, dim, largest, sorted,
          values_out_tensor, indices_out_tensor, &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        if (owns_indices_tensor && indices_out_tensor != nullptr) {
          retire_acl_tensor(indices_out_tensor);
        }
        if (temp_indices_buffer != nullptr) {
          RetireTemporaryDeviceMemory(temp_indices_buffer);
        }
        return absl::InternalError(absl::StrFormat(
            "aclnnTopkGetWorkspaceSize failed. Error code: %d", ret));
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnTopk);

      if (exec_ret.ok() && needs_cast) {
        uint64_t cast_workspace_size = 0;
        aclOpExecutor* cast_executor = nullptr;
        ret = dep::aclnnCastGetWorkspaceSize(indices_out_tensor, ACL_INT32,
                                             acl_tensor_ptr_vector[2],
                                             &cast_workspace_size,
                                             &cast_executor);
        if (ret != ACL_SUCCESS) {
          exec_ret = absl::InternalError(absl::StrFormat(
              "TopK: aclnnCastGetWorkspaceSize (I64->I32) failed: %d", ret));
        } else {
          exec_ret = ExecuteAclnn(acl_stream, cast_workspace_size,
                                  cast_executor, dep::aclnnCast);
        }
      }

      if (owns_indices_tensor) {
        if (indices_out_tensor != nullptr) {
          retire_acl_tensor(indices_out_tensor);
        }
        if (temp_indices_buffer != nullptr) {
          RetireTemporaryDeviceMemory(temp_indices_buffer);
        }
      }
      break;
    }

    case HloOpcode::kReverse: {
      // Reverse operation: docking aclnnFlip - flips tensor elements along a specified dimension
      // HLO reverse.dimensions are stored in thunk_info().dimensions
      const auto& reverse_dims = thunk_info().dimensions;
      if (reverse_dims.empty()) {
        const std::string err_msg = "Reverse: dimensions is empty";
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        return absl::InternalError(err_msg);
      }

      // Zero element tensors are skipped directly.
      if (ShapeUtil::ElementsIn(param_shapes[0]) == 0) {
        exec_ret = absl::OkStatus();
        break;
      }

      aclIntArray* dims_arr = dep::aclCreateIntArray(
          reverse_dims.data(), reverse_dims.size());
      if (dims_arr == nullptr) {
        return absl::InternalError("Reverse: failed to create aclIntArray");
      }

      // Input and output may share buffer (aliasing), aclnnFlip is out-of-place,
      // ACLNN can safely handle input==output internally
      ret = dep::aclnnFlipGetWorkspaceSize(
          acl_tensor_ptr_vector[0], dims_arr,
          acl_tensor_ptr_vector.back(), &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        const std::string err_msg = absl::StrFormat(
            "aclnnFlipGetWorkspaceSize failed. Error code: %d", ret);
        NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
        retire_acl_int_array(dims_arr);
        return absl::InternalError(err_msg);
      }

      exec_ret =
          ExecuteAclnn(acl_stream, workspaceSize, executor, dep::aclnnFlip);

      retire_acl_int_array(dims_arr);
      break;
    }

    case HloOpcode::kPad: {
      const Shape& input_shape = param_shapes[0];
      const Shape& pad_value_shape = param_shapes[1];
      const Shape& output_shape = param_shapes[2];
      if (!ShapeUtil::IsScalar(pad_value_shape)) {
        return absl::InternalError(
            "Pad currently requires a scalar padding value operand");
      }

      const int64_t rank = input_shape.dimensions_size();
      if (thunk_info().pad_edge_padding_low.size() != rank ||
          thunk_info().pad_edge_padding_high.size() != rank ||
          thunk_info().pad_interior_padding.size() != rank) {
        return absl::InternalError("Pad metadata rank mismatch");
      }

      for (int64_t i = 0; i < rank; ++i) {
        if (thunk_info().pad_edge_padding_low[i] < 0 ||
            thunk_info().pad_edge_padding_high[i] < 0) {
          return absl::UnimplementedError(absl::StrFormat(
              "Pad with negative edge padding is unsupported: dim=%ld, low=%ld, high=%ld",
              i, thunk_info().pad_edge_padding_low[i],
              thunk_info().pad_edge_padding_high[i]));
        }
      }

      TF_ASSIGN_OR_RETURN(
          aclScalar * pad_scalar,
          CreateScalarFromDeviceBuffer(
              allocations->GetDeviceAddress(args_[1]).opaque(),
              pad_value_shape.element_type()));
      auto destroy_pad_scalar = absl::MakeCleanup(
          [&] { retire_acl_scalar(pad_scalar); });

      // 1. Fill the whole output with pad_value.
      ret = dep::aclnnInplaceFillScalarGetWorkspaceSize(
          acl_tensor_ptr_vector[2], pad_scalar, &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrFormat(
            "aclnnInplaceFillScalarGetWorkspaceSize failed. Error code: %d",
            ret));
      }
      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnInplaceFillScalar);
      if (!exec_ret.ok()) {
        break;
      }

      // Zero-element operand: output is just the filled padding value tensor.
      if (ShapeUtil::ElementsIn(input_shape) == 0) {
        break;
      }

      // 2. Create a strided view over the output buffer so that copying the
      //    operand into it materializes both edge padding and interior padding:
      //    output_index = low + operand_index * (interior + 1)
      std::vector<int64_t> output_dims =
          NpuTool::ConvertXlaShape2VectorShape(output_shape);
      std::vector<int64_t> output_strides(output_dims.size(), 1);
      for (int64_t i = static_cast<int64_t>(output_dims.size()) - 2; i >= 0;
           --i) {
        output_strides[i] = output_dims[i + 1] * output_strides[i + 1];
      }

      std::vector<int64_t> input_dims =
          NpuTool::ConvertXlaShape2VectorShape(input_shape);
      std::vector<int64_t> padded_view_strides(input_dims.size(), 1);
      int64_t element_offset = 0;
      for (int64_t i = 0; i < rank; ++i) {
        padded_view_strides[i] =
            output_strides[i] * (thunk_info().pad_interior_padding[i] + 1);
        element_offset +=
            thunk_info().pad_edge_padding_low[i] * output_strides[i];
      }

      aclDataType dtype =
          NpuTool::ConvertXlaType2AclDataType(input_shape.element_type());
      auto output_addr = allocations->GetDeviceAddress(args_[2]);
      aclTensor* output_window = dep::aclCreateTensor(
          input_dims.data(), input_dims.size(), dtype,
          padded_view_strides.data(), element_offset, ACL_FORMAT_ND,
          output_dims.data(), output_dims.size(), output_addr.opaque());
      if (output_window == nullptr) {
        return absl::InternalError("Pad: failed to create strided output view");
      }
      auto destroy_output_window = absl::MakeCleanup(
          [&] { retire_acl_tensor(output_window); });

      ret = dep::aclnnInplaceCopyGetWorkspaceSize(
          output_window, acl_tensor_ptr_vector[0], &workspaceSize, &executor);
      if (ret != ACL_SUCCESS) {
        return absl::InternalError(absl::StrFormat(
            "Pad aclnnInplaceCopyGetWorkspaceSize failed. Error code: %d",
            ret));
      }

      exec_ret = ExecuteAclnn(acl_stream, workspaceSize, executor,
                              dep::aclnnInplaceCopy);
      break;
    }

    default: {
      const std::string err_msg =
          absl::StrFormat("Op code(%v) does not support execute now",
                          HloOpcodeString(thunk_info().op_code));
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
  }

tril_dispatch_done:

  if (exec_ret.ok()) {
    acl_resources.device_ptrs.insert(acl_resources.device_ptrs.end(),
                                     temporary_device_ptrs.begin(),
                                     temporary_device_ptrs.end());
    acl_resources.tensors.insert(acl_resources.tensors.end(),
                                 acl_tensor_ptr_vector.begin(),
                                 acl_tensor_ptr_vector.end());
    acl_tensor_ptr_vector.clear();
    TF_RETURN_IF_ERROR(
        NpuStreamOrderedDeviceMemoryPool::GetInstance()->RetireAclResources(
            acl_stream, std::move(acl_resources)));
    acl_resources_retired = true;
  }

  // Control whether to print input and output values based on environment variables (including D2H operations, affecting performance)
  // Printing the output value requires synchronizing the stream first to ensure that the data has been copied from the device to the host
  if (NPU_CONFIG_GET_BOOL(NpuConfig::Key::ENABLE_PRINT_IO_VALUES)) {
    // Synchronize the stream first to ensure that the operator execution is completed
    TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
    int device_ordinal = params.stream->parent()->device_ordinal();
    PrintOutputTensor(allocations, param_shapes, device_ordinal);
  }

  return exec_ret;
}

absl::StatusOr<std::vector<aclTensor*>>
NpuAclnnThunk::ExtractAclInputOutTensorPtr(
    const NpuBufferAllocations* allocations,
    const std::vector<Shape>& param_shapes) {
  std::vector<aclTensor*> acl_tensor_ptr_vector;

  // First get the device address of all parameters
  std::vector<void*> param_addrs;
  param_addrs.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    auto addr = allocations->GetDeviceAddress(args_[i]);
    param_addrs.push_back(addr.opaque());
  }

  // Print the addresses of all parameters for diagnostic purposes
  std::string addr_debug = "Param addrs: ";
  for (size_t i = 0; i < param_addrs.size(); ++i) {
    addr_debug += absl::StrFormat("[%d]=%p; ", i, param_addrs[i]);
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, addr_debug);

  for (int i = 0; i < args_.size(); ++i) {
    void* device_addr = param_addrs[i];

    auto tensor_ptr = NpuTool::CreateAclTensorPtr(
        param_shapes[i], param_shapes[i].element_type(), device_addr);
    acl_tensor_ptr_vector.push_back(tensor_ptr);
  }

  return acl_tensor_ptr_vector;
}

absl::Status NpuAclnnThunk::ExecuteAclnn(aclrtStream& stream,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         AclnnExecFunc aclnnExecFunc) {
  aclnnStatus ret = 0;

  // Obtain memory from the Workspace memory pool instead of dynamic application
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    NPU_LOG_FORMAT_PRINT(DEBUG,
                         "Additional aclnn WorkspaceSize is %d, get from pool",
                         workspaceSize);
    auto* pool = NpuWorkspaceMemoryPool::GetInstance();
    if (pool == nullptr) {
      const std::string err_msg = "Workspace memory pool not available";
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }

    auto alloc_result = pool->AllocateForStream(stream, workspaceSize);
    if (!alloc_result.ok()) {
      const std::string err_msg =
          absl::StrFormat("Failed to allocate workspace from pool: %s",
                          alloc_result.status().ToString().c_str());
      NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
      return absl::InternalError(err_msg);
    }
    workspaceAddr = alloc_result.value();
  } else {
    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Additional aclnn WorkspaceSize is 0, no need to apply");
  }

  // Call aclnnExecFunc second section interface
  ret = aclnnExecFunc(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    const std::string err_msg =
        absl::StrFormat("Execute aclnnExecFunc failed. Error code: %d", ret);
    NPU_LOG_FORMAT_PRINT(ERROR, err_msg);
    return absl::InternalError(err_msg);
  }

  return absl::OkStatus();
}

absl::Status NpuAclnnThunk::PrintOutputTensor(
    const NpuBufferAllocations* allocations,
    const std::vector<Shape>& param_shapes, int device_ordinal) {
  // For the simple operator currently processed by ACLNN, only the last parameter is the output parameter.
  const auto& output_arg = args_.back();
  const auto& output_shape = param_shapes.back();

  auto addr = allocations->GetDeviceAddress(output_arg);
  void* device_addr = addr.opaque();

  // Use unified printing functions
  PrintParamInfo(0, "output", device_ordinal, output_arg, output_shape,
                 device_addr);

  return absl::OkStatus();
}

}  // namespace xla
