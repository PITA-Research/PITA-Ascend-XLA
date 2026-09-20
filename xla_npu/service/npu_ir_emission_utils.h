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

#ifndef XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_
#define XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "tsl/platform/protobuf.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

// // <HLO computation fingerprint, serialized compiled object>.
// using BinaryMap = absl::flat_hash_map<std::string, std::string>;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
inline constexpr int64_t kNpuMinDimensionToTransposeTiled = 16;
// But if both swap dimensions are larger than 'kNpuMinDimensionToTransposeTiled2',
// and the product of the dimensions to be swapped is larger than
// 'kNpuMinTotalDimensionsToTransposeTiled', tiled transposition may be more
// efficient.
inline constexpr int64_t kNpuMinDimensionToTransposeTiled2 = 8;
inline constexpr int64_t kNpuMinTotalDimensionsToTransposeTiled = 64 * 128;
// As the amount of shared memory is limited, we need to make sure that we don't
// detect 102 transposes that would require too much bytes for the most minor
// dimension.
inline constexpr int64_t kNpuMaxBitsInMostMinorDimension = 8 * 8;

// The number of shared memory banks.
inline constexpr int64_t kNpuNumShmemBanks = 32;

// The bitwidth of a shared memory bank.
inline constexpr int64_t kNpuBankBitwidth = 32;

absl::StatusOr<BufferAllocation::Slice> NpuGetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index);

// Returns the first hero instruction reachable from `instr` as root. Hero
// instruction can be in a different computation if the parent HloFusionAdaptor
// is a producer-consumer fusion.
HloInstructionAdaptor NpuFindNonTrivialHero(const HloInstructionAdaptor& instr);

// Same as above, but fusion is the parent computation of the hlo instruction.
const HloInstruction& NpuFindNonTrivialHero(const HloInstruction& instr);

std::optional<TransposeDescription> NpuGetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero);

// Canonical transpose permutes the input shape
// <D_0 x ... x D_n x T2 x D_{n+1} x ... x D_m x A x T1 x B> into
// <D'_0 x ... x D'_n' x T1 x D'_{n'+1} x ... x D'_m x A x T2 x B>.
// Note that the `D` dimensions are batch dimensions. They can also be
// permuted, but they are tiled by 1.
// 1. <8x32> -> <32x8> will be canonicalized to <8x1x32x1> -> <32x1x8x1>.
// 2. <8x2x32> -> <32x2x8> will be canonicalized to <8x2x32x1> -> <32x2x8x1>.
// 3. <8x2x32x7x6> -> <6x32x2x7x8> becomes <8x2x32x7x6x1> -> <6x32x2x7x8x1>.
struct NpuTransposeSpec {
  PrimitiveType elem_type() const { return input_shape().element_type(); }

  const Shape& input_shape() const { return transpose->operand(0)->shape(); }
  const Shape& output_shape() const { return transpose->shape(); }

  int64_t rank() const { return input_shape().dimensions().size(); }
  int64_t canonical_rank() const { return canonical_input_shape.size(); }

  int64_t dim_A() const { return canonical_input_shape[dim_A_id()]; }
  int64_t dim_A_id() const { return canonical_rank() - 3; }

  int64_t dim_B() const { return canonical_input_shape.back(); }
  int64_t dim_B_id() const { return canonical_rank() - 1; }

  int64_t dim_T1() const { return canonical_input_shape[dim_T1_input_id()]; }
  int64_t dim_T1_input_id() const { return canonical_rank() - 2; }
  int64_t dim_T1_output_id() const {
    return canonical_inv_permutation[canonical_rank() - 2];
  }

  int64_t dim_T2() const { return canonical_input_shape[dim_T2_input_id()]; }
  int64_t dim_T2_input_id() const {
    return canonical_permutation[canonical_rank() - 2];
  }
  int64_t dim_T2_output_id() const { return canonical_rank() - 2; }

  std::string ToString() const;

  const HloTransposeInstruction* transpose;

  llvm::SmallVector<int64_t, 3> permutation;
  llvm::SmallVector<int64_t, 3> inv_permutation;

  llvm::SmallVector<int64_t, 3> canonical_output_shape;
  llvm::SmallVector<int64_t, 3> canonical_permutation;
  llvm::SmallVector<int64_t, 3> canonical_inv_permutation;
  llvm::SmallVector<int64_t, 3> canonical_input_shape;
};

NpuTransposeSpec NpuGetTransposeSpec(const HloTransposeInstruction* transpose);

// Returns the default tile sizes for the packed transpose emitter.
absl::StatusOr<absl::InlinedVector<int64_t, 3>> NpuGetPackedTransposeTileSizes(
    const NpuTransposeSpec& spec);

// This class stores either a non-owning reference or owns data that represents
// a dense array in XLA format. It is used for intermediate storage during IR
// constant emission.
class NpuDenseDataIntermediate {
 public:
  // Creates an instance of NpuDenseDataIntermediate that owns the provided vector.
  static NpuDenseDataIntermediate Own(std::vector<uint8_t> owned) {
    NpuDenseDataIntermediate di;
    di.data_ = std::move(owned);
    return di;
  }

  // Creates an instance of NpuDenseDataIntermediate that aliases the input.
  static NpuDenseDataIntermediate Alias(absl::Span<const uint8_t> aliased) {
    NpuDenseDataIntermediate di;
    di.data_ = aliased;
    return di;
  }

  // Returns a reference to the data this object represents.
  absl::Span<const uint8_t> span() const {
    return data_.index() == 0 ? absl::Span<const uint8_t>(std::get<0>(data_))
                              : std::get<1>(data_);
  }

 private:
  std::variant<std::vector<uint8_t>, absl::Span<const uint8_t>> data_;
};

absl::StatusOr<NpuDenseDataIntermediate> NpuLiteralToXlaFormat(
    const Literal& literal);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_NPU_IR_EMISSION_UTILS_H_
