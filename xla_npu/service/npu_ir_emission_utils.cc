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

#include "xla/xla_npu/service/npu_ir_emission_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "tsl/platform/protobuf.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
absl::StatusOr<BufferAllocation::Slice> NpuGetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index) {
  return buffer_assignment.GetUniqueSlice(instr, index);
}

static bool IsNormalized(const HloTransposeInstruction& transpose) {
  const auto& permutation = transpose.dimensions();
  for (int i = 0; i < permutation.size() - 1; ++i) {
    if (permutation[i] + 1 == permutation[i + 1]) {
      return false;
    }
  }
  return true;
}

static bool CanEmitPackedTranspose(const HloTransposeInstruction& transpose) {
  // Support only normalized transposes.
  if (!IsNormalized(transpose)) {
    return false;
  }
  const auto& spec = NpuGetTransposeSpec(&transpose);
  return NpuGetPackedTransposeTileSizes(spec).ok();
}

std::optional<TransposeDescription> NpuGetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero) {
  if (hero.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  // We can assume that TransposeDimensionGrouper pass has run, so no need to
  // call GetNormalizedLogicalTransposeShape here.
  absl::InlinedVector<int64_t, 3> permutation(hero.dimensions().begin(),
                                              hero.dimensions().end());
  // A real transpose needs at least 2 transpose dimensions.
  if (permutation.size() < 2) {
    return std::nullopt;
  }
  auto bit_width = GetBitwidth(hero.shape().element_type());
  absl::InlinedVector<int64_t, 3> dimensions(hero.shape().dimensions().begin(),
                                             hero.shape().dimensions().end());
  int64_t operand_most_minor_dim = hero.operand(0)->shape().dimensions().back();
  if (CanEmitPackedTranspose(*Cast<HloTransposeInstruction>(&hero))) {
    int64_t vector_size =
        kNpuBankBitwidth / GetBitwidth(hero.shape().element_type());
    int64_t shmem_usage_bytes = kNpuNumShmemBanks * (kNpuBankBitwidth / 8) *
                                kNpuNumShmemBanks * vector_size;
    return TransposeDescription{&hero, dimensions, permutation,
                                shmem_usage_bytes};
  }
  if (permutation.back() == dimensions.size() - 1) {
    operand_most_minor_dim =
        hero.operand(0)->shape().dimensions(dimensions.size() - 2);
    if (bit_width * dimensions.back() <= kNpuMaxBitsInMostMinorDimension &&
        bit_width * dimensions.back() *
                std::min(operand_most_minor_dim,
                         dimensions[dimensions.size() - 2]) >=
            8 * kNpuMinDimensionToTransposeTiled) {
      // Tile size for transposition.
      int64_t shmem_usage_bytes =
          CeilOfRatio(kNpuNumShmemBanks * (kNpuNumShmemBanks + 1LL) *
                          bit_width * dimensions.back(),
                      8LL);
      return TransposeDescription{&hero, dimensions, permutation,
                                  shmem_usage_bytes};
    }
  } else if ((operand_most_minor_dim >= kNpuMinDimensionToTransposeTiled &&
              dimensions.back() >= kNpuMinDimensionToTransposeTiled) ||
             (operand_most_minor_dim >= kNpuMinDimensionToTransposeTiled2 &&
              dimensions.back() >= kNpuMinDimensionToTransposeTiled2 &&
              operand_most_minor_dim * dimensions.back() >=
                  kNpuMinTotalDimensionsToTransposeTiled)) {
    if (bit_width == 4) {
      return std::nullopt;
    }
    int64_t shmem_usage_bytes = CeilOfRatio(
        kNpuNumShmemBanks * (kNpuNumShmemBanks + 1LL) * bit_width, 8LL);
    return TransposeDescription{&hero, dimensions, permutation,
                                shmem_usage_bytes};
  }
  return std::nullopt;
}

NpuTransposeSpec NpuGetTransposeSpec(const HloTransposeInstruction* transpose) {
  auto inv_permutation = InversePermutation(transpose->dimensions());
  auto& output_shape = transpose->shape();
  llvm::SmallVector<int64_t, 3> canonical_output_shape =
      llvm::to_vector<3>(output_shape.dimensions());
  llvm::SmallVector<int64_t, 3> canonical_permutation =
      llvm::to_vector<3>(transpose->dimensions());

  // If the last dimension is transposed, add a size-1 B dimension.
  if (canonical_permutation.back() != canonical_output_shape.size() - 1) {
    canonical_permutation.push_back(output_shape.dimensions().size());
    canonical_output_shape.push_back(1);
  }
  int64_t dim_t1 = -1;
  int64_t dim_t2 = -1;
  for (int64_t i = canonical_permutation.size() - 1; i >= 0; --i) {
    if (canonical_permutation[i] != i) {
      dim_t2 = canonical_permutation[i];
      dim_t1 = i;
      break;
    }
  }
  // Insert size-1 A dimension if necessary.
  auto rank = canonical_output_shape.size();
  if (canonical_permutation[rank - 3] != rank - 3) {
    canonical_output_shape.insert(canonical_output_shape.begin() + dim_t1, 1);
    for (auto& p : canonical_permutation) {
      if (p > rank - 3) p++;
    }
    canonical_permutation.insert(canonical_permutation.begin() + dim_t1,
                                 dim_t1);
  }
  auto canonical_inv_permutation = InversePermutation(canonical_permutation);
  auto canonical_input_shape =
      Permute(canonical_output_shape, canonical_inv_permutation);
  return NpuTransposeSpec{
      transpose,
      llvm::to_vector<3>(transpose->dimensions()),
      llvm::to_vector<3>(inv_permutation),
      canonical_output_shape,
      canonical_permutation,
      llvm::to_vector<3>(canonical_inv_permutation),
      llvm::to_vector<3>(canonical_input_shape),
  };
}

std::string NpuTransposeSpec::ToString() const {
  return absl::Substitute(R"(
transpose: $0
canonical_input_shape: $1
canonical_output_shape: $2
canonical_permutation: $3
canonical_inv_permutation: $4
[T2, A, T1, B] = [$5, $6, $7, $8]
)",
                          transpose->ToString(),
                          absl::StrJoin(canonical_input_shape, ","),
                          absl::StrJoin(canonical_output_shape, ","),
                          absl::StrJoin(canonical_permutation, ","),
                          absl::StrJoin(canonical_inv_permutation, ","),
                          dim_T2(), dim_A(), dim_T1(), dim_B());
}

absl::StatusOr<absl::InlinedVector<int64_t, 3>> NpuGetPackedTransposeTileSizes(
    const NpuTransposeSpec& spec) {
  // Check the side outputs, etc.
  int64_t bits_per_element = GetBitwidth(spec.elem_type());
  if (bits_per_element >= kNpuBankBitwidth) {
    return absl::InvalidArgumentError("Element type is too large");
  }
  absl::InlinedVector<int64_t, 3> tile_sizes(spec.canonical_rank(), 1);
  int64_t vector_size = kNpuBankBitwidth / bits_per_element;

  // The shmem size is `shmem_dim x shmem_dim`.
  int64_t shmem_dim = kNpuNumShmemBanks * vector_size;
  int64_t tile_size_T1 = std::min(spec.dim_T1(), shmem_dim);
  int64_t tile_size_A = std::min(spec.dim_A(), shmem_dim / tile_size_T1);
  int64_t tile_size_T2 = std::min(spec.dim_T2(), shmem_dim);
  int64_t populated_shmem_rows = tile_size_T2;
  int64_t populated_shmem_cols = tile_size_A * tile_size_T1;

  // Do not use the packed transpose if there are not enough populated rows or
  // columns in shmem.
  const int64_t kNumMinPopulatedRowsOrColumns = 10 * vector_size;
  if (populated_shmem_cols < kNumMinPopulatedRowsOrColumns ||
      populated_shmem_rows < kNumMinPopulatedRowsOrColumns) {
    return absl::InvalidArgumentError("Not enough rows or columns in shmem");
  }

  // These divisibility constrains are too strict, we can do better.
  if (spec.dim_B() != 1 || populated_shmem_rows % vector_size != 0 ||
      populated_shmem_cols % vector_size != 0 ||
      spec.dim_T2() % tile_size_T2 % vector_size != 0) {
    return absl::InvalidArgumentError("The shape is not supported");
  }
  tile_sizes[spec.dim_T1_output_id()] = tile_size_T1;
  tile_sizes[spec.dim_T2_output_id()] = tile_size_T2;
  tile_sizes[spec.dim_A_id()] = tile_size_A;
  return tile_sizes;
}

HloInstructionAdaptor NpuFindNonTrivialHero(
    const HloInstructionAdaptor& instr) {
  HloInstructionAdaptor hero = instr;

  // Go up the chain of trivial element-wise(+bitcast, -copy) operations. Note
  // that no memoization is needed due to number of operands constraints: we
  // never have to revisit same nodes.
  while (IsIntermediate(&hero.instruction(), /*allowed_operand_count=*/1) &&
         hero.parent().ContainsInstruction(hero.GetOperand(0))) {
    hero = hero.GetOperand(0);
  }

  // Try a bit harder to find a transpose or concat hero. The shared memory
  // transpose and concat emitters also work if there are elementwise ops with
  // more than 1 operand on the path between root and the root op.
  auto is_transpose = [](const HloInstruction& node) {
    return NpuGetDescriptionForTiledTransposeEmitter(node).has_value();
  };
  if (auto transpose = FindHero(hero, std::move(is_transpose))) {
    return *transpose;
  }
  auto is_concatenate = [](const HloInstruction& node) {
    return node.opcode() == HloOpcode::kConcatenate;
  };
  if (auto concatenate = FindHero(hero, std::move(is_concatenate))) {
    return *concatenate;
  }
  if (hero.opcode() != HloOpcode::kReduce) {
    return instr;
  }
  return hero;
}

const HloInstruction& NpuFindNonTrivialHero(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion);
  auto fusion_adaptor = HloFusionAdaptor::ForComputation(instr.parent());
  HloInstructionAdaptor instr_adaptor(instr, fusion_adaptor.get());
  return NpuFindNonTrivialHero(instr_adaptor).instruction();
}

absl::StatusOr<NpuDenseDataIntermediate> NpuLiteralToXlaFormat(
    const Literal& literal) {
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return Internal("Unsupported type in NpuLiteralToXlaFormat");
  }

  int64_t byte_size = literal.size_bytes();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    auto bit_width = primitive_util::BitWidth(element_type);
    std::vector<uint8_t> output(CeilOfRatio<int64_t>(byte_size, 8 / bit_width));
    absl::Span<char> output_span =
        absl::MakeSpan(reinterpret_cast<char*>(output.data()), output.size());
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(literal.untyped_data()),
                       byte_size),
        output_span);
    return NpuDenseDataIntermediate::Own(std::move(output));
  }

  return NpuDenseDataIntermediate::Alias(absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(literal.untyped_data()), byte_size));
}
}  // namespace xla
