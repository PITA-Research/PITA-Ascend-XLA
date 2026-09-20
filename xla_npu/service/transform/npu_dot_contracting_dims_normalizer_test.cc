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

#include "xla/xla_npu/service/transform/npu_dot_contracting_dims_normalizer.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/shape_util.h"
#include "xla/service/hlo_verifier.h"

namespace xla {
namespace {

using NpuDotContractingDimsNormalizerTest =
    HloHardwareIndependentTestBase;

template <typename Range>
std::vector<int64_t> ToVector(const Range& range) {
  return std::vector<int64_t>(range.begin(), range.end());
}

void ExpectMaterializedTranspose(const HloInstruction* transpose) {
  ASSERT_EQ(transpose->opcode(), HloOpcode::kTranspose);
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(transpose->shape().layout()));
  EXPECT_FALSE(ShapeUtil::TransposeIsBitcast(
      transpose->operand(0)->shape(), transpose->shape(),
      transpose->dimensions()));
}

TEST_F(NpuDotContractingDimsNormalizerTest, MaterializesVjpTransposes) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[64,32]{1,0} parameter(0)
      rhs = f32[16,64]{1,0} parameter(1)
      ROOT dot = f32[32,16]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* dot = module->entry_computation()->root_instruction();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);
  EXPECT_EQ(ToVector(dot->dot_dimension_numbers()
                         .lhs_contracting_dimensions()),
            (std::vector<int64_t>{1}));
  EXPECT_EQ(ToVector(dot->dot_dimension_numbers()
                         .rhs_contracting_dimensions()),
            (std::vector<int64_t>{0}));
  ExpectMaterializedTranspose(dot->operand(0));
  ExpectMaterializedTranspose(dot->operand(1));
  EXPECT_EQ(ToVector(dot->operand(0)->dimensions()),
            (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(ToVector(dot->operand(1)->dimensions()),
            (std::vector<int64_t>{1, 0}));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       NormalizesVjpResultTransposeLayout) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[1,64]{1,0} parameter(0)
      rhs = f32[1,64]{1,0} parameter(1)
      dot = f32[64,64]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
      ROOT result = f32[64,64]{0,1} transpose(dot), dimensions={1,0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* result =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(result->opcode(), HloOpcode::kTranspose);
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(result->shape().layout()));
  EXPECT_FALSE(ShapeUtil::TransposeIsBitcast(
      result->operand(0)->shape(), result->shape(), result->dimensions()));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       NormalizesStandaloneTransposeLayout) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      input = f32[2,3]{1,0} parameter(0)
      ROOT result = f32[3,2]{0,1} transpose(input), dimensions={1,0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* result =
      module->entry_computation()->root_instruction();
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(result->shape().layout()));
  EXPECT_FALSE(ShapeUtil::TransposeIsBitcast(
      result->operand(0)->shape(), result->shape(), result->dimensions()));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       PropagatesTransposeLayoutThroughNestedTupleAliases) {
  constexpr char kHlo[] = R"(
    HloModule test, entry_computation_layout={(f32[2,3]{1,0})->(f32[2,3]{1,0}, f32[3,2]{1,0})}

    ENTRY main {
      input = f32[2,3]{1,0} parameter(0)
      transpose = f32[3,2]{0,1} transpose(input), dimensions={1,0}
      inner = (f32[2,3]{1,0}, f32[3,2]{0,1}) tuple(input, transpose)
      outer = ((f32[2,3]{1,0}, f32[3,2]{0,1})) tuple(inner)
      extracted = (f32[2,3]{1,0}, f32[3,2]{0,1}) get-tuple-element(outer), index=0
      value = f32[3,2]{0,1} get-tuple-element(extracted), index=1
      ROOT result = (f32[2,3]{1,0}, f32[3,2]{0,1}) tuple(input, value)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(
      root->shape().tuple_shapes(1).layout()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape().tuple_shapes(0),
                              root->operand(0)->shape()));
  TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                           /*allow_mixed_precision=*/false)
                   .Run(module.get()).status());
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       PreservesBatchAndContractPairing) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[3,5,2,7]{3,2,1,0} parameter(0)
      rhs = f32[7,3,11,5]{3,2,1,0} parameter(1)
      ROOT dot = f32[3,2,11]{2,1,0} dot(lhs, rhs),
        lhs_batch_dims={0}, lhs_contracting_dims={1,3},
        rhs_batch_dims={1}, rhs_contracting_dims={3,0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* dot = module->entry_computation()->root_instruction();
  const DotDimensionNumbers& dims = dot->dot_dimension_numbers();
  EXPECT_EQ(ToVector(dims.lhs_batch_dimensions()),
            (std::vector<int64_t>{0}));
  EXPECT_EQ(ToVector(dims.rhs_batch_dimensions()),
            (std::vector<int64_t>{0}));
  EXPECT_EQ(ToVector(dims.lhs_contracting_dimensions()),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(ToVector(dims.rhs_contracting_dimensions()),
            (std::vector<int64_t>{1, 2}));
  ExpectMaterializedTranspose(dot->operand(0));
  ExpectMaterializedTranspose(dot->operand(1));
}

TEST_F(NpuDotContractingDimsNormalizerTest, LeavesCanonicalDotUnchanged) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[32,64]{1,0} parameter(0)
      rhs = f32[64,16]{1,0} parameter(1)
      ROOT dot = f32[32,16]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(false));

  const HloInstruction* dot = module->entry_computation()->root_instruction();
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kParameter);
}


TEST_F(NpuDotContractingDimsNormalizerTest, NormalizesDotResultAndTupleLayout) {
  constexpr char kHlo[] = R"(
    HloModule test, entry_computation_layout={(f32[2,3]{1,0}, f32[3,5]{1,0})->(f32[2,5]{1,0})}
    ENTRY main {
      lhs = f32[2,3]{1,0} parameter(0)
      rhs = f32[3,5]{1,0} parameter(1)
      dot = f32[2,5]{0,1} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT result = (f32[2,5]{0,1}) tuple(dot)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const auto* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(
      root->operand(0)->shape().layout()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape().tuple_shapes(0),
                              root->operand(0)->shape()));
  TF_ASSERT_OK(HloVerifier(true, false).Run(module.get()).status());
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       RelayoutsBothConstantOperandsWithoutChangingSharedUsers) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      lhs = f32[2,3]{0,1} constant({{1,2,3},{4,5,6}})
      rhs = f32[2,3]{0,1} constant({{7,8,9},{10,11,12}})
      dot = f32[2]{0} dot(lhs, rhs), lhs_batch_dims={0},
        rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
      ROOT result = (f32[2]{0}, f32[2,3]{0,1}) tuple(dot, lhs)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(auto expected, evaluator.Evaluate(*module, {}));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const auto* root = module->entry_computation()->root_instruction();
  const auto* dot = root->operand(0);
  for (const auto* operand : dot->operands()) {
    EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout()));
    EXPECT_TRUE(ShapeUtil::Equal(operand->shape(), operand->literal().shape()));
  }
  // Check physical bytes, not only shape tags or evaluator logical values.
  const auto bytes = dot->operand(0)->literal().data<float>();
  EXPECT_EQ(std::vector<float>(bytes.begin(), bytes.end()),
            (std::vector<float>{1,2,3,4,5,6}));
  EXPECT_NE(dot->operand(0), root->operand(1));
  EXPECT_FALSE(LayoutUtil::IsMonotonicWithDim0Major(root->operand(1)->shape().layout()));
  TF_ASSERT_OK_AND_ASSIGN(auto actual, evaluator.Evaluate(*module, {}));
  EXPECT_EQ(expected, actual);
  TF_ASSERT_OK(HloVerifier(true, false).Run(module.get()).status());
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NpuDotContractingDimsNormalizerTest,
       RelayoutsConstantBeforeNoncanonicalTranspose) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      lhs = f32[3,2]{0,1} constant({{1,2},{3,4},{5,6}})
      rhs = f32[3,2]{0,1} constant({{7,8},{9,10},{11,12}})
      ROOT dot = f32[2,2]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(auto expected, evaluator.Evaluate(*module, {}));
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const auto* dot = module->entry_computation()->root_instruction();
  ASSERT_EQ(dot->operand(0)->opcode(), HloOpcode::kTranspose);
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(
      dot->operand(0)->operand(0)->shape().layout()));
  TF_ASSERT_OK_AND_ASSIGN(auto actual, evaluator.Evaluate(*module, {}));
  EXPECT_EQ(expected, actual);
  TF_ASSERT_OK(HloVerifier(true, false).Run(module.get()).status());
  EXPECT_THAT(NpuDotContractingDimsNormalizer().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
