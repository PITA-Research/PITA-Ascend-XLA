/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#include "xla/xla_npu/service/transform/npu_right_triangular_solve_expander.h"

#include <string>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"

namespace xla {
namespace {
using NpuRightTriangularSolveExpanderTest = HloHardwareIndependentTestBase;
TEST_F(NpuRightTriangularSolveExpanderTest, BatchedRectangularOptions) {
  for (const std::string transpose : {"NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"}) {
    for (const std::string lower : {"true", "false"}) {
      for (const std::string unit : {"true", "false"}) {
        const std::string hlo = R"(
          HloModule test
          ENTRY main {
            a = f32[2,5,5]{2,1,0} parameter(0)
            b = f32[2,3,5]{2,1,0} parameter(1)
            ROOT solve = f32[2,3,5]{2,1,0} triangular-solve(a,b),
              left_side=false, lower=)" + lower + ", unit_diagonal=" + unit +
              ", transpose_a=" + transpose + "\n}";
        TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
        EXPECT_THAT(NpuRightTriangularSolveExpander().Run(module.get()),
                    absl_testing::IsOkAndHolds(true));
        const auto* result = module->entry_computation()->root_instruction();
        ASSERT_EQ(result->opcode(), HloOpcode::kTranspose);
        EXPECT_EQ(result->dimensions(0), 0);
        EXPECT_EQ(result->dimensions(1), 2);
        EXPECT_EQ(result->dimensions(2), 1);
        const auto* solve = result->operand(0);
        ASSERT_EQ(solve->opcode(), HloOpcode::kTriangularSolve);
        const auto& options = solve->triangular_solve_options();
        EXPECT_TRUE(options.left_side());
        EXPECT_EQ(options.lower(), lower == "true");
        EXPECT_EQ(options.unit_diagonal(), unit == "true");
        EXPECT_EQ(options.transpose_a(), transpose == "NO_TRANSPOSE" ?
            TriangularSolveOptions::TRANSPOSE : TriangularSolveOptions::NO_TRANSPOSE);
        EXPECT_EQ(solve->shape().dimensions(1), 5);
        EXPECT_EQ(solve->shape().dimensions(2), 3);
        EXPECT_EQ(solve->operand(0)->opcode(), HloOpcode::kParameter);
        ASSERT_EQ(solve->operand(1)->opcode(), HloOpcode::kTranspose);
        TF_ASSERT_OK(HloVerifier(true, false).Run(module.get()).status());
        EXPECT_THAT(NpuRightTriangularSolveExpander().Run(module.get()),
                    absl_testing::IsOkAndHolds(false));
      }
    }
  }
}
TEST_F(NpuRightTriangularSolveExpanderTest, LeavesLeftSolveUnchanged) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      a = f32[5,5] parameter(0)
      b = f32[5,3] parameter(1)
      ROOT solve = f32[5,3] triangular-solve(a,b), left_side=true,
        lower=true, transpose_a=NO_TRANSPOSE
    }
  )"));
  EXPECT_THAT(NpuRightTriangularSolveExpander().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}
}  // namespace
}  // namespace xla
