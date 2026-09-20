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

#include "xla/xla_npu/service/transform/npu_scalar_fma_compensation.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using NpuScalarFmaCompensationTest = HloHardwareIndependentTestBase;

int CountOpcode(const HloComputation* computation, HloOpcode opcode) {
  int count = 0;
  for (const HloInstruction* instr : computation->instructions()) {
    if (instr->opcode() == opcode) {
      ++count;
    }
  }
  return count;
}

bool HasConstantValue(const HloComputation* computation, float value) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == F32 &&
        instr->literal().Get<float>({}) == value) {
      return true;
    }
  }
  return false;
}

TEST_F(NpuScalarFmaCompensationTest, RewritesScalarF32MultiplyAdd) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      other = f32[] parameter(2)
      mul = f32[] multiply(lhs, rhs)
      ROOT add = f32[] add(mul, other)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  const int add_count_before = CountOpcode(entry, HloOpcode::kAdd);
  const int multiply_count_before = CountOpcode(entry, HloOpcode::kMultiply);

  EXPECT_THAT(NpuScalarFmaCompensation().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* root = entry->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_NE(root->operand(0)->opcode(), HloOpcode::kMultiply);
  EXPECT_NE(root->operand(1)->opcode(), HloOpcode::kMultiply);
  EXPECT_GT(CountOpcode(entry, HloOpcode::kAdd), add_count_before);
  EXPECT_GT(CountOpcode(entry, HloOpcode::kMultiply), multiply_count_before);
  EXPECT_GT(CountOpcode(entry, HloOpcode::kSubtract), 0);
  EXPECT_TRUE(HasConstantValue(entry, 4097.0f));
}

TEST_F(NpuScalarFmaCompensationTest, LeavesVectorMultiplyAddUnchanged) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[4]{0} parameter(0)
      rhs = f32[4]{0} parameter(1)
      other = f32[4]{0} parameter(2)
      mul = f32[4]{0} multiply(lhs, rhs)
      ROOT add = f32[4]{0} add(mul, other)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuScalarFmaCompensation().Run(module.get()),
              absl_testing::IsOkAndHolds(false));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kMultiply);
}

TEST_F(NpuScalarFmaCompensationTest, LeavesScalarAddWithoutMultiplyUnchanged) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuScalarFmaCompensation().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
