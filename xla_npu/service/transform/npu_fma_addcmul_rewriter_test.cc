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

#include "xla/xla_npu/service/transform/npu_fma_addcmul_rewriter.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using NpuFmaAddcmulRewriterTest = HloHardwareIndependentTestBase;

uint32_t ValueBits(const HloInstruction* instr) {
  uint32_t bits = 0;
  const std::string opaque(instr->opaque());
  EXPECT_EQ(sscanf(opaque.c_str(), "value_bits=%u", &bits), 1);
  return bits;
}

TEST_F(NpuFmaAddcmulRewriterTest, RewritesScalarMultiplyAdd) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      self = f32[] parameter(2)
      mul = f32[] multiply(x, y)
      ROOT add = f32[] add(mul, self)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuFmaAddcmulRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__npu_fma_addcmul");
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_EQ(ValueBits(root), 0x3f800000u);
}

TEST_F(NpuFmaAddcmulRewriterTest, RewritesVectorScaledMultiplyAdd) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      alpha = f32[4]{0} parameter(0)
      q2 = f32[4]{0} parameter(1)
      q1 = f32[4]{0} parameter(2)
      two = f32[] constant(2)
      two_broadcast = f32[4]{0} broadcast(two), dimensions={}
      scaled = f32[4]{0} multiply(alpha, two_broadcast)
      product = f32[4]{0} multiply(scaled, q2)
      ROOT add = f32[4]{0} add(product, q1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuFmaAddcmulRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->operand(0)->name(), "q1");
  EXPECT_EQ(root->operand(1)->name(), "alpha");
  EXPECT_EQ(root->operand(2)->name(), "q2");
  EXPECT_EQ(ValueBits(root), 0x40000000u);
}

TEST_F(NpuFmaAddcmulRewriterTest, RewritesChainedMultiplyAdds) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      c = f32[] parameter(2)
      d = f32[] parameter(3)
      e = f32[] parameter(4)
      first_product = f32[] multiply(a, b)
      first_add = f32[] add(first_product, c)
      second_product = f32[] multiply(first_add, d)
      ROOT second_add = f32[] add(second_product, e)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuFmaAddcmulRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__npu_fma_addcmul");
  ASSERT_EQ(root->operand(1)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->operand(1)->custom_call_target(), "__npu_fma_addcmul");
}

TEST_F(NpuFmaAddcmulRewriterTest, SkipsNonCOrderBuffers) {
  constexpr char kHlo[] = R"(
    HloModule test
    ENTRY main {
      x = f32[2,3]{0,1} parameter(0)
      y = f32[2,3]{0,1} parameter(1)
      self = f32[2,3]{0,1} parameter(2)
      mul = f32[2,3]{0,1} multiply(x, y)
      ROOT add = f32[2,3]{0,1} add(mul, self)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(NpuFmaAddcmulRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

}  // namespace
}  // namespace xla
