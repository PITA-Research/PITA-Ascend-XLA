/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */
#include "xla/xla_npu/service/transform/npu_scatter_window_expander.h"

#include <string>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/literal.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"

namespace xla {
namespace {
class NpuScatterWindowExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  void Check(const std::string& body) {
    const std::string hlo = R"(
      HloModule test
      add {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT sum = f32[] add(x,y)
      }
      ENTRY main {
        base = f32[4,6] constant({{1,2,3,4,5,6},{7,8,9,10,11,12},
                                 {13,14,15,16,17,18},{19,20,21,22,23,24}})
    )" + body + "\n}";
    CheckModule(hlo, 25);
  }
  void CheckModule(const std::string& hlo, int64_t padded_size) {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloEvaluator evaluator;
    TF_ASSERT_OK_AND_ASSIGN(auto expected, evaluator.Evaluate(*module, {}));
    EXPECT_THAT(NpuScatterWindowExpander().Run(module.get()),
                absl_testing::IsOkAndHolds(true));
    TF_ASSERT_OK(HloVerifier(true, false).Run(module.get()).status());
    TF_ASSERT_OK_AND_ASSIGN(auto actual, evaluator.Evaluate(*module, {}));
    EXPECT_EQ(expected, actual);
    for (const auto* instr : module->entry_computation()->instructions()) {
      if (instr->opcode() != HloOpcode::kScatter) continue;
      const auto& dn = static_cast<const HloScatterInstruction*>(instr)
                           ->scatter_dimension_numbers();
      EXPECT_EQ(dn.update_window_dims_size(), 0);
      EXPECT_EQ(dn.scatter_dims_to_operand_dims_size(), 1);
      EXPECT_EQ(instr->shape().dimensions(0), padded_size);
      EXPECT_EQ(instr->get_frontend_attribute("npu.scatter_add_normalized"), "1");
      EXPECT_EQ(instr->operand(1)->opcode(), HloOpcode::kConstant);
    }
    EXPECT_THAT(NpuScatterWindowExpander().Run(module.get()),
                absl_testing::IsOkAndHolds(false));
  }
};

TEST_F(NpuScatterWindowExpanderTest, PartialOverlappingAndInvalidWindows) {
  Check(R"(
    indices = s32[5,2] constant({{1,0},{1,1},{-1,0},{3,4},{2,3}})
    updates = f32[5,3] constant({{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}})
    ROOT result = f32[4,6] scatter(base,indices,updates),
      update_window_dims={1}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=add
  )");
}
TEST_F(NpuScatterWindowExpanderTest, PermutedMapVectorAxisAndUpdateAxis) {
  Check(R"(
    indices = s64[2,2] constant({{0,3},{1,2}})
    updates = f32[3,2] constant({{1,2},{3,4},{5,6}})
    ROOT result = f32[4,6] scatter(base,indices,updates),
      update_window_dims={0}, inserted_window_dims={0},
      scatter_dims_to_operand_dims={1,0}, index_vector_dim=0, to_apply=add
  )");
}
TEST_F(NpuScatterWindowExpanderTest, TwoDimensionalWindowsAndLargeIndices) {
  Check(R"(
    indices = s64[3,2] constant({{1,2},{2,3},{4294967297,0}})
    updates = f32[3,2,3] constant({{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},
                                  {{13,14,15},{16,17,18}}})
    ROOT result = f32[4,6] scatter(base,indices,updates),
      update_window_dims={1,2}, inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=add
  )");
}
TEST_F(NpuScatterWindowExpanderTest, SingleAxisMultidimensionalIndexGrid) {
  Check(R"(
    indices = s32[2,2,1] constant({{{1},{1}},{{5},{-1}}})
    updates = f32[4,2,2] constant({{{1,2},{3,4}},{{5,6},{7,8}},
                                  {{9,10},{11,12}},{{13,14},{15,16}}})
    ROOT result = f32[4,6] scatter(base,indices,updates),
      update_window_dims={0}, inserted_window_dims={1},
      scatter_dims_to_operand_dims={1}, index_vector_dim=2, to_apply=add
  )");
}
TEST_F(NpuScatterWindowExpanderTest, ImplicitIndexVectorWithGrid) {
  Check(R"(
    indices = s32[2,2] constant({{1,1},{5,-1}})
    updates = f32[4,2,2] constant({{{1,2},{3,4}},{{5,6},{7,8}},
                                  {{9,10},{11,12}},{{13,14},{15,16}}})
    ROOT result = f32[4,6] scatter(base,indices,updates),
      update_window_dims={0}, inserted_window_dims={1},
      scatter_dims_to_operand_dims={1}, index_vector_dim=2, to_apply=add
  )");
}
TEST_F(NpuScatterWindowExpanderTest, UnitWindowAndAlreadyPointShapedInputs) {
  for (bool window : {false, true}) {
    CheckModule(std::string(R"(
      HloModule test
      add {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT sum = f32[] add(x,y)
      }
      ENTRY main {
        base = f32[5] constant({1,2,3,4,5})
        indices = s32[4,1] constant({{1},{1},{-1},{2147483647}})
    )") + (window ?
        "updates = f32[4,1] constant({{6},{7},{8},{9}})\n" :
        "updates = f32[4] constant({6,7,8,9})\n") +
        "ROOT result = f32[5] scatter(base,indices,updates), " +
        (window ? "update_window_dims={1}, inserted_window_dims={}, " :
                  "update_window_dims={}, inserted_window_dims={0}, ") +
        "scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add\n}", 6);
  }
}
TEST_F(NpuScatterWindowExpanderTest, ScalarOperandWithEmptyIndexVectors) {
  CheckModule(R"(
    HloModule test
    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x,y)
    }
    ENTRY main {
      base = f32[] constant(5)
      indices = s32[3,0] constant({})
      updates = f32[3] constant({1,2,3})
      ROOT result = f32[] scatter(base,indices,updates),
        update_window_dims={}, inserted_window_dims={},
        scatter_dims_to_operand_dims={}, index_vector_dim=1, to_apply=add
    }
  )", 2);
}
}  // namespace
}  // namespace xla
