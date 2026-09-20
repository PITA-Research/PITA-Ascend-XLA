/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved. */

#include "xla/xla_npu/service/transform/npu_variadic_reduce_expander.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace {

class NpuVariadicReduceExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  void CheckExpansion(const std::string& hlo) {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloEvaluator evaluator;
    TF_ASSERT_OK_AND_ASSIGN(auto expected, evaluator.Evaluate(*module, {}));
    EXPECT_THAT(NpuVariadicReduceExpander().Run(module.get()),
                absl_testing::IsOkAndHolds(true));
    TF_ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                     .Run(module.get()).status());
    TF_ASSERT_OK_AND_ASSIGN(auto actual, evaluator.Evaluate(*module, {}));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
    for (const HloInstruction* instr :
         module->entry_computation()->instructions()) {
      EXPECT_NE(instr->opcode(), HloOpcode::kReduce);
      if (instr->opcode() == HloOpcode::kSlice) {
        for (int64_t stride : instr->slice_strides()) EXPECT_EQ(stride, 1);
      }
    }
    EXPECT_THAT(NpuVariadicReduceExpander().Run(module.get()),
                absl_testing::IsOkAndHolds(false));
  }
};

// This is JAX's actual coupled reducer, including its separate predicates for
// the value and index outputs. Names intentionally do not identify argmin/max.
std::string ArgModule(const std::string& input_dims,
                      const std::string& output_dims,
                      const std::string& dimensions, int iota_dimension,
                      const std::string& literal, bool maximum) {
  const std::string reducer = absl::StrReplaceAll(R"(
    reducer {
      lv = f32[] parameter(0)
      li = s32[] parameter(1)
      rv = f32[] parameter(2)
      ri = s32[] parameter(3)
      better = pred[] compare(lv, rv), direction=$DIRECTION
      nan = pred[] compare(lv, lv), direction=NE
      value_pred = pred[] or(better, nan)
      equal = pred[] compare(lv, rv), direction=EQ
      first = pred[] compare(li, ri), direction=LT
      tie = pred[] and(equal, first)
      index_pred = pred[] or(value_pred, tie)
      value = f32[] select(value_pred, lv, rv)
      index = s32[] select(index_pred, li, ri)
      ROOT result = (f32[], s32[]) tuple(value, index)
    }
  )", {{"$DIRECTION", maximum ? "GT" : "LT"}});
  return absl::StrCat(
      "HloModule test\n", reducer, "\nENTRY main {\n",
      "  input = f32", input_dims, " constant(", literal, ")\n",
      "  index = s32", input_dims, " iota(), iota_dimension=", iota_dimension, "\n",
      "  init = f32[] constant(", maximum ? "-inf" : "inf", ")\n",
      "  zero = s32[] constant(0)\n",
      "  ROOT result = (f32", output_dims, ", s32", output_dims,
      ") reduce(input, index, init, zero), dimensions={", dimensions,
      "}, to_apply=reducer\n}");
}

TEST_F(NpuVariadicReduceExpanderTest, TiesNaNsOddEvenAndSingleton) {
  for (bool maximum : {false, true}) {
    for (const std::string& values : {
             "{2, 7, 7, -4, -4}", "{0, nan, 2, nan, 4}",
             "{nan, nan, nan, nan, nan}", "{inf, inf, inf, inf, inf}",
             "{-inf, -inf, -inf, -inf, -inf}", "{0, -0, 0, -0, 0}"}) {
      SCOPED_TRACE(values);
      CheckExpansion(ArgModule("[5]", "[]", "0", 0, values, maximum));
    }
    CheckExpansion(ArgModule("[6]", "[]", "0", 0, "{0, 1, 2, 3, 4, 5}", maximum));
    CheckExpansion(ArgModule("[1]", "[]", "0", 0, "{3}", maximum));
  }
}

TEST_F(NpuVariadicReduceExpanderTest, BatchAndNonTrailingAxis) {
  for (bool maximum : {false, true}) {
    const std::string values = "{{0, 4, 2}, {5, 3, 2}, {1, 4, 7}}";
    CheckExpansion(ArgModule("[3,3]", "[3]", "1", 1, values, maximum));
    CheckExpansion(ArgModule("[3,3]", "[3]", "0", 0, values, maximum));
    CheckExpansion(ArgModule("[3,3]", "[]", "0,1", 1, values, maximum));
  }
}

TEST_F(NpuVariadicReduceExpanderTest, MixedReducersAndInitialValues) {
  CheckExpansion(R"(
    HloModule test
    reducer {
      a = f32[] parameter(0)
      b = s32[] parameter(1)
      c = f32[] parameter(2)
      d = s32[] parameter(3)
      sum = f32[] add(a, c)
      max = s32[] maximum(b, d)
      ROOT result = (f32[], s32[]) tuple(sum, max)
    }
    ENTRY main {
      x = f32[5] constant({1, 2, 3, 4, 5})
      y = s32[5] constant({4, 8, 2, 1, 0})
      init_x = f32[] constant(3)
      init_y = s32[] constant(10)
      ROOT result = (f32[], s32[]) reduce(x, y, init_x, init_y), dimensions={0}, to_apply=reducer
    }
  )");
}

TEST_F(NpuVariadicReduceExpanderTest, EmptyReductionReturnsInitialValues) {
  CheckExpansion(ArgModule("[0]", "[]", "0", 0, "{}", true));
}

TEST_F(NpuVariadicReduceExpanderTest, RefusesNonElementwiseReducer) {
  constexpr char kHlo[] = R"(
    HloModule test
    identity {
      ROOT p = f32[] parameter(0)
    }
    reducer {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      c = f32[] parameter(2)
      d = f32[] parameter(3)
      called = f32[] call(a), to_apply=identity
      ROOT result = (f32[], f32[]) tuple(called, b)
    }
    ENTRY main {
      x = f32[6] parameter(0)
      zero = f32[] constant(0)
      ROOT result = (f32[], f32[]) reduce(x, x, zero, zero), dimensions={0}, to_apply=reducer
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  auto result = NpuVariadicReduceExpander().Run(module.get());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace xla
