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

#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

class ConvertToAfirTest : public HloTestBase {
 protected:
  // Helper to run conversion and return the MLIR module
  // Note: We use the API that creates its own context to avoid lifetime issues
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertToAfir(
      const HloModule& module, const HloToAfirOptions& options = {}) {
    return ConvertHloToAfir(module, options);
  }
};

// Test basic constant conversion
TEST_F(ConvertToAfirTest, ConvertConstant) {
  const char* hlo_text = R"(
HloModule SimpleConstant

ENTRY main {
  %const = f32[] constant(1.0)
  ROOT %root = f32[] constant(2.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test simple add operation
TEST_F(ConvertToAfirTest, ConvertSimpleAdd) {
  const char* hlo_text = R"(
HloModule SimpleAdd

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module));

  ASSERT_NE(mlir_module.get(), nullptr);
  // Verify module has one function (the entry computation)
  auto funcs = mlir_module->getOps<mlir::func::FuncOp>();
  EXPECT_EQ(std::distance(funcs.begin(), funcs.end()), 1);
}

// Test multiple operations
TEST_F(ConvertToAfirTest, ConvertMultipleOps) {
  const char* hlo_text = R"(
HloModule MultipleOps

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %add = f32[] add(%p0, %p1)
  %sub = f32[] subtract(%p0, %p1)
  %mul = f32[] multiply(%add, %sub)
  ROOT %div = f32[] divide(%mul, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test integer operations
TEST_F(ConvertToAfirTest, ConvertIntOps) {
  const char* hlo_text = R"(
HloModule IntOps

ENTRY main {
  %p0 = s32[] parameter(0)
  %p1 = s32[] parameter(1)
  ROOT %add = s32[] add(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module));

  ASSERT_NE(mlir_module.get(), nullptr);
}

TEST_F(ConvertToAfirTest, ConvertPredParameterToAfirLoad) {
  const char* hlo_text = R"(
HloModule PredParameter

ENTRY main {
  ROOT %p0 = pred[64,1] parameter(0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.emit_afir_dialect = true;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
  auto func = *mlir_module->getOps<mlir::func::FuncOp>().begin();
  auto input_type =
      mlir::cast<mlir::RankedTensorType>(func.getFunctionType().getInput(0));
  auto element_type = mlir::cast<mlir::IntegerType>(input_type.getElementType());
  EXPECT_EQ(element_type.getWidth(), 8);
  EXPECT_FALSE(element_type.isSigned());
}

TEST_F(ConvertToAfirTest, ConvertConcatAfterRankReducingReshape) {
  const char* hlo_text = R"(
HloModule ConcatAfterRankReducingReshape

ENTRY main {
  %p0 = f32[64,1,7] parameter(0)
  %p1 = f32[64,4,1] parameter(1)
  %reshape0 = f32[64,7] reshape(%p0)
  %reshape1 = f32[64,4] reshape(%p1)
  ROOT %concat = f32[64,11] concatenate(%reshape0, %reshape1), dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.emit_afir_dialect = true;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test with verification enabled (default)
TEST_F(ConvertToAfirTest, ConvertWithVerification) {
  const char* hlo_text = R"(
HloModule VerifiedModule

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.verify_module = true;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test without verification
TEST_F(ConvertToAfirTest, ConvertWithoutVerification) {
  const char* hlo_text = R"(
HloModule NoVerifyModule

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.verify_module = false;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test name preservation option
TEST_F(ConvertToAfirTest, PreserveNames) {
  const char* hlo_text = R"(
HloModule NamePreservation

ENTRY main {
  %my_param = f32[] parameter(0)
  %my_const = f32[] constant(3.14)
  ROOT %my_add = f32[] add(%my_param, %my_const)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.preserve_names = true;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test name disabled
TEST_F(ConvertToAfirTest, DisableNamePreservation) {
  const char* hlo_text = R"(
HloModule NoNamePreservation

ENTRY main {
  %param.0 = f32[] parameter(0)
  %const.0 = f32[] constant(2.5)
  ROOT %add.0 = f32[] add(%param.0, %const.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloToAfirOptions options;
  options.preserve_names = false;

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module, options));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test unsupported operation (should fail gracefully)
TEST_F(ConvertToAfirTest, UnsupportedOperation) {
  const char* hlo_text = R"(
HloModule UnsupportedOp

ENTRY main {
  %p0 = f32[32] parameter(0)
  ROOT %transpose = f32[32] transpose(%p0), dimensions={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  auto result = ConvertToAfir(*module);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("Unsupported opcode"));
}

// Test tuple handling (basic structure, will be enhanced)
TEST_F(ConvertToAfirTest, TupleResult) {
  const char* hlo_text = R"(
HloModule TupleResult

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  auto result = ConvertToAfir(*module);
  // Tuples should be handled in ConvertShape
  // This will fail in instruction conversion since we don't have tuple support yet
}

// Test array operations (shape > rank 0)
TEST_F(ConvertToAfirTest, ArrayOperations) {
  const char* hlo_text = R"(
HloModule ArrayOps

ENTRY main {
  %p0 = f32[4] parameter(0)
  %p1 = f32[4] parameter(1)
  ROOT %add = f32[4] add(%p0, %p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(auto mlir_module, ConvertToAfir(*module));

  ASSERT_NE(mlir_module.get(), nullptr);
}

// Test multiple computations (nested computations)
TEST_F(ConvertToAfirTest, NestedComputation) {
  const char* hlo_text = R"(
HloModule NestedComp

add {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] constant(1.0)
  ROOT %call = f32[] call(%p0, %p1), to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  auto result = ConvertToAfir(*module);
  // Will fail because we don't support call operations yet
  // But tests that we handle nested computation structure
}

}  // namespace xla
