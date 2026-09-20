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

#include "./test_mair_gen.h"

using namespace mlir;
using namespace mlir::air;

void GraphGenTester::buildGeluModule(mlir::ModuleOp& module,
                                     mlir::MLIRContext& context) {
  // Input and output shapes
  std::vector<int> broadcastTensorShape{3};

  // Create function: accept tensor input and return tensor
  auto f32Type = mlir::Float32Type::get(&context);
  auto inputOutputTensorType = mlir::RankedTensorType::get({3}, f32Type);
  auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType},
                                          {inputOutputTensorType});
  auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                         "main", funcType);

  // Create function body
  auto* block = func.addEntryBlock();
  mlir::OpBuilder builder(block, block->begin());
  auto loc = mlir::UnknownLoc::get(&context);

  // scalar type
  auto scalarType = mlir::RankedTensorType::get({}, f32Type);

  auto i32Type = builder.getI32Type();

  mlir::RankedTensorType shapeTensorType =
      mlir::RankedTensorType::get({1}, i32Type);

  std::vector<int32_t> shapeValue{3};

  auto shapeAttr = mlir::DenseIntElementsAttr::get(shapeTensorType, shapeValue);

  mlir::Value broadcastShapeValue =
      ConstantOp::create(builder, loc, shapeTensorType, shapeAttr).getResult();

  // Get input parameter x: tensor<10x20xf32>
  auto x = block->getArgument(0);

  auto x_squared = MulOp::create(builder, loc, inputOutputTensorType, x, x);
  auto x_cube =
      MulOp::create(builder, loc, inputOutputTensorType, x_squared, x);

  auto a_scalar = ConstantOp::create(builder, loc, scalarType,
                                     builder.getF32FloatAttr(0.044715f));
  auto a_scalar_tensor = BroadcastOp::create(
      builder, loc, inputOutputTensorType, a_scalar, broadcastShapeValue);

  auto x3_mul_scalar =
      MulOp::create(builder, loc, inputOutputTensorType,
                    a_scalar_tensor.getResult(), x_cube.getResult());

  auto x3_mul_scalar_add_x = AddOp::create(builder, loc, inputOutputTensorType,
                                           x3_mul_scalar.getResult(), x);

  auto scalar_sqrt_2_dev_pi = ConstantOp::create(
      builder, loc, scalarType, builder.getF32FloatAttr(0.797884583f));
  auto scalar_sqrt_2_dev_pi_tensor =
      BroadcastOp::create(builder, loc, inputOutputTensorType,
                          scalar_sqrt_2_dev_pi, broadcastShapeValue);

  auto x_calc_1 = MulOp::create(builder, loc, inputOutputTensorType,
                                scalar_sqrt_2_dev_pi_tensor.getResult(),
                                x3_mul_scalar_add_x.getResult());

  auto tanh_x_calc_1 =
      TanhOp::create(builder, loc, inputOutputTensorType, x_calc_1.getResult());

  auto scalar_1 = ConstantOp::create(builder, loc, scalarType,
                                     builder.getF32FloatAttr(1.0f));
  auto scalar_1_tensor = BroadcastOp::create(
      builder, loc, inputOutputTensorType, scalar_1, broadcastShapeValue);

  auto x_calc_2 =
      AddOp::create(builder, loc, inputOutputTensorType,
                    scalar_1_tensor.getResult(), tanh_x_calc_1.getResult());

  auto scalar_1_dev_2 = ConstantOp::create(builder, loc, scalarType,
                                           builder.getF32FloatAttr(0.5f));
  auto scalar_1_dev_2_tensor = BroadcastOp::create(
      builder, loc, inputOutputTensorType, scalar_1_dev_2, broadcastShapeValue);

  auto result = MulOp::create(
      builder, loc, inputOutputTensorType,
      MulOp::create(builder, loc, inputOutputTensorType,
                    scalar_1_dev_2_tensor.getResult(), x_calc_2.getResult()),
      x);

  // Return results
  mlir::func::ReturnOp::create(builder, loc, result.getResult());

  // Add functions to modules
  module.getBody()->push_back(func);

  return;
}

void GraphGenTester::buildAddMulModule(mlir::ModuleOp& module,
                                       mlir::MLIRContext& context) {
  std::cout << "Start build addNul mlir" << std::endl;
  // Input and output shapes
  std::vector<int> broadcastTensorShape{10, 20};

  // Create function: accept tensor input and return tensor
  auto f32Type = mlir::Float32Type::get(&context);
  auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
  auto funcType = mlir::FunctionType::get(
      &context, {inputOutputTensorType, inputOutputTensorType},
      {inputOutputTensorType});
  auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                         "main", funcType);

  // Create function body
  auto* block = func.addEntryBlock();
  mlir::OpBuilder builder(block, block->begin());
  auto loc = mlir::UnknownLoc::get(&context);

  // scalar type
  auto scalarType = mlir::RankedTensorType::get({}, f32Type);

  // The target shape for subsequent broadcast operations
  auto broadcastShapeValue = func.getArgument(0);

  // Get input parameter x: tensor<10x20xf32>
  auto x1 = block->getArgument(0);
  auto x2 = block->getArgument(1);

  // add
  auto add = AddOp::create(builder, loc, inputOutputTensorType, x1, x2);

  auto a_scalar = ConstantOp::create(builder, loc, scalarType,
                                     builder.getF32FloatAttr(0.044715f));
  auto a_scalar_tensor = BroadcastOp::create(
      builder, loc, inputOutputTensorType, a_scalar, broadcastShapeValue);

  auto result = MulOp::create(builder, loc, inputOutputTensorType,
                              a_scalar_tensor.getResult(), add.getResult());

  // Return results
  mlir::func::ReturnOp::create(builder, loc, result.getResult());

  // Add functions to modules
  module.getBody()->push_back(func);

  return;
}

void GraphGenTester::buildAddModule(mlir::ModuleOp& module,
                                    mlir::MLIRContext& context) {
  // Input and output shapes
  std::cout << "Start build add mlir" << std::endl;
  // Create function: accept tensor input and return tensor
  auto f32Type = mlir::Float32Type::get(&context);
  auto tensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
  auto funcType =
      mlir::FunctionType::get(&context, {tensorType, tensorType}, {tensorType});
  auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                         "main", funcType);

  // Create function body
  auto* block = func.addEntryBlock();
  mlir::OpBuilder builder(block, block->begin());
  auto loc = mlir::UnknownLoc::get(&context);

  // Get input parameter x: tensor<10x20xf32>
  auto x1 = block->getArgument(0);
  auto x2 = block->getArgument(1);

  // add
  auto result = AddOp::create(builder, loc, tensorType, x1, x2);

  // Return results
  mlir::func::ReturnOp::create(builder, loc, result.getResult());

  // Add functions to modules
  module.getBody()->push_back(func);

  return;
}

void GraphGenTester::buildAEmptyModule(mlir::ModuleOp& module,
                                       mlir::MLIRContext& context) {
  // Input and output shapes
  std::cout << "Start build empty mlir" << std::endl;
  // Create function: accept tensor input and return tensor
  auto f32Type = mlir::Float32Type::get(&context);
  auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
  auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType},
                                          {inputOutputTensorType});
  auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                         "main", funcType);

  // Create function body
  auto* block = func.addEntryBlock();
  mlir::OpBuilder builder(block, block->begin());
  auto loc = mlir::UnknownLoc::get(&context);

  // Get input parameter x: tensor<10x20xf32>
  auto x1 = block->getArgument(0);

  // Return results
  mlir::func::ReturnOp::create(builder, loc, x1);

  // Add functions to modules
  module.getBody()->push_back(func);

  return;
}

void GraphGenTester::buildConstModule(mlir::ModuleOp& module,
                                      mlir::MLIRContext& context) {
  // Input and output shapes
  std::cout << "Start build Const mlir" << std::endl;
  // Create function: accept tensor input and return tensor
  // 1. Floating point type
  auto f32Type = mlir::Float32Type::get(&context);
  auto f64Type = mlir::Float64Type::get(&context);

  // 2. Signed integer type (default)
  auto i8Type = mlir::IntegerType::get(&context, 8);    // signed i8
  auto i16Type = mlir::IntegerType::get(&context, 16);  // signed i16
  auto i32Type = mlir::IntegerType::get(&context, 32);  // signed i32
  auto i64Type = mlir::IntegerType::get(&context, 64);  // signed i64

  // 3. Unsigned integer type (sign attribute needs to be specified)
  auto ui8Type =
      mlir::IntegerType::get(&context, 8, mlir::IntegerType::Unsigned);
  auto ui16Type =
      mlir::IntegerType::get(&context, 16, mlir::IntegerType::Unsigned);
  auto ui32Type =
      mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned);
  auto ui64Type =
      mlir::IntegerType::get(&context, 64, mlir::IntegerType::Unsigned);

  // 4. Create a scalar tensor type (0 dimension)
  auto _0scalarF32Type = mlir::RankedTensorType::get({}, f32Type);
  auto _0scalarF64Type = mlir::RankedTensorType::get({}, f64Type);
  auto _0scalarI8Type = mlir::RankedTensorType::get({}, i8Type);
  auto _0scalarUI8Type = mlir::RankedTensorType::get({}, ui8Type);
  auto _0scalarI16Type = mlir::RankedTensorType::get({}, i16Type);
  auto _0scalarUI16Type = mlir::RankedTensorType::get({}, ui16Type);
  auto _0scalarI32Type = mlir::RankedTensorType::get({}, i32Type);
  auto _0scalarUI32Type = mlir::RankedTensorType::get({}, ui32Type);
  auto _0scalarI64Type = mlir::RankedTensorType::get({}, i64Type);
  auto _0scalarU642Type = mlir::RankedTensorType::get({}, ui64Type);

  auto _1scalarF32Type = mlir::RankedTensorType::get({1}, f32Type);
  auto _1scalarF64Type = mlir::RankedTensorType::get({1}, f64Type);
  auto _1scalarI8Type = mlir::RankedTensorType::get({1}, i8Type);
  auto _1scalarUI8Type = mlir::RankedTensorType::get({1}, ui8Type);
  auto _1scalarI16Type = mlir::RankedTensorType::get({1}, i16Type);
  auto _1scalarUI16Type = mlir::RankedTensorType::get({1}, ui16Type);
  auto _1scalarI32Type = mlir::RankedTensorType::get({1}, i32Type);
  auto _1scalarUI32Type = mlir::RankedTensorType::get({1}, ui32Type);
  auto _1scalarI64Type = mlir::RankedTensorType::get({1}, i64Type);
  auto _1scalarU642Type = mlir::RankedTensorType::get({1}, ui64Type);

  auto inputOutputTensorType = mlir::RankedTensorType::get({10}, f32Type);
  auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType},
                                          {inputOutputTensorType});
  auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                         "main", funcType);

  // Create function body
  auto* block = func.addEntryBlock();
  mlir::OpBuilder builder(block, block->begin());
  auto loc = mlir::UnknownLoc::get(&context);

  // Get input parameter x: tensor<10x20xf32>
  auto x1 = block->getArgument(0);

  // scalar
  auto _0_const_scalarF32Type = ConstantOp::create(
      builder, loc, _0scalarF32Type,
      mlir::DenseElementsAttr::get(
          _0scalarF32Type,
          llvm::ArrayRef<float>{static_cast<float>(0.044715f)}));
  auto _0_const_scalarF64Type = ConstantOp::create(
      builder, loc, _0scalarF64Type,
      mlir::DenseElementsAttr::get(
          _0scalarF64Type,
          llvm::ArrayRef<float>{static_cast<double>(0.044715)}));
  auto _0_const_scalarI8Type = ConstantOp::create(
      builder, loc, _0scalarI8Type,
      mlir::DenseElementsAttr::get(
          _0scalarI8Type, llvm::ArrayRef<float>{static_cast<int8_t>(42)}));
  auto _0_const_scalarI16Type = ConstantOp::create(
      builder, loc, _0scalarI16Type,
      mlir::DenseElementsAttr::get(
          _0scalarI16Type, llvm::ArrayRef<float>{static_cast<int16_t>(42)}));
  auto _0_const_scalarI32Type = ConstantOp::create(
      builder, loc, _0scalarI32Type,
      mlir::DenseElementsAttr::get(
          _0scalarI32Type, llvm::ArrayRef<float>{static_cast<int32_t>(42)}));
  auto _0_const_scalarI64Type = ConstantOp::create(
      builder, loc, _0scalarI64Type,
      mlir::DenseElementsAttr::get(
          _0scalarI64Type, llvm::ArrayRef<float>{static_cast<int64_t>(42)}));
  auto _0_const_scalarUI8Type = ConstantOp::create(
      builder, loc, _0scalarUI8Type,
      mlir::DenseElementsAttr::get(
          _0scalarUI8Type, llvm::ArrayRef<float>{static_cast<uint8_t>(42)}));
  auto _0_const_scalarUI16Type = ConstantOp::create(
      builder, loc, _0scalarUI16Type,
      mlir::DenseElementsAttr::get(
          _0scalarUI16Type, llvm::ArrayRef<float>{static_cast<uint16_t>(42)}));
  auto _0_const_scalarUI32Type = ConstantOp::create(
      builder, loc, _0scalarUI32Type,
      mlir::DenseElementsAttr::get(
          _0scalarUI32Type, llvm::ArrayRef<float>{static_cast<uint32_t>(42)}));
  auto _0_const_scalarU642Type = ConstantOp::create(
      builder, loc, _0scalarU642Type,
      mlir::DenseElementsAttr::get(
          _0scalarU642Type, llvm::ArrayRef<float>{static_cast<uint64_t>(42)}));

  auto _1_const_scalarF32Type = ConstantOp::create(
      builder, loc, _1scalarF32Type,
      mlir::DenseElementsAttr::get(
          _1scalarF32Type,
          llvm::ArrayRef<float>{static_cast<float>(0.044715f)}));
  auto _1_const_scalarF64Type = ConstantOp::create(
      builder, loc, _1scalarF64Type,
      mlir::DenseElementsAttr::get(
          _1scalarF64Type,
          llvm::ArrayRef<float>{static_cast<double>(0.044715)}));
  auto _1_const_scalarI8Type = ConstantOp::create(
      builder, loc, _1scalarI8Type,
      mlir::DenseElementsAttr::get(
          _1scalarI8Type, llvm::ArrayRef<float>{static_cast<int8_t>(42)}));
  auto _1_const_scalarI16Type = ConstantOp::create(
      builder, loc, _1scalarI16Type,
      mlir::DenseElementsAttr::get(
          _1scalarI16Type, llvm::ArrayRef<float>{static_cast<int16_t>(42)}));
  auto _1_const_scalarI32Type = ConstantOp::create(
      builder, loc, _1scalarI32Type,
      mlir::DenseElementsAttr::get(
          _1scalarI32Type, llvm::ArrayRef<float>{static_cast<int32_t>(42)}));
  auto _1_const_scalarI64Type = ConstantOp::create(
      builder, loc, _1scalarI64Type,
      mlir::DenseElementsAttr::get(
          _1scalarI64Type, llvm::ArrayRef<float>{static_cast<int64_t>(42)}));
  auto _1_const_scalarUI8Type = ConstantOp::create(
      builder, loc, _1scalarUI8Type,
      mlir::DenseElementsAttr::get(
          _1scalarUI8Type, llvm::ArrayRef<float>{static_cast<uint8_t>(42)}));
  auto _1_const_scalarUI16Type = ConstantOp::create(
      builder, loc, _1scalarUI16Type,
      mlir::DenseElementsAttr::get(
          _1scalarUI16Type, llvm::ArrayRef<float>{static_cast<uint16_t>(42)}));
  auto _1_const_scalarUI32Type = ConstantOp::create(
      builder, loc, _1scalarUI32Type,
      mlir::DenseElementsAttr::get(
          _1scalarUI32Type, llvm::ArrayRef<float>{static_cast<uint32_t>(42)}));
  auto _1_const_scalarU642Type = ConstantOp::create(
      builder, loc, _1scalarU642Type,
      mlir::DenseElementsAttr::get(
          _1scalarU642Type, llvm::ArrayRef<float>{static_cast<uint64_t>(42)}));

  // Return results
  mlir::func::ReturnOp::create(builder, loc, x1);

  // Add functions to modules
  module.getBody()->push_back(func);

  return;
}

Status GraphGenTester::Convert(mlir::ModuleOp& module, ge::GraphPtr& graphPtr) {
  Exporter exporter;
  return exporter.Convert(module, graphPtr);
}

int main(int argc, char** argv) {
  static mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::air::AirDialect>();

  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  GraphGenTester::buildConstModule(module, context);

  std::cout << "Generated MLIR:\n";
  module.print(llvm::outs());
  std::cout << std::endl;

  ge::GraphPtr graphPtr = std::make_shared<ge::Graph>("XLAGraph");
  ;
  if (!module) {
    log("null module.");
  }

  GraphGenTester::Convert(module, graphPtr);
  std::cout << "Finished convert to GE Grap:\n";
  std::cout << std::endl;

  return 0;
}