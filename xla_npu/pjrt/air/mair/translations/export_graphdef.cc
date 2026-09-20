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

#include "./export_graphdef.h"
#include "./status.h"
#include "./convert_type.h"
#include "./convert_tensor.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <iomanip>

#include "../ir/mair_ops.h"
#include <utility>
#include <map>
namespace mlir::air {
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::Value;

const std::unordered_map<std::string, std::vector<std::string>>
    Exporter::inputNameMap = {
        {"Add", {"x1", "x2"}},
        {"Mul", {"x1", "x2"}},
        {"Tanh", {"x"}},
        {"BroadcastTo", {"x", "shape"}},
        {"Div", {"x1", "x2"}},
        {"Exp", {"x"}},
        {"Maximum", {"x1", "x2"}},
        {"Neg", {"x"}},
        {"Sub", {"x1", "x2"}},
        {"MatMul", {"x1", "x2"}},
        {"Reshape", {"x", "shape"}},
        {"ReduceSum", {"x", "axes"}},
        {"ReduceMax", {"x", "axes"}},
        {"ReduceMin", {"x", "axes"}},
        {"ConcatV2", {"x", "concat_dim"}},
        {"Cast", {"x"}},
        {"UnsqueezeV3", {"x", "axes"}},
        {"SqueezeV3", {"x", "axes"}},
        {"LogicalAnd", {"x1", "x2"}},
        {"Equal", {"x1", "x2"}},
        {"NotEqual", {"x1", "x2"}},
        {"Greater", {"x1", "x2"}},
        {"GreaterEqual", {"x1", "x2"}},
        {"Less", {"x1", "x2"}},
        {"LessEqual", {"x1", "x2"}},
        {"Cos", {"x"}},
        {"Gather", {"x", "indices"}},
        {"GatherV2", {"x", "indices", "axis", "batch_dims"}},
        {"Pow", {"x1", "x2"}},
        {"Rsqrt", {"x"}},
        {"Select", {"condition", "x1", "x2"}},
        {"Sin", {"x"}},
        {"Slice", {"x", "offsets", "size"}},
        {"Transpose", {"x", "perm"}},
        {"ReduceAll", {"x", "axes"}},
        {"ReduceAny", {"x", "axes"}},
        {"BatchMatMul", {"x1", "x2"}},
};

const std::map<ge::DataType, std::string> DataTypeNames = {
    {ge::DataType::DT_FLOAT, "DT_FLOAT         "},    // float type
    {ge::DataType::DT_FLOAT16, "DT_FLOAT16       "},  // fp16 type
    {ge::DataType::DT_INT8, "DT_INT8          "},     // int8 type
    {ge::DataType::DT_INT32, "DT_INT32         "},    // int32 type
    {ge::DataType::DT_UINT8, "DT_UINT8         "},    // uint8 type
                                                      // reserved
    {ge::DataType::DT_INT16, "DT_INT16         "},    // int16 type
    {ge::DataType::DT_UINT16, "DT_UINT16        "},   // uint16 type
    {ge::DataType::DT_UINT32, "DT_UINT32        "},   // unsigned int32
    {ge::DataType::DT_INT64, "DT_INT64         "},    // int64 type
    {ge::DataType::DT_UINT64, "DT_UINT64        "},   // unsigned int64
    {ge::DataType::DT_DOUBLE, "DT_DOUBLE        "},   // double type
    {ge::DataType::DT_BOOL, "DT_BOOL          "},     // bool type
    {ge::DataType::DT_STRING, "DT_STRING        "},   // string type
    {ge::DataType::DT_DUAL_SUB_INT8,
     "DT_DUAL_SUB_INT8 "},  // dual output int8 type
    {ge::DataType::DT_DUAL_SUB_UINT8,
     "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {ge::DataType::DT_COMPLEX64, "DT_COMPLEX64     "},   // complex64 type
    {ge::DataType::DT_COMPLEX128, "DT_COMPLEX128    "},  // complex128 type
    {ge::DataType::DT_QINT8, "DT_QINT8         "},       // qint8 type
    {ge::DataType::DT_QINT16, "DT_QINT16        "},      // qint16 type
    {ge::DataType::DT_QINT32, "DT_QINT32        "},      // qint32 type
    {ge::DataType::DT_QUINT8, "DT_QUINT8        "},      // quint8 type
    {ge::DataType::DT_QUINT16, "DT_QUINT16       "},     // quint16 type
    {ge::DataType::DT_RESOURCE, "DT_RESOURCE      "},    // resource type
    {ge::DataType::DT_STRING_REF, "DT_STRING_REF    "},  // string ref type
    {ge::DataType::DT_DUAL, "DT_DUAL          "},        // dual output type
    {ge::DataType::DT_VARIANT, "DT_VARIANT       "},     // dt_variant type
    {ge::DataType::DT_BF16, "DT_BF16          "},        // bf16 type
    {ge::DataType::DT_UNDEFINED,
     "DT_UNDEFINED     "},  // Used to indicate a DataType field has not been set.
    {ge::DataType::DT_INT4, "DT_INT4          "},         // int4 type
    {ge::DataType::DT_UINT1, "DT_UINT1         "},        // uint1 type
    {ge::DataType::DT_INT2, "DT_INT2          "},         // int2 type
    {ge::DataType::DT_UINT2, "DT_UINT2         "},        // uint2 type
    {ge::DataType::DT_COMPLEX32, "DT_COMPLEX32     "},    // complex32 type
    {ge::DataType::DT_HIFLOAT8, "DT_HIFLOAT8      "},     // hifloat8 type
    {ge::DataType::DT_FLOAT8_E5M2, "DT_FLOAT8_E5M2   "},  // float8_e5m2 type
    {ge::DataType::DT_FLOAT8_E4M3FN,
     "DT_FLOAT8_E4M3FN "},                                // float8_e4m3fn type
    {ge::DataType::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0   "},  // float8_e8m0 type
    {ge::DataType::DT_FLOAT6_E3M2, "DT_FLOAT6_E3M2   "},  // float6_e3m2 type
    {ge::DataType::DT_FLOAT6_E2M3, "DT_FLOAT6_E2M3   "},  // float6_e2m3 type
    {ge::DataType::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1   "},  // float4_e2m1 type
    {ge::DataType::DT_FLOAT4_E1M2, "DT_FLOAT4_E1M2   "},  // float4_e1m2 type
    {ge::DataType::DT_MAX,
     "DT_MAX           "}  // Mark the boundaries of data types
};

Status Exporter::Convert(mlir::ModuleOp& module, ge::GraphPtr& graphPtr) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "Start to covert module to ge graph.");
  std::optional<mlir::func::FuncOp> entry_func;
  mlir::StringAttr entry_func_id =
      mlir::StringAttr::get(module.getContext(), "main");
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "External functions not supported.");
      return Status::OK();
    }
    if (function.getName() == entry_func_id) {
      entry_func.emplace(function);
    }
  }

  if (!entry_func.has_value()) {
    NPU_LOG_FORMAT_PRINT(ERROR, "entry function `main` must be present.");
    return Status::OK();
  }

  // start to build ge graph
  mlir::func::FuncOp function = entry_func.value();
  mlir::Block& entryBlock = function.getBody().front();
  NPU_LOG_FORMAT_PRINT(
      DEBUG, "Get mlir function done. Start to get arguments of the function.");

  // build input arguments
  for (int i = 0; i < entryBlock.getNumArguments(); i++) {
    mlir::BlockArgument arg = entryBlock.getArgument(i);
    std::string mlirSymbol = "%arg" + std::to_string(arg.getArgNumber());
    std::string realSymbol = "Data_" + std::to_string(i);
    NPU_LOG_FORMAT_PRINT(DEBUG, "Add arguments:%s ; parse to :%s",
                         mlirSymbol.c_str(), realSymbol.c_str());

    // Get parameter type and shape
    mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(arg.getType());

    ge::DataType dtype;
    ConvertToDataType(arg_type.getElementType(), &dtype);

    auto shape = ge::Shape();
    ConvertToShape(arg_type, &shape);

    auto format = ge::FORMAT_ND;
    ge::TensorDesc data_desc(shape, format, dtype);
    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Gen ArgumentNode Data:[%s], InputDesc/OutputDesc[%s]",
        realSymbol.c_str(), mlir::debugString(arg_type).c_str());
    auto data = ge::OperatorFactory::CreateOperator(realSymbol.c_str(), "Data");
    data.SetAttr("index", i);
    data.UpdateInputDesc("x", data_desc);
    data.UpdateOutputDesc("y", data_desc);

    // Record input parameter information
    inputInfos.push_back({mlirSymbol, data, data_desc});

    // Create mapping
    symbolMap[mlirSymbol] = {
        .orgSymbol = mlirSymbol, .geOp = data, .tensorDesc = data_desc};

    NPU_LOG_FORMAT_PRINT(DEBUG, "mlirSymbol:[%s], realSymbol[%s]",
                         mlirSymbol.c_str(), realSymbol.c_str());
  }

  std::vector<ge::Operator> inputs;
  for (const auto& inputInfo : inputInfos) {
    inputs.emplace_back(inputInfo.geOp);
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "--Set graph inputs. The inputs size =[%d]",
                       inputs.size());
  for (const auto& inputOp : inputs) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "--Input operator name=[%s]",
                         std::string(getOpName(inputOp)).c_str());
  }

  // ==================================
  // build ge oprators
  // ==================================
  // Operator counter (used to generate unique symbols)
  int opCounter = 0;
  std::vector<ge::Operator> outputs;

  for (mlir::Operation& op : entryBlock.getOperations()) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "Start to deal with the op:[%s]",
                         op.getName().getStringRef().str().c_str());
    if (mlir::isa<mlir::func::ReturnOp>(op)) {
      SymbolInfo symbolInfo = processReturnOp(op);
      NPU_LOG_FORMAT_PRINT(DEBUG, "Return op is:[%s]",
                           std::string(getOpName(symbolInfo.geOp)).c_str());
      outputs.emplace_back(symbolInfo.geOp);
      continue;
    }

    // Handle common operations
    processOperation(op, opCounter, graphPtr);
    opCounter++;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "--Set graph outputs. The outputs size =[%d]",
                       outputs.size());
  for (const auto& outputOp : outputs) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "--Output operator name=[%s]",
                         std::string(getOpName(outputOp)).c_str());
  }

  graphPtr->SetInputs(inputs).SetOutputs(outputs);
  NPU_LOG_FORMAT_PRINT(DEBUG, "Finish All Operator");
  std::vector<GNode> nodes = graphPtr->GetAllNodes();
  NPU_LOG_FORMAT_PRINT(DEBUG, "Get All GNodes size =[%d]", nodes.size());
  for (size_t i = 0; i < nodes.size(); i++) {
    ge::AscendString nodeName;
    nodes.at(i).GetName(nodeName);
    NPU_LOG_FORMAT_PRINT(DEBUG, "Nodes:[%d], name =[%s]", (int)i,
                         nodeName.GetString());
  }
  return Status::OK();
}

Exporter::SymbolInfo Exporter::processReturnOp(mlir::Operation& returnOp) {
  mlir::func::ReturnOp returnOpCast =
      mlir::cast<mlir::func::ReturnOp>(returnOp);

  mlir::Value returnValue = returnOpCast.getOperand(0);
  NPU_LOG_FORMAT_PRINT(DEBUG, "returnOp operand:[%s]",
                       mlir::debugString(returnValue).c_str());
  std::string mlirSymbol = getMlirSymbol(returnValue);

  // Find the true output symbol
  auto it = symbolMap.find(mlirSymbol);
  if (it == symbolMap.end()) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Find [%s] failed !!!", mlirSymbol.c_str());
  }

  const SymbolInfo& symbolInfo = it->second;
  printSymbolInfo(symbolInfo);
  NPU_LOG_FORMAT_PRINT(DEBUG, "graph output mlirSymbol:[%s]",
                       mlirSymbol.c_str());
  return symbolInfo;
}

void Exporter::printSymbolInfo(const SymbolInfo& symbolInfo) {
  NPU_LOG_FORMAT_PRINT(DEBUG, "SymbolInfo = mlirSymbol:[%s] opName:[%s]",
                       symbolInfo.orgSymbol.c_str(),
                       getOpName(symbolInfo.geOp));
}

bool Exporter::processOperation(mlir::Operation& mlirOp, int opCounter,
                                ge::GraphPtr& graphPtr) {
  // 1. Get the operation name (remove the "mair." prefix)
  std::string opName = mlirOp.getName().getStringRef().str();
  if (opName.find("mair.") != 0) {
    NPU_LOG_FORMAT_PRINT(WARN, "Skip non-mair operations, op is:[%s]",
                         opName.c_str());
    return true;
  }

  std::string opType = opName.substr(5);  // Remove "mair."

  // 2. Create target operations
  ge::Operator geOp = ge::OperatorFactory::CreateOperator(
      (opType + std::to_string(opCounter)).c_str(), opType.c_str());
  // 3. Process input
  if (!processOperationInputs(mlirOp, geOp, opType)) {
    return false;
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "Input number of this op is:[%d]",
                       geOp.GetInputsSize());

  // 4. Process output
  if (!processOperationOutputs(mlirOp, geOp, opCounter, opType)) {
    return false;
  }

  NPU_LOG_FORMAT_PRINT(DEBUG, "Output number of this op is:[%d]",
                       geOp.GetOutputsSize());

  return true;
}
bool Exporter::processConcatV2Inputs(mlir::Operation& mlirOp,
                                     ge::Operator& geOp, std::string opType) {
  auto concatV2Op = dyn_cast<mlir::air::ConcatV2Op>(mlirOp);
  mlir::ValueRange x_value = concatV2Op.getX();
  geOp.DynamicInputRegisterByIndex("x", x_value.size(), 0);
  unsigned i = 0;
  for (mlir::Value val : x_value) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "----------trans the input x of ConcatV2: %s",
                         mlir::debugString(val).c_str());
    std::string mlirSymbol = getMlirSymbol(val);
    if (mlirSymbol.empty()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Error: Unable to obtain operand symbols");
      return false;
    }
    auto it = symbolMap.find(mlirSymbol);
    if (it == symbolMap.end()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Error: Value symbol mapping not found: %s",
                           mlirSymbol.c_str());
      return false;
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Find input :");
    printSymbolInfo(it->second);
    ge::Operator inputOp = it->second.geOp;
    geOp.SetInput(i++, inputOp, 0U);
  }

  mlir::Value dim_value = concatV2Op.getConcatDim();
  const std::string* inputName = getGeOpInputName(opType, 1);
  NPU_LOG_FORMAT_PRINT(DEBUG, "----------trans the input dim of ConcatV2: %s",
                       mlir::debugString(dim_value).c_str());

  std::string mlirSymbol = getMlirSymbol(dim_value);
  if (mlirSymbol.empty()) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Error: Unable to obtain operand symbols");
    return false;
  }

  auto it = symbolMap.find(mlirSymbol);
  if (it == symbolMap.end()) {
    NPU_LOG_FORMAT_PRINT(ERROR, "Error: DIM symbol mapping not found: %s",
                         mlirSymbol.c_str());
    return false;
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "Find input :");
  printSymbolInfo(it->second);
  ge::Operator inputOp = it->second.geOp;
  geOp.SetInput(inputName->c_str(), inputOp);
  auto nAttr = mlirOp.getAttr("N");
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(nAttr)) {
    geOp.SetAttr("N", intAttr.getInt());
    return true;
  }
  NPU_LOG_FORMAT_PRINT(ERROR, "ERROR: N value is illegal.");
  return false;
}

bool Exporter::processOperationInputs(mlir::Operation& mlirOp,
                                      ge::Operator& geOp, std::string opType) {
  if (mlirOp.getName().getStringRef() == "mair.ConcatV2") {
    return processConcatV2Inputs(mlirOp, geOp, opType);
  }

  unsigned numOperands = mlirOp.getNumOperands();
  for (unsigned i = 0; i < numOperands; i++) {
    mlir::Value operand = mlirOp.getOperand(i);
    NPU_LOG_FORMAT_PRINT(DEBUG, "trans the input: %s",
                         mlir::debugString(operand).c_str());

    std::string mlirSymbol = getMlirSymbol(operand);
    if (mlirSymbol.empty()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Error: Unable to obtain operand symbols");
      return false;
    }

    auto it = symbolMap.find(mlirSymbol);
    if (it == symbolMap.end()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Error: Symbol mapping not found: %s",
                           mlirSymbol.c_str());
      return false;
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Find input :");
    printSymbolInfo(it->second);

    const std::string* inputName = getGeOpInputName(opType, i);

    ge::Operator inputOp = it->second.geOp;
    geOp.SetInput(inputName->c_str(), inputOp);

    NPU_LOG_FORMAT_PRINT(
        DEBUG, "Set opInput[%d]: origin=%s, inputName=[%s] opName=[%s]", i,
        mlirSymbol.c_str(), inputName->c_str(), getOpName(inputOp));
  }

  return true;
}

const char* Exporter::getOpName(const ge::Operator& geOp) {
  ge::AscendString opName;
  geOp.GetName(opName);
  return opName.GetString();
}

// Processing operation output
bool Exporter::processOperationOutputs(mlir::Operation& mlirOp,
                                       ge::Operator& geOp, int opCounter,
                                       std::string opType) {
  unsigned numResults = mlirOp.getNumResults();

  for (unsigned i = 0; i < numResults; i++) {
    mlir::Value result = mlirOp.getResult(i);

    // Get MLIR symbols
    std::string mlirSymbol = getMlirSymbol(result);
    if (mlirSymbol.empty()) {
      NPU_LOG_FORMAT_PRINT(ERROR, "Error: Unable to get result symbols");
      return false;
    }
    if (opType == "Const") {
      mlir::Attribute valueAttr = mlirOp.getAttr("value");
      if (auto denseAttr = mlir::dyn_cast<mlir::ElementsAttr>(valueAttr)) {
        ge::Tensor valueTensor;
        if (!ConvertToTensor(denseAttr, &valueTensor).ok()) {
          NPU_LOG_FORMAT_PRINT(ERROR, "Fail to convert to tensor!!");
        }
        ge::TensorDesc desc = valueTensor.GetTensorDesc();
        ge::DataType dt = desc.GetDataType();
        ge::Shape shape = desc.GetShape();
        std::vector<int64_t> dims = shape.GetDims();

        std::ostringstream oss;
        oss << "Const ge tensor : "
            << "DataType:[" << DataTypeNames.find(dt)->second
            << "]; Shape dims=[";
        for (auto dim : dims) {
          oss << dim << ",";
        }
        oss << "]";
        NPU_LOG_FORMAT_PRINT(DEBUG, "%s", oss.str().c_str());

        geOp.SetAttr("value", valueTensor);
        geOp.UpdateOutputDesc("y", desc);
      } else {
        NPU_LOG_FORMAT_PRINT(ERROR, "Failed Const operatro. Not ElementsAttr");
      }
    }

    // Create mapping
    symbolMap[mlirSymbol] = {
        .orgSymbol = mlirSymbol,
        .geOp = geOp,
    };

    NPU_LOG_FORMAT_PRINT(DEBUG, "output[%d]: MLIR symbol=%s", i,
                         mlirSymbol.c_str());
  }

  return true;
}

float Exporter::extractFloatValueSimple(mlir::Attribute valueAttr) {
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valueAttr)) {
    // Convert directly to float
    return floatAttr.getValue().convertToFloat();
  }

  // if DenseElementsAttr(scalar tensor)
  if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
    if (denseAttr.isSplat() && denseAttr.getElementType().isF32()) {
      return denseAttr.getSplatValue<float>();
    }
    // If it is f64, convert to float
    if (denseAttr.isSplat() && denseAttr.getElementType().isF64()) {
      return static_cast<float>(denseAttr.getSplatValue<double>());
    }
  }

  // if IntegerAttr (integer converted to floating point)
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
    return static_cast<float>(intAttr.getValue().getSExtValue());
  }

  std::string attrStr;
  llvm::raw_string_ostream oss(attrStr);
  oss << valueAttr;
  NPU_LOG_FORMAT_PRINT(WARN, "Fail to get value from %s", attrStr.c_str());
  return 0.0f;
}

std::string Exporter::getMlirSymbol(mlir::Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {  // input
    return "%arg" + std::to_string(blockArg.getArgNumber());
  } else if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {  // output
    mlir::Operation* definingOp = opResult.getDefiningOp();
    if (!definingOp) {
      return "";
    }

    // Find the index of the result in the operation
    unsigned resultIndex = opResult.getResultNumber();
    // Try to get symbols in standard format
    if (definingOp->getNumResults() == 1) {
      return "%" + std::to_string(getOperationNumber(definingOp));
    } else {
      return "%" + std::to_string(getOperationNumber(definingOp)) + "." +
             std::to_string(resultIndex);
    }
  }
  return "";
}

// Get the operation sequence number (used to generate symbols)
int Exporter::getOperationNumber(mlir::Operation* op) {
  // Simple implementation: traverse block count
  mlir::Block* block = op->getBlock();
  int counter = 0;

  for (mlir::Operation& blockOp : *block) {
    if (&blockOp == op) {
      return counter;
    }

    // Skip return operation
    if (!mlir::isa<mlir::func::ReturnOp>(blockOp)) {
      counter++;
    }
  }

  return -1;
}

// Generate output symbols
std::string Exporter::generateOutputSymbol(int opIndex, int outputIndex) {
  return "op_" + std::to_string(opIndex) + "_out_" +
         std::to_string(outputIndex);
}

const std::string* Exporter::getGeOpInputName(const std::string& opName,
                                              int index) {
  auto it = inputNameMap.find(opName);
  if (it == inputNameMap.end()) {
    return nullptr;
  }

  const std::vector<std::string>& inputNames = it->second;
  if (inputNames.size() >= index + 1) {
    return &inputNames[index];
  }

  return nullptr;
}

}  // namespace mlir::air