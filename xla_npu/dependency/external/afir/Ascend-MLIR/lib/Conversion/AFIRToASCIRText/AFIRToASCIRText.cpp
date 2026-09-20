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

#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "Dialect/AFIR/AFIR.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "xla/xla_npu/common/npu_config.h"

#define GEN_PASS_DECL_CONVERTAFIRTOASCIRTEXTPASS
#define GEN_PASS_DEF_CONVERTAFIRTOASCIRTEXTPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir {
namespace afir {

struct AFIRToASCIRTextPass
    : public ::impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass> {
  using Base = ::impl::ConvertAFIRToASCIRTextPassBase<AFIRToASCIRTextPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string output = convertModuleToASCIRText(module);

    if (!ascirPath.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(ascirPath, ec);
      if (ec) {
        llvm::errs() << "Error opening file '" << ascirPath
                     << "': " << ec.message() << "\n";
        return signalPassFailure();
      }
      os << output;
      os.close();
    } else {
      llvm::errs() << output;
    }
  }

 private:
  struct OpInfo {
    int apiType;
    int apiUnit;
    int computeType;
    bool isBroadcast;
  };

  // Helper function to format special floating point values
  std::string formatFloatValue(double val, Type type);
  int64_t calculateStrideForShapeIndex(int64_t index, ArrayRef<int64_t> shape);
  void generateMemQueBufOpsOutputsAttr(llvm::raw_ostream& os, int64_t tensorId,
                                       int64_t queDepth, int queBufNum,
                                       int64_t positionId, int64_t reuseId);
  OpInfo getOpInfo(StringRef opName);
  std::string convertModuleToASCIRText(ModuleOp module);
  std::string convertFuncToASCIRText(func::FuncOp func, Attribute graphAttr);
  std::string convertOpToASCIRText(
      Operation* op, int& nodeCounter,
      llvm::StringMap<std::string>& ssaToNodeName, const std::string& graphName,
      const llvm::StringMap<SmallVector<Operation*, 4>>& valueToConsumers);
  std::string generateGraphAttr(Attribute graphAttr);
  std::string generateNodeAttr(Operation* op, const std::string& nodeName,
                               int nodeCounter);
  std::string generateTensorAttr(
      Operation* op,
      const llvm::StringMap<SmallVector<Operation*, 4>>& valueToConsumers);
  std::string generateApiAttr(Operation* op);
  std::string generateIrAttrDef(Operation* op);
  std::string generateIrDef(Operation* op);
  std::string generateInputSrc(
      Operation* op, const llvm::StringMap<std::string>& ssaToNodeName);
  std::string generateDataNode(
      int nodeIndex, int argIndex, Type argType, const std::string& graphName,
      Attribute argAttr = nullptr, Attribute graphAttr = nullptr,
      const llvm::StringMap<SmallVector<Operation*, 4>>* valueToConsumers =
          nullptr,
      const std::string* argSSAName = nullptr);
  std::string generateOutputNode(int index, const std::string& lastNodeName,
                                 const std::string& graphName);
  int parseDataType(Type type);
  std::string calculateStrides(ArrayRef<int64_t> shape);
};

AFIRToASCIRTextPass::OpInfo AFIRToASCIRTextPass::getOpInfo(StringRef opName) {
  static const llvm::StringMap<OpInfo> opInfoMap = {
      {"load", {1, 2, 0, false}},        {"store", {1, 2, 1, false}},
      {"scalar", {1, 1, 11, false}},     {"index_expr", {1, 1, 11, false}},
      {"broadcast", {2, 7, 4, true}},    {"cast", {1, 5, 3, false}},
      {"select", {1, 5, 3, false}},      {"where", {1, 5, 3, false}},
      {"concat", {1, 5, 7, false}},      {"max", {1, 5, 5, false}},
      {"min", {1, 5, 3, false}},         {"sum", {1, 5, 5, false}},
      {"mean", {1, 5, 5, false}},        {"prod", {1, 5, 5, false}},
      {"any", {1, 5, 5, false}},         {"all", {1, 5, 5, false}},
      {"abs", {1, 5, 3, false}},         {"exp", {1, 5, 3, false}},
      {"ln", {1, 5, 3, false}},          {"sqrt", {1, 5, 3, false}},
      {"rsqrt", {1, 5, 3, false}},       {"reciprocal", {1, 5, 3, false}},
      {"erf", {1, 5, 3, false}},         {"tanh", {1, 5, 3, false}},
      {"relu", {1, 5, 3, false}},        {"neg", {1, 5, 3, false}},
      {"sigmoid", {1, 5, 3, false}},     {"logical_not", {1, 5, 3, false}},
      {"isnan", {1, 5, 3, false}},       {"isfinite", {1, 5, 3, false}},
      {"leaky_relu", {1, 5, 3, false}},  {"add", {1, 5, 3, false}},
      {"sub", {1, 5, 3, false}},         {"mul", {1, 5, 3, false}},
      {"div", {1, 5, 3, false}},         {"minimum", {1, 5, 3, false}},
      {"maximum", {1, 5, 3, false}},     {"truediv", {1, 5, 3, false}},
      {"pow", {1, 5, 3, false}},         {"bitwise_and", {1, 5, 3, false}},
      {"floor_div", {1, 5, 3, false}},   {"gelu", {1, 5, 3, false}},
      {"sign", {1, 5, 3, false}},        {"logical_or", {1, 5, 3, false}},
      {"logical_and", {1, 5, 3, false}}, {"ge", {1, 5, 3, false}},
      {"eq", {1, 5, 3, false}},          {"ne", {1, 5, 3, false}},
      {"gt", {1, 5, 3, false}},          {"le", {1, 5, 3, false}},
      {"lt", {1, 5, 3, false}},          {"clip_by_value", {1, 5, 3, false}},
      {"transpose", {1, 3, 6, false}}};

  auto it = opInfoMap.find(opName);
  if (it != opInfoMap.end()) {
    return it->second;
  }

  return {1, 5, 11, false};
}

std::string AFIRToASCIRTextPass::calculateStrides(ArrayRef<int64_t> shape) {
  std::string result;
  llvm::raw_string_ostream os(result);

  int rank = shape.size();
  // Calculate strides using column-major layout
  // For shape [d0, d1, d2, ...], strides are [d1*d2*..., d2*..., ..., 1]
  // Note: Size-1 dimensions are treated as broadcast dimensions with stride=0
  for (int i = 0; i < rank; i++) {
    if (shape[i] == 1) {
      // Size-1 dimension - broadcast dimension, stride is 0
      os << "      strides: \"0\"\n";
    } else {
      // Calculate stride for this dimension (column-major)
      int64_t stride = calculateStrideForShapeIndex(i, shape);
      os << "      strides: \"" << stride << "\"\n";
    }
  }

  os.flush();

  return result;
}

int64_t AFIRToASCIRTextPass::calculateStrideForShapeIndex(
    int64_t index, ArrayRef<int64_t> shape) {
  int64_t stride = 1;
  int rank = shape.size();
  for (int j = index + 1; j < rank; j++) {
    if (shape[j] > 0) {
      stride *= shape[j];
    }
  }
  return stride;
}

std::string AFIRToASCIRTextPass::convertModuleToASCIRText(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);

  Attribute graphAttr = module->getAttr("afir.asc_graph_attr");

  for (Operation& op : module.getBody()->getOperations()) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      os << convertFuncToASCIRText(func, graphAttr);
    }
  }

  return result;
}

std::string AFIRToASCIRTextPass::convertFuncToASCIRText(func::FuncOp func,
                                                        Attribute graphAttr) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string graphName = func.getName().str();

  os << "asc_graph_attr {\n";

  if (graphAttr) {
    os << generateGraphAttr(graphAttr);
  } else {
    os << "  tiling_key: -1\n";
  }

  os << "}\n";

  int nodeCounter = 0;
  llvm::StringMap<std::string> ssaToNodeName;
  llvm::StringMap<int> argToDataCounter;
  llvm::StringMap<Type> argToType;
  int dataCounter = 0;

  // Map arguments to their data counter
  llvm::StringMap<Attribute> argToAttr;

  // Get argument attributes from function
  auto argAttrsAttr = func.getArgAttrsAttr();
  if (argAttrsAttr) {
    for (auto arg : func.getArguments()) {
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << arg.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;
      argToDataCounter[ssaName] = dataCounter;
      argToType[ssaName] = arg.getType();

      // Get argument attributes
      if (auto argAttrs = dyn_cast<DictionaryAttr>(argAttrsAttr[dataCounter])) {
        auto outputsAttr = argAttrs.get("outputs");
        if (!outputsAttr) {
          dataCounter++;
          continue;
        }
        auto outputsArrayAttr = dyn_cast<ArrayAttr>(outputsAttr);
        if (outputsArrayAttr && !outputsArrayAttr.empty()) {
          argToAttr[ssaName] = outputsArrayAttr[0];
        }
      }
      dataCounter++;
    }
  } else {
    for (auto arg : func.getArguments()) {
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << arg.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;
      argToDataCounter[ssaName] = dataCounter;
      argToType[ssaName] = arg.getType();
      dataCounter++;
    }
  }

  // Generate compute nodes
  Block& block = func.getBody().front();

  // Build a map from SSA value to its consuming operation(s)
  llvm::StringMap<SmallVector<Operation*, 4>> valueToConsumers;
  for (Operation& op : block) {
    for (Value operand : op.getOperands()) {
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;
      valueToConsumers[ssaName].push_back(&op);
    }
  }

  for (Operation& op : block) {
    if (!isa<func::ReturnOp>(op)) {
      // Generate Data nodes for arguments that are used for the first time
      for (Value operand : op.getOperands()) {
        std::string ptrStr;
        llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
        std::string ssaName = "ssa_" + ptrStr;

        if (argToDataCounter.count(ssaName) && !ssaToNodeName.count(ssaName)) {
          int idx = argToDataCounter[ssaName];
          Type argType = argToType[ssaName];
          Attribute argAttr =
              argToAttr.count(ssaName) ? argToAttr[ssaName] : nullptr;
          std::string dataNodeName =
              graphName + "/Data_" + std::to_string(nodeCounter);
          ssaToNodeName[ssaName] = dataNodeName;

          std::string dataNodeText =
              generateDataNode(nodeCounter, idx, argType, graphName, argAttr,
                               graphAttr, &valueToConsumers, &ssaName);
          os << dataNodeText;
          nodeCounter++;
        }
      }

      if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
        if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
          std::string inputPtrStr;
          llvm::raw_string_ostream inputOs(inputPtrStr);
          inputOs << op.getOperand(0).getAsOpaquePointer();
          std::string inputSsaName = "ssa_" + inputPtrStr;

          std::string resultPtrStr;
          llvm::raw_string_ostream resultOs(resultPtrStr);
          resultOs << op.getResult(0).getAsOpaquePointer();
          std::string resultSsaName = "ssa_" + resultPtrStr;

          if (ssaToNodeName.count(inputSsaName)) {
            ssaToNodeName[resultSsaName] = ssaToNodeName.lookup(inputSsaName);
          }
        }
        continue;
      }

      std::string nodeText = convertOpToASCIRText(
          &op, nodeCounter, ssaToNodeName, graphName, valueToConsumers);
      os << nodeText;
    }
  }

  // Generate Output node
  if (block.getTerminator() && isa<func::ReturnOp>(block.getTerminator())) {
    func::ReturnOp returnOp = cast<func::ReturnOp>(block.getTerminator());
    if (returnOp.getNumOperands() > 0) {
      Value lastValue = returnOp.getOperand(0);
      std::string ptrStr;
      llvm::raw_string_ostream(ptrStr) << lastValue.getAsOpaquePointer();
      std::string ssaName = "ssa_" + ptrStr;

      if (ssaToNodeName.count(ssaName)) {
        os << generateOutputNode(nodeCounter, ssaToNodeName.lookup(ssaName),
                                 graphName);
        nodeCounter++;
      }
    }
  }

  os << "graph_name: \"" << graphName << "\"";

  os.flush();

  return result;
}

std::string AFIRToASCIRTextPass::convertOpToASCIRText(
    Operation* op, int& nodeCounter,
    llvm::StringMap<std::string>& ssaToNodeName, const std::string& graphName,
    const llvm::StringMap<SmallVector<Operation*, 4>>& valueToConsumers) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  std::string nodeName;
  std::string ssaName;

  if (op->getNumResults() > 0) {
    Value result = op->getResult(0);
    std::string ptrStr;
    llvm::raw_string_ostream(ptrStr) << result.getAsOpaquePointer();
    ssaName = "ssa_" + ptrStr;
  }

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  if (baseOpName == "load") {
    nodeName = "Load_" + std::to_string(nodeCounter);
  } else if (baseOpName == "store") {
    nodeName = "Store_" + std::to_string(nodeCounter);
  } else {
    nodeName = capitalizedOpName + "_" + std::to_string(nodeCounter);
  }

  std::string fullNodeName = graphName + "/" + nodeName;

  if (!ssaName.empty()) {
    ssaToNodeName[ssaName] = fullNodeName;
  }

  os << "asc_node {\n";
  os << generateInputSrc(op, ssaToNodeName);
  os << generateTensorAttr(op, valueToConsumers);
  os << generateNodeAttr(op, fullNodeName, nodeCounter);
  os << generateIrDef(op);
  os << "}\n";

  nodeCounter++;

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateGraphAttr(Attribute graphAttr) {
  std::string result;
  llvm::raw_string_ostream os(result);

  auto ascGraphAttr = dyn_cast<afir::AscGraphAttrGroupsAttr>(graphAttr);
  if (!ascGraphAttr) {
    return result;
  }

  int64_t tilingKey = ascGraphAttr.getTilingKey();
  os << "  tiling_key: " << tilingKey << "\n";

  auto axes = ascGraphAttr.getAxes();
  bool use_old_cann_style =
      NPU_CONFIG_GET_BOOL(xla::NpuConfig::Key::USE_OLD_CANN_STYLE);
  for (auto axis : axes) {
    int64_t id = axis.getId();
    std::string name = axis.getName().str();
    std::string size = axis.getSize().str();
    std::string align = axis.getAlign() ? axis.getAlign().str() : "1";

    os << "  axis {\n";
    if (id > 0) {
      os << "    id: " << id << "\n";
    }
    os << "    name: \"" << name << "\"\n";
    os << "    size: \"" << size << "\"\n";
    // Note that the type of the align parameter is defined differently in the proto files of the old and new versions of the CANN package. The new version of the CANN package is defined as string type (double quotes are required)
    // The old version of CANN package is defined as int32 type (no need to add double quotes)
    if (use_old_cann_style) {
      os << "    align: " << align << "\n";
    } else {
      os << "    align: \"" << align << "\"\n";
    }
    os << "    allow_unaligned_tail: true\n";
    os << "  }\n";
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateNodeAttr(Operation* op,
                                                  const std::string& nodeName,
                                                  int nodeCounter) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName =
      opName.find("afir.") == 0 ? opName.substr(5) : opName;

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  OpInfo opInfo = getOpInfo(baseOpName);

  int execOrder = nodeCounter;
  if (opInfo.isBroadcast) {
    execOrder = -1;
  } else if (baseOpName == "load") {
    execOrder = nodeCounter;
  } else {
    execOrder = nodeCounter - 1;
  }

  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"" << capitalizedOpName << "\"\n";
  os << "    sched {\n";
  os << "      exec_order: " << execOrder << "\n";

  auto indexingMapsAttr = op->getAttrOfType<ArrayAttr>("indexing_maps");
  if (indexingMapsAttr) {
    for (auto attr : indexingMapsAttr) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
        auto affineMap = affineMapAttr.getValue();
        for (unsigned i = 0; i < affineMap.getNumResults(); i++) {
          os << "      axis: " << i << "\n";
        }
        break;
      }
    }
  }

  auto loopAxisAttr = op->getAttrOfType<IntegerAttr>("loop_axis");
  int32_t loopAxis = loopAxisAttr ? loopAxisAttr.getInt() : -1;
  os << "      loop_axis: " << loopAxis << "\n";

  os << "    }\n";
  os << generateApiAttr(op);
  os << generateIrAttrDef(op);
  os << "  }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateTensorAttr(
    Operation* op,
    const llvm::StringMap<SmallVector<Operation*, 4>>& valueToConsumers) {
  std::string result;
  llvm::raw_string_ostream os(result);

  auto outputsAttr = op->getAttrOfType<ArrayAttr>("outputs");
  if (!outputsAttr || outputsAttr.empty()) {
    os.flush();
    return result;
  }

  auto ascTensorAttr = dyn_cast<afir::AscTensorGroupsAttr>(outputsAttr[0]);
  if (!ascTensorAttr) {
    os.flush();
    return result;
  }

  int64_t tensorId = ascTensorAttr.getTensorId();
  int64_t reuseId = ascTensorAttr.getReuseId();
  int64_t positionId = ascTensorAttr.getPositionId();

  auto positionConfig = ascTensorAttr.getPositionConfig();
  int64_t queDepth = -1;
  int queBufNum = -1;

  uint32_t depth = positionConfig.getDepth();
  bool isDoubleBuffer = positionConfig.getIsDoubleBuffer();

  if (depth > 0) {
    queDepth = depth;
  }
  if (isDoubleBuffer) {
    queBufNum = 2;
  }

  Type tensorType = op->getResult(0).getType();

  os << "  outputs {\n";
  os << "    attr {\n";

  // Generate dtype first
  if (auto shapedType = dyn_cast<ShapedType>(tensorType)) {
    Type elementType = shapedType.getElementType();
    int dtype = parseDataType(elementType);
    os << "      dtype: " << dtype << "\n";
  }

  // Then generate axis_ids, repeats, strides
  if (auto shapedType = dyn_cast<ShapedType>(tensorType)) {
    int rank = shapedType.getRank();
    auto shape = shapedType.getShape();

    // Get axis information from indexing_maps to match sched.axis
    // This ensures consistency between outputs.axis_ids and sched.axis
    auto indexingMapsAttr = op->getAttrOfType<ArrayAttr>("indexing_maps");
    int numAxes = rank;  // Default to tensor rank
    bool hasIndexingMaps = false;

    if (indexingMapsAttr) {
      for (auto attr : indexingMapsAttr) {
        if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
          auto affineMap = affineMapAttr.getValue();
          // Use the number of results from the last affine map as axis count
          // This matches the logic used for sched.axis generation
          numAxes = affineMap.getNumResults();
          hasIndexingMaps = true;
          break;
        }
      }
    }

    // Fallback: if still 0, default to 1 axis
    if (numAxes == 0) {
      numAxes = 1;
    }

    // Generate axis_ids based on the determined axis count
    for (int i = 0; i < numAxes; i++) {
      os << "      axis_ids: " << i << "\n";
    }

    // Generate repeats and strides - always match numAxes count
    if (rank == 0) {
      // Scalar case - always generate default repeats/strides
      for (int i = 0; i < numAxes; i++) {
        os << "      repeats: \"1\"\n";
      }
      for (int i = 0; i < numAxes; i++) {
        os << "      strides: \"0\"\n";
      }
    } else {
      // Normal case: generate repeats/strides matching numAxes
      // First use actual tensor shape for available dimensions
      for (int i = 0; i < std::min(numAxes, rank); i++) {
        int64_t size = shapedType.getDimSize(i);
        if (size == ShapedType::kDynamic) {
          os << "      repeats: \"-1\"\n";
        } else {
          os << "      repeats: \"" << size << "\"\n";
        }
      }
      // If numAxes > rank, fill with default "1"
      for (int i = rank; i < numAxes; i++) {
        os << "      repeats: \"1\"\n";
      }

      // For slice-derived LoadOp, strides are computed from the original
      // source shape (stored in ir_attr_def["source_shape"]), while the
      // zero-gate is keyed on the result shape (shape[i]).
      // For all other ops, strideShape == shape (result shape).
      llvm::SmallVector<int64_t> sourceShapeVec;
      ArrayRef<int64_t> strideShape = shape;
      if (auto irAttrDef = op->getAttrOfType<DictionaryAttr>("ir_attr_def")) {
        if (auto srcShapeAttr =
                dyn_cast_or_null<StringAttr>(irAttrDef.get("source_shape"))) {
          llvm::StringRef s = srcShapeAttr.getValue();
          while (!s.empty()) {
            llvm::StringRef token;
            std::tie(token, s) = s.split(',');
            int64_t v = 0;
            token.trim().getAsInteger(10, v);
            sourceShapeVec.push_back(v);
          }
          if ((int)sourceShapeVec.size() == rank) strideShape = sourceShapeVec;
        }
      }

      // Generate strides using column-major layout.
      // Zero-gate: if result shape[i] == 1, stride = 0 (keyed on result shape).
      // Otherwise: stride computed from strideShape (source shape for slice LoadOp).
      for (int i = 0; i < std::min(numAxes, rank); i++) {
        int64_t repeat_val = shape[i];  // zero-gate uses result shape

        if (repeat_val == 1) {
          // dimension with repeat 1, stride is 0
          os << "      strides: \"0\"\n";
        } else {
          // Calculate stride from strideShape (source shape for slice LoadOp,
          // result shape for all other ops)
          int64_t stride = calculateStrideForShapeIndex(i, strideShape);
          os << "      strides: \"" << stride << "\"\n";
        }
      }
      // If numAxes > rank, fill with default "0" (broadcast dimensions)
      for (int i = rank; i < numAxes; i++) {
        os << "      strides: \"0\"\n";
      }
    }
  }
  generateMemQueBufOpsOutputsAttr(os, tensorId, queDepth, queBufNum, positionId,
                                  reuseId);
  os << "    }\n";
  os << "  }\n";

  os.flush();
  return result;
}

void AFIRToASCIRTextPass::generateMemQueBufOpsOutputsAttr(
    llvm::raw_ostream& os, int64_t tensorId, int64_t queDepth, int queBufNum,
    int64_t positionId, int64_t reuseId) {
  os << "      mem {\n";
  os << "        tensor_id: " << tensorId << "\n";
  os << "      }\n";
  os << "      que {\n";
  os << "        id: -1\n";
  os << "        depth: " << queDepth << "\n";
  os << "        buf_num: " << queBufNum << "\n";
  os << "      }\n";
  os << "      buf {\n";
  os << "        id: " << positionId << "\n";
  os << "      }\n";
  os << "      opt {\n";
  os << "        reuse_id: " << reuseId << "\n";
  os << "        ref_tensor: -1\n";
  os << "        merge_scope: -1\n";
  os << "      }\n";
}

std::string AFIRToASCIRTextPass::generateApiAttr(Operation* op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  OpInfo opInfo = getOpInfo(baseOpName);

  os << "    api {\n";

  auto apiTypeAttr = op->getAttrOfType<IntegerAttr>("api_type");
  int apiType = apiTypeAttr ? apiTypeAttr.getInt() : opInfo.apiType;
  os << "      type: " << apiType << "\n";

  auto computeTypeAttr = op->getAttrOfType<IntegerAttr>("compute_type");
  int computeType =
      computeTypeAttr ? computeTypeAttr.getInt() : opInfo.computeType;
  if (computeType > 0 && baseOpName != "load") {
    os << "      compute_type: " << computeType << "\n";
  }

  auto apiUnitAttr = op->getAttrOfType<IntegerAttr>("api_unit");
  int apiUnit = apiUnitAttr ? apiUnitAttr.getInt() : opInfo.apiUnit;
  os << "      unit: " << apiUnit << "\n";

  os << "    }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateIrAttrDef(Operation* op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  bool needsEmptyIrAttrDef = (baseOpName == "load" || baseOpName == "store");
  auto irAttrDefAttr = op->getAttrOfType<DictionaryAttr>("ir_attr_def");

  if (needsEmptyIrAttrDef || (irAttrDefAttr && !irAttrDefAttr.empty()) ||
      baseOpName == "scalar") {
    os << "    ir_attr_def {\n";

    // For scalar operations, add value attribute
    if (baseOpName == "scalar") {
      auto valueAttr = op->getAttr("value");
      if (valueAttr) {
        os << "      attr {\n";
        os << "        key: \"value\"\n";
        os << "        value {\n";

        if (auto elementsAttr = dyn_cast<ElementsAttr>(valueAttr)) {
          auto shapedType = dyn_cast<ShapedType>(elementsAttr.getType());
          if (shapedType && shapedType.getRank() == 0) {
            // Scalar value - get element type
            Type elemType = shapedType.getElementType();
            if (elemType.isF32() || elemType.isF16() || elemType.isBF16()) {
              float val = elementsAttr.getSplatValue<float>();
              os << "          s: \"" << formatFloatValue(val, elemType)
                 << "\"\n";
            } else if (elemType.isF64()) {
              double val = elementsAttr.getSplatValue<double>();
              os << "          s: \"" << formatFloatValue(val, elemType)
                 << "\"\n";
            } else if (elemType.isInteger(32)) {
              int32_t val = elementsAttr.getSplatValue<int32_t>();
              os << "          s: \"" << val << "\"\n";
            } else {
              os << "          s: \"" << elementsAttr << "\"\n";
            }
          } else {
            os << "          s: \"" << elementsAttr << "\"\n";
          }
        } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
          double val = floatAttr.getValueAsDouble();
          Type type = floatAttr.getType();
          os << "          s: \"" << formatFloatValue(val, type) << "\"\n";
        } else if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
          os << "          s: \"" << intAttr.getInt() << "\"\n";
        } else {
          os << "          s: \"" << valueAttr << "\"\n";
        }

        os << "        }\n";
        os << "      }\n";
      }
    }

    if (irAttrDefAttr && !irAttrDefAttr.empty()) {
      for (auto attr : irAttrDefAttr) {
        os << "      attr {\n";
        os << "        key: \"" << attr.getName().strref() << "\"\n";
        os << "        value {\n";

        auto value = attr.getValue();
        if (auto stringAttr = dyn_cast<StringAttr>(value)) {
          os << "          expression: \"" << stringAttr.getValue() << "\"\n";
        } else if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
          os << "          i: " << intAttr.getInt() << "\n";
        } else if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
          os << "          b: " << (boolAttr.getValue() ? "true" : "false")
             << "\n";
        } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
          os << "          f: " << floatAttr.getValueAsDouble() << "\n";
        } else {
          os << "          s: \"" << value << "\"\n";
        }

        os << "        }\n";
        os << "      }\n";
      }
    }

    os << "    }\n";
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateIrDef(Operation* op) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string opName = op->getName().getStringRef().str();
  std::string baseOpName;

  if (opName.find("afir.") == 0) {
    baseOpName = opName.substr(5);
  } else {
    baseOpName = opName;
  }

  int inputNums = op->getNumOperands();
  int outputNums = op->getNumResults();

  os << "  ir_def {\n";

  // For concat operations, always use a single "x" input name
  // regardless of the number of operands, because concat is registered
  // with a single dynamic input in autofuse
  if (baseOpName == "concat") {
    os << "    input_names: \"x\"\n";
  } else {
    for (int i = 0; i < inputNums; i++) {
      if (inputNums == 1) {
        os << "    input_names: \"x\"\n";
      } else {
        os << "    input_names: \"x" << (i + 1) << "\"\n";
      }
    }
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_names: \"y\"\n";
  }

  // For concat operations, only generate one input_ir_type
  if (baseOpName == "concat") {
    os << "    input_ir_type: 2\n";
  } else {
    for (int i = 0; i < inputNums; i++) {
      os << "    input_ir_type: 0\n";
    }
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_ir_type: 0\n";
  }

  std::string capitalizedOpName = baseOpName;
  if (!baseOpName.empty()) {
    capitalizedOpName[0] = ::toupper(baseOpName[0]);
  }

  os << "    type: \"" << capitalizedOpName << "\"\n";

  // For concat operations, only generate one input_nums
  if (baseOpName == "concat") {
    os << "    input_nums: " << inputNums
       << "\n";  // The single input has inputNums tensors
  } else {
    for (int i = 0; i < inputNums; i++) {
      os << "    input_nums: 1\n";
    }
  }

  for (int i = 0; i < outputNums; i++) {
    os << "    output_nums: 1\n";
  }

  os << "  }\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateInputSrc(
    Operation* op, const llvm::StringMap<std::string>& ssaToNodeName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (Value operand : op->getOperands()) {
    std::string ptrStr;
    llvm::raw_string_ostream(ptrStr) << operand.getAsOpaquePointer();
    std::string ssaName = "ssa_" + ptrStr;
    if (ssaToNodeName.count(ssaName)) {
      os << "  input_src {\n";
      os << "    src_node_name: \"" << ssaToNodeName.lookup(ssaName) << "\"\n";
      os << "  }\n";
    }
  }

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateDataNode(
    int nodeIndex, int argIndex, Type argType, const std::string& graphName,
    Attribute argAttr, Attribute graphAttr,
    const llvm::StringMap<SmallVector<Operation*, 4>>* valueToConsumers,
    const std::string* argSSAName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string nodeName = graphName + "/Data_" + std::to_string(nodeIndex);

  // Get number of axes from graph attribute
  int numGraphAxes = 0;
  if (graphAttr) {
    auto ascGraphAttr = dyn_cast<afir::AscGraphAttrGroupsAttr>(graphAttr);
    if (ascGraphAttr) {
      numGraphAxes = ascGraphAttr.getAxes().size();
    }
  }

  // Try to get axis count from consumer's indexing_maps
  int numConsumerAxes = 0;
  if (valueToConsumers && argSSAName) {
    auto it = valueToConsumers->find(*argSSAName);
    if (it != valueToConsumers->end() && !it->second.empty()) {
      // Get the first consumer
      Operation* consumer = it->second[0];
      auto indexingMapsAttr =
          consumer->getAttrOfType<ArrayAttr>("indexing_maps");
      if (indexingMapsAttr) {
        // Find the indexing map for this operand
        unsigned operandIndex = 0;
        for (unsigned i = 0; i < consumer->getNumOperands(); ++i) {
          std::string ptrStr;
          llvm::raw_string_ostream(ptrStr)
              << consumer->getOperand(i).getAsOpaquePointer();
          std::string operandSSAName = "ssa_" + ptrStr;
          if (operandSSAName == *argSSAName) {
            operandIndex = i;
            break;
          }
        }

        if (operandIndex < indexingMapsAttr.size()) {
          if (auto affineMapAttr =
                  dyn_cast<AffineMapAttr>(indexingMapsAttr[operandIndex])) {
            auto affineMap = affineMapAttr.getValue();
            numConsumerAxes = affineMap.getNumResults();
          }
        }
      }
    }
  }

  // Use graph axes count for all nodes to ensure consistency
  // If graphAttr has axes defined, use that count; otherwise use consumer info or rank
  int numOutputAxes = numGraphAxes;
  if (numOutputAxes == 0 && numConsumerAxes > 0) {
    // Fallback: use consumer axes if no graph axes
    numOutputAxes = numConsumerAxes;
  }
  if (numOutputAxes == 0) {
    // Final fallback: at least 1 axis
    numOutputAxes = 1;
  }

  os << "asc_node {\n";
  os << "  outputs {\n";
  os << "    attr {\n";

  int64_t tensorId = -1;
  int64_t reuseId = -1;
  int64_t positionId = -1;
  int64_t queDepth = -1;
  int queBufNum = -1;

  if (argAttr) {
    auto ascTensorAttr = dyn_cast<afir::AscTensorGroupsAttr>(argAttr);
    if (ascTensorAttr) {
      tensorId = ascTensorAttr.getTensorId();
      reuseId = ascTensorAttr.getReuseId();
      positionId = ascTensorAttr.getPositionId();

      auto positionConfig = ascTensorAttr.getPositionConfig();
      uint32_t depth = positionConfig.getDepth();
      bool isDoubleBuffer = positionConfig.getIsDoubleBuffer();

      if (depth > 0) {
        queDepth = depth;
      }
      if (isDoubleBuffer) {
        queBufNum = 2;
      }
    }
  }

  // First, generate dtype at the beginning
  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    Type elementType = shapedType.getElementType();
    int dtype = parseDataType(elementType);
    os << "      dtype: " << dtype << "\n";
  }

  // Then generate axis_ids, repeats, strides based on type
  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    int rank = shapedType.getRank();
    auto shape = shapedType.getShape();

    // Special handling for scalar (rank 0) tensors
    if (rank == 0) {
      // Use the determined axis count (from consumer or graph)
      for (int i = 0; i < numOutputAxes; i++) {
        os << "      axis_ids: " << i << "\n";
      }
      for (int i = 0; i < numOutputAxes; i++) {
        os << "      repeats: \"1\"\n";
      }
      for (int i = 0; i < numOutputAxes; i++) {
        os << "      strides: \"0\"\n";
      }
    } else {
      // Normal case: non-scalar tensor
      for (int i = 0; i < numOutputAxes; i++) {
        os << "      axis_ids: " << i << "\n";
      }

      for (int i = 0; i < rank; i++) {
        int64_t size = shapedType.getDimSize(i);
        if (size == ShapedType::kDynamic) {
          os << "      repeats: \"-1\"\n";
        } else {
          os << "      repeats: \"" << size << "\"\n";
        }
      }
      for (int i = rank; i < numOutputAxes; i++) {
        os << "      repeats: \"1\"\n";
      }

      os << calculateStrides(shape);
      for (int i = rank; i < numOutputAxes; i++) {
        os << "      strides: \"0\"\n";
      }
    }
  } else {
    // Non-ShapedType: use the determined axis count
    if (numOutputAxes == 0) {
      numOutputAxes = 1;
    }
    for (int i = 0; i < numOutputAxes; i++) {
      os << "      axis_ids: " << i << "\n";
    }
    for (int i = 0; i < numOutputAxes; i++) {
      os << "      repeats: \"1\"\n";
    }
    for (int i = 0; i < numOutputAxes; i++) {
      os << "      strides: \"0\"\n";
    }
  }

  generateMemQueBufOpsOutputsAttr(os, tensorId, queDepth, queBufNum, positionId,
                                  reuseId);
  os << "    }\n";
  os << "  }\n";
  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"Data\"\n";
  os << "    sched {\n";
  if (nodeIndex > 0) {
    os << "      exec_order: " << nodeIndex << "\n";
  }

  if (auto shapedType = dyn_cast<ShapedType>(argType)) {
    int rank = shapedType.getRank();

    // Use the same numOutputAxes that was calculated for outputs.attr
    // This ensures consistency between outputs.axis_ids and sched.axis
    int numSchedAxes = numOutputAxes;
    if (numSchedAxes == 0) {
      numSchedAxes = rank;
    }
    // Final fallback: if still 0 (scalar with no consumer info), default to 1
    if (numSchedAxes == 0) {
      numSchedAxes = 1;
    }

    for (int i = 0; i < numSchedAxes; i++) {
      os << "      axis: " << i << "\n";
    }
  } else {
    // Use the same numOutputAxes for non-ShapedType
    int numSchedAxes = numOutputAxes;
    if (numSchedAxes == 0) {
      numSchedAxes = 1;
    }
    for (int i = 0; i < numSchedAxes; i++) {
      os << "      axis: " << i << "\n";
    }
  }

  os << "      loop_axis: -1\n";
  os << "    }\n";
  os << "    api {\n";
  os << "      compute_type: 11\n";
  os << "    }\n";
  os << "    ir_attr_def {\n";
  os << "      attr {\n";
  os << "        key: \"index\"\n";
  os << "        value {\n";
  os << "          i: " << argIndex << "\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  ir_def {\n";
  os << "    output_names: \"y\"\n";
  os << "    output_ir_type: 0\n";
  os << "    type: \"Data\"\n";
  os << "    output_nums: 1\n";
  os << "  }\n";
  os << "}\n";

  os.flush();
  return result;
}

std::string AFIRToASCIRTextPass::generateOutputNode(
    int index, const std::string& lastNodeName, const std::string& graphName) {
  std::string result;
  llvm::raw_string_ostream os(result);

  std::string nodeName = graphName + "/Output_" + std::to_string(index);

  os << "asc_node {\n";
  os << "  input_src {\n";
  os << "    src_node_name: \"" << lastNodeName << "\"\n";
  os << "  }\n";
  os << "  outputs {\n";
  os << "    attr {\n";
  os << "      mem {\n";
  os << "        tensor_id: -1\n";
  os << "      }\n";
  os << "      que {\n";
  os << "        id: -1\n";
  os << "        depth: -1\n";
  os << "        buf_num: -1\n";
  os << "      }\n";
  os << "      buf {\n";
  os << "        id: -1\n";
  os << "      }\n";
  os << "      opt {\n";
  os << "        reuse_id: -1\n";
  os << "        ref_tensor: -1\n";
  os << "        merge_scope: -1\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  attr {\n";
  os << "    name: \"" << nodeName << "\"\n";
  os << "    type: \"Output\"\n";
  os << "    sched {\n";
  os << "      exec_order: 8\n";
  os << "      axis: 0\n";
  os << "      axis: 1\n";
  os << "      loop_axis: -1\n";
  os << "    }\n";
  os << "    api {\n";
  os << "      compute_type: 11\n";
  os << "    }\n";
  os << "    ir_attr_def {\n";
  os << "      attr {\n";
  os << "        key: \"index\"\n";
  os << "        value {\n";
  os << "          i: 0\n";
  os << "        }\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "  ir_def {\n";
  os << "    input_names: \"x\"\n";
  os << "    output_names: \"y\"\n";
  os << "    input_ir_type: 0\n";
  os << "    output_ir_type: 0\n";
  os << "    type: \"Output\"\n";
  os << "    input_nums: 1\n";
  os << "    output_nums: 1\n";
  os << "  }\n";
  os << "}\n";

  os.flush();
  return result;
}

int AFIRToASCIRTextPass::parseDataType(Type type) {
  return llvm::TypeSwitch<Type, int>(type)
      .Case<Float32Type>([](Float32Type) { return 0; })
      .Case<Float16Type>([](Float16Type) { return 1; })
      .Case<Float64Type>([](Float64Type) { return 11; })
      .Case<IntegerType>([](IntegerType intType) {
        if (intType.getWidth() == 8) {
          return intType.isUnsigned() ? 4 : 2;
        }
        if (intType.getWidth() == 16) {
          return intType.isUnsigned() ? 7 : 6;
        }
        if (intType.getWidth() == 32) {
          return intType.isUnsigned() ? 8 : 3;
        }
        if (intType.getWidth() == 64) {
          return intType.isUnsigned() ? 10 : 9;
        }
        return 0;
      })
      .Default([](Type) { return 0; });
}

std::string AFIRToASCIRTextPass::formatFloatValue(double val, Type type) {
  // Handle special floating point values
  if (std::isinf(val) && val < 0) {
    if (type.isF16() || type.isBF16()) {
      return "-65504.0f";
    } else if (type.isF64()) {
      return "-1.7976931348623157e+308";
    } else {
      // Default to F32
      return "-3.4028234663852886e+38f";
    }
  } else if (std::isinf(val)) {
    if (type.isF16() || type.isBF16()) {
      return "65504.0f";
    } else if (type.isF64()) {
      return "1.7976931348623157e+308";
    } else {
      // Default to F32
      return "3.4028234663852886e+38f";
    }
  } else if (std::isnan(val)) {
    return "0.0f/0.0f";
  } else {
    // Regular value, format with appropriate suffix
    if (type.isF64()) {
      std::ostringstream oss;
      oss << std::showpoint << val;
      return oss.str();
    } else {
      std::ostringstream oss;
      oss << std::showpoint << val << "f";
      return oss.str();
    }
  }
}

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass() {
  return std::unique_ptr<Pass>(new AFIRToASCIRTextPass());
}

}  // namespace afir
}  // namespace mlir
