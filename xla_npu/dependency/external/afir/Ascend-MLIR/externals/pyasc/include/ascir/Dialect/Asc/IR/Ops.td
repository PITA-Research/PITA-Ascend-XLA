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

#ifndef ASC_OPS_TD
#define ASC_OPS_TD

include "Adv/Activation.td"
include "Adv/Kfc.td"
include "Adv/Math.td"
include "Adv/Matmul.td"
include "Adv/Normalization.td"
include "Adv/Quantization.td"
include "Adv/Sort.td"
include "Base.td"
include "Basic/Common.td"
include "Basic/OpAipp.td"
include "Basic/OpAtomic.td"
include "Basic/OpBlockSync.td"
include "Basic/OpCache.td"
include "Basic/OpCommon.td"
include "Basic/OpConv2d.td"
include "Basic/OpCopy.td"
include "Basic/OpDataCache.td"
include "Basic/OpDataConversion.td"
include "Basic/OpDataCopy.td"
include "Basic/OpDetermineComputeSync.td"
include "Basic/OpDumpTensor.td"
include "Basic/OpFixpipe.td"
include "Basic/OpGemm.td"
include "Basic/OpHcclContext.td"
include "Basic/OpLimits.td"
include "Basic/OpListTensor.td"
include "Basic/OpMm.td"
include "Basic/OpProposal.td"
include "Basic/OpScalar.td"
include "Basic/OpSetAtomic.td"
include "Basic/OpSwapMem.td"
include "Basic/OpSysVar.td"
include "Basic/OpVecBilinearInterpolation.td"
include "Basic/OpVecBinary.td"
include "Basic/OpVecBinaryScalar.td"
include "Basic/OpVecBrcb.td"
include "Basic/OpVecCmpsel.td"
include "Basic/OpVecCreatevecindex.td"
include "Basic/OpVecDuplicate.td"
include "Basic/OpVecGather.td"
include "Basic/OpVecGatherMask.td"
include "Basic/OpVecMulCast.td"
include "Basic/OpVecReduce.td"
include "Basic/OpVecScatter.td"
include "Basic/OpVecTensor.td"
include "Basic/OpVecTernaryScalar.td"
include "Basic/OpVecTranspose.td"
include "Basic/OpVecUnary.td"
include "Basic/OpVecVconv.td"
include "Basic/OpVecVpadding.td"
include "Basic/ProfTrace.td"
include "Core/Attributes.td"
include "Core/Tensor.td"
include "Fwk/TBuf.td"
include "Fwk/TBufPool.td"
include "Fwk/TPipe.td"
include "Fwk/TQue.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_NoOp : AscendC_Op<"noop"> {
  let summary = "noop";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

def AscendC_ConstructOp : AscendC_Op<"construct", [Pure]> {
  let summary = "Construct Ascend C object (struct or enum)";
  let arguments = (ins Variadic<AnyType>:$operands,
                   OptionalAttr<TypeArrayAttr>:$types, UnitAttr:$isConstexpr,
                   UnitAttr:$isStatic);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Type":$result), [{
      build($_builder, $_state, result, ValueRange{});
    }]>,
    OpBuilder<(ins "Type":$result, "ValueRange":$operands), [{
      build($_builder, $_state, result, operands, ArrayAttr{});
    }]>,
  ];
  let assemblyFormat = [{
    type($result) `(` ($operands^)? `)` ($types^)? (`constexpr` $isConstexpr^)?
    (`static` $isStatic^)? attr-dict (`:` type($operands)^)?
  }];
}

def AscendC_AscendIsAICOp : APIOp<"ascend_is_aic", "AscendIsAIC"> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def AscendC_AscendIsAIVOp : APIOp<"ascend_is_aiv", "AscendIsAIV"> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def AscendC_FftsCrossCoreSyncOp : APIOp<"ffts_cross_core_sync", "FftsCrossCoreSync"> {
  let arguments = (ins AscendC_PipeAttr:$pipe, AnyType:$config);
}

def AscendC_SetFftsBaseAddrOp : APIOp<"set_ffts_base_addr", "SetFftsBaseAddr"> {
  let arguments = (ins AnyMemRef);
}

def AscendC_PopStackBufferOp : APIOp<"pop_stack_buffer", "PopStackBuffer"> {
  let arguments = (ins AscendC_TPositionAttr:$pos, AscendC_LocalTensor:$tensor);
  let assemblyFormat = "$pos $tensor attr-dict `:` qualified(type($tensor))";
}

def AscendC_LocalTensorAutoOp : APIOp<"local_tensor_auto", "LocalTensorAuto"> {
  let summary = "Create virtual tensor with automatic allocation semantic";
  let arguments = (ins UnitAttr:$input, UnitAttr:$output, Variadic<I64>:$dynamicShape);
  let results = (outs AscendC_LocalTensor:$result);
  let assemblyFormat = [{
    `(` $dynamicShape `)` (`input` $input^)? (`output` $output^)? attr-dict `:`
    type($result)
  }];
  let builders = [
    OpBuilder<(ins "Type":$result), [{
      build($_builder, $_state, result, false, false, ValueRange{});
    }]>,
  ];
}

#endif // ASC_OPS_TD
