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

#ifndef ASC_ADV_MATMUL_TD
#define ASC_ADV_MATMUL_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_MatmulInitOp : APIOp<"matmul.init", "Init"> {
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_TCubeTiling:$cubeTiling,
                    Optional<AscendC_Pipe>:$pipe);
}

def AscendC_MatmulSetTensorAOp
    : APIOp<"matmul.set_tensor_a", "SetTensorA", [AscMemberFunc]> {
  let description = "Matmul set left tensor input";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    AnyType:$transpose);
  let paramTypeLists = [0, 0, 0];
}

def AscendC_MatmulSetTensorAScalarOp
    : APIOp<"matmul.set_tensor_a_scalar", "SetTensorA", [AscMemberFunc]> {
  let description = "Matmul set left scalar input";
  let arguments = (ins AscendC_Matmul:$matmul, AnyType:$aScalar);
  let paramTypeLists = [0, 0];
}

def AscendC_MatmulSetTensorBOp
    : APIOp<"matmul.set_tensor_b", "SetTensorB", [AscMemberFunc]> {
  let description = "Matmul set right tensor input";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    AnyType:$transpose);
  let paramTypeLists = [0, 0, 0];
}

def AscendC_MatmulSetTensorBScalarOp
    : APIOp<"matmul.set_tensor_b_scalar", "SetTensorB", [AscMemberFunc]> {
  let description = "Matmul set right scalar input";
  let arguments = (ins AscendC_Matmul:$matmul, AnyType:$bScalar);
  let paramTypeLists = [0, 0];
}

def AscendC_MatmulSetBiasOp
    : APIOp<"matmul.set_bias", "SetBias", [AscMemberFunc]> {
  let description = "Matmul set bias tensor";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor);
  let paramTypeLists = [0, 0];
}

def AscendC_MatmulDisableBiasOp
    : APIOp<"matmul.disable_bias", "DisableBias", [AscMemberFunc]> {
  let description = "Matmul disable bias flag";
  let arguments = (ins AscendC_Matmul:$matmul);
  let paramTypeLists = [0];
}

def AscendC_MatmulSetSelfDefineDataOp
    : APIOp<"matmul.set_self_define_data", "SetSelfDefineData", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul, AnyType:$dataPtr);
  let paramTypeLists = [0, -2];
}

def AscendC_MatmulSetSparseIndexOp
    : APIOp<"matmul.set_sparse_index", "SetSparseIndex", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul, AscendC_GlobalTensor:$indexGlobal);
}

def AscendC_MatmulSetUserDefInfoOp
    : APIOp<"matmul.set_user_def_info", "SetUserDefInfo", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul, AnyMemRef:$tilingPtr);
  let paramTypeLists = [0, -2];
}

def AscendC_MatmulGetMatmulApiTilingOp : APIOp<"get_matmul_api_tiling", "GetMatmulApiTiling"> {
  let arguments = (ins AscendC_MatmulConfig:$mmCFG, AnyType:$l1Size, TypeAttr:$matmulType);
  let results = (outs AscendC_MatmulApiStaticTiling:$result);
}

def AscendC_MatmulIterateOp
    : APIOp<"matmul.iterate", "Iterate", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$partialsum,
                    Optional<AscendC_LocalTensor>:$localCmaxtrix,
                    AnyType:$sync);
  let results = (outs AnyType:$result);
  let paramTypeLists = [0, 0, 0, 4];
  let assemblyFormat = [{
    $matmul `,` $partialsum `,` $sync attr-dict (`,` $localCmaxtrix^)? `:`
    qualified(type($matmul)) `,` type($partialsum) `,` type($sync) (`,` type($localCmaxtrix)^)? `,` type($result)
  }];
}

def AscendC_MatmulGetTensorCOp
    : APIOp<"matmul.get_tensor_c", "GetTensorC", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    Optional<AscendC_BaseTensorTypeInterface>:$tensor2,
                    AnyType:$atomic,
                    AnyType:$sequentialwrite,
                    AnyType:$sync);
  let paramTypeLists = [0, 0, 0, 0, 0, 4];
}

def AscendC_MatmulGetTensorCReturnOp
    : APIOp<"matmul.get_tensor_c_return", "GetTensorC", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$atomic,
                    AnyType:$sequentialwrite,
                    AnyType:$sync);
  let results = (outs AscendC_GlobalTensor:$result);
  let paramTypeLists = [0, 0, 0, 4];
  let assemblyFormat = [{
    $matmul `,` $atomic `,` $sequentialwrite `,` $sync attr-dict `:`
    qualified(type($matmul)) `,` type($atomic) `,` type($sequentialwrite) `,`
    type($sync) `,` qualified(type($result))
  }];
}

def AscendC_MatmulIterateAllOp
    : APIOp<"matmul.iterate_all", "IterateAll", [AscMemberFunc, AttrSizedOperandSegments]> {
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    Optional<AnyType>:$enAtomic,
                    Optional<AnyType>:$enSenquentialWrite,
                    Optional<AnyType>:$waitIterateAll,
                    Optional<AnyType>:$fakeMsg,
                    AnyType:$sync);
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 4];
}

def AscendC_MatmulWaitIterateAllOp
  : APIOp<"matmul.wait_iterate_all", "WaitIterateAll", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul);
  let paramTypeLists = [0];
}


def AscendC_MatmulIterateBatchOp
    : APIOp<"matmul.iterate_batch", "IterateBatch", [AscMemberFunc, AttrSizedOperandSegments]> {
  let description = "Matmul iterate batch interface for mix scence";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    AnyType:$batchA,
                    AnyType:$batchB,
                    AnyType:$enSenquentialWrite,
                    AnyType:$maxtrixStrideA,
                    AnyType:$maxtrixStrideB,
                    AnyType:$maxtrixStrideC,
                    Optional<AnyType>:$enPartialSum,
                    Optional<AnyType>:$enAtomic,
                    AnyType:$sync,
                    Optional<AnyType>:$waitIterateBatch);
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4];
}

def AscendC_MatmulIterateBatchCubeOnlyOp
    : APIOp<"matmul.iterate_batch_cube_only", "IterateBatch", [AscMemberFunc]> {
  let description = "Matmul iterate batch interface for cube-only scence";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_BaseTensorTypeInterface:$tensor,
                    AnyType:$enPartialSum,
                    AnyType:$enAtomic,
                    AnyType:$enSenquentialWrite,
                    AnyType:$maxtrixStrideA,
                    AnyType:$maxtrixStrideB,
                    AnyType:$maxtrixStrideC);
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, 0];
}

def AscendC_MatmulWaitIterateBatchOp 
  : APIOp<"matmul.wait_iterate_batch", "WaitIterateBatch", [AscMemberFunc]> {
  let arguments = (ins AscendC_Matmul:$matmul);
}

def AscendC_MatmulGetBatchTensorCOp
    : APIOp<"matmul.get_batch_tensor_c", "GetBatchTensorC", [AscMemberFunc]> {
  let description = "Matmul get batch tensor C result";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$batchA,
                    AnyType:$batchB,
                    AnyType:$sequentialwrite,
                    AnyType:$sync);
  let results = (outs AscendC_GlobalTensor:$result);
  let paramTypeLists = [0, 0, 0, 0, 4];
  let assemblyFormat = [{
    $matmul `,` $batchA `,` $batchB `,` $sequentialwrite `,` $sync attr-dict `:`
    qualified(type($matmul)) `,` type($batchA) `,` type($batchB) `,` type($sequentialwrite) `,`
    type($sync) `,` qualified(type($result))
  }];
}

def AscendC_MatmulGetBatchTensorCLocalMemOp
    : APIOp<"matmul.get_batch_tensor_c_local_mem", "GetBatchTensorC", [AscMemberFunc]> {
  let description = "Matmul get batch tensor C result use local memory";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_LocalTensor:$c,
                    AnyType:$batchA,
                    AnyType:$batchB,
                    AnyType:$sequentialwrite,
                    AnyType:$sync);
  let paramTypeLists = [0, 0, 0, 0, 0, 4];
}

def AscendC_MatmulEndOp : APIOp<"matmul.end", "End"> {
  let arguments = (ins AscendC_Matmul:$matmul);
}

def AscendC_RegistMatmulObjOp : APIOp<"regist_matmul_obj", "REGIST_MATMUL_OBJ"> {
  let arguments = (ins AscendC_Pipe:$pipe, AscendC_Matmul:$matmul,
                   Optional<AscendC_TCubeTiling>:$tiling);
}

def AscendC_MatmulIterateNBatchOp : APIOp<"matmul.iterate_n_batch", "IterateNBatch", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::IterateNBatch` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AnyType:$batchLoop,
                    AnyType:$batchA,
                    AnyType:$batchB,
                    AnyType:$enSequentialWrite,
                    AnyType:$matrixStrideA,
                    AnyType:$matrixStrideB,
                    AnyType:$matrixStrideC,
                    AnyType:$sync,
                    AnyType:$waitIterateBatch);
  let paramTypeLists = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4];
}

def AscendC_MatmulSetHF32Op : APIOp<"matmul.set_hf32", "SetHF32", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetHF32` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AnyType:$enableHF32,
                    AnyType:$transMode);
}

def AscendC_MatmulSetTailOp : APIOp<"matmul.set_tail", "SetTail", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetTail` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AnyType:$tailM,
                    AnyType:$tailN,
                    AnyType:$tailK);
}

def AscendC_MatmulSetBatchNumOp : APIOp<"matmul.set_batch_num", "SetBatchNum", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetBatchNum` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AnyType:$batchA,
                    AnyType:$batchB);
}

def AscendC_MatmulSetWorkspaceOp : APIOp<"matmul.set_workspace", "SetWorkspace", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetWorkspace` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AnyType:$tensor,
                    Optional<AnyType>:$size);
}

def AscendC_MatmulWaitGetTensorCOp : APIOp<"matmul.wait_get_tensor_c", "WaitGetTensorC", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::WaitGetTensorC` method";
  let arguments = (ins AscendC_Matmul:$aux);
}

def AscendC_MatmulGetOffsetCOp : APIOp<"matmul.get_offset_c", "GetOffsetC", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::GetOffsetC` method";
  let arguments = (ins AscendC_Matmul:$aux);
  let results = (outs AscendC_MatrixOffset:$result);
}

def AscendC_MatmulAsyncGetTensorCOp : APIOp<"matmul.async_get_tensor_c", "AsyncGetTensorC", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::AsyncGetTensorC` method";
  let arguments = (ins AscendC_Matmul:$aux,
                    AscendC_LocalTensor:$tensor);
}

def AscendC_MatmulSetQuantScalarOp : APIOp<"matmul.set_quant_scalar", "SetQuantScalar", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetQuantScalar` method";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$quantscalar);
}

def AscendC_MatmulSetQuantVectorOp : APIOp<"matmul.set_quant_vector", "SetQuantVector", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetQuantVector` method";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AscendC_GlobalTensor:$quantvector);
  
}

def AscendC_MatmulSetOrgShapeOp : APIOp<"matmul.set_org_shape", "SetOrgShape", [AscMemberFunc, AttrSizedOperandSegments]> {
  let summary = "Call `AscendC::Matmul::SetOrgShape` method";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$orgm,
                    AnyType:$orgn,
                    AnyType:$orgka,
                    Optional<AnyType>:$orgkb,
                    Optional<AnyType>:$orgkc);
}

def AscendC_MatmulSetSingleShapeOp : APIOp<"matmul.set_single_shape", "SetSingleShape", [AscMemberFunc]> {
  let summary = "Call `AscendC::Matmul::SetSingleShape` method";
  let arguments = (ins AscendC_Matmul:$matmul,
                    AnyType:$singlem,
                    AnyType:$singlen,
                    AnyType:$singlek);
}

#endif //ASC_ADV_MATMUL_TD
