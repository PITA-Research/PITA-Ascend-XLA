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

#ifndef ASC_CORE_TYPES_TD
#define ASC_CORE_TYPES_TD

include "Core/Base.td"

def AscendC_BaseGlobalTensor
    : AscendC_BaseTensorType<"BaseGlobalTensor", "base_global_tensor"> {
  let description = "Represents AscendC::BaseGlobalTensor";
}

def AscendC_BaseLocalTensor
    : AscendC_BaseTensorType<"BaseLocalTensor", "base_local_tensor"> {
  let description = "Represents AscendC::BaseLocalTensor";
}

def AscendC_TBuf : AscendC_BaseQueueType<"TBuf", "tbuf"> {
  let description = "Represents AscendC::TBuf";
  let parameters = (ins "TPositionAttr":$tPositionAttr);
  let assemblyFormat = "`<` custom<PrettyTPosition>($tPositionAttr) `>`";
  let builders = [
    TypeBuilder<(ins "TPosition":$position), [{
      return $_get($_ctxt, TPositionAttr::get($_ctxt, position));
    }]>,
  ];
  let extraClassDeclaration = [{
    TPosition getTPosition() { return getTPositionAttr().getValue(); }
  }];
}

def AscendC_DataCopyPadExtParams : AscendC_Type<"DataCopyPadExtParams", "data_copy_pad_ext_params"> {
  let description = "Represents AscendC::DataCopyPadExtParams";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def AscendC_FixpipeParams : AscendC_Type<"FixpipeParams", "fixpipe_params"> {
  let description = "Represents AscendC::FixpipeParams";
  let parameters = (ins "Type":$type);
  let assemblyFormat = "`<`  $type `>`";
}

def AscendC_GlobalTensor : AscendC_BaseTensorType<"GlobalTensor", "global_tensor"> {
  let summary = "Global tensor from GM (OUT)";
}

def AscendC_LoadData2DMxParams
    : AscendC_BaseLoadDataParamsType<"LoadData2DMxParams", "load_data_2d_mx_params"> {
  let summary = "Represents AscendC::LoadData2DMxParams";
}

def AscendC_LoadData3DParamsV2
    : AscendC_BaseLoadDataParamsType<"LoadData3DParamsV2", "load_data_3d_params_v2"> {
  let summary = "Represents AscendC::LoadData3DParamsV2";
  let parameters = (ins "Type":$type);
  let assemblyFormat = "`<`  $type `>`";
}

def AscendC_LocalTensor : AscendC_BaseTensorType<"LocalTensor", "local_tensor"> {
  let summary = "Local tensor from UB";
}

def AscendC_Mask : AscendC_Type<"Mask", "mask"> {
  let summary = "Represents vector mask (bit mode)";
}

def AscendC_Matmul : AscendC_Type<"Matmul", "matmul"> {
  let description = "Represents matmul::Matmul";
  let parameters = (ins "TPositionAttr":$srcAPositionAttr,
                        "CubeFormatAttr":$cubeFormatAAttr, "Type":$typeA,
                        "bool":$isTransA, "LayoutModeAttr":$layoutModeAAttr,
                        "TPositionAttr":$srcBPositionAttr,
                        "CubeFormatAttr":$cubeFormatBAttr, "Type":$typeB,
                        "bool":$isTransB, "LayoutModeAttr":$layoutModeBAttr,
                        "TPositionAttr":$srcCPositionAttr,
                        "CubeFormatAttr":$cubeFormatCAttr, "Type":$typeC,
                        "bool":$isTransC, "LayoutModeAttr":$layoutModeCAttr,
                        "TPositionAttr":$srcBiasPositionAttr,
                        "CubeFormatAttr":$cubeFormatBiasAttr,
                        "Type":$typeBias, "MatmulConfigAttr":$matmulConfig);
  let assemblyFormat = [{
    `<` custom<PrettyTPosition>($srcAPositionAttr) `,`
    $cubeFormatAAttr `,` $typeA `,` $isTransA `,` $layoutModeAAttr `,` custom<PrettyTPosition>($srcBPositionAttr) `,`
    $cubeFormatBAttr `,` $typeB `,` $isTransB `,` $layoutModeBAttr `,` custom<PrettyTPosition>($srcCPositionAttr) `,`
    $cubeFormatCAttr `,` $typeC `,` $isTransC `,` $layoutModeCAttr `,` custom<PrettyTPosition>($srcBiasPositionAttr) `,`
    $cubeFormatBiasAttr `,` $typeBias `,` $matmulConfig `>`
  }];
  let builders = [
    TypeBuilder<(ins "TPosition":$srcAPosition, "CubeFormat":$cubeFormatA,
                 "Type":$typeA, "bool":$isTransA, "LayoutMode":$layoutModeA, "TPosition":$srcBPosition, "CubeFormat":$cubeFormatB,
                 "Type":$typeB, "bool":$isTransB, "LayoutMode":$layoutModeB, "TPosition":$srcCPosition, "CubeFormat":$cubeFormatC,
                 "Type":$typeC, "bool":$isTransC, "LayoutMode":$layoutModeC, "TPosition":$srcBiasPosition,
                 "CubeFormat":$cubeFormatBias, "Type":$typeBias, "MatmulConfigAttr":$matmulConfig), [{
      auto srcA = TPositionAttr::get($_ctxt, srcAPosition);
      auto formatA = CubeFormatAttr::get($_ctxt, cubeFormatA);
      auto layoutA = LayoutModeAttr::get($_ctxt, layoutModeA);
      auto srcB = TPositionAttr::get($_ctxt, srcBPosition);
      auto formatB = CubeFormatAttr::get($_ctxt, cubeFormatB);
      auto layoutB = LayoutModeAttr::get($_ctxt, layoutModeB);
      auto srcC = TPositionAttr::get($_ctxt, srcCPosition);
      auto formatC = CubeFormatAttr::get($_ctxt, cubeFormatC);
      auto layoutC = LayoutModeAttr::get($_ctxt, layoutModeC);
      auto srcBias = TPositionAttr::get($_ctxt, srcBiasPosition);
      auto formatBias = CubeFormatAttr::get($_ctxt, cubeFormatBias);
      return $_get($_ctxt, srcA, formatA, typeA, isTransA, layoutA,
                    srcB, formatB, typeB, isTransB, layoutB,
                    srcC, formatC, typeC, isTransC, layoutC,
                    srcBias, formatBias, typeBias, matmulConfig);
    }]>,
  ];
  let extraClassDeclaration = [{
    TPosition getSrcAPosition() { return getSrcAPositionAttr().getValue(); }
    TPosition getSrcBPosition() { return getSrcBPositionAttr().getValue(); }
    TPosition getSrcCPosition() { return getSrcCPositionAttr().getValue(); }
    TPosition getSrcBiasPosition() { return getSrcBiasPositionAttr().getValue(); }
    CubeFormat getCubeFormatA() { return getCubeFormatAAttr().getValue(); }
    CubeFormat getCubeFormatB() { return getCubeFormatBAttr().getValue(); }
    CubeFormat getCubeFormatC() { return getCubeFormatCAttr().getValue(); }
    LayoutMode getLayoutModeA() { return getLayoutModeAAttr().getValue(); }
    LayoutMode getLayoutModeB() { return getLayoutModeBAttr().getValue(); }
    LayoutMode getLayoutModeC() { return getLayoutModeCAttr().getValue(); }
    CubeFormat getCubeFormatBias() { return getCubeFormatBiasAttr().getValue(); }
  }];
}

def AscendC_QueBind : AscendC_BaseQueueType<"QueBind", "que_bind"> {
  let description = "Represents AscendC::TQueBind";
  let parameters = (ins "TPositionAttr":$srcPositionAttr,
                        "TPositionAttr":$dstPositionAttr, "int64_t":$depth);
  let assemblyFormat = [{
    `<` custom<PrettyTPosition>($srcPositionAttr) `,`
    custom<PrettyTPosition>($dstPositionAttr) `,` $depth `>`
  }];
  let builders = [
    TypeBuilder<(ins "TPosition":$srcPosition, "TPosition":$dstPosition,
                 "int64_t":$depth), [{
      auto src = TPositionAttr::get($_ctxt, srcPosition);
      auto dst = TPositionAttr::get($_ctxt, dstPosition);
      return $_get($_ctxt, src, dst, depth);
    }]>,
  ];
  let extraClassDeclaration = [{
    TPosition getSrcPosition() { return getSrcPositionAttr().getValue(); }
    TPosition getDstPosition() { return getDstPositionAttr().getValue(); }
  }];
}

def AscendC_Queue : AscendC_BaseQueueType<"Queue", "queue"> {
  let description = "Represents AscendC::TQue";
  let parameters = (ins "TPositionAttr":$tPositionAttr, "int64_t":$depth);
  let assemblyFormat = [{
    `<` custom<PrettyTPosition>($tPositionAttr) `,` $depth `>`
  }];
  let builders = [
    TypeBuilder<(ins "TPosition":$position, "int64_t":$depth), [{
      return $_get($_ctxt, TPositionAttr::get($_ctxt, position), depth);
    }]>,
  ];
  let extraClassDeclaration = [{
    TPosition getPosition() { return getTPositionAttr().getValue(); }
  }];
}

def AscendC_TBufPool : AscendC_Type<"TBufPool", "tbuf_pool"> {
  let description = "Represents AscendC::TBufPool";
  let parameters = (ins "TPositionAttr":$tPositionAttr, "uint32_t": $bufIDSize);
  let assemblyFormat = "`<` custom<PrettyTPosition>($tPositionAttr) `,` $bufIDSize `>`";
  let builders = [
    TypeBuilder<(ins "TPosition":$position, "uint32_t": $bufIDSize), [{
      return $_get($_ctxt, TPositionAttr::get($_ctxt, position), bufIDSize);
    }]>,
  ];
  let extraClassDeclaration = [{
    TPosition getTPosition() { return getTPositionAttr().getValue(); }
  }];
}

include "ascir/API/Types.td.inc"

#endif // ASC_CORE_TYPES_TD
