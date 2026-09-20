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

#ifndef ASC_BASIC_OP_DATACACHE_TD
#define ASC_BASIC_OP_DATACACHE_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Data Cache Clean And Invalid Ops
//===----------------------------------------------------------------------===//

def AscendC_DataCacheCleanAndInvalidGlobalOp
    : APIOp<"data_cache_clean_invalid_global", "DataCacheCleanAndInvalid", [AscFunc]> {
  let description = "Clean and invalidate data cache for GlobalTensor (with DcciDst).";
  let arguments = (ins
    AscendC_GlobalTensor:$dst,
    AscendC_CacheLineAttr:$entireType,
    AscendC_DcciDstAttr:$dcciDst
  );
  let paramTypeLists = [2, 3, 3];
}

def AscendC_DataCacheCleanAndInvalidGlobalNoDcciDstOp
    : APIOp<"data_cache_clean_invalid_global_nodccidst", "DataCacheCleanAndInvalid", [AscFunc]> {
  let description = "Clean and invalidate data cache for GlobalTensor.";
  let arguments = (ins
    AscendC_GlobalTensor:$dst,
    AscendC_CacheLineAttr:$entireType
  );
  let paramTypeLists = [2, 3];
}

def AscendC_GetICachePreloadStatusOp : APIOp<"get_icache_preload_status", "GetICachePreloadStatus", [AscFunc]> {
  let description = "Get the status of ICache preload: 0 indicates idle, 1 indicates busy.";
  let results = (outs I64:$status);
}

def AscendC_ICachePreLoadOp : APIOp<"icache_preload", "ICachePreLoad", [AscFunc]> {
  let description = "Preload instructions from DDR address to ICache.";
  let arguments = (ins
    I64:$preFetchLen
  );
  let paramTypeLists = [0];
}
#endif // ASC_BASIC_OP_DATACACHE_TD
