/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attributes.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::ascendc {

::llvm::StringRef stringifyAddressSpace(AddressSpace val) {
  switch (val) {
    case AddressSpace::Default: return "Default";
    case AddressSpace::gm: return "gm";
    case AddressSpace::ca: return "ca";
    case AddressSpace::cb: return "cb";
    case AddressSpace::cc: return "cc";
    case AddressSpace::ubuf: return "ubuf";
    case AddressSpace::cbuf: return "cbuf";
    case AddressSpace::fbuf: return "fbuf";
  }
  return "";
}

::std::optional<AddressSpace> symbolizeAddressSpace(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(str)
      .Case("Default", AddressSpace::Default)
      .Case("gm", AddressSpace::gm)
      .Case("ca", AddressSpace::ca)
      .Case("cb", AddressSpace::cb)
      .Case("cc", AddressSpace::cc)
      .Case("ubuf", AddressSpace::ubuf)
      .Case("cbuf", AddressSpace::cbuf)
      .Case("fbuf", AddressSpace::fbuf)
      .Default(::std::nullopt);
}
::std::optional<AddressSpace> symbolizeAddressSpace(uint8_t value) {
  switch (value) {
  case 0: return AddressSpace::Default;
  case 22: return AddressSpace::gm;
  case 23: return AddressSpace::ca;
  case 24: return AddressSpace::cb;
  case 25: return AddressSpace::cc;
  case 26: return AddressSpace::ubuf;
  case 27: return AddressSpace::cbuf;
  case 28: return AddressSpace::fbuf;
  default: return ::std::nullopt;
  }
}

bool AddressSpaceAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 22)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 23)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 24)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 25)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 26)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 27)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 28)))));
}
AddressSpaceAttr AddressSpaceAttr::get(::mlir::MLIRContext *context, AddressSpace val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<AddressSpaceAttr>(baseAttr);
}
AddressSpace AddressSpaceAttr::getValue() const {
  return static_cast<AddressSpace>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyAippInputFormat(AippInputFormat val) {
  switch (val) {
    case AippInputFormat::YUV420SP_U8: return "YUV420SP_U8";
    case AippInputFormat::XRGB8888_U8: return "XRGB8888_U8";
    case AippInputFormat::RGB888_U8: return "RGB888_U8";
    case AippInputFormat::YUV400_U8: return "YUV400_U8";
  }
  return "";
}

::std::optional<AippInputFormat> symbolizeAippInputFormat(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AippInputFormat>>(str)
      .Case("YUV420SP_U8", AippInputFormat::YUV420SP_U8)
      .Case("XRGB8888_U8", AippInputFormat::XRGB8888_U8)
      .Case("RGB888_U8", AippInputFormat::RGB888_U8)
      .Case("YUV400_U8", AippInputFormat::YUV400_U8)
      .Default(::std::nullopt);
}
::std::optional<AippInputFormat> symbolizeAippInputFormat(uint8_t value) {
  switch (value) {
  case 0: return AippInputFormat::YUV420SP_U8;
  case 1: return AippInputFormat::XRGB8888_U8;
  case 4: return AippInputFormat::RGB888_U8;
  case 9: return AippInputFormat::YUV400_U8;
  default: return ::std::nullopt;
  }
}

bool AippInputFormatAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))));
}
AippInputFormatAttr AippInputFormatAttr::get(::mlir::MLIRContext *context, AippInputFormat val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<AippInputFormatAttr>(baseAttr);
}
AippInputFormat AippInputFormatAttr::getValue() const {
  return static_cast<AippInputFormat>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyBatchMode(BatchMode val) {
  switch (val) {
    case BatchMode::NONE: return "NONE";
    case BatchMode::BATCH_LESS_THAN_L1: return "BATCH_LESS_THAN_L1";
    case BatchMode::BATCH_LARGE_THAN_L1: return "BATCH_LARGE_THAN_L1";
    case BatchMode::SINGLE_LARGE_THAN_L1: return "SINGLE_LARGE_THAN_L1";
  }
  return "";
}

::std::optional<BatchMode> symbolizeBatchMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<BatchMode>>(str)
      .Case("NONE", BatchMode::NONE)
      .Case("BATCH_LESS_THAN_L1", BatchMode::BATCH_LESS_THAN_L1)
      .Case("BATCH_LARGE_THAN_L1", BatchMode::BATCH_LARGE_THAN_L1)
      .Case("SINGLE_LARGE_THAN_L1", BatchMode::SINGLE_LARGE_THAN_L1)
      .Default(::std::nullopt);
}
::std::optional<BatchMode> symbolizeBatchMode(uint8_t value) {
  switch (value) {
  case 0: return BatchMode::NONE;
  case 1: return BatchMode::BATCH_LESS_THAN_L1;
  case 2: return BatchMode::BATCH_LARGE_THAN_L1;
  case 3: return BatchMode::SINGLE_LARGE_THAN_L1;
  default: return ::std::nullopt;
  }
}

bool BatchModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
BatchModeAttr BatchModeAttr::get(::mlir::MLIRContext *context, BatchMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<BatchModeAttr>(baseAttr);
}
BatchMode BatchModeAttr::getValue() const {
  return static_cast<BatchMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyBatchOutMode(BatchOutMode val) {
  switch (val) {
    case BatchOutMode::SINGLE_BATCH: return "SINGLE_BATCH";
    case BatchOutMode::MULTI_BATCH: return "MULTI_BATCH";
    case BatchOutMode::DYNAMIC: return "DYNAMIC";
  }
  return "";
}

::std::optional<BatchOutMode> symbolizeBatchOutMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<BatchOutMode>>(str)
      .Case("SINGLE_BATCH", BatchOutMode::SINGLE_BATCH)
      .Case("MULTI_BATCH", BatchOutMode::MULTI_BATCH)
      .Case("DYNAMIC", BatchOutMode::DYNAMIC)
      .Default(::std::nullopt);
}
::std::optional<BatchOutMode> symbolizeBatchOutMode(uint8_t value) {
  switch (value) {
  case 0: return BatchOutMode::SINGLE_BATCH;
  case 1: return BatchOutMode::MULTI_BATCH;
  case 2: return BatchOutMode::DYNAMIC;
  default: return ::std::nullopt;
  }
}

bool BatchOutModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))));
}
BatchOutModeAttr BatchOutModeAttr::get(::mlir::MLIRContext *context, BatchOutMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<BatchOutModeAttr>(baseAttr);
}
BatchOutMode BatchOutModeAttr::getValue() const {
  return static_cast<BatchOutMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCMPMODE(CMPMODE val) {
  switch (val) {
    case CMPMODE::LT: return "LT";
    case CMPMODE::GT: return "GT";
    case CMPMODE::EQ: return "EQ";
    case CMPMODE::LE: return "LE";
    case CMPMODE::GE: return "GE";
    case CMPMODE::NE: return "NE";
  }
  return "";
}

::std::optional<CMPMODE> symbolizeCMPMODE(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CMPMODE>>(str)
      .Case("LT", CMPMODE::LT)
      .Case("GT", CMPMODE::GT)
      .Case("EQ", CMPMODE::EQ)
      .Case("LE", CMPMODE::LE)
      .Case("GE", CMPMODE::GE)
      .Case("NE", CMPMODE::NE)
      .Default(::std::nullopt);
}
::std::optional<CMPMODE> symbolizeCMPMODE(uint64_t value) {
  switch (value) {
  case 0: return CMPMODE::LT;
  case 1: return CMPMODE::GT;
  case 2: return CMPMODE::EQ;
  case 3: return CMPMODE::LE;
  case 4: return CMPMODE::GE;
  case 5: return CMPMODE::NE;
  default: return ::std::nullopt;
  }
}

bool CMPMODEAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 5)))));
}
CMPMODEAttr CMPMODEAttr::get(::mlir::MLIRContext *context, CMPMODE val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<CMPMODEAttr>(baseAttr);
}
CMPMODE CMPMODEAttr::getValue() const {
  return static_cast<CMPMODE>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCO2Layout(CO2Layout val) {
  switch (val) {
    case CO2Layout::NZ: return "NZ";
    case CO2Layout::ROW_MAJOR: return "ROW_MAJOR";
    case CO2Layout::COLUMN_MAJOR: return "COLUMN_MAJOR";
  }
  return "";
}

::std::optional<CO2Layout> symbolizeCO2Layout(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CO2Layout>>(str)
      .Case("NZ", CO2Layout::NZ)
      .Case("ROW_MAJOR", CO2Layout::ROW_MAJOR)
      .Case("COLUMN_MAJOR", CO2Layout::COLUMN_MAJOR)
      .Default(::std::nullopt);
}
::std::optional<CO2Layout> symbolizeCO2Layout(uint8_t value) {
  switch (value) {
  case 0: return CO2Layout::NZ;
  case 1: return CO2Layout::ROW_MAJOR;
  case 2: return CO2Layout::COLUMN_MAJOR;
  default: return ::std::nullopt;
  }
}

bool CO2LayoutAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))));
}
CO2LayoutAttr CO2LayoutAttr::get(::mlir::MLIRContext *context, CO2Layout val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<CO2LayoutAttr>(baseAttr);
}
CO2Layout CO2LayoutAttr::getValue() const {
  return static_cast<CO2Layout>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCacheLine(CacheLine val) {
  switch (val) {
    case CacheLine::SINGLE_CACHE_LINE: return "single_cache_line";
    case CacheLine::ENTIRE_DATA_CACHE: return "entire_data_cache";
  }
  return "";
}

::std::optional<CacheLine> symbolizeCacheLine(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CacheLine>>(str)
      .Case("single_cache_line", CacheLine::SINGLE_CACHE_LINE)
      .Case("entire_data_cache", CacheLine::ENTIRE_DATA_CACHE)
      .Default(::std::nullopt);
}
::std::optional<CacheLine> symbolizeCacheLine(uint8_t value) {
  switch (value) {
  case 0: return CacheLine::SINGLE_CACHE_LINE;
  case 1: return CacheLine::ENTIRE_DATA_CACHE;
  default: return ::std::nullopt;
  }
}

bool CacheLineAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
CacheLineAttr CacheLineAttr::get(::mlir::MLIRContext *context, CacheLine val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<CacheLineAttr>(baseAttr);
}
CacheLine CacheLineAttr::getValue() const {
  return static_cast<CacheLine>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCacheMode(CacheMode val) {
  switch (val) {
    case CacheMode::CACHE_MODE_DISABLE: return "CACHE_MODE_DISABLE";
    case CacheMode::CACHE_MODE_NORMAL: return "CACHE_MODE_NORMAL";
  }
  return "";
}

::std::optional<CacheMode> symbolizeCacheMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CacheMode>>(str)
      .Case("CACHE_MODE_DISABLE", CacheMode::CACHE_MODE_DISABLE)
      .Case("CACHE_MODE_NORMAL", CacheMode::CACHE_MODE_NORMAL)
      .Default(::std::nullopt);
}
::std::optional<CacheMode> symbolizeCacheMode(uint8_t value) {
  switch (value) {
  case 0: return CacheMode::CACHE_MODE_DISABLE;
  case 1: return CacheMode::CACHE_MODE_NORMAL;
  default: return ::std::nullopt;
  }
}

bool CacheModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
CacheModeAttr CacheModeAttr::get(::mlir::MLIRContext *context, CacheMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<CacheModeAttr>(baseAttr);
}
CacheMode CacheModeAttr::getValue() const {
  return static_cast<CacheMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCacheRwMode(CacheRwMode val) {
  switch (val) {
    case CacheRwMode::READ: return "READ";
    case CacheRwMode::WRITE: return "WRITE";
    case CacheRwMode::RW: return "RW";
  }
  return "";
}

::std::optional<CacheRwMode> symbolizeCacheRwMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CacheRwMode>>(str)
      .Case("READ", CacheRwMode::READ)
      .Case("WRITE", CacheRwMode::WRITE)
      .Case("RW", CacheRwMode::RW)
      .Default(::std::nullopt);
}
::std::optional<CacheRwMode> symbolizeCacheRwMode(uint8_t value) {
  switch (value) {
  case 1: return CacheRwMode::READ;
  case 2: return CacheRwMode::WRITE;
  case 3: return CacheRwMode::RW;
  default: return ::std::nullopt;
  }
}

bool CacheRwModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
CacheRwModeAttr CacheRwModeAttr::get(::mlir::MLIRContext *context, CacheRwMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<CacheRwModeAttr>(baseAttr);
}
CacheRwMode CacheRwModeAttr::getValue() const {
  return static_cast<CacheRwMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCopyDirection(CopyDirection val) {
  switch (val) {
    case CopyDirection::Unknown: return "Unknown";
    case CopyDirection::gm_ubuf: return "gm_ubuf";
    case CopyDirection::ubuf_gm: return "ubuf_gm";
    case CopyDirection::ubuf_ubuf: return "ubuf_ubuf";
    case CopyDirection::gm_gm: return "gm_gm";
  }
  return "";
}

::std::optional<CopyDirection> symbolizeCopyDirection(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CopyDirection>>(str)
      .Case("Unknown", CopyDirection::Unknown)
      .Case("gm_ubuf", CopyDirection::gm_ubuf)
      .Case("ubuf_gm", CopyDirection::ubuf_gm)
      .Case("ubuf_ubuf", CopyDirection::ubuf_ubuf)
      .Case("gm_gm", CopyDirection::gm_gm)
      .Default(::std::nullopt);
}
::std::optional<CopyDirection> symbolizeCopyDirection(uint32_t value) {
  switch (value) {
  case 0: return CopyDirection::Unknown;
  case 1: return CopyDirection::gm_ubuf;
  case 2: return CopyDirection::ubuf_gm;
  case 3: return CopyDirection::ubuf_ubuf;
  case 4: return CopyDirection::gm_gm;
  default: return ::std::nullopt;
  }
}

bool CopyDirectionAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))));
}
CopyDirectionAttr CopyDirectionAttr::get(::mlir::MLIRContext *context, CopyDirection val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<CopyDirectionAttr>(baseAttr);
}
CopyDirection CopyDirectionAttr::getValue() const {
  return static_cast<CopyDirection>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyCubeFormat(CubeFormat val) {
  switch (val) {
    case CubeFormat::ND: return "nd";
    case CubeFormat::NZ: return "nz";
    case CubeFormat::ZN: return "zn";
    case CubeFormat::ZZ: return "zz";
    case CubeFormat::NN: return "nn";
    case CubeFormat::ND_ALIGN: return "nd_align";
    case CubeFormat::SCALAR: return "scalar";
    case CubeFormat::VECTOR: return "vector";
  }
  return "";
}

::std::optional<CubeFormat> symbolizeCubeFormat(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CubeFormat>>(str)
      .Case("nd", CubeFormat::ND)
      .Case("nz", CubeFormat::NZ)
      .Case("zn", CubeFormat::ZN)
      .Case("zz", CubeFormat::ZZ)
      .Case("nn", CubeFormat::NN)
      .Case("nd_align", CubeFormat::ND_ALIGN)
      .Case("scalar", CubeFormat::SCALAR)
      .Case("vector", CubeFormat::VECTOR)
      .Default(::std::nullopt);
}
::std::optional<CubeFormat> symbolizeCubeFormat(uint8_t value) {
  switch (value) {
  case 0: return CubeFormat::ND;
  case 1: return CubeFormat::NZ;
  case 2: return CubeFormat::ZN;
  case 3: return CubeFormat::ZZ;
  case 4: return CubeFormat::NN;
  case 5: return CubeFormat::ND_ALIGN;
  case 6: return CubeFormat::SCALAR;
  case 7: return CubeFormat::VECTOR;
  default: return ::std::nullopt;
  }
}

bool CubeFormatAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))));
}
CubeFormatAttr CubeFormatAttr::get(::mlir::MLIRContext *context, CubeFormat val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<CubeFormatAttr>(baseAttr);
}
CubeFormat CubeFormatAttr::getValue() const {
  return static_cast<CubeFormat>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyDcciDst(DcciDst val) {
  switch (val) {
    case DcciDst::CACHELINE_ALL: return "cacheline_all";
    case DcciDst::CACHELINE_UB: return "cacheline_ub";
    case DcciDst::CACHELINE_OUT: return "cacheline_out";
    case DcciDst::CACHELINE_ATOMIC: return "cacheline_atomic";
  }
  return "";
}

::std::optional<DcciDst> symbolizeDcciDst(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DcciDst>>(str)
      .Case("cacheline_all", DcciDst::CACHELINE_ALL)
      .Case("cacheline_ub", DcciDst::CACHELINE_UB)
      .Case("cacheline_out", DcciDst::CACHELINE_OUT)
      .Case("cacheline_atomic", DcciDst::CACHELINE_ATOMIC)
      .Default(::std::nullopt);
}
::std::optional<DcciDst> symbolizeDcciDst(uint8_t value) {
  switch (value) {
  case 0: return DcciDst::CACHELINE_ALL;
  case 1: return DcciDst::CACHELINE_UB;
  case 2: return DcciDst::CACHELINE_OUT;
  case 3: return DcciDst::CACHELINE_ATOMIC;
  default: return ::std::nullopt;
  }
}

bool DcciDstAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
DcciDstAttr DcciDstAttr::get(::mlir::MLIRContext *context, DcciDst val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<DcciDstAttr>(baseAttr);
}
DcciDst DcciDstAttr::getValue() const {
  return static_cast<DcciDst>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyFmatrixMode(FmatrixMode val) {
  switch (val) {
    case FmatrixMode::FMATRIX_LEFT: return "fmatrix_left";
    case FmatrixMode::FMATRIX_RIGHT: return "fmatrix_right";
  }
  return "";
}

::std::optional<FmatrixMode> symbolizeFmatrixMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<FmatrixMode>>(str)
      .Case("fmatrix_left", FmatrixMode::FMATRIX_LEFT)
      .Case("fmatrix_right", FmatrixMode::FMATRIX_RIGHT)
      .Default(::std::nullopt);
}
::std::optional<FmatrixMode> symbolizeFmatrixMode(uint8_t value) {
  switch (value) {
  case 0: return FmatrixMode::FMATRIX_LEFT;
  case 1: return FmatrixMode::FMATRIX_RIGHT;
  default: return ::std::nullopt;
  }
}

bool FmatrixModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
FmatrixModeAttr FmatrixModeAttr::get(::mlir::MLIRContext *context, FmatrixMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<FmatrixModeAttr>(baseAttr);
}
FmatrixMode FmatrixModeAttr::getValue() const {
  return static_cast<FmatrixMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyGatherMaskMode(GatherMaskMode val) {
  switch (val) {
    case GatherMaskMode::DEFAULT: return "DEFAULT";
  }
  return "";
}

::std::optional<GatherMaskMode> symbolizeGatherMaskMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<GatherMaskMode>>(str)
      .Case("DEFAULT", GatherMaskMode::DEFAULT)
      .Default(::std::nullopt);
}
::std::optional<GatherMaskMode> symbolizeGatherMaskMode(uint8_t value) {
  switch (value) {
  case 0: return GatherMaskMode::DEFAULT;
  default: return ::std::nullopt;
  }
}

bool GatherMaskModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0))));
}
GatherMaskModeAttr GatherMaskModeAttr::get(::mlir::MLIRContext *context, GatherMaskMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<GatherMaskModeAttr>(baseAttr);
}
GatherMaskMode GatherMaskModeAttr::getValue() const {
  return static_cast<GatherMaskMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyHardEvent(HardEvent val) {
  switch (val) {
    case HardEvent::MTE2_MTE1: return "mte2_mte1";
    case HardEvent::MTE1_MTE2: return "mte1_mte2";
    case HardEvent::MTE1_M: return "mte1_m";
    case HardEvent::M_MTE1: return "m_mte1";
    case HardEvent::MTE2_V: return "mte2_v";
    case HardEvent::V_MTE2: return "v_mte2";
    case HardEvent::MTE3_V: return "mte3_v";
    case HardEvent::V_MTE3: return "v_mte3";
    case HardEvent::M_V: return "m_v";
    case HardEvent::V_M: return "v_m";
    case HardEvent::V_V: return "v_v";
    case HardEvent::MTE3_MTE1: return "mte3_mte1";
    case HardEvent::MTE1_MTE3: return "mte1_mte3";
    case HardEvent::MTE1_V: return "mte1_v";
    case HardEvent::MTE2_M: return "mte2_m";
    case HardEvent::M_MTE2: return "m_mte2";
    case HardEvent::V_MTE1: return "v_mte1";
    case HardEvent::M_FIX: return "m_fix";
    case HardEvent::FIX_M: return "fix_m";
    case HardEvent::MTE3_MTE2: return "mte3_mte2";
    case HardEvent::MTE2_MTE3: return "mte2_mte3";
    case HardEvent::S_V: return "s_v";
    case HardEvent::V_S: return "v_s";
    case HardEvent::S_MTE2: return "s_mte2";
    case HardEvent::MTE2_S: return "mte2_s";
    case HardEvent::S_MTE3: return "s_mte3";
    case HardEvent::MTE3_S: return "mte3_s";
    case HardEvent::MTE2_FIX: return "mte2_fix";
    case HardEvent::FIX_MTE2: return "fix_mte2";
    case HardEvent::FIX_S: return "fix_s";
    case HardEvent::M_S: return "m_s";
    case HardEvent::FIX_MTE3: return "fix_mte3";
    case HardEvent::MTE1_FIX: return "mte1_fix";
    case HardEvent::FIX_MTE1: return "fix_mte1";
    case HardEvent::FIX_FIX: return "fix_fix";
    case HardEvent::MAX: return "max";
  }
  return "";
}

::std::optional<HardEvent> symbolizeHardEvent(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<HardEvent>>(str)
      .Case("mte2_mte1", HardEvent::MTE2_MTE1)
      .Case("mte1_mte2", HardEvent::MTE1_MTE2)
      .Case("mte1_m", HardEvent::MTE1_M)
      .Case("m_mte1", HardEvent::M_MTE1)
      .Case("mte2_v", HardEvent::MTE2_V)
      .Case("v_mte2", HardEvent::V_MTE2)
      .Case("mte3_v", HardEvent::MTE3_V)
      .Case("v_mte3", HardEvent::V_MTE3)
      .Case("m_v", HardEvent::M_V)
      .Case("v_m", HardEvent::V_M)
      .Case("v_v", HardEvent::V_V)
      .Case("mte3_mte1", HardEvent::MTE3_MTE1)
      .Case("mte1_mte3", HardEvent::MTE1_MTE3)
      .Case("mte1_v", HardEvent::MTE1_V)
      .Case("mte2_m", HardEvent::MTE2_M)
      .Case("m_mte2", HardEvent::M_MTE2)
      .Case("v_mte1", HardEvent::V_MTE1)
      .Case("m_fix", HardEvent::M_FIX)
      .Case("fix_m", HardEvent::FIX_M)
      .Case("mte3_mte2", HardEvent::MTE3_MTE2)
      .Case("mte2_mte3", HardEvent::MTE2_MTE3)
      .Case("s_v", HardEvent::S_V)
      .Case("v_s", HardEvent::V_S)
      .Case("s_mte2", HardEvent::S_MTE2)
      .Case("mte2_s", HardEvent::MTE2_S)
      .Case("s_mte3", HardEvent::S_MTE3)
      .Case("mte3_s", HardEvent::MTE3_S)
      .Case("mte2_fix", HardEvent::MTE2_FIX)
      .Case("fix_mte2", HardEvent::FIX_MTE2)
      .Case("fix_s", HardEvent::FIX_S)
      .Case("m_s", HardEvent::M_S)
      .Case("fix_mte3", HardEvent::FIX_MTE3)
      .Case("mte1_fix", HardEvent::MTE1_FIX)
      .Case("fix_mte1", HardEvent::FIX_MTE1)
      .Case("fix_fix", HardEvent::FIX_FIX)
      .Case("max", HardEvent::MAX)
      .Default(::std::nullopt);
}
::std::optional<HardEvent> symbolizeHardEvent(uint8_t value) {
  switch (value) {
  case 0: return HardEvent::MTE2_MTE1;
  case 1: return HardEvent::MTE1_MTE2;
  case 2: return HardEvent::MTE1_M;
  case 3: return HardEvent::M_MTE1;
  case 4: return HardEvent::MTE2_V;
  case 5: return HardEvent::V_MTE2;
  case 6: return HardEvent::MTE3_V;
  case 7: return HardEvent::V_MTE3;
  case 8: return HardEvent::M_V;
  case 9: return HardEvent::V_M;
  case 10: return HardEvent::V_V;
  case 11: return HardEvent::MTE3_MTE1;
  case 12: return HardEvent::MTE1_MTE3;
  case 13: return HardEvent::MTE1_V;
  case 14: return HardEvent::MTE2_M;
  case 15: return HardEvent::M_MTE2;
  case 16: return HardEvent::V_MTE1;
  case 17: return HardEvent::M_FIX;
  case 18: return HardEvent::FIX_M;
  case 19: return HardEvent::MTE3_MTE2;
  case 20: return HardEvent::MTE2_MTE3;
  case 21: return HardEvent::S_V;
  case 22: return HardEvent::V_S;
  case 23: return HardEvent::S_MTE2;
  case 24: return HardEvent::MTE2_S;
  case 25: return HardEvent::S_MTE3;
  case 26: return HardEvent::MTE3_S;
  case 27: return HardEvent::MTE2_FIX;
  case 28: return HardEvent::FIX_MTE2;
  case 29: return HardEvent::FIX_S;
  case 30: return HardEvent::M_S;
  case 31: return HardEvent::FIX_MTE3;
  case 32: return HardEvent::MTE1_FIX;
  case 33: return HardEvent::FIX_MTE1;
  case 34: return HardEvent::FIX_FIX;
  case 35: return HardEvent::MAX;
  default: return ::std::nullopt;
  }
}

bool HardEventAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 12)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 13)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 14)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 15)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 16)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 17)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 18)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 19)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 20)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 21)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 22)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 23)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 24)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 25)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 26)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 27)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 28)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 29)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 30)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 31)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 32)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 33)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 34)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 35)))));
}
HardEventAttr HardEventAttr::get(::mlir::MLIRContext *context, HardEvent val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<HardEventAttr>(baseAttr);
}
HardEvent HardEventAttr::getValue() const {
  return static_cast<HardEvent>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyIterateMode(IterateMode val) {
  switch (val) {
    case IterateMode::ITERATE_MODE_NORMAL: return "ITERATE_MODE_NORMAL";
    case IterateMode::ITERATE_MODE_ALL: return "ITERATE_MODE_ALL";
    case IterateMode::ITERATE_MODE_BATCH: return "ITERATE_MODE_BATCH";
    case IterateMode::ITERATE_MODE_N_BATCH: return "ITERATE_MODE_N_BATCH";
    case IterateMode::ITERATE_MODE_DEFAULT: return "ITERATE_MODE_DEFAULT";
  }
  return "";
}

::std::optional<IterateMode> symbolizeIterateMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<IterateMode>>(str)
      .Case("ITERATE_MODE_NORMAL", IterateMode::ITERATE_MODE_NORMAL)
      .Case("ITERATE_MODE_ALL", IterateMode::ITERATE_MODE_ALL)
      .Case("ITERATE_MODE_BATCH", IterateMode::ITERATE_MODE_BATCH)
      .Case("ITERATE_MODE_N_BATCH", IterateMode::ITERATE_MODE_N_BATCH)
      .Case("ITERATE_MODE_DEFAULT", IterateMode::ITERATE_MODE_DEFAULT)
      .Default(::std::nullopt);
}
::std::optional<IterateMode> symbolizeIterateMode(uint8_t value) {
  switch (value) {
  case 1: return IterateMode::ITERATE_MODE_NORMAL;
  case 2: return IterateMode::ITERATE_MODE_ALL;
  case 4: return IterateMode::ITERATE_MODE_BATCH;
  case 8: return IterateMode::ITERATE_MODE_N_BATCH;
  case 255: return IterateMode::ITERATE_MODE_DEFAULT;
  default: return ::std::nullopt;
  }
}

bool IterateModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 255)))));
}
IterateModeAttr IterateModeAttr::get(::mlir::MLIRContext *context, IterateMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<IterateModeAttr>(baseAttr);
}
IterateMode IterateModeAttr::getValue() const {
  return static_cast<IterateMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyIterateOrder(IterateOrder val) {
  switch (val) {
    case IterateOrder::ORDER_M: return "ORDER_M";
    case IterateOrder::ORDER_N: return "ORDER_N";
    case IterateOrder::UNDEF: return "UNDEF";
  }
  return "";
}

::std::optional<IterateOrder> symbolizeIterateOrder(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<IterateOrder>>(str)
      .Case("ORDER_M", IterateOrder::ORDER_M)
      .Case("ORDER_N", IterateOrder::ORDER_N)
      .Case("UNDEF", IterateOrder::UNDEF)
      .Default(::std::nullopt);
}
::std::optional<IterateOrder> symbolizeIterateOrder(uint8_t value) {
  switch (value) {
  case 0: return IterateOrder::ORDER_M;
  case 1: return IterateOrder::ORDER_N;
  case 2: return IterateOrder::UNDEF;
  default: return ::std::nullopt;
  }
}

bool IterateOrderAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))));
}
IterateOrderAttr IterateOrderAttr::get(::mlir::MLIRContext *context, IterateOrder val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<IterateOrderAttr>(baseAttr);
}
IterateOrder IterateOrderAttr::getValue() const {
  return static_cast<IterateOrder>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyLayoutMode(LayoutMode val) {
  switch (val) {
    case LayoutMode::NONE: return "none";
    case LayoutMode::NORMAL: return "normal";
    case LayoutMode::BSNGD: return "bsngd";
    case LayoutMode::SBNGD: return "sbngd";
    case LayoutMode::BNGS1S2: return "bngs1s2";
  }
  return "";
}

::std::optional<LayoutMode> symbolizeLayoutMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<LayoutMode>>(str)
      .Case("none", LayoutMode::NONE)
      .Case("normal", LayoutMode::NORMAL)
      .Case("bsngd", LayoutMode::BSNGD)
      .Case("sbngd", LayoutMode::SBNGD)
      .Case("bngs1s2", LayoutMode::BNGS1S2)
      .Default(::std::nullopt);
}
::std::optional<LayoutMode> symbolizeLayoutMode(uint8_t value) {
  switch (value) {
  case 0: return LayoutMode::NONE;
  case 1: return LayoutMode::NORMAL;
  case 2: return LayoutMode::BSNGD;
  case 3: return LayoutMode::SBNGD;
  case 4: return LayoutMode::BNGS1S2;
  default: return ::std::nullopt;
  }
}

bool LayoutModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))));
}
LayoutModeAttr LayoutModeAttr::get(::mlir::MLIRContext *context, LayoutMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<LayoutModeAttr>(baseAttr);
}
LayoutMode LayoutModeAttr::getValue() const {
  return static_cast<LayoutMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyMaskMode(MaskMode val) {
  switch (val) {
    case MaskMode::NORMAL: return "NORMAL";
    case MaskMode::COUNTER: return "COUNTER";
  }
  return "";
}

::std::optional<MaskMode> symbolizeMaskMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MaskMode>>(str)
      .Case("NORMAL", MaskMode::NORMAL)
      .Case("COUNTER", MaskMode::COUNTER)
      .Default(::std::nullopt);
}
::std::optional<MaskMode> symbolizeMaskMode(uint8_t value) {
  switch (value) {
  case 0: return MaskMode::NORMAL;
  case 1: return MaskMode::COUNTER;
  default: return ::std::nullopt;
  }
}

bool MaskModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
MaskModeAttr MaskModeAttr::get(::mlir::MLIRContext *context, MaskMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<MaskModeAttr>(baseAttr);
}
MaskMode MaskModeAttr::getValue() const {
  return static_cast<MaskMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace Detail::Impl::AscendC {

::llvm::StringRef stringifyMatmulPolicy(MatmulPolicy val) {
  switch (val) {
    case MatmulPolicy::MatmulPolicy: return "MatmulPolicy";
    case MatmulPolicy::TrianUpperMatmulPolicy: return "TrianUpperMatmulPolicy";
    case MatmulPolicy::TrianLowerMatmulPolicy: return "TrianLowerMatmulPolicy";
    case MatmulPolicy::NBuffer33MatmulPolicy: return "NBuffer33MatmulPolicy";
  }
  return "";
}

::std::optional<MatmulPolicy> symbolizeMatmulPolicy(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MatmulPolicy>>(str)
      .Case("MatmulPolicy", MatmulPolicy::MatmulPolicy)
      .Case("TrianUpperMatmulPolicy", MatmulPolicy::TrianUpperMatmulPolicy)
      .Case("TrianLowerMatmulPolicy", MatmulPolicy::TrianLowerMatmulPolicy)
      .Case("NBuffer33MatmulPolicy", MatmulPolicy::NBuffer33MatmulPolicy)
      .Default(::std::nullopt);
}
::std::optional<MatmulPolicy> symbolizeMatmulPolicy(uint8_t value) {
  switch (value) {
  case 0: return MatmulPolicy::MatmulPolicy;
  case 1: return MatmulPolicy::TrianUpperMatmulPolicy;
  case 2: return MatmulPolicy::TrianLowerMatmulPolicy;
  case 3: return MatmulPolicy::NBuffer33MatmulPolicy;
  default: return ::std::nullopt;
  }
}

bool MatmulPolicyAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
MatmulPolicyAttr MatmulPolicyAttr::get(::mlir::MLIRContext *context, MatmulPolicy val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<MatmulPolicyAttr>(baseAttr);
}
MatmulPolicy MatmulPolicyAttr::getValue() const {
  return static_cast<MatmulPolicy>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace Detail::Impl::AscendC
namespace mlir::ascendc {

::llvm::StringRef stringifyMemDsbT(MemDsbT val) {
  switch (val) {
    case MemDsbT::ALL: return "all";
    case MemDsbT::DDR: return "ddr";
    case MemDsbT::UB: return "ub";
    case MemDsbT::SEQ: return "seq";
  }
  return "";
}

::std::optional<MemDsbT> symbolizeMemDsbT(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MemDsbT>>(str)
      .Case("all", MemDsbT::ALL)
      .Case("ddr", MemDsbT::DDR)
      .Case("ub", MemDsbT::UB)
      .Case("seq", MemDsbT::SEQ)
      .Default(::std::nullopt);
}
::std::optional<MemDsbT> symbolizeMemDsbT(uint8_t value) {
  switch (value) {
  case 0: return MemDsbT::ALL;
  case 1: return MemDsbT::DDR;
  case 2: return MemDsbT::UB;
  case 3: return MemDsbT::SEQ;
  default: return ::std::nullopt;
  }
}

bool MemDsbTAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
MemDsbTAttr MemDsbTAttr::get(::mlir::MLIRContext *context, MemDsbT val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<MemDsbTAttr>(baseAttr);
}
MemDsbT MemDsbTAttr::getValue() const {
  return static_cast<MemDsbT>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyPipe(Pipe val) {
  switch (val) {
    case Pipe::PIPE_S: return "pipe_s";
    case Pipe::PIPE_V: return "pipe_v";
    case Pipe::PIPE_M: return "pipe_m";
    case Pipe::PIPE_MTE1: return "pipe_mte1";
    case Pipe::PIPE_MTE2: return "pipe_mte2";
    case Pipe::PIPE_MTE3: return "pipe_mte3";
    case Pipe::PIPE_ALL: return "pipe_all";
    case Pipe::PIPE_MTE4: return "pipe_mte4";
    case Pipe::PIPE_MTE5: return "pipe_mte5";
    case Pipe::PIPE_V2: return "pipe_v2";
    case Pipe::PIPE_FIX: return "pipe_fix";
  }
  return "";
}

::std::optional<Pipe> symbolizePipe(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Pipe>>(str)
      .Case("pipe_s", Pipe::PIPE_S)
      .Case("pipe_v", Pipe::PIPE_V)
      .Case("pipe_m", Pipe::PIPE_M)
      .Case("pipe_mte1", Pipe::PIPE_MTE1)
      .Case("pipe_mte2", Pipe::PIPE_MTE2)
      .Case("pipe_mte3", Pipe::PIPE_MTE3)
      .Case("pipe_all", Pipe::PIPE_ALL)
      .Case("pipe_mte4", Pipe::PIPE_MTE4)
      .Case("pipe_mte5", Pipe::PIPE_MTE5)
      .Case("pipe_v2", Pipe::PIPE_V2)
      .Case("pipe_fix", Pipe::PIPE_FIX)
      .Default(::std::nullopt);
}
::std::optional<Pipe> symbolizePipe(uint8_t value) {
  switch (value) {
  case 0: return Pipe::PIPE_S;
  case 1: return Pipe::PIPE_V;
  case 2: return Pipe::PIPE_M;
  case 3: return Pipe::PIPE_MTE1;
  case 4: return Pipe::PIPE_MTE2;
  case 5: return Pipe::PIPE_MTE3;
  case 6: return Pipe::PIPE_ALL;
  case 7: return Pipe::PIPE_MTE4;
  case 8: return Pipe::PIPE_MTE5;
  case 9: return Pipe::PIPE_V2;
  case 10: return Pipe::PIPE_FIX;
  default: return ::std::nullopt;
  }
}

bool PipeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))));
}
PipeAttr PipeAttr::get(::mlir::MLIRContext *context, Pipe val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<PipeAttr>(baseAttr);
}
Pipe PipeAttr::getValue() const {
  return static_cast<Pipe>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyQuantMode(QuantMode val) {
  switch (val) {
    case QuantMode::NoQuant: return "no_quant";
    case QuantMode::F322F16: return "f322f16";
    case QuantMode::VQF322HIF8_PRE: return "vqf322hif8_pre";
    case QuantMode::QF322HIF8_PRE: return "qf322hif8_pre";
    case QuantMode::VQF322HIF8_PRE_HYBRID: return "vqf322hif8_pre_hybrid";
    case QuantMode::QF322HIF8_PRE_HYBRID: return "qf322hif8_pre_hybrid";
    case QuantMode::AttachF16Mul: return "attachf16mul";
    case QuantMode::VREQ8: return "vreq8";
    case QuantMode::REQ8: return "req8";
    case QuantMode::VDEQF16: return "vdeqf16";
    case QuantMode::DEQF16: return "deqf16";
    case QuantMode::VQF322FP8_PRE: return "vqf322fp8_pre";
    case QuantMode::QF322FP8_PRE: return "qf322fp8_pre";
    case QuantMode::VQF322F32_PRE: return "vqf322f32_pre";
    case QuantMode::QF322F32_PRE: return "qf322f32_pre";
    case QuantMode::F322BF16: return "f322bf16";
    case QuantMode::VQF162B8_PRE: return "vqf162b8_pre";
    case QuantMode::QF162B8_PRE: return "qf162b8_pre";
    case QuantMode::VQF162S4_PRE: return "vqf162s4_pre";
    case QuantMode::QF162S4_PRE: return "qf162s4_pre";
    case QuantMode::VREQ4: return "vreq4";
    case QuantMode::REQ4: return "req4";
    case QuantMode::VQF322B8_PRE: return "vqf322b8_pre";
    case QuantMode::QF322B8_PRE: return "qf322b8_pre";
    case QuantMode::VQF322S4_PRE: return "vqf322s4_pre";
    case QuantMode::QF322S4_PRE: return "qf322s4_pre";
    case QuantMode::VDEQS16: return "vdeqs16";
    case QuantMode::DEQS16: return "deqs16";
    case QuantMode::VQF162S16_PRE: return "vqf162s16_pre";
    case QuantMode::QF162S16_PRE: return "qf162s16_pre";
    case QuantMode::VQF322F16_PRE: return "vqf322f16_pre";
    case QuantMode::QF322F16_PRE: return "qf322f16_pre";
    case QuantMode::VQF322BF16_PRE: return "vqf322bf16_pre";
    case QuantMode::QF322BF16_PRE: return "qf322bf16_pre";
    case QuantMode::VQS322BF16_PRE: return "vqs322bf16_pre";
    case QuantMode::QS322BF16_PRE: return "qs322bf16_pre";
  }
  return "";
}

::std::optional<QuantMode> symbolizeQuantMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<QuantMode>>(str)
      .Case("no_quant", QuantMode::NoQuant)
      .Case("f322f16", QuantMode::F322F16)
      .Case("vqf322hif8_pre", QuantMode::VQF322HIF8_PRE)
      .Case("qf322hif8_pre", QuantMode::QF322HIF8_PRE)
      .Case("vqf322hif8_pre_hybrid", QuantMode::VQF322HIF8_PRE_HYBRID)
      .Case("qf322hif8_pre_hybrid", QuantMode::QF322HIF8_PRE_HYBRID)
      .Case("attachf16mul", QuantMode::AttachF16Mul)
      .Case("vreq8", QuantMode::VREQ8)
      .Case("req8", QuantMode::REQ8)
      .Case("vdeqf16", QuantMode::VDEQF16)
      .Case("deqf16", QuantMode::DEQF16)
      .Case("vqf322fp8_pre", QuantMode::VQF322FP8_PRE)
      .Case("qf322fp8_pre", QuantMode::QF322FP8_PRE)
      .Case("vqf322f32_pre", QuantMode::VQF322F32_PRE)
      .Case("qf322f32_pre", QuantMode::QF322F32_PRE)
      .Case("f322bf16", QuantMode::F322BF16)
      .Case("vqf162b8_pre", QuantMode::VQF162B8_PRE)
      .Case("qf162b8_pre", QuantMode::QF162B8_PRE)
      .Case("vqf162s4_pre", QuantMode::VQF162S4_PRE)
      .Case("qf162s4_pre", QuantMode::QF162S4_PRE)
      .Case("vreq4", QuantMode::VREQ4)
      .Case("req4", QuantMode::REQ4)
      .Case("vqf322b8_pre", QuantMode::VQF322B8_PRE)
      .Case("qf322b8_pre", QuantMode::QF322B8_PRE)
      .Case("vqf322s4_pre", QuantMode::VQF322S4_PRE)
      .Case("qf322s4_pre", QuantMode::QF322S4_PRE)
      .Case("vdeqs16", QuantMode::VDEQS16)
      .Case("deqs16", QuantMode::DEQS16)
      .Case("vqf162s16_pre", QuantMode::VQF162S16_PRE)
      .Case("qf162s16_pre", QuantMode::QF162S16_PRE)
      .Case("vqf322f16_pre", QuantMode::VQF322F16_PRE)
      .Case("qf322f16_pre", QuantMode::QF322F16_PRE)
      .Case("vqf322bf16_pre", QuantMode::VQF322BF16_PRE)
      .Case("qf322bf16_pre", QuantMode::QF322BF16_PRE)
      .Case("vqs322bf16_pre", QuantMode::VQS322BF16_PRE)
      .Case("qs322bf16_pre", QuantMode::QS322BF16_PRE)
      .Default(::std::nullopt);
}
::std::optional<QuantMode> symbolizeQuantMode(uint32_t value) {
  switch (value) {
  case 0: return QuantMode::NoQuant;
  case 1: return QuantMode::F322F16;
  case 2: return QuantMode::VQF322HIF8_PRE;
  case 3: return QuantMode::QF322HIF8_PRE;
  case 4: return QuantMode::VQF322HIF8_PRE_HYBRID;
  case 5: return QuantMode::QF322HIF8_PRE_HYBRID;
  case 6: return QuantMode::AttachF16Mul;
  case 8: return QuantMode::VREQ8;
  case 9: return QuantMode::REQ8;
  case 10: return QuantMode::VDEQF16;
  case 11: return QuantMode::DEQF16;
  case 12: return QuantMode::VQF322FP8_PRE;
  case 13: return QuantMode::QF322FP8_PRE;
  case 14: return QuantMode::VQF322F32_PRE;
  case 15: return QuantMode::QF322F32_PRE;
  case 16: return QuantMode::F322BF16;
  case 17: return QuantMode::VQF162B8_PRE;
  case 18: return QuantMode::QF162B8_PRE;
  case 19: return QuantMode::VQF162S4_PRE;
  case 20: return QuantMode::QF162S4_PRE;
  case 21: return QuantMode::VREQ4;
  case 22: return QuantMode::REQ4;
  case 23: return QuantMode::VQF322B8_PRE;
  case 24: return QuantMode::QF322B8_PRE;
  case 25: return QuantMode::VQF322S4_PRE;
  case 26: return QuantMode::QF322S4_PRE;
  case 27: return QuantMode::VDEQS16;
  case 28: return QuantMode::DEQS16;
  case 29: return QuantMode::VQF162S16_PRE;
  case 30: return QuantMode::QF162S16_PRE;
  case 31: return QuantMode::VQF322F16_PRE;
  case 32: return QuantMode::QF322F16_PRE;
  case 33: return QuantMode::VQF322BF16_PRE;
  case 34: return QuantMode::QF322BF16_PRE;
  case 35: return QuantMode::VQS322BF16_PRE;
  case 36: return QuantMode::QS322BF16_PRE;
  default: return ::std::nullopt;
  }
}

bool QuantModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 12)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 13)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 14)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 15)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 16)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 17)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 18)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 19)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 20)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 21)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 22)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 23)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 24)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 25)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 26)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 27)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 28)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 29)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 30)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 31)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 32)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 33)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 34)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 35)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 36)))));
}
QuantModeAttr QuantModeAttr::get(::mlir::MLIRContext *context, QuantMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<QuantModeAttr>(baseAttr);
}
QuantMode QuantModeAttr::getValue() const {
  return static_cast<QuantMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyReduceOrder(ReduceOrder val) {
  switch (val) {
    case ReduceOrder::ORDER_VALUE_INDEX: return "order_value_index";
    case ReduceOrder::ORDER_INDEX_VALUE: return "order_index_value";
    case ReduceOrder::ORDER_ONLY_VALUE: return "order_only_value";
    case ReduceOrder::ORDER_ONLY_INDEX: return "order_only_index";
  }
  return "";
}

::std::optional<ReduceOrder> symbolizeReduceOrder(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ReduceOrder>>(str)
      .Case("order_value_index", ReduceOrder::ORDER_VALUE_INDEX)
      .Case("order_index_value", ReduceOrder::ORDER_INDEX_VALUE)
      .Case("order_only_value", ReduceOrder::ORDER_ONLY_VALUE)
      .Case("order_only_index", ReduceOrder::ORDER_ONLY_INDEX)
      .Default(::std::nullopt);
}
::std::optional<ReduceOrder> symbolizeReduceOrder(uint8_t value) {
  switch (value) {
  case 0: return ReduceOrder::ORDER_VALUE_INDEX;
  case 1: return ReduceOrder::ORDER_INDEX_VALUE;
  case 2: return ReduceOrder::ORDER_ONLY_VALUE;
  case 3: return ReduceOrder::ORDER_ONLY_INDEX;
  default: return ::std::nullopt;
  }
}

bool ReduceOrderAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))));
}
ReduceOrderAttr ReduceOrderAttr::get(::mlir::MLIRContext *context, ReduceOrder val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<ReduceOrderAttr>(baseAttr);
}
ReduceOrder ReduceOrderAttr::getValue() const {
  return static_cast<ReduceOrder>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyRoundMode(RoundMode val) {
  switch (val) {
    case RoundMode::CAST_NONE: return "CAST_NONE";
    case RoundMode::CAST_RINT: return "CAST_RINT";
    case RoundMode::CAST_FLOOR: return "CAST_FLOOR";
    case RoundMode::CAST_CEIL: return "CAST_CEIL";
    case RoundMode::CAST_ROUND: return "CAST_ROUND";
    case RoundMode::CAST_TRUNC: return "CAST_TRUNC";
    case RoundMode::CAST_ODD: return "CAST_ODD";
  }
  return "";
}

::std::optional<RoundMode> symbolizeRoundMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RoundMode>>(str)
      .Case("CAST_NONE", RoundMode::CAST_NONE)
      .Case("CAST_RINT", RoundMode::CAST_RINT)
      .Case("CAST_FLOOR", RoundMode::CAST_FLOOR)
      .Case("CAST_CEIL", RoundMode::CAST_CEIL)
      .Case("CAST_ROUND", RoundMode::CAST_ROUND)
      .Case("CAST_TRUNC", RoundMode::CAST_TRUNC)
      .Case("CAST_ODD", RoundMode::CAST_ODD)
      .Default(::std::nullopt);
}
::std::optional<RoundMode> symbolizeRoundMode(uint8_t value) {
  switch (value) {
  case 0: return RoundMode::CAST_NONE;
  case 1: return RoundMode::CAST_RINT;
  case 2: return RoundMode::CAST_FLOOR;
  case 3: return RoundMode::CAST_CEIL;
  case 4: return RoundMode::CAST_ROUND;
  case 5: return RoundMode::CAST_TRUNC;
  case 6: return RoundMode::CAST_ODD;
  default: return ::std::nullopt;
  }
}

bool RoundModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))));
}
RoundModeAttr RoundModeAttr::get(::mlir::MLIRContext *context, RoundMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<RoundModeAttr>(baseAttr);
}
RoundMode RoundModeAttr::getValue() const {
  return static_cast<RoundMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyScheduleType(ScheduleType val) {
  switch (val) {
    case ScheduleType::INNER_PRODUCT: return "INNER_PRODUCT";
    case ScheduleType::OUTER_PRODUCT: return "OUTER_PRODUCT";
  }
  return "";
}

::std::optional<ScheduleType> symbolizeScheduleType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ScheduleType>>(str)
      .Case("INNER_PRODUCT", ScheduleType::INNER_PRODUCT)
      .Case("OUTER_PRODUCT", ScheduleType::OUTER_PRODUCT)
      .Default(::std::nullopt);
}
::std::optional<ScheduleType> symbolizeScheduleType(uint8_t value) {
  switch (value) {
  case 0: return ScheduleType::INNER_PRODUCT;
  case 1: return ScheduleType::OUTER_PRODUCT;
  default: return ::std::nullopt;
  }
}

bool ScheduleTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
ScheduleTypeAttr ScheduleTypeAttr::get(::mlir::MLIRContext *context, ScheduleType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<ScheduleTypeAttr>(baseAttr);
}
ScheduleType ScheduleTypeAttr::getValue() const {
  return static_cast<ScheduleType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifySelMode(SelMode val) {
  switch (val) {
    case SelMode::VSEL_CMPMASK_SPR: return "vsel_cmpmask_spr";
    case SelMode::VSEL_TENSOR_SCALAR_MODE: return "vsel_tensor_scalar_mode";
    case SelMode::VSEL_TENSOR_TENSOR_MODE: return "vsel_tensor_tensor_mode";
  }
  return "";
}

::std::optional<SelMode> symbolizeSelMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SelMode>>(str)
      .Case("vsel_cmpmask_spr", SelMode::VSEL_CMPMASK_SPR)
      .Case("vsel_tensor_scalar_mode", SelMode::VSEL_TENSOR_SCALAR_MODE)
      .Case("vsel_tensor_tensor_mode", SelMode::VSEL_TENSOR_TENSOR_MODE)
      .Default(::std::nullopt);
}
::std::optional<SelMode> symbolizeSelMode(uint8_t value) {
  switch (value) {
  case 0: return SelMode::VSEL_CMPMASK_SPR;
  case 1: return SelMode::VSEL_TENSOR_SCALAR_MODE;
  case 2: return SelMode::VSEL_TENSOR_TENSOR_MODE;
  default: return ::std::nullopt;
  }
}

bool SelModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))));
}
SelModeAttr SelModeAttr::get(::mlir::MLIRContext *context, SelMode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<SelModeAttr>(baseAttr);
}
SelMode SelModeAttr::getValue() const {
  return static_cast<SelMode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyTPosition(TPosition val) {
  switch (val) {
    case TPosition::GM: return "gm";
    case TPosition::A1: return "a1";
    case TPosition::A2: return "a2";
    case TPosition::B1: return "b1";
    case TPosition::B2: return "b2";
    case TPosition::C1: return "c1";
    case TPosition::C2: return "c2";
    case TPosition::CO1: return "co1";
    case TPosition::CO2: return "co2";
    case TPosition::VECIN: return "vecin";
    case TPosition::VECOUT: return "vecout";
    case TPosition::VECCALC: return "veccalc";
  }
  return "";
}

::std::optional<TPosition> symbolizeTPosition(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TPosition>>(str)
      .Case("gm", TPosition::GM)
      .Case("a1", TPosition::A1)
      .Case("a2", TPosition::A2)
      .Case("b1", TPosition::B1)
      .Case("b2", TPosition::B2)
      .Case("c1", TPosition::C1)
      .Case("c2", TPosition::C2)
      .Case("co1", TPosition::CO1)
      .Case("co2", TPosition::CO2)
      .Case("vecin", TPosition::VECIN)
      .Case("vecout", TPosition::VECOUT)
      .Case("veccalc", TPosition::VECCALC)
      .Default(::std::nullopt);
}
::std::optional<TPosition> symbolizeTPosition(uint8_t value) {
  switch (value) {
  case 0: return TPosition::GM;
  case 1: return TPosition::A1;
  case 2: return TPosition::A2;
  case 3: return TPosition::B1;
  case 4: return TPosition::B2;
  case 5: return TPosition::C1;
  case 6: return TPosition::C2;
  case 7: return TPosition::CO1;
  case 8: return TPosition::CO2;
  case 9: return TPosition::VECIN;
  case 10: return TPosition::VECOUT;
  case 11: return TPosition::VECCALC;
  default: return ::std::nullopt;
  }
}

bool TPositionAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))));
}
TPositionAttr TPositionAttr::get(::mlir::MLIRContext *context, TPosition val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<TPositionAttr>(baseAttr);
}
TPosition TPositionAttr::getValue() const {
  return static_cast<TPosition>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
namespace mlir::ascendc {

::llvm::StringRef stringifyTransposeType(TransposeType val) {
  switch (val) {
    case TransposeType::TRANSPOSE_TYPE_NONE: return "TRANSPOSE_TYPE_NONE";
    case TransposeType::TRANSPOSE_NZ2ND_0213: return "TRANSPOSE_NZ2ND_0213";
    case TransposeType::TRANSPOSE_NZ2NZ_0213: return "TRANSPOSE_NZ2NZ_0213";
    case TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N: return "TRANSPOSE_NZ2NZ_012_WITH_N";
    case TransposeType::TRANSPOSE_NZ2ND_012_WITH_N: return "TRANSPOSE_NZ2ND_012_WITH_N";
    case TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N: return "TRANSPOSE_NZ2ND_012_WITHOUT_N";
    case TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N: return "TRANSPOSE_NZ2NZ_012_WITHOUT_N";
    case TransposeType::TRANSPOSE_ND2ND_ONLY: return "TRANSPOSE_ND2ND_ONLY";
    case TransposeType::TRANSPOSE_ND_UB_GM: return "TRANSPOSE_ND_UB_GM";
    case TransposeType::TRANSPOSE_GRAD_ND_UB_GM: return "TRANSPOSE_GRAD_ND_UB_GM";
    case TransposeType::TRANSPOSE_ND2ND_B16: return "TRANSPOSE_ND2ND_B16";
    case TransposeType::TRANSPOSE_NCHW2NHWC: return "TRANSPOSE_NCHW2NHWC";
    case TransposeType::TRANSPOSE_NHWC2NCHW: return "TRANSPOSE_NHWC2NCHW";
  }
  return "";
}

::std::optional<TransposeType> symbolizeTransposeType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TransposeType>>(str)
      .Case("TRANSPOSE_TYPE_NONE", TransposeType::TRANSPOSE_TYPE_NONE)
      .Case("TRANSPOSE_NZ2ND_0213", TransposeType::TRANSPOSE_NZ2ND_0213)
      .Case("TRANSPOSE_NZ2NZ_0213", TransposeType::TRANSPOSE_NZ2NZ_0213)
      .Case("TRANSPOSE_NZ2NZ_012_WITH_N", TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N)
      .Case("TRANSPOSE_NZ2ND_012_WITH_N", TransposeType::TRANSPOSE_NZ2ND_012_WITH_N)
      .Case("TRANSPOSE_NZ2ND_012_WITHOUT_N", TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N)
      .Case("TRANSPOSE_NZ2NZ_012_WITHOUT_N", TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N)
      .Case("TRANSPOSE_ND2ND_ONLY", TransposeType::TRANSPOSE_ND2ND_ONLY)
      .Case("TRANSPOSE_ND_UB_GM", TransposeType::TRANSPOSE_ND_UB_GM)
      .Case("TRANSPOSE_GRAD_ND_UB_GM", TransposeType::TRANSPOSE_GRAD_ND_UB_GM)
      .Case("TRANSPOSE_ND2ND_B16", TransposeType::TRANSPOSE_ND2ND_B16)
      .Case("TRANSPOSE_NCHW2NHWC", TransposeType::TRANSPOSE_NCHW2NHWC)
      .Case("TRANSPOSE_NHWC2NCHW", TransposeType::TRANSPOSE_NHWC2NCHW)
      .Default(::std::nullopt);
}
::std::optional<TransposeType> symbolizeTransposeType(uint8_t value) {
  switch (value) {
  case 0: return TransposeType::TRANSPOSE_TYPE_NONE;
  case 1: return TransposeType::TRANSPOSE_NZ2ND_0213;
  case 2: return TransposeType::TRANSPOSE_NZ2NZ_0213;
  case 3: return TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N;
  case 4: return TransposeType::TRANSPOSE_NZ2ND_012_WITH_N;
  case 5: return TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N;
  case 6: return TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N;
  case 7: return TransposeType::TRANSPOSE_ND2ND_ONLY;
  case 8: return TransposeType::TRANSPOSE_ND_UB_GM;
  case 9: return TransposeType::TRANSPOSE_GRAD_ND_UB_GM;
  case 10: return TransposeType::TRANSPOSE_ND2ND_B16;
  case 11: return TransposeType::TRANSPOSE_NCHW2NHWC;
  case 12: return TransposeType::TRANSPOSE_NHWC2NCHW;
  default: return ::std::nullopt;
  }
}

bool TransposeTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 6)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 7)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 12)))));
}
TransposeTypeAttr TransposeTypeAttr::get(::mlir::MLIRContext *context, TransposeType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint8_t>(val));
  return ::llvm::cast<TransposeTypeAttr>(baseAttr);
}
TransposeType TransposeTypeAttr::getValue() const {
  return static_cast<TransposeType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::ascendc
