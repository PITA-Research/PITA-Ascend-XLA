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

#ifndef ASCIR_DIALECT_UTILS_INLINING_H
#define ASCIR_DIALECT_UTILS_INLINING_H

#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace ascir {

template <typename... AllowedOpT>
struct AllowlistInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *, bool,
                       IRMapping &) const override {
    return isa<AllowedOpT...>(op);
  }
};

struct PermissiveInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Operation *, bool) const override {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    return true;
  }

  void handleTerminator(Operation *, Block *) const override {}

  void handleTerminator(Operation *, ValueRange) const override {}
};

}  // namespace ascir
}  // namespace mlir

#endif  // ASCIR_DIALECT_UTILS_INLINING_H
