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

#include "./status.h"

#include <stdio.h>

#include <deque>
#include <map>

namespace mlir::air {

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

const std::string& Status::empty_string() {
  static std::string* empty = new std::string;
  return *empty;
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    std::string result = "status error";
    return result;
  }
}

void Status::ReplaceAllPayloads(
    const std::unordered_map<std::string, std::string>& payloads) {
  if (ok() || payloads.empty()) {
    return;
  }
  if (state_ == nullptr) {
    state_ = std::make_unique<State>();
  }
  state_->payloads = payloads;
}

void Status::IgnoreError() const {
  // no-op
}

const std::unordered_map<std::string, std::string> Status::GetAllPayloads()
    const {
  if (ok()) {
    return {};
  }
  return state_->payloads;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

// kDerivedMarker is appended to the Status message string to indicate whether a
// Status object is the root cause of an error or if it has been triggered by
// cancelling/aborting a step.
static const char* kDerivedMarker = "[_Derived_]";

void StatusGroup::Update(const Status& s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    children_.push_back(s);
    ok_ = false;
  }
}

bool StatusGroup::IsDerived(const Status& s) {
  return s.error_message().find(kDerivedMarker) != std::string::npos;
}
}  // namespace mlir::air
