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

#ifndef AIR_TRANSLATIONS_STATUS_H_
#define AIR_TRANSLATIONS_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir::air {

/// @ingroup core
/// Denotes success or failure of a call.
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }

  const std::string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator!=(const Status& x) const;
  bool operator==(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of: `if (overall_status.ok()) overall_status = new_status`
  /// Use: `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  // Returns all the payload information.
  // Returns an empty result if status is ok.
  const std::unordered_map<std::string, std::string> GetAllPayloads() const;

  // Copies all the payloads using the input and discards existing payloads.
  // Does nothing if status is ok or 'payloads' is empty.
  void ReplaceAllPayloads(
      const std::unordered_map<std::string, std::string>& payloads);

 private:
  static const std::string& empty_string();

  struct State {
    std::string msg;
    std::unordered_map<std::string, std::string> payloads;
  };

  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

class StatusGroup {
 public:
  static bool IsDerived(const Status& s);

  Status as_summary_status() const;
  Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  void Update(const Status& status);

  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  std::vector<Status> children_;
  std::vector<std::string> recent_logs_;  // recent warning and error logs
  size_t num_ok_ = 0;
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

inline Status& Status::operator=(const Status& s) {
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

using StatusCallback = std::function<void(const Status&)>;

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace mlir::air

#endif  // AIR_TRANSLATIONS_STATUS_H_
