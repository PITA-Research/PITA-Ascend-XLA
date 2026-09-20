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

#ifndef XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_PYTHON_GIL_GUARD_H_
#define XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_PYTHON_GIL_GUARD_H_

#include <signal.h>
#include <sys/time.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

// Python C API
#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace xla {

// ============================================================================
// RAII wrapper to ensure Python GIL is always released
// ============================================================================
class PyGILStateGuard {
 public:
  PyGILStateGuard();
  ~PyGILStateGuard();

  PyGILStateGuard(const PyGILStateGuard&) = delete;
  PyGILStateGuard& operator=(const PyGILStateGuard&) = delete;

 private:
  PyGILState_STATE gstate_;
};

// ============================================================================
// Timeout protection for Python calls
// ============================================================================

// Guard object that sets up a timer and SIGALRM handler for timeout
class TimeoutGuard {
 public:
  explicit TimeoutGuard(int timeout_seconds);
  ~TimeoutGuard();

  // Returns true if timeout occurred during the guarded section
  bool TimedOut() const;

  TimeoutGuard(const TimeoutGuard&) = delete;
  TimeoutGuard& operator=(const TimeoutGuard&) = delete;

 private:
  int timeout_seconds_;
  struct sigaction old_sa_;
  struct itimerval old_timer_;
};

// ============================================================================
// Safe wrapper for Python calls with timeout protection
// ============================================================================

// Safe wrapper for PyObject_CallObject with timeout protection
// This function prevents the process from hanging indefinitely if Python code
// gets stuck or enters an infinite loop.
// Args: callable: Python callable object (function/method)
//       args: Arguments tuple to pass to the callable
//       name: Name of the function being called (for logging)
//       timed_out_flag: Pointer to atomic bool flag to set on timeout
//       timeout_seconds: Maximum time to wait before timeout (default: 600 seconds)
// Returns: PyObject result from the callable, or nullptr on timeout/error
// Note: The caller is responsible for Py_DECREF on the returned object if it is not nullptr.
PyObject* SafePythonCall(PyObject* callable, PyObject* args, const char* name,
                         std::atomic<bool>* timed_out_flag,
                         int timeout_seconds = 600);

}  // namespace xla

#endif  // XLA_XLA_NPU_SERVICE_CONVERT_TO_AFIR_PYTHON_GIL_GUARD_H_
