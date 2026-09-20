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

#include "xla/xla_npu/service/convert_to_afir/python_gil_guard.h"

#include <signal.h>
#include <sys/time.h>

#include "absl/strings/str_format.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

// ============================================================================
// PyGILStateGuard Implementation
// ============================================================================

PyGILStateGuard::PyGILStateGuard() : gstate_(PyGILState_Ensure()) {
  NPU_LOG_PRINT(DEBUG, "PyGILStateGuard: GIL acquired");
}

PyGILStateGuard::~PyGILStateGuard() {
  PyGILState_Release(gstate_);
  NPU_LOG_PRINT(DEBUG, "PyGILStateGuard: GIL released");
}

// ============================================================================
// TimeoutGuard Implementation
// ============================================================================

// Signal-safe flag to indicate if timeout occurred
static volatile sig_atomic_t g_timeout_flag = 0;

// Global flag for timeout detection - using atomic for thread safety
static std::atomic<bool> g_python_call_timed_out{false};

// Timeout handler - will be called when timer expires
// Note: This handler must be signal-safe (no malloc, no locks)
static void TimeoutHandler(int sig) {
  (void)sig;
  g_timeout_flag = 1;
  g_python_call_timed_out.store(true, std::memory_order_release);
}

TimeoutGuard::TimeoutGuard(int timeout_seconds)
    : timeout_seconds_(timeout_seconds) {
  if (timeout_seconds_ > 0) {
    NPU_LOG_FORMAT_PRINT(DEBUG, "TimeoutGuard: Setting timeout for %d seconds",
                         timeout_seconds_);

    // Set up signal handler for SIGALRM
    struct sigaction sa;
    sa.sa_handler = TimeoutHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, &old_sa_);

    // Set the timer (ITIMER_REAL uses real time)
    struct itimerval timer;
    timer.it_value.tv_sec = timeout_seconds_;
    timer.it_value.tv_usec = 0;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;
    setitimer(ITIMER_REAL, &timer, &old_timer_);
  }
}

TimeoutGuard::~TimeoutGuard() {
  if (timeout_seconds_ > 0) {
    // Restore old timer and signal handler
    setitimer(ITIMER_REAL, &old_timer_, nullptr);
    sigaction(SIGALRM, &old_sa_, nullptr);
    g_timeout_flag = 0;

    NPU_LOG_PRINT(DEBUG, "TimeoutGuard: Timer cancelled");
  }
}

bool TimeoutGuard::TimedOut() const { return g_timeout_flag != 0; }

// ============================================================================
// SafePythonCall Implementation
// ============================================================================

PyObject* SafePythonCall(PyObject* callable, PyObject* args, const char* name,
                         std::atomic<bool>* timed_out_flag,
                         int timeout_seconds) {
  PyObject* result = nullptr;
  bool call_success = false;

  {
    TimeoutGuard to_guard(timeout_seconds);

    // Perform the Python call under timeout protection
    result = PyObject_CallObject(callable, args);

    if (to_guard.TimedOut()) {
      // Timeout occurred - signal was received during Python call
      NPU_LOG_FORMAT_PRINT(ERROR, "Python call '%s' timed out after %d seconds",
                           name ? name : "unknown", timeout_seconds);

      if (timed_out_flag) {
        timed_out_flag->store(true, std::memory_order_release);
      }

      // Clean up result if any
      if (result) {
        Py_DECREF(result);
        result = nullptr;
      }

      // Set Python error to indicate timeout
      PyErr_SetString(PyExc_RuntimeError, "Python call timed out");
    } else {
      call_success = true;
      NPU_LOG_FORMAT_PRINT(DEBUG, "Python call '%s' completed successfully",
                           name ? name : "unknown");
    }
  }

  // After timeout guard destruction, timer is disabled
  if (!call_success && timed_out_flag &&
      timed_out_flag->load(std::memory_order_acquire)) {
    // We already handled the timeout above
  }

  return result;
}

}  // namespace xla
