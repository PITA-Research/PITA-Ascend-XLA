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

#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"

#include <unistd.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/xla_npu/common/npu_config.h"
#include "xla/xla_npu/common/npu_constant.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"
#include "xla/xla_npu/service/convert_to_afir/python_gil_guard.h"

#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"

namespace xla {

namespace {

// Helper function to execute a shell command and get output
absl::StatusOr<std::string> ExecuteCommand(const std::string& command) {
  char buffer[128];
  std::string result;

  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return absl::InternalError(
        absl::StrCat("Failed to execute command: ", command));
  }

  try {
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
  } catch (...) {
    pclose(pipe);
    return absl::InternalError("Error reading command output");
  }

  int return_code = pclose(pipe);
  if (return_code != 0) {
    return absl::InternalError(
        absl::StrCat("Command failed with code ", return_code, ": ", command));
  }

  return result;
}

// Helper function to write string to file
absl::Status WriteFile(absl::string_view content, const std::string& filepath) {
  std::ofstream ofs(filepath);
  if (!ofs) {
    return absl::InternalError(absl::StrCat("Failed to open file: ", filepath));
  }
  ofs << content;
  ofs.close();
  return absl::OkStatus();
}

// Helper function to read file to string
absl::StatusOr<std::string> ReadFile(const std::string& filepath) {
  std::ifstream ifs(filepath);
  if (!ifs) {
    return absl::InternalError(absl::StrCat("Failed to open file: ", filepath));
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

std::string FetchPythonErrorString() {
  if (!PyErr_Occurred()) {
    return "unknown Python error";
  }

  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string error;
  PyObject* traceback_module = PyImport_ImportModule("traceback");
  if (traceback_module) {
    PyObject* formatted =
        PyObject_CallMethod(traceback_module, "format_exception", "OOO",
                            type ? type : Py_None, value ? value : Py_None,
                            traceback ? traceback : Py_None);
    if (formatted) {
      PyObject* separator = PyUnicode_FromString("");
      PyObject* joined =
          separator ? PyUnicode_Join(separator, formatted) : nullptr;
      if (joined) {
        const char* text = PyUnicode_AsUTF8(joined);
        if (text) {
          error = text;
        }
        Py_DECREF(joined);
      }
      Py_XDECREF(separator);
      Py_DECREF(formatted);
    }
    Py_DECREF(traceback_module);
  }

  if (error.empty() && value) {
    PyObject* value_str = PyObject_Str(value);
    if (value_str) {
      const char* text = PyUnicode_AsUTF8(value_str);
      if (text) {
        error = text;
      }
      Py_DECREF(value_str);
    }
  }

  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);

  return error.empty() ? "failed to format Python exception" : error;
}

// Helper function to extract graph_name from ascir text
absl::StatusOr<std::string> ExtractGraphName(const std::string& ascir_text) {
  // Find the last occurrence of "graph_name:"
  size_t pos = ascir_text.rfind("graph_name:");
  if (pos == std::string::npos) {
    return absl::InternalError("graph_name not found in ascir text");
  }

  // Extract the line containing graph_name
  size_t line_start = ascir_text.rfind('\n', pos);
  if (line_start == std::string::npos) {
    line_start = 0;
  } else {
    line_start += 1;  // Skip the newline
  }

  size_t line_end = ascir_text.find('\n', pos);
  if (line_end == std::string::npos) {
    line_end = ascir_text.length();
  }

  std::string graph_line = ascir_text.substr(line_start, line_end - line_start);

  // Parse: graph_name: "name"
  size_t quote_start = graph_line.find('"', pos - line_start);
  if (quote_start == std::string::npos) {
    return absl::InternalError(
        "Invalid graph_name format: missing opening quote");
  }

  size_t quote_end = graph_line.find('"', quote_start + 1);
  if (quote_end == std::string::npos) {
    return absl::InternalError(
        "Invalid graph_name format: missing closing quote");
  }

  std::string graph_name =
      graph_line.substr(quote_start + 1, quote_end - quote_start - 1);
  return graph_name;
}

// Helper function to fix sched.axis and outputs.attr.axis_ids to ensure consistency
std::string FixNodeAxisInfo(const std::string& ascir_text) {
  std::string result = ascir_text;

  // Strategy: Count how many axis definitions are in asc_graph_attr
  // Then for any node with sched that's missing axis, add all axes
  // Also fix outputs.attr.axis_ids to match sched.axis count
  // Helper function: Count axes in asc_graph_attr
  auto countGlobalAxes = [&result]() -> size_t {
    size_t graph_attr_pos = result.find("asc_graph_attr {");
    if (graph_attr_pos == std::string::npos) {
      return 0;
    }

    // Find the end of asc_graph_attr
    size_t graph_attr_end = result.find("\n}\n", graph_attr_pos);
    if (graph_attr_end == std::string::npos) {
      graph_attr_end = result.find("}\nasc_node", graph_attr_pos);
    }

    if (graph_attr_end == std::string::npos) {
      return 0;
    }

    std::string graph_attr_section =
        result.substr(graph_attr_pos, graph_attr_end - graph_attr_pos);

    // Count axis definitions
    size_t num_axes = 0;
    size_t pos = 0;
    while ((pos = graph_attr_section.find("axis {", pos)) !=
           std::string::npos) {
      num_axes++;
      pos += 7;
    }

    return num_axes;
  };

  // Helper function: Count axis fields in a sched section
  auto countSchedAxes = [](const std::string& sched_content) -> size_t {
    size_t num_axes = 0;
    size_t search_pos = 0;

    while ((search_pos = sched_content.find("axis:", search_pos)) !=
           std::string::npos) {
      // Make sure it's not part of "loop_axis:"
      if (search_pos < 4 || sched_content.substr(search_pos - 4, 4) != "loop") {
        // Check if preceded by newline (with optional spaces), or at the start
        bool is_valid_axis = false;
        if (search_pos == 0) {
          is_valid_axis = true;
        } else {
          // Check if preceded only by whitespace or newline
          bool found_newline = false;
          bool only_whitespace = true;
          for (size_t i = 1; i <= search_pos; i++) {
            if (sched_content[search_pos - i] == '\n') {
              found_newline = true;
              break;
            }
            if (sched_content[search_pos - i] != ' ' &&
                sched_content[search_pos - i] != '\t') {
              only_whitespace = false;
              break;
            }
          }
          is_valid_axis = found_newline || only_whitespace;
        }
        if (is_valid_axis) {
          num_axes++;
        }
      }
      search_pos += 5;
    }

    return num_axes;
  };

  // Helper function: Fix sched sections to match global axis count
  auto fixSchedSections = [&result,
                           &countSchedAxes](size_t num_axes) -> size_t {
    size_t pos = 0;
    size_t fixed_count = 0;
    size_t total_count = 0;

    while ((pos = result.find("sched {\n", pos)) != std::string::npos) {
      total_count++;
      size_t sched_end = result.find("}", pos);
      if (sched_end == std::string::npos || sched_end > pos + 200) {
        pos += 8;
        continue;
      }

      std::string sched_content = result.substr(pos + 8, sched_end - (pos + 8));
      size_t num_sched_axis = countSchedAxes(sched_content);
      if (num_sched_axis < num_axes) {
        size_t num_missing = num_axes - num_sched_axis;

        // Find insertion point (before closing brace)
        size_t insert_pos = sched_end;
        size_t newline_before_brace = result.rfind("\n", sched_end);
        if (newline_before_brace != std::string::npos &&
            newline_before_brace > pos) {
          insert_pos = newline_before_brace + 1;
        }

        // Add missing axes
        std::string axis_fields;
        for (size_t i = num_sched_axis; i < num_axes; ++i) {
          axis_fields += absl::StrFormat("      axis: %zu\n", i);
        }

        result.insert(insert_pos, axis_fields);
        sched_end += axis_fields.length();
        fixed_count++;

        NPU_LOG_FORMAT_PRINT(DEBUG, "Fixed sched at %d: %d -> %d axes", pos,
                             num_sched_axis, num_axes);
      }

      pos = sched_end + 1;
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Processed %d sched sections, fixed %d",
                         total_count, fixed_count);
    return fixed_count;
  };

  // Helper function: Find matching closing brace for attr block
  auto findAttrBlockEnd = [&result](size_t attr_start) -> size_t {
    size_t attr_content_start = attr_start + 7;  // After "attr {\n"
    int brace_depth = 1;  // We're inside "attr {" already

    for (size_t i = attr_content_start;
         i < result.length() && i < attr_content_start + 2000; ++i) {
      if (result[i] == '{') {
        brace_depth++;
      } else if (result[i] == '}') {
        brace_depth--;
        if (brace_depth == 0) {
          return i;
        }
      }
    }

    return std::string::npos;
  };

  // Helper function: Count occurrences of a field in output_attr_section
  auto countField = [](const std::string& section,
                       const std::string& field_name) -> size_t {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = section.find(field_name, pos)) != std::string::npos) {
      count++;
      pos += field_name.length();
    }
    return count;
  };

  // Helper function: Parse field values from output_attr_section
  auto parseFieldValues = [](const std::string& section,
                             const std::string& field_name)
      -> std::pair<std::vector<std::string>, std::vector<size_t>> {
    std::vector<std::string> values;
    std::vector<size_t> positions;
    size_t pos = 0;

    while ((pos = section.find(field_name, pos)) != std::string::npos) {
      size_t val_start = pos + field_name.length();
      size_t val_end = section.find("\"", val_start + 1);
      if (val_end != std::string::npos && val_end > val_start) {
        std::string val =
            section.substr(val_start + 1, val_end - val_start - 1);
        values.push_back(val);
        positions.push_back(pos);
      }
      pos += field_name.length();
    }

    return {values, positions};
  };

  // Helper function: Fix incorrect stride values
  auto fixStrideValues =
      [&result, &parseFieldValues](
          const std::string& output_attr_section, size_t output_attr_start,
          const std::vector<std::string>& repeats,
          const std::vector<size_t>& stride_positions) -> void {
    auto [strides, _] = parseFieldValues(output_attr_section, "strides:");

    for (size_t i = 0; i < strides.size() && i < repeats.size(); i++) {
      if (strides[i] == "0" && repeats[i] != "1") {
        // Compute correct stride (column-major: stride[i] = product of repeats[i+1:])
        int64_t stride = 1;
        for (size_t j = i + 1; j < repeats.size(); j++) {
          try {
            stride *= std::stoll(repeats[j]);
          } catch (...) {
            stride = 0;
            break;
          }
        }

        // Replace the stride value in result
        if (i < stride_positions.size()) {
          size_t stride_pos = stride_positions[i];
          size_t val_start = result.find("\"", stride_pos + 8) + 1;
          size_t val_end = result.find("\"", val_start);
          if (val_end != std::string::npos) {
            result.replace(val_start, val_end - val_start,
                           std::to_string(stride));
          }
        }
      }
    }
  };

  // Helper function: Fix outputs.attr sections to match global axis count
  auto fixOutputAttrSections = [&result, &findAttrBlockEnd, &countField,
                                &parseFieldValues, &fixStrideValues,
                                &countSchedAxes]() -> size_t {
    size_t pos = 0;
    size_t fixed_count = 0;

    while ((pos = result.find("asc_node {", pos)) != std::string::npos) {
      size_t node_start = pos;

      // Find outputs section
      size_t outputs_pos = result.find("outputs {\n", pos);
      if (outputs_pos == std::string::npos || outputs_pos > node_start + 500) {
        pos += 9;
        continue;
      }

      // Find attr section inside outputs
      size_t output_attr_start = result.find("attr {\n", outputs_pos);
      if (output_attr_start == std::string::npos ||
          output_attr_start > outputs_pos + 50) {
        pos += 9;
        continue;
      }

      // Find attr block end
      size_t output_attr_end = findAttrBlockEnd(output_attr_start);
      if (output_attr_end == std::string::npos) {
        pos += 9;
        continue;
      }

      std::string output_attr_section = result.substr(
          output_attr_start, output_attr_end - output_attr_start + 1);
      size_t num_axis_ids = countField(output_attr_section, "axis_ids:");
      size_t num_repeats = countField(output_attr_section, "repeats:");
      size_t num_strides = countField(output_attr_section, "strides:");

      // Find sched section
      size_t sched_pos = result.find("sched {\n", pos);
      if (sched_pos == std::string::npos || sched_pos > node_start + 2000) {
        pos += 9;
        continue;
      }

      size_t sched_end = result.find("}\n", sched_pos);
      if (sched_end == std::string::npos || sched_end > sched_pos + 500) {
        pos += 9;
        continue;
      }

      std::string sched_section =
          result.substr(sched_pos, sched_end - sched_pos);
      size_t num_sched_axis = countSchedAxes(sched_section);

      // Fix outputs.attr if needed. Keep axis_ids, repeats, and strides in
      // lockstep with sched.axis; Autofuse rejects partially padded attrs.
      if (num_axis_ids < num_sched_axis || num_repeats < num_sched_axis ||
          num_strides < num_sched_axis) {
        // Find insertion position
        size_t last_field_pos = 0;

        // Find last field position
        std::vector<std::string> fields = {"axis_ids:", "repeats:", "strides:"};
        for (const auto& field : fields) {
          size_t search_pos = 0;
          while ((search_pos = output_attr_section.find(field, search_pos)) !=
                 std::string::npos) {
            size_t field_pos = output_attr_start + search_pos;
            if (field_pos > last_field_pos) last_field_pos = field_pos;
            search_pos += field.length();
          }
        }

        // If no fields found, insert after dtype
        if (last_field_pos == 0) {
          size_t dtype_pos = output_attr_section.find("dtype:");
          if (dtype_pos != std::string::npos) {
            last_field_pos = output_attr_start + dtype_pos;
          }
        }

        if (last_field_pos > 0) {
          size_t line_end = result.find("\n", last_field_pos);
          if (line_end != std::string::npos && line_end < output_attr_end) {
            // Fix incorrect stride values
            auto [repeats, repeat_pos] =
                parseFieldValues(output_attr_section, "repeats:");
            auto [strides, stride_pos] =
                parseFieldValues(output_attr_section, "strides:");
            fixStrideValues(output_attr_section, output_attr_start, repeats,
                            stride_pos);

            // Generate additional axes
            std::string additional_axes;
            for (size_t i = num_axis_ids; i < num_sched_axis; i++) {
              additional_axes += "      axis_ids: " + std::to_string(i) + "\n";
            }
            for (size_t i = num_repeats; i < num_sched_axis; i++) {
              additional_axes += "      repeats: \"1\"\n";
            }
            for (size_t i = num_strides; i < num_sched_axis; i++) {
              additional_axes += "      strides: \"0\"\n";
            }

            if (!additional_axes.empty()) {
              result.insert(line_end + 1, additional_axes);
              output_attr_end += additional_axes.length();
              fixed_count++;
            }
          }
        }
      }

      pos = node_start + 9;
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Fixed %d outputs.attr sections", fixed_count);
    return fixed_count;
  };

  // Main logic
  NPU_LOG_PRINT(DEBUG, "FixNodeAxisInfo: Starting to process ASCIR text");

  size_t num_axes = countGlobalAxes();
  NPU_LOG_FORMAT_PRINT(DEBUG, "Found %d axes in asc_graph_attr", num_axes);

  if (num_axes == 0) {
    NPU_LOG_PRINT(DEBUG, "No axes found, returning without modifications");
    return result;
  }

  // Step 1: Fix sched sections
  fixSchedSections(num_axes);

  // Step 2: Fix outputs.attr sections
  fixOutputAttrSections();

  return result;
}

}  // namespace
AscirBackend::AscirBackend() = default;

AscirBackend::~AscirBackend() = default;

absl::StatusOr<std::vector<std::string>> AscirBackend::ConvertHloToAfirModule(
    const std::unordered_map<std::string, const HloComputation*>& computations,
    const AscirBackendOptions& options) {
  std::vector<std::string> afir_strings;

  // Process all computations in the map
  for (const auto& [name, computation] : computations) {
    NPU_LOG_FORMAT_PRINT(INFO, "Converting computation: %s", name.c_str());

    // Create MLIR context for this computation
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::tensor::TensorDialect>();
    context->loadDialect<mlir::afir::AFIRDialect>();

    // Convert this single computation (with all its dependencies) to AFIR
    HloToAfirOptions afir_options;
    afir_options.preserve_names = true;
    afir_options.verify_module = true;
    afir_options.emit_afir_dialect = true;

    TF_ASSIGN_OR_RETURN(
        auto mlir_module,
        ConvertHloComputationToAfir(*computation, context.get(), afir_options));

    // Serialize MLIR module to string immediately
    std::string afir_string;
    llvm::raw_string_ostream stream(afir_string);
    mlir_module->print(stream);

    afir_strings.push_back(afir_string);
  }

  NPU_LOG_FORMAT_PRINT(INFO, "Converted %d computations to AFIR",
                       afir_strings.size());

  return afir_strings;
}

absl::StatusOr<std::string> AscirBackend::ConvertAfirToAscir(
    const std::string& afir_string, const AscirBackendOptions& options) {
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Starting conversion");

  // Create MLIR context for parsing AFIR
  auto context = std::make_unique<mlir::MLIRContext>();
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Created MLIR context");

  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::arith::ArithDialect>();
  context->loadDialect<mlir::tensor::TensorDialect>();
  context->loadDialect<mlir::afir::AFIRDialect>();
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Loaded dialects");

  // Parse AFIR string to MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(afir_string),
                                              context.get());
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Parsed AFIR string");

  if (!module) {
    return absl::InternalError("Failed to parse AFIR string");
  }

  // Create temporary file for ASCIR output
  std::string temp_ascir_file =
      absl::StrCat(options.output_dir, "/temp_output.ascir");
  NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertAfirToAscir: Created temp file path: %s",
                       temp_ascir_file.c_str());

  // Create PassManager to run the conversion pass
  mlir::PassManager pm(context.get());
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Created PassManager");

  pm.addPass(mlir::afir::createConvertAFIRToASCIRTextPass());
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Added conversion pass");

  // Capture stderr to get the output (since pass outputs to stderr by default)
  // Use dup2 to properly save and restore stderr
  int stderr_fd = dup(fileno(stderr));
  if (stderr_fd == -1) {
    return absl::InternalError("Failed to duplicate stderr file descriptor");
  }

  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: About to redirect stderr");

  // Redirect stderr to a file
  FILE* temp_file = fopen(temp_ascir_file.c_str(), "w");
  if (!temp_file) {
    close(stderr_fd);
    return absl::InternalError(
        absl::StrCat("Failed to open temp file: ", temp_ascir_file));
  }

  dup2(fileno(temp_file), fileno(stderr));

  // Run the pass
  mlir::LogicalResult result = pm.run(*module);

  // Flush and close the temp file
  fflush(stderr);
  fclose(temp_file);

  // Restore stderr using dup2
  dup2(stderr_fd, fileno(stderr));
  close(stderr_fd);

  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Stderr restored");

  if (mlir::failed(result)) {
    return absl::InternalError("Failed to run AFIR to ASCIR conversion pass");
  }
  NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Pass succeeded");

  // Read the generated ASCIR file
  TF_ASSIGN_OR_RETURN(auto ascir_text, ReadFile(temp_ascir_file));

  // Clean up temporary file if not preserving intermediates
  if (!options.preserve_intermediates) {
    std::remove(temp_ascir_file.c_str());
  }

  return ascir_text;
}

absl::StatusOr<SingleComputationResult> AscirBackend::GenerateKernel(
    const std::string& ascir_text, const std::string& computation_name,
    const AscirBackendOptions& options) {
  SingleComputationResult result;
  result.computation_name = computation_name;

  // Extract graph_name from ascir text
  TF_ASSIGN_OR_RETURN(auto graph_name, ExtractGraphName(ascir_text));
  result.graph_name = graph_name;
  NPU_LOG_FORMAT_PRINT(DEBUG, "Computation: %s, Extracted graph_name: %s",
                       computation_name.c_str(), graph_name.c_str());

  // Create computation-specific output directory
  std::string computation_output_dir =
      absl::StrCat(options.output_dir, "/", computation_name);
  std::string mkdir_cmd = absl::StrCat("mkdir -p ", computation_output_dir);
  TF_ASSIGN_OR_RETURN(auto _, ExecuteCommand(mkdir_cmd));

  // Save ASCIR to file for reference (use graph_name as filename)
  std::string ascir_file =
      absl::StrCat(computation_output_dir, "/", graph_name, ".ascir");
  TF_RETURN_IF_ERROR(WriteFile(ascir_text, ascir_file));
  result.ascir_file = ascir_file;
  NPU_LOG_FORMAT_PRINT(DEBUG, "ASCIR saved to: %s", ascir_file.c_str());

  // Initialize Python interpreter
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "About to initialize Python interpreter for %s...",
                       computation_name.c_str());
  fflush(stderr);
  if (!Py_IsInitialized()) {
    Py_Initialize();
    if (!Py_IsInitialized()) {
      return absl::InternalError("Failed to initialize Python interpreter");
    }
  }
  NPU_LOG_PRINT(DEBUG, "Python initialized successfully");

  // Use RAII guard to ensure GIL is always released
  PyGILStateGuard gil_guard;

  // Set environment variable for SOC version
  std::string set_soc_cmd = absl::StrFormat(
      "import os; os.environ['SOC_VERSION'] = '%s'", options.soc_version);
  PyRun_SimpleString(set_soc_cmd.c_str());

  // Add runtime path to Python path
  const char* ascend_mlir_path = std::getenv("ASCEND_MLIR_PYTHON_PATH");
  std::string runtime_path = ascend_mlir_path ? ascend_mlir_path : "";
  if (runtime_path.empty()) {
    return absl::InternalError(
        "ASCEND_MLIR_PYTHON_PATH environment variable not set. "
        "Please set it to {xla-npu code repository "
        "path}/xla_npu/dependency/external/afir/Ascend-MLIR/python");
  }
  NPU_LOG_FORMAT_PRINT(DEBUG, "Adding runtime path: %s", runtime_path.c_str());
  std::string import_cmd =
      absl::StrFormat("import sys; sys.path.insert(0, '%s')", runtime_path);
  PyRun_SimpleString(import_cmd.c_str());
  NPU_LOG_PRINT(DEBUG, "Runtime path added");

  // Import runtime module
  NPU_LOG_PRINT(DEBUG, "Importing runtime module...");
  PyObject* runtime_module = PyImport_ImportModule("runtime");
  NPU_LOG_PRINT(DEBUG, "Runtime module imported");
  if (!runtime_module) {
    PyErr_Print();
    return absl::InternalError("Failed to import runtime module");
  }

  // Get AscGen class
  PyObject* ascgen_class = PyObject_GetAttrString(runtime_module, "AscGen");
  Py_DECREF(runtime_module);
  if (!ascgen_class) {
    PyErr_Print();
    return absl::InternalError("Failed to get AscGen class");
  }

  // Create AscGen instance with soc_version parameter
  // Build argument tuple: (ascend_root=None, soc_version=options.soc_version)
  PyObject* args = PyTuple_New(0);  // No positional args

  // Build keyword arguments dict: {"soc_version": options.soc_version}
  PyObject* kwargs = PyDict_New();
  PyObject* soc_version_str = PyUnicode_FromString(options.soc_version.c_str());
  PyDict_SetItemString(kwargs, "soc_version", soc_version_str);
  Py_XDECREF(soc_version_str);

  NPU_LOG_FORMAT_PRINT(INFO, "Creating AscGen instance with soc_version: %s",
                       options.soc_version.c_str());

  // Call AscGen constructor with kwargs
  PyObject* ascgen_instance = PyObject_Call(ascgen_class, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(ascgen_class);

  if (!ascgen_instance) {
    PyErr_Print();
    return absl::InternalError("Failed to create AscGen instance");
  }

  // Prepare compile_graph arguments
  // Create Path object for output_dir
  PyObject* sys_module = PyImport_ImportModule("pathlib");
  if (!sys_module) {
    Py_DECREF(ascgen_instance);
    PyErr_Print();
    return absl::InternalError("Failed to import pathlib");
  }
  PyObject* path_class = PyObject_GetAttrString(sys_module, "Path");
  Py_DECREF(sys_module);
  if (!path_class) {
    Py_DECREF(ascgen_instance);
    PyErr_Print();
    return absl::InternalError("Failed to get Path class");
  }

  PyObject* output_dir_py =
      PyObject_CallFunction(path_class, "s", computation_output_dir.c_str());
  if (!output_dir_py) {
    Py_DECREF(path_class);
    Py_DECREF(ascgen_instance);
    PyErr_Print();
    return absl::InternalError("Failed to create Path object");
  }

  // Get compile_graph method
  PyObject* compile_graph_method =
      PyObject_GetAttrString(ascgen_instance, "compile_graph");
  if (!compile_graph_method) {
    Py_DECREF(ascgen_instance);
    Py_DECREF(output_dir_py);
    PyErr_Print();
    return absl::InternalError("Failed to get compile_graph method");
  }

  // Call compile_graph(graph_text, output_dir, graph_name) with timeout
  PyObject* compile_args = PyTuple_New(3);
  PyTuple_SetItem(compile_args, 0, PyUnicode_FromString(ascir_text.c_str()));
  PyTuple_SetItem(compile_args, 1, output_dir_py);  // Steals reference
  PyTuple_SetItem(compile_args, 2, PyUnicode_FromString(graph_name.c_str()));

  // Use SafePythonCall with timeout (default 600 seconds = 10 min)
  std::atomic<bool> compile_timed_out{false};
  PyObject* compile_result =
      SafePythonCall(compile_graph_method, compile_args, "compile_graph",
                     &compile_timed_out, 600);
  Py_DECREF(compile_args);
  Py_DECREF(compile_graph_method);

  if (compile_timed_out.load(std::memory_order_acquire)) {
    Py_DECREF(ascgen_instance);
    NPU_LOG_FORMAT_PRINT(ERROR, "compile_graph timed out for computation %s",
                         computation_name.c_str());
    return absl::InternalError(absl::StrCat(
        "compile_graph timed out for computation: ", computation_name));
  }

  if (!compile_result) {
    Py_DECREF(ascgen_instance);
    std::string python_error = FetchPythonErrorString();
    NPU_LOG_FORMAT_PRINT(ERROR, "compile_graph failed: %s",
                         python_error.c_str());
    return absl::InternalError(absl::StrCat(
        "Failed to call compile_graph for computation '", computation_name,
        "', graph '", graph_name, "'\nASCIR file: ", ascir_file,
        "\nOutput dir: ", computation_output_dir, "\n", python_error));
  }

  // Parse result: (host_file, device_file, host_tiling, tiling_def)
  if (!PyTuple_Check(compile_result) || PyTuple_Size(compile_result) != 4) {
    Py_DECREF(compile_result);
    Py_DECREF(ascgen_instance);
    return absl::InternalError("Invalid compile_graph result format");
  }

  // Extract host_file path (returns Path object, need to convert to string)
  PyObject* host_file_obj = PyTuple_GetItem(compile_result, 0);  // Borrowed
  PyObject* host_file_str_obj =
      PyObject_Str(host_file_obj);  // Convert Path to str
  if (host_file_str_obj) {
    const char* host_file_str = PyUnicode_AsUTF8(host_file_str_obj);
    if (host_file_str) {
      result.host_file = host_file_str;
    }
    Py_DECREF(host_file_str_obj);
  }

  // Extract device_file path (returns Path object, need to convert to string)
  PyObject* device_file_obj = PyTuple_GetItem(compile_result, 1);  // Borrowed
  PyObject* device_file_str_obj =
      PyObject_Str(device_file_obj);  // Convert Path to str
  if (device_file_str_obj) {
    const char* device_file_str = PyUnicode_AsUTF8(device_file_str_obj);
    if (device_file_str) {
      result.device_file = device_file_str;
    }
    Py_DECREF(device_file_str_obj);
  }

  // Extract host_tiling and tiling_def for jit_compile
  PyObject* host_tiling_obj = PyTuple_GetItem(compile_result, 2);  // Borrowed
  PyObject* tiling_def_obj = PyTuple_GetItem(compile_result, 3);   // Borrowed

  // Both should be strings (unicode)
  if (!PyUnicode_Check(host_tiling_obj) || !PyUnicode_Check(tiling_def_obj)) {
    Py_DECREF(compile_result);
    Py_DECREF(ascgen_instance);
    Py_DECREF(path_class);
    return absl::InternalError("host_tiling and tiling_def must be strings");
  }

  // Increment references since we need them after compile_result is decref'd
  Py_INCREF(host_tiling_obj);
  Py_INCREF(tiling_def_obj);

  // Read device_file content for jit_compile (device_impl is not in compile_result)
  std::string device_impl;
  std::ifstream device_file_stream(result.device_file);
  if (!device_file_stream.is_open()) {
    Py_DECREF(compile_result);
    Py_DECREF(ascgen_instance);
    Py_DECREF(path_class);
    return absl::InternalError(
        absl::StrCat("Failed to open device_file: ", result.device_file));
  }
  std::stringstream device_buffer;
  device_buffer << device_file_stream.rdbuf();
  device_impl = device_buffer.str();
  device_file_stream.close();

  NPU_LOG_FORMAT_PRINT(DEBUG, "Read device_file (%s) size: %d",
                       result.device_file.c_str(), device_impl.size());
  NPU_LOG_PRINT(DEBUG, "Using host_tiling_obj and tiling_def_obj from memory");

  // Clean up compile_result and related objects
  Py_DECREF(compile_result);
  Py_DECREF(ascgen_instance);
  Py_DECREF(path_class);

  // Step 4: Compile device kernel using autofuse.compile_adapter.jit_compile
  // Import autofuse.compile_adapter module
  PyObject* autofuse_module = PyImport_ImportModule("autofuse.compile_adapter");
  if (!autofuse_module) {
    PyErr_Print();
    return absl::InternalError(
        "Failed to import autofuse.compile_adapter module");
  }

  // Get jit_compile function from autofuse.compile_adapter module
  PyObject* jit_compile_func =
      PyObject_GetAttrString(autofuse_module, "jit_compile");
  Py_DECREF(autofuse_module);
  if (!jit_compile_func) {
    PyErr_Print();
    return absl::InternalError(
        "Failed to get jit_compile function from autofuse.compile_adapter");
  }

  // Check if it's callable
  if (!PyCallable_Check(jit_compile_func)) {
    Py_DECREF(jit_compile_func);
    return absl::InternalError("jit_compile is not callable");
  }

  // Prepare output_file parameter
  // Use absolute paths to avoid issues when ascendc_compile.main calls os.chdir()
  char abs_output_dir_buf[PATH_MAX];
  std::string abs_output_dir = options.output_dir;
  if (realpath(options.output_dir.c_str(), abs_output_dir_buf) != nullptr) {
    abs_output_dir = abs_output_dir_buf;
  }
  // output_file: final .so location (in abs_output_dir root)
  // output_path: temp build working dir for jit_compile (computation subdir),
  //   must differ from the directory containing output_file to avoid SameFileError
  std::string output_file =
      absl::StrCat(abs_output_dir, "/", graph_name, ".so");
  std::string jit_output_path =
      absl::StrCat(abs_output_dir, "/", computation_name);

  // Prepare parameters list: [graph_name, output_file, output_path, "--force_unknown=True", "--soc_version", "--compile_options"]
  // NOTE: CANN <= 8.5.0 compile_adapter.py does not accept --soc_version as a
  // CLI argument (SOC_VERSION is passed via environment variable instead).
  // When use_old_cann_style is true we omit --soc_version; when false we
  // include both --soc_version and --compile_options.
  bool use_old_cann_style =
      NPU_CONFIG_GET_BOOL(xla::NpuConfig::Key::USE_OLD_CANN_STYLE);
  int args_list_size = use_old_cann_style ? 4 : 6;
  PyObject* args_list = PyList_New(args_list_size);
  if (!args_list) {
    Py_DECREF(jit_compile_func);
    return absl::InternalError("Failed to create args_list");
  }

  // Set list items
  std::string graph_name_arg = absl::StrCat("--graph_name=", graph_name);
  std::string output_file_arg = absl::StrCat("--output_file=", output_file);
  std::string output_path_arg = absl::StrCat("--output_path=", jit_output_path);
  const char* force_unknown_arg = "--force_unknown=True";

  PyList_SetItem(args_list, 0, PyUnicode_FromString(graph_name_arg.c_str()));
  PyList_SetItem(args_list, 1, PyUnicode_FromString(output_file_arg.c_str()));
  PyList_SetItem(args_list, 2, PyUnicode_FromString(output_path_arg.c_str()));
  PyList_SetItem(args_list, 3, PyUnicode_FromString(force_unknown_arg));

  if (!use_old_cann_style) {
    // New CANN style: include --soc_version and --compile_options
    std::string soc_version_arg =
        absl::StrCat("--soc_version=", options.soc_version);
    const char* compile_options_arg = "--compile_options=-Wno-unused-parameter";
    PyList_SetItem(args_list, 4, PyUnicode_FromString(soc_version_arg.c_str()));
    PyList_SetItem(args_list, 5, PyUnicode_FromString(compile_options_arg));
  }

  // Prepare positional arguments: tiling_def, host_impl, device_impl, args_list
  PyObject* kernel_args = PyTuple_New(4);
  if (!kernel_args) {
    Py_DECREF(args_list);
    Py_DECREF(jit_compile_func);
    return absl::InternalError("Failed to create kernel_args tuple");
  }

  // Set tuple items (steals references)
  // Note: tiling_def_obj and host_tiling_obj already have their references incremented
  // at line 383-384, so PyTuple_SetItem will steal those references correctly

  PyObject* device_impl_py = PyUnicode_FromString(device_impl.c_str());
  if (!device_impl_py) {
    Py_DECREF(tiling_def_obj);
    Py_DECREF(host_tiling_obj);
    Py_DECREF(args_list);
    Py_DECREF(kernel_args);
    Py_DECREF(jit_compile_func);
    return absl::InternalError("Failed to create device_impl string");
  }

  // PyTuple_SetItem steals the reference, so we don't decref separately
  PyTuple_SetItem(kernel_args, 0, tiling_def_obj);
  PyTuple_SetItem(kernel_args, 1, host_tiling_obj);
  PyTuple_SetItem(kernel_args, 2, device_impl_py);
  PyTuple_SetItem(kernel_args, 3, args_list);

  NPU_LOG_FORMAT_PRINT(DEBUG, "Calling jit_compile with graph_name: %s",
                       graph_name.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "output_file: %s", output_file.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "output_path: %s", options.output_dir.c_str());

  // Call jit_compile(tiling_def, host_impl, device_impl, args_list) with timeout
  // Use SafePythonCall with timeout (default 600 seconds = 10 min)
  std::atomic<bool> jit_timed_out{false};
  PyObject* kernel_result = SafePythonCall(jit_compile_func, kernel_args,
                                           "jit_compile", &jit_timed_out, 600);
  Py_DECREF(kernel_args);
  Py_DECREF(jit_compile_func);

  if (jit_timed_out.load(std::memory_order_acquire)) {
    NPU_LOG_FORMAT_PRINT(ERROR, "jit_compile timed out for computation %s",
                         computation_name.c_str());
    return absl::InternalError(absl::StrCat(
        "jit_compile timed out for computation: ", computation_name));
  }

  if (!kernel_result) {
    std::string python_error = FetchPythonErrorString();
    NPU_LOG_FORMAT_PRINT(ERROR, "jit_compile failed: %s",
                         python_error.c_str());
    return absl::InternalError(
        absl::StrCat("Failed to call jit_compile:\n", python_error));
  }

  // jit_compile returns None, the compiled .so file is at output_file path
  Py_DECREF(kernel_result);

  // Set device_binary_file to the output .so file
  result.device_binary_file = output_file;

  NPU_LOG_FORMAT_PRINT(DEBUG, "jit_compile completed, output at: %s",
                       result.device_binary_file.c_str());

  // PyGILStateGuard destructor will automatically release GIL here
  return result;
}

absl::StatusOr<NpuCompileResult> AscirBackend::Compile(
    const std::unordered_map<std::string, const HloComputation*>& computations,
    const AscirBackendOptions& options) {
  NpuCompileResult final_result;

  // Only process entry computation, ignore other computations (such as reduce's to_apply)
  const std::string& entry_key = NpuConstant::entryComputationKeyName;
  auto it = computations.find(entry_key);
  if (it == computations.end()) {
    return absl::InternalError(absl::StrCat("Entry computation with key '",
                                            entry_key,
                                            "' not found in computations map"));
  }

  const HloComputation* entry_computation = it->second;
  NPU_LOG_FORMAT_PRINT(INFO, "Processing entry computation: %s",
                       entry_key.c_str());

  // Create a map containing only entry computation for conversion
  std::unordered_map<std::string, const HloComputation*> entry_computation_map;
  entry_computation_map[entry_key] = entry_computation;

  // Step 1: Convert HLO to AFIR for entry computation only
  TF_ASSIGN_OR_RETURN(auto afir_strings,
                      ConvertHloToAfirModule(entry_computation_map, options));

  // Create output directory if it doesn't exist
  std::string mkdir_cmd = absl::StrCat("mkdir -p ", options.output_dir);
  TF_ASSIGN_OR_RETURN(auto _, ExecuteCommand(mkdir_cmd));

  // Process entry computation only
  const std::string& afir_string = afir_strings[0];

  NPU_LOG_FORMAT_PRINT(INFO, "Processing entry computation: %s",
                       entry_key.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "About to call ConvertAfirToAscir for %s",
                       entry_key.c_str());
  NPU_LOG_FORMAT_PRINT(DEBUG, "AFIR string size: %d bytes", afir_string.size());

  // Step 2: Convert AFIR to AscIR
  TF_ASSIGN_OR_RETURN(auto ascir_text,
                      ConvertAfirToAscir(afir_string, options));
  NPU_LOG_FORMAT_PRINT(DEBUG,
                       "ConvertAfirToAscir completed, ASCIR size: %d bytes",
                       ascir_text.size());

  // Fix sched.axis and outputs.attr.axis for all nodes in order to using one set of axes for the entire graph.
  ascir_text = FixNodeAxisInfo(ascir_text);
  NPU_LOG_PRINT(DEBUG,
                "Fix sched.axis and outputs.attr.axis for all nodes in order to"
                "using one set of axes for the entire graph.");

  // Step 3: Generate kernel for entry computation
  TF_ASSIGN_OR_RETURN(auto comp_result,
                      GenerateKernel(ascir_text, entry_key, options));

  NpuArtifactInfo npu_artifact_info;
  // Use the graph name as the product prefix name. Currently, this naming is based on the computation name. Replace. with _
  npu_artifact_info.name = comp_result.graph_name;
  npu_artifact_info.dir = options.output_dir;
  final_result.artifact_infos[entry_key] = npu_artifact_info;

  NPU_LOG_FORMAT_PRINT(INFO, "Successfully compiled entry computation: %s",
                       entry_key.c_str());

  return final_result;
}

}  // namespace xla
