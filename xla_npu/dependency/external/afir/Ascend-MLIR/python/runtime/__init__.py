"""
Ascend Runtime Library - Ascend NPU Kernel execution library without PyAsc dependency

This library provides a complete Ascend NPU kernel development and execution process:
1. Source code generation (AscenGen.save_source_files)
2. Host Tiling compilation and execution (compile_and_execute_host_tiling)
3. Bisheng Compiler (BishengCompiler)
4. Kernel execution (KernelExecutor, AscendRuntime)
5. Tiling data calculation (calculate_tiling)

Main features:
- Completely remove PyAsc dependency
- Call libruntime_camodel.so directly using ctypes
- Simple and easy-to-use API design
- Complete compilation-execution process
- Support dynamic compilation of host tiling C++ code

Example:
    >>> from runtime import (
    ...     save_source_files,
    ...     compile_and_execute_host_tiling,
    ...     compile_kernel,
    ...     execute_kernel,
    ... )
    >>>
    >>> # 1. Generate source code from graph (returns 4 values)
    >>> host_file, device_file, _, tiling_def = save_source_files(
    ...     graph_text=graph_def,
    ...     output_dir="./build",
    ...     graph_name="my_kernel"
    ... )
    >>>
    >>> # tiling_def contains the C++ definition of the tiling data structure
    >>> # For example: "struct AutofuseTilingData { uint32_t block_dim;... };"
    >>>
    >>> # 2. Compile and execute host tiling to generate tiling data
    >>> tiling_data = calc_tiling_data(
    ...     host_cpp_file=host_file,
    ...     output_shape=(20, 31)
    ... )
    >>>
    >>> # 3. Compile device kernel
    >>> binary_path = compile_kernel(
    ...     str(device_file),
    ...     "./build",
    ...     "my_kernel"
    ... )
    >>>
    >>> # 4. Read binary
    >>> binary_data = read_binary(binary_path)
    >>>
    >>> # 5. Execute kernel
    >>> outputs = execute_kernel(
    ...     binary_data=binary_data,
    ...     function_name="my_kernel",
    ...     inputs=[input_array],
    ...     output_shapes=[(20, 31)],
    ...     tiling_data=tiling_data
    ... )
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

# ================================================================
# Version information
# ================================================================

from . import utils
from . import compiler
from . import executor
from . import ascgen

# ================================================================
# Main export
# ================================================================

# tool module
from .utils import (
    find_ascend_root,
    find_runtime_library,
    setup_environment,
    logger,
    Logger,
    ProgressBar,
    # Abnormal
    AscendRuntimeError,
    CompilerError,
    ExecutorError,
    BinaryNotFoundError,
    InvalidBinaryError,
    MemoryAllocationError,
    KernelLaunchError,
    get_platform,
    clean_dump,
)

# compiler module
from .compiler import (
    BishengCompiler,
    compile_kernel as compile_kernel_with_bisheng,
)

# Ascgen compile module
from .ascgen import (
    AscGen,
    save_source_files,
    calc_tiling_data,
)

# Actuator module
from .executor import (
    DevBinary,
    AscendRuntime,
    KernelExecutor,
    execute_kernel,
)

# ================================================================
# Convenience function
# ================================================================

def read_binary(file_path: str) -> bytes:
    """
    Read ELF binary file

    Args:
        file_path: file path

    Returns:
        Binary data
    """
    with open(file_path, "rb") as f:
        return f.read()


def compile_kernel(src_file: str,
                   output_dir: str,
                   kernel_name: str,
                   **kwargs) -> str:
    """
    Compile the AscendC kernel (convenience function)

    This is a simplified wrapper around compiler.compile_kernel.

    Args:
        src_file: source file path
        output_dir: output directory
        kernel_name: kernel name
        **kwargs: other parameters passed to BishengCompiler

    Returns:
        ELF binary file path

    Example:
        >>> binary_path = compile_kernel(
        ...     src_file="device/kernel.cpp",
        ...     output_dir="./build",
        ...     kernel_name="my_kernel"
        ... )
    """
    from pathlib import Path
    result = compile_kernel_with_bisheng(
        src_file=Path(src_file),
        output_dir=Path(output_dir),
        kernel_name=kernel_name,
        **kwargs
    )
    return str(result)


# ================================================================
# Module information
# ================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Tools
    "get_platform",
    "find_ascend_root",
    "find_runtime_library",
    "setup_environment",
    "logger",
    "Logger",
    "ProgressBar",
    "clean_dump",

    # Abnormal
    "AscendRuntimeError",
    "CompilerError",
    "ExecutorError",
    "BinaryNotFoundError",
    "InvalidBinaryError",
    "MemoryAllocationError",
    "KernelLaunchError",

    # compiler
    "BishengCompiler",
    "compile_kernel_with_bisheng",

    # Ascgen compile
    "AscGen",
    "save_source_files",
    "calc_tiling_data",

    # actuator
    "DevBinary",
    "AscendRuntime",
    "KernelExecutor",
    "execute_kernel",

    # Convenience function
    "read_binary",
    "compile_kernel",
]
