"""
AscendC source code generation interface

Provides the function of using Autofuse to generate source code and compile and execute host tiling.

Main functions:
1. AscGen - source code generator package class
2. compile_and_execute_host_tiling - Compile and execute host tiling C++ code
3. save_source_files - save source code files
"""
import ctypes
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .utils import (
    get_platform,
    find_ascend_root,
    setup_environment,
    logger,
    CompilerError,
    ASCEND_A2,
)


# ================================================================
# Source code generator class
# ================================================================

class AscGen:
    """
    AscendC source code generator wrapper class

    Provides Autofuse source code generation and host tiling compilation and execution functions.
    """

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2):
        """
        Initialize the source code generator

        Args:
            ascend_root: Ascend root directory
            soc_version: SOC version

        Raises:
            CompilerError: initialization failed
        """
        # Set up the environment
        if ascend_root is None:
            ascend_root = find_ascend_root()

        setup_environment(ascend_root, soc_version)

        self.ascend_root = ascend_root
        self.soc_version = soc_version
        self.platform = get_platform()

        # Load libgraph.so
        self._load_libgraph()

        # Import the autofuse module
        self._import_autofuse()

    def _load_libgraph(self):
        """Load libgraph.so"""
        libgraph_path = self.ascend_root / "lib64" / "libgraph.so"

        if not libgraph_path.exists():
            raise CompilerError(f"Cannot find libgraph.so: {libgraph_path}")

        try:
            ctypes.CDLL(str(libgraph_path), mode=ctypes.RTLD_GLOBAL)
            logger.info(f"Loaded libgraph.so: {libgraph_path}")
        except OSError as e:
            raise CompilerError(f"Failed to load libgraph.so: {e}")

    def _import_autofuse(self):
        """Import autofuse related modules"""
        try:
            from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions
            self.ascir = ascir
            self.Autofuser = Autofuser
            self.AutofuserOptions = AutofuserOptions
            logger.info("Autofuse modules imported successfully")
        except ImportError as e:
            raise CompilerError(f"Failed to import autofuse modules: {e}")

    def compile_graph(self,
                      graph_text: str,
                      output_dir: Path,
                      graph_name: str = "kernel") -> Tuple[Path, Path, str, str]:
        """
        Generate source code from AscendC graph

        Args:
            graph_text: AscendC graph text definition
            output_dir: output directory
            graph_name: graph name

        Returns:
            (host source file path, device source file path, host_tiling source code, tiling_def structure definition)

        Raises:
            CompilerError: Build failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Deserialize graph
        graph = self._deserialize_graph(graph_text)

        # Execute autofuse
        try:
            tiling_def, host_tiling, op_kernel = self._run_autofuse(graph)
        except Exception as e:
            failed_graph_file = output_dir / f"{graph_name}_failed.ascir"
            failed_debug_file = output_dir / f"{graph_name}_failed_debug.txt"
            try:
                failed_graph_file.write_text(graph_text, encoding="utf-8")
                failed_debug_file.write_text(
                    self.ascir.utils.debug_str(graph), encoding="utf-8"
                )
            except Exception as dump_error:
                logger.error(f"Failed to dump failed graph: {dump_error}")
            raise CompilerError(
                f"Autofuse schedule/codegen failed for graph '{graph_name}'. "
                f"Input graph dumped to {failed_graph_file}, "
                f"debug graph dumped to {failed_debug_file}: {e}"
            ) from e

        # Save source file
        host_file, device_file = self._save_source_files(
            tiling_def, host_tiling, op_kernel,
            output_dir, graph_name
        )

        return host_file, device_file, host_tiling, tiling_def

    def _deserialize_graph(self, graph_text: str):
        """Deserialize ascgraph text"""
        graph = self.ascir.utils.deserialize("asc_graph", graph_text)
        debug_str = self.ascir.utils.debug_str(graph)

        if not debug_str:
            raise CompilerError("Failed to deserialize ascgraph")

        logger.debug(f"Graph deserialized: \n{debug_str}")
        return graph

    def _run_autofuse(self, graph):
        """Execute autofuse process"""
        options = self.AutofuserOptions()
        fuser = self.Autofuser(options)

        sched_result = fuser.schedule(graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)

        if not all([tiling_def, host_tiling, op_kernel]):
            raise CompilerError("Autofuse codegen failed")

        logger.info("Autofuse pipeline completed")
        return tiling_def, host_tiling, op_kernel

    def _save_source_files(self, tiling_def, host_tiling, op_kernel,
                           output_dir: Path, graph_name: str) -> Tuple[Path, Path]:
        """
        Save source code files to the output directory

        Args:
            tiling_def: tiling definition
            host_tiling: host tiling source code
            op_kernel: device kernel source code
            output_dir: output directory
            graph_name: graph name

        Returns:
            (host source file path, device source file path)
        """
        TILING_DATA_HEADER_NAME = 'autofuse_tiling_data.h'

        # Create subdirectory
        host_dir = output_dir / "host"
        device_dir = output_dir / "device"
        host_dir.mkdir(parents=True, exist_ok=True)
        device_dir.mkdir(parents=True, exist_ok=True)

        # Save host tiling source file
        host_file = host_dir / f"{graph_name}_tiling.cpp"
        host_file.write_text(host_tiling, encoding="utf-8")
        tiling_def_file = host_dir / TILING_DATA_HEADER_NAME
        tiling_def_file.write_text(tiling_def, encoding="utf-8")
        logger.info(f"Saved host tiling: {host_file}")

        # Save device kernel source file
        device_file = device_dir / f"{graph_name}_op_kernel.cpp"
        device_file.write_text(op_kernel, encoding="utf-8")
        tiling_def_file = device_dir / TILING_DATA_HEADER_NAME
        tiling_def_file.write_text(tiling_def, encoding="utf-8")
        logger.info(f"Saved device kernel: {device_file}")

        return host_file, device_file

    def calc_tiling_data(
            self,
            tiling_def: str,
            host_tiling: str,
            build_dir: Optional[Path] = None,
    ) -> bytes:
        """
        Compile and execute host tiling C++ code to generate tiling data

        Args:
            tiling_def: tiling data structure definition (C++ code)
            host_tiling: host tiling C++ source code
            build_dir: Compilation output directory, if None, a temporary directory is used (for debugging)

        Returns:
            tiling data (bytes)
        """
        # Decide which directory to use
        if build_dir is None:
            # Use temporary directory (old behavior)
            work_dir = Path(tempfile.mkdtemp())
            cleanup = True
        else:
            # Use user-specified directory (easy for debugging)
            work_dir = Path(build_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
            logger.info(f"Using build directory for debugging: {work_dir}")

        try:
            # Compile host tiling to.so
            so_file = self._compile_host_tiling_to_so(
                tiling_def, host_tiling, work_dir
            )
            logger.info(f"Compiled host tiling to: {so_file}")
            if so_file is None:
                raise CompilerError("Failed to compile host tiling to .so")

            # Call the function in.so to obtain tiling data
            tiling_data = self._call_tiling_functions(so_file)

            return tiling_data
        finally:
            # Clean up temporary directory
            if cleanup and work_dir.exists():
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

    def _compile_host_tiling_to_so(
            self,
            tiling_def: str,
            host_tiling: str,
            output_dir: Path
    ) -> Optional[Path]:
        """
        Compile host tiling C++ source code into.so file

        Use CMake to build the system because host tiling uses the ASC language,
        Requires Ascend specific compilers and libraries.

        Args:
            tiling_def: tiling data structure definition (C++ code)
            host_tiling: host tiling C++ source code
            output_dir: output directory

        Returns:
            Compiled.so file path, returns None on failure
        """
        output_dir = Path(output_dir)

        # Save source file
        source_file = output_dir / "host_tiling_combined.cpp"
        source_content = f"""// Auto-generated host tiling library
// Tiling data structure definition
{tiling_def}

// Host tiling implementation
{host_tiling}
"""
        source_file.write_text(source_content, encoding="utf-8")

        # Save tiling_def header file (may be required at compile time)
        tiling_header = output_dir / "autofuse_tiling_data.h"
        tiling_header.write_text(tiling_def, encoding="utf-8")

        # Find ASC CMake package path
        asc_cmake_path = None
        possible_paths = [
            self.ascend_root / "tools" / "tikcpp" / "ascendc_kernel_cmake",
            self.ascend_root / "compiler" / "tikcpp" / "ascendc_kernel_cmake",
            self.ascend_root / "ascendc_devkit" / "tikcpp" / "samples" / "cmake",
        ]
        for path in possible_paths:
            if path.exists():
                asc_cmake_path = path
                break

        # Generate CMakeLists.txt
        cmake_file = output_dir / "CMakeLists.txt"
        if asc_cmake_path is not None:
            logger.info(f"Using ASC CMake package from: {asc_cmake_path}")
            # CMake configuration using ASC language support
            cmake_content = f"""cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED HINTS {asc_cmake_path})
project(HostTiling LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION {self.soc_version} CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "{self.ascend_root}" CACHE PATH "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# Set library search path (must precede add_library)
link_directories(
    {self.ascend_root}/lib64
)

# Set source file properties to ASC language
set_source_files_properties(
    {source_file.name}
    PROPERTIES LANGUAGE ASC
)

# Create shared library
add_library(host_tiling SHARED
    {source_file.name}
)

# Set the output name (using the standard lib prefix and.so suffix)
set_target_properties(host_tiling PROPERTIES
    OUTPUT_NAME host_tiling
)

# Link required libraries
target_link_libraries(host_tiling PRIVATE
    c_sec
    ascendalog
    platform
    tiling_api
    graph
)

# Set include path
target_include_directories(host_tiling PRIVATE
    {output_dir}
    {self.ascend_root}/include
    {self.ascend_root}/include/graph
    {self.ascend_root}/include/experiment
    {self.ascend_root}/pkg_inc/base
    {self.ascend_root}/{self.platform}-linux/include
    {self.ascend_root}/{self.platform}-linux/ascendc/include/highlevel_api/tiling/platform
)

# Set compilation options
target_compile_options(host_tiling PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    -O2
    -fno-common
    -Wextra
    -Wfloat-equal
    -fvisibility=default
    -DLOG_CPP
    -ffile-prefix-map=${{CMAKE_CURRENT_LIST_DIR}}/=
)
"""
        else:
            # Fallback: Instead of using ASC language, try using standard C++ (may fail)
            logger.warning(f"ASC CMake package not found, using fallback C++ mode (may not work)")
            cmake_content = f"""cmake_minimum_required(VERSION 3.14)
project(HostTiling CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_ROOT "{self.ascend_root}")

# Set library search path
link_directories(${{ASCEND_ROOT}}/lib64)

# Create shared library
add_library(host_tiling SHARED
    {source_file.name}
)

# Set output name
set_target_properties(host_tiling PROPERTIES
    OUTPUT_NAME host_tiling
)

# Set include path
target_include_directories(host_tiling PRIVATE
    {output_dir}
    ${{ASCEND_ROOT}}/include
    ${{ASCEND_ROOT}}/include/graph
    ${{ASCEND_ROOT}}/include/experiment
    ${{ASCEND_ROOT}}/pkg_inc/base
    ${{ASCEND_ROOT}}/{self.platform}-linux/pkg_inc/base
    ${{ASCEND_ROOT}}/{self.platform}-linux/include
    ${{ASCEND_ROOT}}/{self.platform}-linux/ascendc/include/highlevel_api/tiling/platform
)

# Link required libraries
target_link_libraries(host_tiling PRIVATE
    c_sec
    ascendalog
    platform
    tiling_api
    graph
)

# Set compilation options
target_compile_options(host_tiling PRIVATE
    -O2
    -fno-common
    -Wextra
    -fvisibility=default
    -DLOG_CPP
)
"""
        cmake_file.write_text(cmake_content, encoding="utf-8")

        # Create build directory
        build_dir = output_dir / "cmake_build"
        build_dir.mkdir(exist_ok=True)

        so_file = build_dir / "libhost_tiling.so"

        # Set environment variables (refer to call_ascgen.py:setup_cmake_environment)
        env = os.environ.copy()
        env["ASCEND_HOME_PATH"] = str(self.ascend_root)
        env["SOC_VERSION"] = self.soc_version

        # Path settings required by CMake
        if asc_cmake_path is not None:
            env["CMAKE_PREFIX_PATH"] = str(asc_cmake_path)
            logger.info(f"Setting CMAKE_PREFIX_PATH: {asc_cmake_path}")

        env["ASCEND_CANN_PACKAGE_PATH"] = str(self.ascend_root)
        logger.info(f"Setting ASCEND_CANN_PACKAGE_PATH: {self.ascend_root}")

        # Set LD_LIBRARY_PATH
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{self.ascend_root}/lib64:{ld_path}"

        try:
            # 1. CMake configuration
            logger.info(f"Configuring CMake project...")
            configure_cmd = [
                "cmake",
                "-S", str(output_dir),
                "-B", str(build_dir),
                f"-DCMAKE_BUILD_TYPE=Release",
            ]

            result = subprocess.run(
                configure_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"CMake configuration failed: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return None

            logger.info(f"CMake configuration successful")

            # 2. CMake build
            logger.info(f"Building with CMake...")
            build_cmd = [
                "cmake",
                "--build", str(build_dir),
                "--parallel",
            ]

            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"CMake build failed: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return None

            # Check if the output file exists
            if so_file.exists():
                logger.info(f"Build successful: {so_file}")
                return so_file
            else:
                # Find possible other output locations
                for possible_so in build_dir.glob("*.so"):
                    logger.info(f"Found .so file: {possible_so}")
                    return possible_so
                logger.error(f"No .so file found in {build_dir}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Build timeout")
            return None
        except FileNotFoundError as e:
            logger.error(f"CMake not found: {e}")
            logger.error("Please ensure CMake is installed and available in PATH")
            return None
        except Exception as e:
            logger.error(f"Build error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _call_tiling_functions(self, so_file: Path) -> bytes:
        """
        Call the function in the.so file to obtain tiling data

        Args:
            so_file:.so file path

        Returns:
            tiling data (bytes)
        """
        # Load.so file
        # Use RTLD_LOCAL to avoid global state conflicts with libruntime_camodel.so
        try:
            so_lib = ctypes.CDLL(str(so_file), mode=ctypes.RTLD_LOCAL)
            logger.info(f"Loaded .so file: {so_file}")
        except OSError as e:
            raise CompilerError(f"Failed to load .so file: {e}")

        # Get GetTilingDataSize function
        try:
            get_tiling_size_func = so_lib.GetTilingDataSize
            get_tiling_size_func.restype = ctypes.c_size_t
            get_tiling_size_func.argtypes = []
        except AttributeError as e:
            raise CompilerError(f"GetTilingDataSize function not found in .so: {e}")

        # Get tiling data size
        tiling_size = get_tiling_size_func()
        logger.info(f"Tiling data size: {tiling_size} bytes")

        # Allocate tiling data buffer
        tiling_buffer = ctypes.create_string_buffer(tiling_size)

        # Get the AutofuseTiling function
        try:
            autofuse_tiling_func = so_lib.AutofuseTiling
            autofuse_tiling_func.restype = ctypes.c_int
            autofuse_tiling_func.argtypes = [
                ctypes.c_void_p,  # tiling_data
                ctypes.POINTER(ctypes.c_uint32),  # workspace_size
                ctypes.POINTER(ctypes.c_uint32),  # block_dim
                ctypes.c_void_p,  # unknown (can be nullptr)
            ]
        except AttributeError as e:
            raise CompilerError(f"AutofuseTiling function not found in .so: {e}")

        # Call AutofuseTiling to calculate tiling data
        workspace_size = ctypes.c_uint32(0)
        block_dim = ctypes.c_uint32(0)

        ret = autofuse_tiling_func(
            ctypes.addressof(tiling_buffer),  # tiling_data
            ctypes.byref(workspace_size),  # workspace_size
            ctypes.byref(block_dim),  # block_dim
            ctypes.c_void_p(0)  # unknown (nullptr)
        )

        if ret != 0:
            raise CompilerError(f"AutofuseTiling failed with return code: {ret}")

        logger.info(
            f"Tiling calculation successful: block_dim={block_dim.value}, workspace_size={workspace_size.value}")

        # Return tiling data
        tiling_data = tiling_buffer.raw[:tiling_size]

        # Parse and print tiling data
        self._parse_and_print_tiling_data(tiling_data)

        return tiling_data

    def _parse_and_print_tiling_data(self, tiling_data: bytes):
        """
        Parse and print tiling data

        Parse binary data based on the AutofuseTilingData structure:
        struct AutofuseTilingData {
            uint32_t block_dim;
            uint32_t corenum;
            uint32_t ub_size;
            uint32_t hbm_size;
            uint32_t tiling_key;
            uint32_t z1t_size;
            uint32_t z0z1Tb_size;
            uint32_t z0t_size;
            uint32_t z0Tb_size;
            uint32_t q0_size;
            uint32_t q1_size;
            uint32_t q2_size;
            uint32_t b0_size;
            uint32_t b1_size;
            uint32_t tmp_tbuf_size;
        };

        Args:
            tiling_data: tiling data binary
        """
        # AutofuseTilingData contains 15 uint32_t fields
        expected_size = 15 * 4  # 60 bytes
        if len(tiling_data) < expected_size:
            logger.warning(
                f"Tiling data size ({len(tiling_data)} bytes) is smaller than expected ({expected_size} bytes)")
            # Only parse available data
            num_fields = len(tiling_data) // 4
        else:
            num_fields = 15

        # Using the struct module to parse uint32_t arrays
        # Format: 15 'I' (unsigned int, 4 bytes)
        fmt = f'{num_fields}I'
        fields = struct.unpack(fmt, tiling_data[:num_fields * 4])

        # Field name mapping
        field_names = [
            "block_dim",
            "corenum",
            "ub_size",
            "hbm_size",
            "tiling_key",
            "z1t_size",
            "z0z1Tb_size",
            "z0t_size",
            "z0Tb_size",
            "q0_size",
            "q1_size",
            "q2_size",
            "b0_size",
            "b1_size",
            "tmp_tbuf_size",
        ]

        logger.info("\n" + "=" * 70)
        logger.info("Tiling Data (AutofuseTilingData):")
        logger.info("=" * 70)

        # print each field
        for i, (name, value) in enumerate(zip(field_names, fields)):
            logger.info(f"  [{i:2d}] {name:20s} = {value:10d} (0x{value:08x})")

        logger.info("=" * 70)
        logger.info(f"Total size: {len(tiling_data)} bytes")
        logger.info("=" * 70 + "\n")


# ================================================================
# Convenience function
# ================================================================

def save_source_files(graph_text: str,
                      output_dir: Path,
                      graph_name: str = "kernel",
                      ascend_root: Optional[Path] = None,
                      soc_version: Optional[str] = None) -> Tuple[Path, Path, str, str]:
    """
    Generate source code files from AscendC graph (convenience function)

    Args:
        graph_text: AscendC graph text definition
        output_dir: output directory
        graph_name: graph name
        ascend_root: Ascend root directory
        soc_version: SOC version

    Returns:
        (host source file path, device source file path, host_tiling source code, tiling_def structure definition)

    Example:
        >>> host_file, device_file, host_tiling, tiling_def = save_source_files(
        ...     graph_text=graph_def,
        ...     output_dir="./build",
        ...     graph_name="my_kernel"
        ... )
    """
    # Use soc_version from parameter, or fall back to environment variable, or use ASCEND_A2 as default
    if soc_version is None:
        soc_version = os.getenv("SOC_VERSION", ASCEND_A2)
    compiler = AscGen(ascend_root, soc_version)
    return compiler.compile_graph(graph_text, output_dir, graph_name)


def calc_tiling_data(
        tiling_def: str,
        host_tiling: str,
        ascend_root: Optional[Path] = None,
        soc_version: Optional[str] = None,
        build_dir: Optional[Path] = None,
) -> bytes:
    """
    Compile and execute host tiling C++ code (convenience function)

    Args:
        tiling_def: tiling data structure definition (C++ code)
        host_tiling: host tiling C++ source code
        ascend_root: Ascend root directory
        soc_version: SOC version
        build_dir: Compilation output directory, if None, a temporary directory is used (for debugging)

    Returns:
        tiling data (bytes)

    Example:
        >>> tiling_data = calc_tiling_data(
        ...     tiling_def=tiling_def_str,
        ...     host_tiling=host_tiling_str
        ... )
        >>>
        >>> # Specify build directory for debugging
        >>> tiling_data = calc_tiling_data(
        ...     tiling_def=tiling_def_str,
        ...     host_tiling=host_tiling_str,
        ...     build_dir="./debug_build"
        ... )
    """
    # Use soc_version from parameter, or fall back to environment variable, or use ASCEND_A2 as default
    if soc_version is None:
        soc_version = os.getenv("SOC_VERSION", ASCEND_A2)
    compiler = AscGen(ascend_root, soc_version)
    return compiler.calc_tiling_data(tiling_def, host_tiling, build_dir)
