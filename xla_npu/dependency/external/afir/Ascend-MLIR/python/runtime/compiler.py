"""
Bisheng compiler interface

Provides the function of using the bisheng compiler to compile AscendC kernel.

Main functions:
1. BishengCompiler - Bisheng compiler package class
2. compile_kernel - Convenient compilation function
"""
import subprocess
from pathlib import Path
from typing import Optional, List

from .utils import (
    find_ascend_root,
    setup_environment,
    logger,
    CompilerError,
    BinaryNotFoundError,
    ASCEND_A2
)


# ================================================================
# Bisheng compiler class
# ================================================================

class BishengCompiler:
    """
    Bisheng compiler wrapper class

    Provides compilation and linking functions for AscendC kernel.
    """

    # target architecture
    ARCH_VEC = "dav-c220-vec"  # VectorCore
    ARCH_CUBE = "dav-c220-cube"  # AiCore

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2):
        """
        Initialize the compiler

        Args:
            ascend_root: Ascend root directory, if it is None, it will be searched automatically.
            soc_version: SOC version

        Raises:
            CompilerError: if the compiler cannot be found
        """
        # Set up the environment
        if ascend_root is None:
            ascend_root = find_ascend_root()

        setup_environment(ascend_root, soc_version)

        self.ascend_root = ascend_root
        self.soc_version = soc_version

        # Find compilers and linkers
        self.bisheng_path = self._find_compiler()
        self.linker_path = self._find_linker()

        # TikCpp package path
        self.tikcpp_path = ascend_root / "compiler" / "tikcpp"

        logger.info(f"Bisheng Compiler: {self.bisheng_path}")
        logger.info(f"Linker: {self.linker_path}")

    def _find_compiler(self) -> Path:
        """Find bisheng compiler"""
        candidates = [
            self.ascend_root / "compiler" / "ccec_compiler" / "bin" / "bisheng",
            self.ascend_root / "toolchain" / "bisheng_compiler" / "bisheng",
            self.ascend_root / "tools" / "bisheng_compiler" / "bin" / "bisheng",
            self.ascend_root / "x86_64-linux" / "bin" / "bisheng",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise CompilerError(f"Cannot find bisheng compiler in {self.ascend_root}")

    def _find_linker(self) -> Path:
        """Find the ld.lld linker"""
        candidates = [
            self.ascend_root / "compiler" / "ccec_compiler" / "bin" / "ld.lld",
            self.ascend_root / "toolchain" / "bisheng_compiler" / "ld.lld",
            self.ascend_root / "tools" / "bisheng_compiler" / "bin" / "ld.lld",
            self.ascend_root / "x86_64-linux" / "bin" / "ld.lld",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise CompilerError(f"Cannot find ld.lld linker in {self.ascend_root}")

    def _get_common_options(self) -> List[str]:
        """
        Get common compilation options

        Refer to the compilation option configuration of PyAsc compiler.py
        """
        platform = self.ascend_root.name if self.ascend_root.name.endswith("-linux") else "x86_64-linux"
        asc_include = self.ascend_root / platform / "asc" / "include"
        ascendc_include = self.ascend_root / platform / "ascendc" / "include"

        options = [
            "-std=c++17",
            "--cce-disable-kernel-global-attr-check",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            "-I", str(self.tikcpp_path / "tikcfw"),
            "-I", str(self.tikcpp_path / "tikcfw" / "impl"),
            "-I", str(self.tikcpp_path / "tikcfw" / "interface"),
            "-I", str(asc_include),
            "-I", str(ascendc_include),
            "-DASCENDC_DUMP=0",
            "-D__NPU_TILING__",
            "-DTILING_KEY_VAR=0",
        ]

        # Verify path exists
        for path in [asc_include, ascendc_include]:
            if not path.exists():
                logger.warning(f"Include path does not exist: {path}")

        return options

    def compile(self,
                src_file: Path,
                output_file: Path,
                arch: str = ARCH_VEC,
                opt_level: int = 3,
                verbose: bool = False) -> Path:
        """
        Compile AscendC source files

        Args:
            src_file: source file path
            output_file: output.o file path
            arch: target architecture (ARCH_VEC or ARCH_CUBE)
            opt_level: optimization level (0-3)
            verbose: whether to display detailed output

        Returns:
            Output file path

        Raises:
            CompilerError: Compilation failed
        """
        src_file = Path(src_file)
        output_file = Path(output_file)

        if not src_file.exists():
            raise CompilerError(f"Source file not found: {src_file}")

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build compile command
        opt = f"-O{opt_level}"
        compile_cmd = [
                          str(self.bisheng_path),
                          "-c",
                          "-x", "cce",
                          opt,
                          src_file.name,  # Use the filename because cwd will be set to src_file.parent
                          f"--cce-aicore-arch={arch}",
                          "--cce-aicore-only",
                          "-o", str(output_file.resolve()),  # Use absolute path
                      ] + self._get_common_options()

        logger.info(f"Compiling: {src_file.name} -> {output_file.name}")

        if verbose:
            logger.debug(f"Command: {' '.join(compile_cmd)}")

        # Execute compilation
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(src_file.parent)
        )

        if result.returncode != 0:
            error_msg = f"Bisheng compilation failed (code {result.returncode})"
            if verbose:
                error_msg += f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
            else:
                error_msg += f"\nstderr: {result.stderr[:500]}"
            raise CompilerError(error_msg)

        logger.info(f"Compilation successful: {output_file}")

        if verbose and result.stdout:
            logger.debug(f"Compiler output:\n{result.stdout}")

        return output_file

    def link(self,
             obj_file: Path,
             output_file: Path,
             verbose: bool = False) -> Path:
        """
        Link generation ELF binary

        Args:
            obj_file: input.o file path
            output_file: output ELF binary path
            verbose: whether to display detailed output

        Returns:
            Output file path

        Raises:
            CompilerError: Link failed
        """
        obj_file = Path(obj_file)
        output_file = Path(output_file)

        if not obj_file.exists():
            raise CompilerError(f"Object file not found: {obj_file}")

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build link command
        link_cmd = [
            str(self.linker_path),
            "-m", "aicorelinux",
            "-Ttext=0",
            str(obj_file),
            "-static",
            "-o", str(output_file),
        ]

        logger.info(f"Linking: {obj_file.name} -> {output_file.name}")

        if verbose:
            logger.debug(f"Command: {' '.join(link_cmd)}")

        # Execute link
        result = subprocess.run(
            link_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Linking failed (code {result.returncode})"
            if verbose:
                error_msg += f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
            else:
                error_msg += f"\nstderr: {result.stderr[:500]}"
            raise CompilerError(error_msg)

        logger.info(f"Linking successful: {output_file}")

        return output_file

    def compile_and_link(self,
                         src_file: Path,
                         output_dir: Path,
                         kernel_name: str,
                         arch: str = ARCH_VEC,
                         opt_level: int = 3,
                         verbose: bool = False) -> Path:
        """
        Compile and link to generate ELF binary

        Args:
            src_file: source file path
            output_dir: output directory
            kernel_name: kernel name
            arch: target architecture
            opt_level: optimization level
            verbose: whether to display detailed output

        Returns:
            ELF binary file path
        """
        src_file = Path(src_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate files and final output
        obj_file = output_dir / f"{kernel_name}.o"
        bin_file = output_dir / f"{kernel_name}.bin"

        # compile
        self.compile(src_file, obj_file, arch, opt_level, verbose)

        # link
        self.link(obj_file, bin_file, verbose)

        # Verify output
        if not bin_file.exists():
            raise BinaryNotFoundError(f"Binary file not generated: {bin_file}")

        binary_size = bin_file.stat().st_size
        logger.info(f"Generated binary: {bin_file} ({binary_size} bytes)")

        return bin_file


# ================================================================
# Convenience function
# ================================================================

def compile_kernel(src_file: Path,
                   output_dir: Path,
                   kernel_name: str,
                   ascend_root: Optional[Path] = None,
                   soc_version: str = ASCEND_A2,
                   arch: str = BishengCompiler.ARCH_VEC,
                   opt_level: int = 3,
                   verbose: bool = False) -> Path:
    """
    Compile AscendC kernel to ELF binary

    Args:
        src_file: source file path
        output_dir: output directory
        kernel_name: kernel name
        ascend_root: Ascend root directory
        soc_version: SOC version
        arch: target architecture
        opt_level: optimization level
        verbose: whether to display detailed output

    Returns:
        ELF binary file path

    Example:
        >>> binary_path = compile_kernel(
        ...     src_file="device/kernel.cpp",
        ...     output_dir="./build",
        ...     kernel_name="my_kernel"
        ... )
    """
    compiler = BishengCompiler(ascend_root, soc_version)
    return compiler.compile_and_link(
        src_file=src_file,
        output_dir=output_dir,
        kernel_name=kernel_name,
        arch=arch,
        opt_level=opt_level,
        verbose=verbose
    )
