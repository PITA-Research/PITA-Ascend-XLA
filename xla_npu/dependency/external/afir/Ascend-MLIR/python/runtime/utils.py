"""
Ascend Runtime utility functions

Provides common functions such as path search and environment settings.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
import platform

ASCEND_A2 = "Ascend910B1"
ASCEND_A5 = "Ascend910_9599"
ASCEND_HOME_PATH = "ASCEND_HOME_PATH"

# ================================================================
# path tool
# ================================================================


def get_platform():
    # Get system architecture
    arch = platform.machine()

    # Choose the right path based on your architecture
    if arch in ['aarch64', 'arm64']:
        return 'aarch64'
    if arch in ['x86_64', 'amd64']:
        return 'x86_64'
    raise ValueError(f"不支持的架构: {arch}")

def find_ascend_root() -> Path:
    """
    Find the Ascend installation root directory: ASCEND_HOME_PATH environment variable

    Returns:
        Ascend root directory path, returns None if not found
    """

    # Check the ASCEND_HOME_PATH environment variable
    if ASCEND_HOME_PATH not in os.environ:
        print(f"❌ 错误: 环境变量 {ASCEND_HOME_PATH} 未设置")
        print("请先执行: source env.sh")
        sys.exit(1)

    ascend_home_path = os.environ[ASCEND_HOME_PATH]
    print(f"✅ {ASCEND_HOME_PATH}: {ascend_home_path}")

    # Check if the installation path exists
    if not os.path.exists(ascend_home_path):
        print(f"❌ 错误: Ascend 安装路径不存在: {ascend_home_path}")
        sys.exit(1)
    print(f"✅ 安装路径存在：{ascend_home_path}")

    return Path(ascend_home_path)


def find_runtime_library(ascend_root: Optional[Path] = None,
                         soc_version: str = ASCEND_A2,
                         simulation_mode: bool = True) -> Optional[Path]:
    """
    Find runtime library path

    Args:
        ascend_root: Ascend root directory, if it is None, it will be searched automatically.
        soc_version: SOC version (default Ascend910B1)
        simulation_mode: whether it is simulation mode

    Returns:
        runtime library path, returns None if not found
    """
    if ascend_root is None:
        ascend_root = find_ascend_root()

    platform = get_platform()
    # In simulation mode, libraries under tools/simulator are used first.
    if simulation_mode:
        candidates = [
            f"tools/simulator/{soc_version}/lib/libruntime_camodel.so",
            f"{platform}-linux/simulator/{soc_version}/lib/libruntime_camodel.so",
            f"{platform}-linux/simulator/{soc_version}/lib/libruntime_cmodel.so",
        ]
    else:
        candidates = [
            f"{platform}-linux/lib64/libruntime.so",
        ]

    for rel_path in candidates:
        lib_path = ascend_root / rel_path
        if lib_path.exists():
            return lib_path

    return None


# ================================================================
# Environment settings
# ================================================================

def setup_environment(ascend_root: Optional[Path] = None,
                      soc_version: str = ASCEND_A2,
                      simulation_mode: bool = True) -> bool:
    """
    Set up the Ascend runtime environment

    Args:
        ascend_root: Ascend root directory, if it is None, it will be searched automatically.
        soc_version: SOC version
        simulation_mode: whether it is simulation mode

    Returns:
        Whether the environment was successfully set up
    """
    if ascend_root is None:
        ascend_root = find_ascend_root()

    # Set environment variables
    os.environ[ASCEND_HOME_PATH] = str(ascend_root)
    os.environ['SOC_VERSION'] = soc_version

    # Simulation mode related environment variables
    if simulation_mode:
        os.environ['ASCEND_CPU_SIMULATION'] = '1'
        os.environ['ASCEND_DEVICE_ID'] = '0'

    # Update LD_LIBRARY_PATH
    lib_paths = [
        ascend_root / 'lib64',
        ascend_root / f'{platform}-linux' / 'lib64',
        ascend_root / f'{platform}-linux' / 'devlib' / 'linux' / f'{platform}',
    ]

    # Add stub runtime path (for hardware-free environment)
    stub_path = ascend_root / f'runtime/lib64/stub/linux/{platform}'
    if stub_path.exists():
        lib_paths.append(stub_path)

    if soc_version:
        lib_paths.extend([
            ascend_root / f'{platform}-linux/simulator/{soc_version}/lib',
        ])

    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld = ':'.join([str(p) for p in lib_paths if p.exists()])
    os.environ['LD_LIBRARY_PATH'] = f"{new_ld}:{current_ld}"

    return True


# ================================================================
# ELF tools
# ================================================================

def read_elf_binary(file_path: Path) -> bytes:
    """
    Reading ELF binary files

    Args:
        file_path: file path

    Returns:
        ELF file content
    """
    with open(file_path, "rb") as f:
        return f.read()


def get_function_symbols(binary_data: bytes) -> List[str]:
    """
    Extract function symbols from ELF binary

    Args:
        binary_data: ELF binary data

    Returns:
        List of function symbols
    """
    # Simplified implementation: finding printable strings
    # In practical applications, the elftools library should be used to parse ELF
    import re

    # Find null-terminated strings
    strings = re.findall(b'[\x20-\x7e]{4,}\x00', binary_data)
    return [s.decode('ascii', errors='ignore').rstrip('\x00') for s in strings]


# ================================================================
# Logging tool
# ================================================================

class Logger:
    """Simple logging tool"""

    LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}

    def __init__(self, level: str = 'INFO', prefix: str = '[Ascend Runtime]'):
        """
        Initialize the logger

        Args:
            level: log level
            prefix: log prefix
        """
        self.level = self.LEVELS.get(level, 1)
        self.prefix = prefix

    def _log(self, level: str, message: str):
        """Internal log function"""
        if self.LEVELS.get(level, 1) >= self.level:
            print(f"{self.prefix} [{level}] {message}")

    def debug(self, message: str):
        """debug log"""
        self._log('DEBUG', message)

    def info(self, message: str):
        """Information log"""
        self._log('INFO', message)

    def warning(self, message: str):
        """warning log"""
        self._log('WARNING', message)

    def error(self, message: str):
        """error log"""
        self._log('ERROR', message)


# Default logger instance
logger = Logger()


# ================================================================
# Verification tool
# ================================================================

def validate_binary(binary_data: bytes,
                    expected_magic: bytes = b'\\x7fELF') -> bool:
    """
    Verify ELF binaries

    Args:
        binary_data: binary data
        expected_magic: expected magic number (default ELF)

    Returns:
        Is it a valid ELF file?
    """
    return binary_data[:4] == expected_magic


def validate_input_shape(shape: tuple) -> bool:
    """
    Validate input shape

    Args:
        shape: input shape

    Returns:
        Is it valid?
    """
    if not isinstance(shape, (tuple, list)):
        return False
    if len(shape) < 2:
        return False
    return all(isinstance(d, int) and d > 0 for d in shape)


# ================================================================
# progress display
# ================================================================

class ProgressBar:
    """simple progress bar"""

    def __init__(self, total: int, width: int = 50):
        """
        Initialization progress bar

        Args:
            total: total
            width: progress bar width
        """
        self.total = total
        self.width = width
        self.current = 0

    def update(self, n: int = 1):
        """
        update progress

        Args:
            n: increased quantity
        """
        self.current += n
        self._draw()

    def _draw(self):
        """Draw progress bar"""
        if self.total == 0:
            return

        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = '█' * filled + '░' * (self.width - filled)
        percent = int(ratio * 100)

        # Use \\r to make the progress bar update on the same line
        sys.stdout.write(f'\\r[{bar}] {percent}%')
        sys.stdout.flush()

    def finish(self):
        """Complete progress bar"""
        self.current = self.total
        self._draw()
        print()  # newline


# ================================================================
# Exception class
# ================================================================

class AscendRuntimeError(Exception):
    """Ascend Runtime basic exception"""
    pass


class CompilerError(AscendRuntimeError):
    """Compiler exception"""
    pass


class ExecutorError(AscendRuntimeError):
    """Executor exception"""
    pass


class BinaryNotFoundError(CompilerError):
    """Binary file not found exception"""
    pass


class InvalidBinaryError(CompilerError):
    """Invalid binary file exception"""
    pass


class MemoryAllocationError(ExecutorError):
    """Memory allocation exception"""
    pass


class KernelLaunchError(ExecutorError):
    """Kernel startup exception"""
    pass


def clean_dump():
    """Clean the core*.dump files generated by the CANN simulator"""
    if os.environ.get("ASCEND_CLEAN_DUMP", "1") == "1":
        patterns = ("*.dump", "*.toml", "*summary_log", "*vcd", "ffts*.log")
        for p in patterns:
            for f in Path(".").glob(p):
                try:
                    f.unlink()
                except OSError:
                    pass