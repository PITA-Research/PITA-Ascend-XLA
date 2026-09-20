"""
Ascend Kernel executor (no PyAsc dependency)

Use ctypes to directly call libruntime_camodel.so to execute the ELF binary compiled by Bisheng.

Main functions:
1. AscendRuntime - Runtime API encapsulation class
2. KernelExecutor - Kernel executor class
3. execute_kernel - Convenient execution function
"""
import ctypes
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

from .utils import (
    find_ascend_root,
    find_runtime_library,
    setup_environment,
    logger,
    ExecutorError,
    MemoryAllocationError,
    KernelLaunchError,
    InvalidBinaryError,
    ASCEND_A2
)


# ================================================================
# DevBinary structure
# ================================================================

class DevBinary(ctypes.Structure):
    """
    Device Binary structure

    The DevBinary structure corresponding to the runtime is used to register the ELF binary.
    Field order must be correct!
    """
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint64),
    ]


# ================================================================
# Ascend Runtime encapsulation class
# ================================================================

class AscendRuntime:
    """
    Ascend Runtime encapsulation class

    Use ctypes to directly call the API of libruntime_camodel.so.
    """

    # ELF Magic value
    MAGIC_ELF_AIVEC = 0x41415246  # "BARF" - VectorCore kernel
    MAGIC_ELF_AICUBE = 0x41494343  # "AICC" - AiCore kernel
    MAGIC_ELF_AICPU = 0x41415243  # "BARC" - AICPU kernel

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2,
                 simulation_mode: bool = True):
        """
        Initialize Runtime

        Args:
            ascend_root: Ascend root directory
            soc_version: SOC version
            simulation_mode: whether to use CPU simulation mode (default True)

        Raises:
            ExecutorError: Runtime initialization failed
        """
        # Set up the environment
        if ascend_root is None:
            ascend_root = find_ascend_root()
            if ascend_root is None:
                raise ExecutorError("Cannot find Ascend installation")

        setup_environment(ascend_root, soc_version, simulation_mode)

        self.ascend_root = ascend_root
        self.soc_version = soc_version
        self._simulation_mode = simulation_mode
        self._initialized = False
        self._device_set = False
        self._context = None

        # Load runtime library
        self._load_runtime()

    def _load_runtime(self):
        """Load libruntime_camodel.so"""
        runtime_path = find_runtime_library(self.ascend_root, self.soc_version)
        if runtime_path is None:
            raise ExecutorError(
                f"Cannot find libruntime_camodel.so for {self.soc_version}"
            )

        # # For CPU emulation mode, the driver library needs to be preloaded
        # if self._simulation_mode:
        #     simulator_lib_dir = runtime_path.parent
        #     driver_libs = [
        #         "libnpu_drv_camodel.so",
        #         "libnpu_drv.so",
        #     ]
        #
        #     for lib_name in driver_libs:
        #         lib_path = simulator_lib_dir / lib_name
        #         if lib_path.exists():
        #             try:
        #                 logger.info(f"Preloading driver library: {lib_path}")
        #                 ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        #             except Exception as e:
        #                 logger.warning(f"Failed to preload {lib_name}: {e}")

        logger.info(f"Loading runtime: {runtime_path}")
        self.runtime = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
        self._setup_api()
        self._initialized = True

    def _setup_api(self):
        """Set API function signature"""
        # rtSetDevice
        self.runtime.rtSetDevice.argtypes = [ctypes.c_int32]
        self.runtime.rtSetDevice.restype = ctypes.c_int

        # rtDevBinaryRegister
        self.runtime.rtDevBinaryRegister.argtypes = [
            ctypes.POINTER(DevBinary),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.runtime.rtDevBinaryRegister.restype = ctypes.c_int

        # rtFunctionRegister
        self.runtime.rtFunctionRegister.argtypes = [
            ctypes.c_void_p,  # binHandle
            ctypes.c_void_p,  # stubFunc
            ctypes.c_char_p,  # stubName
            ctypes.c_void_p,  # kernelInfoExt
            ctypes.c_uint32,  # funcMode
        ]
        self.runtime.rtFunctionRegister.restype = ctypes.c_int

        # rtKernelLaunch
        self.runtime.rtKernelLaunch.argtypes = [
            ctypes.c_void_p,  # stubFunc
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # args
            ctypes.c_uint32,  # argsSize
            ctypes.c_void_p,  # smDesc
            ctypes.c_void_p,  # stream
        ]
        self.runtime.rtKernelLaunch.restype = ctypes.c_int

        # rtMalloc
        self.runtime.rtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint16,
        ]
        self.runtime.rtMalloc.restype = ctypes.c_int

        # rtFree
        self.runtime.rtFree.argtypes = [ctypes.c_void_p]
        self.runtime.rtFree.restype = ctypes.c_int

        # rtMemcpy - signature: (dst, destMax, src, cnt, kind)
        self.runtime.rtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.runtime.rtMemcpy.restype = ctypes.c_int

        # rtCtxCreate
        self.runtime.rtCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # pctx
            ctypes.c_uint32,  # flags
            ctypes.c_int32,  # device
        ]
        self.runtime.rtCtxCreate.restype = ctypes.c_int

        # rtCtxDestroy
        self.runtime.rtCtxDestroy.argtypes = [ctypes.c_void_p]
        self.runtime.rtCtxDestroy.restype = ctypes.c_int

        # rtCtxSetCurrent
        self.runtime.rtCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.runtime.rtCtxSetCurrent.restype = ctypes.c_int

        # rtStreamCreate - Create stream (do not use get_default_stream directly)
        self.runtime.rtStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self.runtime.rtStreamCreate.restype = ctypes.c_int

        # rtStreamDestroy - destroy stream
        self.runtime.rtStreamDestroy.argtypes = [ctypes.c_void_p]
        self.runtime.rtStreamDestroy.restype = ctypes.c_int

        # rtCtxGetCurrentDefaultStream (only as a fallback, not usually used directly)
        self.runtime.rtCtxGetCurrentDefaultStream.argtypes = []
        self.runtime.rtCtxGetCurrentDefaultStream.restype = ctypes.c_void_p

        # rtDeviceSynchronize
        self.runtime.rtDeviceSynchronize.argtypes = []
        self.runtime.rtDeviceSynchronize.restype = ctypes.c_int

    # ============================================================
    # Device management
    # ============================================================

    def set_device(self, device_id: int = 0):
        """
        Set up device

        Args:
            device_id: device ID

        Note:
            Refer to the implementation of test_runtime_basic.py:
            In camodel simulation mode, rtSetDevice may return an error code, but this does not affect subsequent operations.
            The simulator automatically initializes the necessary context, eliminating the need to manually create the context.
            The key is not to throw an exception or try to create the context manually because of errors in rtSetDevice.
        """
        ret = self.runtime.rtSetDevice(ctypes.c_int32(device_id))

        # The return value is not checked, refer to the implementation of test_runtime_basic.py
        # Even if an error code is returned, camodel will automatically handle it and subsequent operations can still work normally.
        if ret != 0:
            logger.info(f"rtSetDevice returned {ret} (normal in simulation mode)")
        else:
            logger.info(f"Device {device_id} set successfully")

        self._device_set = True

    # ============================================================
    # Memory management
    # ============================================================

    def malloc(self, size: int) -> int:
        """
        Allocate device memory (512-byte aligned)

        Args:
            size: size (bytes)

        Returns:
            Device memory address (aligned)

        Raises:
            MemoryAllocationError: Memory allocation failed

        Note:
            Refer to pyasc implementation, the memory needs to be 512 bytes aligned
        """
        # Allocate 512 more bytes for alignment
        real_size = size + 512
        c_memory_p = ctypes.c_void_p()
        ret = self.runtime.rtMalloc(
            ctypes.byref(c_memory_p),
            ctypes.c_uint64(real_size),
            ctypes.c_uint32(0),  # RT_MEMORY_DEFAULT
            ctypes.c_uint16(33),  # moduleId
        )

        if ret != 0:
            raise MemoryAllocationError(f"rtMalloc failed: {ret}")

        # Aligned to 512 byte boundary
        raw_addr = c_memory_p.value
        aligned_addr = 512 * ((raw_addr + 512 - 1) // 512)

        # Save the original address for release
        if not hasattr(self, '_alloc_map'):
            self._alloc_map = {}
        self._alloc_map[aligned_addr] = raw_addr

        return aligned_addr

    def free(self, ptr: int):
        """
        Release device memory

        Args:
            ptr: device memory address (aligned address)
        """
        # Get the original address for release
        raw_addr = self._alloc_map.get(ptr, ptr)
        self.runtime.rtFree(ctypes.c_void_p(raw_addr))

    def memcpy(self, dst: int, src: int, size: int, kind: int = 1):
        """
        memory copy

        Args:
            dst: destination address
            src: source address
            size: size (bytes)
            kind: copy type
                0 = HOST_TO_HOST
                1 = HOST_TO_DEVICE
                2 = DEVICE_TO_HOST
                3 = DEVICE_TO_DEVICE

        Raises:
            ExecutorError: Copy failed
        """
        ret = self.runtime.rtMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_uint64(size),
            ctypes.c_void_p(src),
            ctypes.c_uint64(size),
            ctypes.c_uint32(kind),
        )
        if ret != 0:
            raise ExecutorError(f"rtMemcpy failed: {ret}")

    def memcpy_h2d(self, dst: int, src: Union[bytes, np.ndarray], size: int):
        """Host to Device memory copy

        Note:
            For large blocks of data, use chunked copy to avoid problems with rtMemcpy
            After testing, the chunk size of 256 bytes is safe
        """
        if isinstance(src, np.ndarray):
            # Convert numpy array to bytes
            src_bytes = src.tobytes()

            # Use chunked copy (up to 256 bytes each time)
            chunk_size = 256
            for offset in range(0, len(src_bytes), chunk_size):
                chunk_bytes = min(chunk_size, len(src_bytes) - offset)
                src_ptr = (ctypes.c_char * chunk_bytes).from_buffer_copy(src_bytes[offset:offset+chunk_bytes])
                self.memcpy(dst + offset, ctypes.addressof(src_ptr), chunk_bytes, kind=1)
        else:
            src_ptr = ctypes.create_string_buffer(src, size)
            self.memcpy(dst, ctypes.addressof(src_ptr), size, kind=1)

    def memcpy_d2h(self, dst: Union[np.ndarray, bytearray], src: int, size: int):
        """
        Device to Host memory copy

        Args:
            dst: destination buffer (numpy array or bytearray)
            src: source device address
            size: copy size (bytes)

        Note:
            For large blocks of data, use element-by-element copy to avoid problems with rtMemcpy
            After testing, 4-byte element copy is safe
        """
        import struct
        output_list = []
        for i in range(0, size, 4):
            elem_ptr = (ctypes.c_char * 4)()
            self.memcpy(
                ctypes.addressof(elem_ptr),
                src + i,
                4,
                kind=2
            )
            output_list.append(bytes(elem_ptr))

        if isinstance(dst, np.ndarray):
            # Copy to numpy array
            dst_bytes = b''.join(output_list)
            dst_view = np.frombuffer(dst_bytes, dtype=dst.dtype).reshape(dst.shape)
            dst[:] = dst_view
        else:
            # Copy to bytearray
            dst[:] = bytearray(b''.join(output_list))

    # ============================================================
    # Kernel management
    # ============================================================

    def register_binary(self, binary_data: bytes, magic: int = MAGIC_ELF_AIVEC) -> int:
        """
        Register ELF binary

        Args:
            binary_data: ELF binary data
            magic: ELF magic value

        Returns:
            Kernel handle

        Raises:
            ExecutorError: Registration failed
        """
        kernel_size = len(binary_data)
        if kernel_size <= 0:
            raise InvalidBinaryError("Kernel size must be greater than 0")

        device_binary = DevBinary(
            magic=ctypes.c_uint32(magic),
            version=ctypes.c_uint32(0),
            data=ctypes.c_char_p(binary_data),
            length=ctypes.c_uint64(kernel_size),
        )

        binary_handle = ctypes.c_void_p()
        ret = self.runtime.rtDevBinaryRegister(
            ctypes.byref(device_binary),
            ctypes.byref(binary_handle),
        )

        if ret != 0:
            raise ExecutorError(f"rtDevBinaryRegister failed: {ret}")

        logger.info(f"Binary registered: 0x{binary_handle.value:X}")
        return binary_handle.value

    def register_function(self, kernel_handle: int, function_name: str, mode: int = 0) -> int:
        """
        Register kernel function

        Args:
            kernel_handle: Kernel handle
            function_name: function name
            mode: mode

        Returns:
            function handle

        Raises:
            ExecutorError: Registration failed
        """
        fn_name_bytes = function_name.encode("utf-8")
        name_ptr = ctypes.c_char_p(fn_name_bytes)
        name_void_ptr = ctypes.cast(name_ptr, ctypes.c_void_p)

        ret = self.runtime.rtFunctionRegister(
            ctypes.c_void_p(kernel_handle),
            name_void_ptr,
            name_ptr,
            name_void_ptr,
            ctypes.c_uint32(mode),
        )

        if ret != 0:
            raise ExecutorError(f"rtFunctionRegister failed: {ret}")

        logger.info(f"Function registered: {function_name} -> 0x{name_void_ptr.value:X}")
        return name_void_ptr.value

    # ============================================================
    # Kernel execution
    # ============================================================

    def create_stream(self, priority: int = 0):
        """
        create stream

        Args:
            priority: Stream priority

        Returns:
            Stream handle
        """
        stream_handle = ctypes.c_void_p()
        ret = self.runtime.rtStreamCreate(
            ctypes.byref(stream_handle),
            ctypes.c_int32(priority),
        )

        if ret != 0:
            raise ExecutorError(f"rtStreamCreate failed: {ret}")

        return stream_handle

    def destroy_stream(self, stream: ctypes.c_void_p):
        """
        destroy stream

        Args:
            stream: Stream handle
        """
        self.runtime.rtStreamDestroy(stream)

    def get_stream(self) -> ctypes.c_void_p:
        """
        Get the default stream - use create_stream instead in emulation mode

        Note: rtCtxGetCurrentDefaultStream will segfault in some cases
        Use create_stream to create a new stream as an alternative
        """
        return self.create_stream()

    def synchronize(self):
        """Sync devices"""
        ret = self.runtime.rtDeviceSynchronize()
        if ret != 0:
            raise ExecutorError(f"rtDeviceSynchronize failed: {ret}")

    def launch_kernel(self,
                      function_handle: int,
                      block_dim: int,
                      args: List[int],
                      stream: Optional[ctypes.c_void_p] = None):
        """
        start kernel

        Args:
            function_handle: function handle
            block_dim: Block dimension
            args: Argument list (will be converted to uint64 array)
            stream: Stream (default stream is used by default)

        Raises:
            KernelLaunchError: Launch failed
        """
        # Convert parameters to uint64 array
        args_array = (ctypes.c_uint64 * len(args))(*args)
        args_size = len(args) * 8

        ret = self.runtime.rtKernelLaunch(
            ctypes.c_void_p(function_handle),
            ctypes.c_uint32(block_dim),
            ctypes.byref(args_array),
            ctypes.c_uint32(args_size),
            ctypes.c_void_p(0),  # smDesc = NULL
            stream if stream is not None else ctypes.c_void_p(0),
        )

        if ret != 0:
            raise KernelLaunchError(f"rtKernelLaunch failed: {ret}")

        logger.debug(f"Kernel launched: block_dim={block_dim}, args={len(args)}")


# ================================================================
# Kernel executor class
# ================================================================

class KernelExecutor:
    """
    Kernel executor

    Provides a simplified kernel execution interface.
    """

    def __init__(self,
                 ascend_root: Optional[Path] = None,
                 soc_version: str = ASCEND_A2,
                 device_id: int = 0,
                 simulation_mode: bool = True):
        """
        Initialize the executor

        Args:
            ascend_root: Ascend root directory
            soc_version: SOC version
            device_id: device ID
            simulation_mode: whether to use CPU simulation mode (default True)
        """
        self.runtime = AscendRuntime(ascend_root, soc_version, simulation_mode)
        self.device_id = device_id
        self._initialized = False

    def initialize(self):
        """Initialize the executor"""
        self.runtime.set_device(self.device_id)
        self._initialized = True

    def execute(self,
                binary_data: bytes,
                function_name: str,
                inputs: List[np.ndarray],
                outputs: List[np.ndarray],
                tiling_data: Optional[bytes] = None,
                block_dim: int = 1,
                magic: int = AscendRuntime.MAGIC_ELF_AIVEC) -> List[np.ndarray]:
        """
        Execute kernel

        Args:
            binary_data: ELF binary data
            function_name: function name
            inputs: list of input tensors
            outputs: list of output tensors (used to allocate memory)
            tiling_data: Tiling data
            block_dim: Block dimension
            magic: ELF magic value

        Returns:
            Output tensor list

        Raises:
            ExecutorError: execution failed
        """
        if not self._initialized:
            self.initialize()

        # Register binary
        kernel_handle = self.runtime.register_binary(binary_data, magic)

        # Register function
        function_handle = self.runtime.register_function(kernel_handle, function_name)

        # Allocate and copy input memory
        input_addrs = []
        for i, inp in enumerate(inputs):
            addr = self.runtime.malloc(inp.nbytes)
            self.runtime.memcpy_h2d(addr, inp, inp.nbytes)
            input_addrs.append(addr)

        # Allocate output memory
        output_addrs = []
        for out in outputs:
            addr = self.runtime.malloc(out.nbytes)
            output_addrs.append(addr)

        # Allocate workspace memory
        workspace_addr = self.runtime.malloc(8192)

        # Build parameters
        args = []
        args.extend(input_addrs)
        args.extend(output_addrs)
        args.append(workspace_addr)  # workspace

        # add tiling data
        if tiling_data:
            for i in range(0, len(tiling_data), 8):
                word = tiling_data[i:i + 8]
                # Pad to 8 bytes
                word = word + b'\x00' * (8 - len(word))
                val = int.from_bytes(word, 'little')
                args.append(val)

        # start kernel
        stream = self.runtime.get_stream()
        self.runtime.launch_kernel(function_handle, block_dim, args, stream)

        # sync
        self.runtime.synchronize()

        # copy output
        for i, (out, addr) in enumerate(zip(outputs, output_addrs)):
            self.runtime.memcpy_d2h(out, addr, out.nbytes)

        # clean up
        for addr in input_addrs + output_addrs:
            self.runtime.free(addr)
        self.runtime.free(workspace_addr)

        return outputs


# ================================================================
# Convenience function
# ================================================================

def execute_kernel(binary_data: bytes,
                   function_name: str,
                   inputs: List[np.ndarray],
                   output_shapes: List[Tuple[int, ...]],
                   tiling_data: Optional[bytes] = None,
                   ascend_root: Optional[Path] = None,
                   soc_version: str = ASCEND_A2,
                   device_id: int = 0,
                   dtype: np.dtype = np.float32,
                   simulation_mode: bool = True) -> List[np.ndarray]:
    """
    Execute kernel (convenience function)

    Args:
        binary_data: ELF binary data
        function_name: function name
        inputs: list of input tensors
        output_shapes: output shape list
        tiling_data: Tiling data
        ascend_root: Ascend root directory
        soc_version: SOC version
        device_id: device ID
        dtype: output data type
        simulation_mode: whether to use CPU simulation mode (default True)

    Returns:
        Output tensor list

    Example:
        >>> outputs = execute_kernel(
        ...     binary_data=binary,
        ...     function_name="my_kernel",
        ...     inputs=[input_array],
        ...     output_shapes=[(20, 31)],
        ...     tiling_data=tiling_bytes
        ... )
    """
    # Allocate output buffer
    outputs = [np.zeros(shape, dtype=dtype) for shape in output_shapes]

    # Create an executor (using CPU emulation mode by default)
    executor = KernelExecutor(ascend_root, soc_version, device_id, simulation_mode)

    # execute
    return executor.execute(binary_data, function_name, inputs, outputs, tiling_data)
