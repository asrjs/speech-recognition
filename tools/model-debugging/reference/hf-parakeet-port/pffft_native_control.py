"""Opt-in trusted local PFFFT binary control; never a browser implementation."""
import ctypes as c
from hashlib import sha256
from pathlib import Path
import numpy as np


def pffft_control(frames, library, expected_sha256):
    if (frames.ndim != 3 or frames.shape[0] != 1 or frames.shape[1] < 1
            or frames.shape[2] != 512 or frames.dtype != np.float32
            or not np.isfinite(frames).all()):
        raise ValueError('PFFFT requires finite fp32 [1, frames, 512] input')
    path = Path(library).resolve(strict=True)
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('PFFFT binary identity differs')
    # Loading an explicitly requested binary executes native code: trusted local builds only.
    dll = c.CDLL(str(path))
    dll.pffft_new_setup.argtypes = [c.c_int, c.c_int]
    dll.pffft_new_setup.restype = c.c_void_p
    dll.pffft_destroy_setup.argtypes = [c.c_void_p]
    dll.pffft_destroy_setup.restype = None
    dll.pffft_aligned_malloc.argtypes = [c.c_size_t]
    dll.pffft_aligned_malloc.restype = c.c_void_p
    dll.pffft_aligned_free.argtypes = [c.c_void_p]
    dll.pffft_aligned_free.restype = None
    dll.pffft_transform_ordered.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p,
                                            c.c_void_p, c.c_int]
    dll.pffft_transform_ordered.restype = None
    dll.pffft_simd_arch.argtypes = []
    dll.pffft_simd_arch.restype = c.c_char_p
    dll.pffft_simd_size.argtypes = []
    dll.pffft_simd_size.restype = c.c_int
    setup = dll.pffft_new_setup(512, 0)  # enum PFFFT_REAL
    if not setup:
        raise MemoryError('PFFFT setup allocation failed')
    pointers = []
    try:
        for _ in range(3):
            pointer = dll.pffft_aligned_malloc(512 * 4)
            if not pointer:
                raise MemoryError('PFFFT aligned allocation failed')
            pointers.append(pointer)
        arrays = [np.ctypeslib.as_array(c.cast(pointer, c.POINTER(c.c_float)),
                                      shape=(512,)) for pointer in pointers]
        input_values, output_values, _ = arrays
        candidate = np.zeros((1, 257, frames.shape[1]), dtype=np.complex64)
        for index in range(frames.shape[1]):
            input_values[:] = frames[0, index]
            dll.pffft_transform_ordered(setup, *pointers, 0)  # PFFFT_FORWARD, unscaled
            if not np.isfinite(output_values).all():
                raise ValueError('PFFFT output is nonfinite')
            candidate.real[0, 0, index] = output_values[0]
            candidate.real[0, 256, index] = output_values[1]
            candidate.real[0, 1:256, index] = output_values[2::2]
            candidate.imag[0, 1:256, index] = output_values[3::2]
        metadata = {'binary_path': str(path), 'binary_sha256': actual,
                    'architecture': dll.pffft_simd_arch().decode('ascii'),
                    'simd_size': dll.pffft_simd_size(), 'length': 512,
                    'transform': 'REAL/FORWARD ordered, unscaled',
                    'scope': 'Native binary control only; compiler/source provenance is separate'}
    finally:
        for pointer in pointers:
            dll.pffft_aligned_free(pointer)
        dll.pffft_destroy_setup(setup)
    if sha256(path.read_bytes()).hexdigest() != actual:
        raise ValueError('PFFFT binary changed during execution')
    return candidate, metadata
