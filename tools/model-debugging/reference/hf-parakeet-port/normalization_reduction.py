"""Diagnostic model of contiguous float32 cascade sums, NOT production code.

Algorithm reference: PyTorch v2.6.0 SumKernel.cpp (BSD-3-Clause),
https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/SumKernel.cpp
Width is an explicit experiment parameter, not inferred from a model name.
Only same-input comparisons against the actual installed engine can validate it.
"""
import numpy as np


def _validate_rows(values, width):
    if type(width) is not int or width not in (8, 16):
        raise ValueError('Diagnostic width must be 8 or 16')
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2:
        raise ValueError('Expected a two-dimensional float32 row matrix')
    if not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('Rows must be nonempty and finite')


def _cascade(groups):
    # Independent elementwise f32 partials; four accumulator levels.
    size = groups.shape[0]
    power = max(4, (size-1).bit_length()//4 if size else 4)
    step = 1 << power
    acc = np.zeros((4,)+groups.shape[1:], dtype=np.float32)
    index = 0
    while index+step <= size:
        for _ in range(step):
            acc[0] = acc[0]+groups[index]
            index += 1
        for level in range(1, 4):
            acc[level] = acc[level]+acc[level-1]
            acc[level-1].fill(0)
            if index & ((step-1) << (level*power)):
                break
    while index < size:
        acc[0] = acc[0]+groups[index]
        index += 1
    for level in range(1, 4):
        acc[0] = acc[0]+acc[level]
    return acc[0]


def _row_partials(items):
    count = items.shape[0]//4
    partials = _cascade(items[:count*4].reshape((count, 4)+items.shape[1:]))
    for index in range(count*4, items.shape[0]):
        partials[0] = partials[0]+items[index]
    for index in range(1, 4):
        partials[0] = partials[0]+partials[index]
    return partials[0]


def cascade_sum_rows(values, width):
    """Contiguous inner-sum candidate, including physical masked-zero tail."""
    _validate_rows(values, width)
    rows, size = values.shape
    transposed = values.T
    if size < width:
        return _row_partials(transposed)
    vectors = size//width
    lanes = _row_partials(transposed[:vectors*width].reshape(vectors, width, rows))
    total = np.zeros(rows, dtype=np.float32)
    # Scalar tail is accumulated before folding vector lanes in this kernel.
    for index in range(vectors*width, size):
        total = total+transposed[index]
    for lane in lanes:
        total = total+lane
    return total


def normalize_candidate(values, valid_length, width):
    """Same-boundary per-feature diagnostic with explicit f32 intermediates."""
    _validate_rows(values, width)
    if type(valid_length) is not int or not 1 <= valid_length <= values.shape[1]:
        raise ValueError('Expected a nonempty integer valid prefix within each row')
    masked = values.copy()
    masked[:, valid_length:] = 0
    mean = cascade_sum_rows(masked, width)/np.float32(valid_length)
    delta = values-mean[:, None]
    delta[:, valid_length:] = 0
    squared = delta*delta
    if valid_length == 1:
        deviation = np.zeros(values.shape[0], dtype=np.float32)
    else:
        deviation = np.sqrt(cascade_sum_rows(squared, width)/np.float32(valid_length-1))
    std = deviation+np.float32(1e-5)
    result = (values-mean[:, None])/std[:, None]
    result[:, valid_length:] = 0
    return result, mean, std
