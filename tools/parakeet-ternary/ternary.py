"""Decode the `thrush-ternary-v2` weight format used by `moondream/parakeet-redux`.

The format is fully described by the repository's `ternary.json`, so the packed
values can be expanded without any model-specific guesswork:

    packing: base 3, 5 elements per byte, least significant digit first,
             element i of a row is digit i % 5 of byte i // 5, pad digits 0
    weight_rule: w = scales[row, col // group_size] * (code - 1), code in {0,1,2}

Every expanded weight is a small-integer code times a stored scale, so expanding
to float32 is exact: no approximation is introduced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELEMENTS_PER_BYTE = 5
CODE_OFFSET = 1
DIGITS_PER_CODE = 3


@dataclass(frozen=True)
class TernaryLayout:
    """Packing geometry for one quantized matrix."""

    in_features: int
    out_features: int
    group_size: int

    @property
    def row_bytes(self) -> int:
        """Bytes per row: the manifest defines this as ``ceil(in/5)``."""

        return -(-self.in_features // ELEMENTS_PER_BYTE)

    @property
    def groups(self) -> int:
        return -(-self.in_features // self.group_size)


def unpack_rows(packed: bytes, layout: TernaryLayout) -> np.ndarray:
    """Expand packed bytes into a ``[out_features, in_features]`` uint8 code grid."""

    expected = layout.row_bytes * layout.out_features
    if len(packed) != expected:
        raise ValueError(
            f"packed block is {len(packed)} bytes, layout needs {expected} "
            f"({layout.row_bytes} per row x {layout.out_features} rows)"
        )

    raw = np.frombuffer(packed, dtype=np.uint8)
    invalid = raw >= DIGITS_PER_CODE ** ELEMENTS_PER_BYTE
    if invalid.any():
        offset = int(invalid.argmax())
        raise ValueError(
            f"invalid packed ternary byte {int(raw[offset])} at offset {offset}; "
            "five base-3 digits must encode a value below 243"
        )
    grid = raw.reshape(layout.out_features, layout.row_bytes)

    # Digit i % 5 of byte i // 5, least significant first, pad digits 0.
    powers = (DIGITS_PER_CODE ** np.arange(ELEMENTS_PER_BYTE)).astype(np.uint16)
    codes = (grid[:, :, None] // powers[None, None, :]) % DIGITS_PER_CODE
    codes = codes.reshape(layout.out_features, layout.row_bytes * ELEMENTS_PER_BYTE)

    # The last byte is zero-padded; drop the padding rather than trusting it.
    return codes[:, : layout.in_features].copy()


def dequantize(codes: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Apply the manifest weight rule: ``w = scales[row, col // group] * (code - 1)``."""

    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got {codes.ndim}-D")
    out_features, in_features = codes.shape
    expected_groups = -(-in_features // group_size)
    if scales.shape != (out_features, expected_groups):
        raise ValueError(
            f"scales shape {scales.shape} does not match "
            f"({out_features}, {expected_groups}) for group_size {group_size}"
        )

    group_of_column = np.arange(in_features) // group_size
    scale_grid = scales[:, group_of_column]
    return (scale_grid * (codes.astype(np.float32) - CODE_OFFSET)).astype(np.float32)


def decode_matrix(
    packed: bytes, scales: np.ndarray, layout: TernaryLayout
) -> np.ndarray:
    """Full decode for one weight matrix."""

    return dequantize(unpack_rows(packed, layout), scales, layout.group_size)


def pack_rows(codes: np.ndarray) -> bytes:
    """Pack a code grid back into bytes. Only used to test the decoder."""

    if codes.ndim != 2 or codes.dtype != np.uint8:
        raise ValueError("codes must be a 2-D uint8 array")
    if codes.size and (codes.min() < 0 or codes.max() > 2):
        raise ValueError("codes must lie in {0, 1, 2}")
    padded = -(-codes.shape[1] // ELEMENTS_PER_BYTE) * ELEMENTS_PER_BYTE
    if padded != codes.shape[1]:
        codes = np.pad(codes, ((0, 0), (0, padded - codes.shape[1])))
    # Each byte holds 5 digits, so reduce groups of 5 down to one byte per group.
    groups = codes.reshape(codes.shape[0], -1, ELEMENTS_PER_BYTE).astype(np.uint32)
    weights = (DIGITS_PER_CODE ** np.arange(ELEMENTS_PER_BYTE)).astype(np.uint32)
    combined = (groups * weights[None, None, :]).sum(axis=2)

    return combined.astype(np.uint8).tobytes()
