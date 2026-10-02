"""Oracle-injection mask guard for frozen spectra; never production inference."""

import hashlib
import numpy as np


def _spectrum(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.complex64
            or values.ndim != 3 or values.shape[0] != 1 or not values.size
            or not np.isfinite(values).all()):
        raise ValueError("Spectrum must be finite nonempty batch-one complex64")


def validate_complex_intervention(candidate, oracle, patched, coordinates, valid_length):
    """Prove selected values are exact and every unselected component is unchanged.

    Coordinates are (bin, frame), bounded by the real valid prefix, not padding.
    Canonical bytes preserve signed zero and support noncontiguous STFT layouts.
    A successful guard is diagnostic evidence, not model or FFT acceptance.
    """
    for values in (candidate, oracle, patched):
        _spectrum(values)
    if candidate.shape != oracle.shape or candidate.shape != patched.shape:
        raise ValueError("Spectrum shapes differ")
    if type(valid_length) is not int or not 0 < valid_length <= candidate.shape[2]:
        raise ValueError("Invalid valid frame length")
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise ValueError("A nonempty explicit coordinate list is required")
    mask = np.zeros(candidate.shape, dtype=bool)
    seen = set()
    for coordinate in coordinates:
        if (not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)):
            raise ValueError("Coordinates must contain integer bin/frame pairs")
        bin_index, frame = coordinate
        key = (bin_index, frame)
        if (not 0 <= bin_index < candidate.shape[1] or not 0 <= frame < valid_length
                or key in seen):
            raise ValueError("Coordinate out of bounds or duplicate")
        seen.add(key)
        mask[0, bin_index, frame] = True
    for component in ("real", "imag"):
        before = np.ascontiguousarray(getattr(candidate, component)).view(np.uint32)
        reference = np.ascontiguousarray(getattr(oracle, component)).view(np.uint32)
        after = np.ascontiguousarray(getattr(patched, component)).view(np.uint32)
        if not np.array_equal(before[~mask], after[~mask]):
            raise ValueError("Intervention changed a component outside the selected mask")
        if not np.array_equal(reference[mask], after[mask]):
            raise ValueError("Selected components differ from the oracle")
    return {
        "status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE",
        "shape": list(candidate.shape), "valid_length": valid_length,
        "coordinates_bin_frame": [list(coordinate) for coordinate in coordinates],
        "outside_mask_bits_unchanged": True, "selected_bits_oracle_exact": True,
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        **{name + "_complex64_le_sha256": hashlib.sha256(
            values.astype("<c8", copy=False).tobytes()).hexdigest()
           for name, values in (("candidate", candidate), ("oracle", oracle), ("patched", patched))},
    }
