"""Fail-closed numerical tensor metrics for tooling, independent of model family."""

import math

import numpy as np


def compare_tensors(reference, candidate, *, atol: float, rtol: float) -> dict:
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("Finite non-negative atol/rtol required")
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(f"Tensor shape/dtype mismatch: {reference.shape}/{reference.dtype} vs {candidate.shape}/{candidate.dtype}")
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Empty/nonfinite tensor cannot prove parity")
    left, right = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    # Integer outputs (lengths/decisions) must be exactly equal.
    if reference.dtype.kind in "iub":
        close = (reference == candidate).ravel()
        error = np.asarray(np.abs(reference.astype(object).ravel() - candidate.astype(object).ravel()), dtype=np.float64)
    else:
        error = np.abs(left - right)
        close = np.isclose(right, left, atol=atol, rtol=rtol)
    divergent = np.flatnonzero(~close)
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / norms) if norms else (1.0 if np.array_equal(left, right) else None)
    return {
        "shape": list(reference.shape), "dtype": str(reference.dtype), "count": int(reference.size),
        "atol": atol, "rtol": rtol, "allclose": bool(close.all()),
        "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
        "max_relative_error": float((error / np.maximum(np.abs(left), np.finfo(np.float64).eps)).max()),
        "cosine_similarity": cosine, "mismatch_count": int(divergent.size),
        "first_divergent_index": int(divergent[0]) if divergent.size else None,
    }
