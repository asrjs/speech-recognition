"""Dependency-light Slaney bank for diagnostics, not a production processor.

Matches the float32 assignment/area-scaling order of librosa.filters.mel.
Callers should bind its bytes to their own reference bank identity.
"""
import hashlib
from numbers import Real
import numpy as np


def slaney_bank(*, sample_rate, n_fft, n_mels, fmin=0.0, fmax=None,
                expected_sha256=None):
    for name, value in [('sample_rate', sample_rate), ('n_fft', n_fft), ('n_mels', n_mels)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    fmax = sample_rate / 2 if fmax is None else fmax
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in (fmin, fmax)):
        raise ValueError('Frequency bounds must be real non-boolean scalars')
    if not np.isfinite([fmin, fmax]).all() or not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError('Invalid frequency bounds')
    if expected_sha256 is not None and (not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Expected bank identity must be lowercase SHA-256')
    slope = 200.0 / 3
    logstep = np.log(6.4) / 27

    def hz_to_mel(value):
        return value / slope if value < 1000 else 15 + np.log(value / 1000) / logstep

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = slope * mels
    mask = mels >= 15
    freqs[mask] = 1000 * np.exp(logstep * (mels[mask] - 15))
    bins = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    ramps = np.subtract.outer(freqs, bins)
    widths = np.diff(freqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / widths[i],
                                             ramps[i + 2] / widths[i + 1]))
    weights *= (2.0 / (freqs[2:] - freqs[:-2]))[:, None]
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite bank')
    actual = hashlib.sha256(weights.astype('<f4').tobytes()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f'Filterbank identity differs: {actual}')
    return weights
