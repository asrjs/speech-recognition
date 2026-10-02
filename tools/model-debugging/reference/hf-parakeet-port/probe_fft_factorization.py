"""One frozen FFT512 frame: complex256 versus two complex128 transforms.

Diagnostic only. No weight loading, production changes or accepted feature port.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from replay_packed_fft import ROOT, digest, read_capture, recombine_packed
from artifact_outputs import require_new_outputs, write_new_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--expected-capture-sha256', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--bin', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__), Path(__file__).with_name('replay_packed_fft.py'),
               ROOT / 'tools/artifact_outputs.py']
    require_new_outputs([args.output], protected_inputs=[args.capture, *sources],
                        protected_directories=[ROOT / 'src', ROOT / 'models'])
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    _, _, _, oracle, frames, _, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    if not 0 <= args.frame < valid or not 1 <= args.bin < 128:
        raise ValueError('Expected valid frame and non-cardinal positive paired bin')
    import torch
    torch.set_num_threads(4)
    frame = frames[:, args.frame:args.frame + 1, :].copy()
    reference = oracle[:, :, args.frame:args.frame + 1]
    actual = torch.fft.rfft(torch.from_numpy(frame), dim=-1).numpy().transpose(0, 2, 1)
    if actual.tobytes() != reference.tobytes():
        raise ValueError('Selected real512 frame does not reproduce capture bytes')
    packed = np.empty((1, 1, 256), np.complex64)
    packed.real, packed.imag = frame[..., ::2], frame[..., 1::2]
    parent = torch.fft.fft(torch.from_numpy(packed), dim=-1).numpy()
    even = torch.fft.fft(torch.from_numpy(packed[..., ::2].copy()), dim=-1).numpy()
    odd = torch.fft.fft(torch.from_numpy(packed[..., 1::2].copy()), dim=-1).numpy()
    k = np.arange(128)
    cosine = np.cos(-2 * np.pi * k / 256).astype(np.float32)
    sine = np.sin(-2 * np.pi * k / 256).astype(np.float32)
    rotated = np.empty_like(odd)
    # Rounded f32 twiddles; single final f32 rounding of each complex product.
    rotated.real = odd.real.astype(np.float64) * cosine - odd.imag.astype(np.float64) * sine
    rotated.imag = odd.real.astype(np.float64) * sine + odd.imag.astype(np.float64) * cosine
    split = np.concatenate([even + rotated, even - rotated], axis=-1)
    if not all(np.isfinite(value).all() for value in (parent, split)):
        raise ValueError('Nonfinite factorization output')

    def pair(value):
        return [float(value.real), float(value.imag)]

    results = {}
    for name, value in [('complex256', parent), ('split128', split)]:
        recombined = recombine_packed(value)[0, args.bin, 0]
        results[name] = {
            'packed_bins': {str(index): pair(value[0, 0, index])
                            for index in (args.bin, 256 - args.bin)},
            'recombined': pair(recombined),
            'absolute_component_errors': pair(complex(
                abs(recombined.real - reference[0, args.bin, 0].real),
                abs(recombined.imag - reference[0, args.bin, 0].imag))),
        }
    if hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise ValueError('Sources changed during diagnostic')
    if digest(args.capture.read_bytes()) != args.expected_capture_sha256:
        raise ValueError('Capture changed during diagnostic')
    receipt = {
        'status': 'DIAGNOSTIC_ONLY', 'sample_id': args.sample_id,
        'capture_sha256': args.expected_capture_sha256, 'sources_sha256': hashes,
        'frame': args.frame, 'bin': args.bin,
        'frame_f32le_sha256': digest(frame.astype('<f4').tobytes()),
        'selected_real512_capture_bits_exact': True,
        'oracle_bin': pair(reference[0, args.bin, 0]),
        'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
        'results': results,
        'scope': 'One frozen frame and fixed split128 factorization; no full-feature, portable FFT, ASR or browser acceptance',
    }
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
