"""One frozen FFT512 frame: packed complex FFTs versus real even/odd halves.

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
    complex_frame = np.empty(frame.shape, np.complex64)
    complex_frame.real, complex_frame.imag = frame, 0
    direct = torch.fft.fft(torch.from_numpy(complex_frame), dim=-1).numpy()
    direct = direct[..., :257].transpose(0, 2, 1).copy()
    if direct.dtype != np.complex64 or direct.shape != reference.shape or not np.isfinite(direct).all():
        raise ValueError('Direct complex512 output contract differs')
    component_error = np.abs(direct.view(np.float32) - reference.copy().view(np.float32))
    results['direct_complex512'] = {
        'bin': pair(direct[0, args.bin, 0]),
        'capture_bytes_exact': direct.tobytes() == reference.tobytes(),
        'component_unequal_count': int(np.count_nonzero(component_error)),
        'max_absolute_component_error': float(component_error.max()),
        'mean_absolute_component_error': float(component_error.mean()),
    }
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
    half_even = torch.fft.rfft(torch.from_numpy(frame[..., ::2].copy()), dim=-1).numpy()[0, 0]
    half_odd = torch.fft.rfft(torch.from_numpy(frame[..., 1::2].copy()), dim=-1).numpy()[0, 0]
    half_results = {}
    f32 = np.float32
    for index in (args.bin, 256 - args.bin):
        z = parent[0, 0]
        er = f32(f32(z[index].real + z[256-index].real) * f32(.5))
        ei = f32(f32(z[index].imag - z[256-index].imag) * f32(.5))
        oddr = f32(f32(z[index].imag + z[256-index].imag) * f32(.5))
        oddi = f32(f32(z[index].real - z[256-index].real) * f32(-.5))
        even_value = half_even[index] if index <= 128 else np.conj(half_even[256-index])
        odd_value = half_odd[index] if index <= 128 else np.conj(half_odd[256-index])

        def combine(real, imag, odd_real, odd_imag):
            cosine = f32(np.cos(-2 * np.pi * index / 512))
            sine = f32(np.sin(-2 * np.pi * index / 512))
            rotated_real = f32(float(odd_real) * float(cosine) - float(odd_imag) * float(sine))
            rotated_imag = f32(float(odd_real) * float(sine) + float(odd_imag) * float(cosine))
            return complex(f32(real + rotated_real), f32(imag + rotated_imag))

        packed_value = combine(er, ei, oddr, oddi)
        direct_value = combine(even_value.real, even_value.imag, odd_value.real, odd_value.imag)
        expected_packed = recombine_packed(parent)[0, index, 0]
        if pair(packed_value) != pair(expected_packed):
            raise ValueError('Component recombination differs from existing packed recipe')
        target = reference[0, index, 0]
        half_results[str(index)] = {
            'packed_even': pair(complex(er, ei)),
            'packed_odd': pair(complex(oddr, oddi)),
            'real_half_even': pair(even_value), 'real_half_odd': pair(odd_value),
            'oracle': pair(target), 'packed_recombined': pair(packed_value),
            'half_recombined': pair(direct_value),
            'packed_absolute_component_errors': [abs(packed_value.real - float(target.real)),
                                                  abs(packed_value.imag - float(target.imag))],
            'half_absolute_component_errors': [abs(direct_value.real - float(target.real)),
                                                abs(direct_value.imag - float(target.imag))],
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
        'real_half_control': half_results,
        'scope': 'One frozen frame, direct complex512, split128 and real even/odd FFT256 controls; no full-feature, portable FFT, ASR or browser acceptance',
    }
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
