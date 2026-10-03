"""Native versus ONNX processor taps, with exact before/after output controls."""
import hashlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from tensor_parity import compare_tensors

NAMES = ('features', 'features_lens', 'spectrum', 'power', 'mel', 'log_mel', 'mean', 'std')
ORDER = ('spectrum', 'power', 'mel', 'log_mel', 'mean', 'std', 'features')


def run_stft_input_control(processor, waveform, length, oracle_signal, oracle_spectrum):
    """Isolate eval preemphasis and STFT with captured, identical inputs."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    f = processor.featurizer

    class PreemphasisBoundary(torch.nn.Module):
        def forward(self, pcm, valid):
            if f.preemph is None:
                return pcm
            mask = torch.arange(pcm.size(1), device=pcm.device)[None, :] < valid[:, None]
            signal = torch.cat((pcm[:, :1], pcm[:, 1:] - f.preemph * pcm[:, :-1]), dim=1)
            return signal.masked_fill(~mask, 0.0)

    class StftBoundary(torch.nn.Module):
        def forward(self, signal):
            return torch.stft(signal, n_fft=f.n_fft, hop_length=f.hop_length,
                win_length=f.win_length, center=True, window=f.window.to(dtype=torch.float),
                return_complex=False, pad_mode='constant')

    def execute(module, example, feeds, input_names, axes):
        buffer = io.BytesIO()
        torch.onnx.export(module.eval(), example, buffer, opset_version=17, dynamo=False,
            input_names=input_names, output_names=['output'], dynamic_axes=axes)
        graph_bytes = buffer.getvalue()
        onnx.checker.check_model(onnx.load_model_from_string(graph_bytes))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Unexpected input/STFT control execution provider')
        required = {value.name for value in session.get_inputs()}
        if set(feeds) != set(input_names) or not required.issubset(feeds):
            raise ValueError('Exported input/STFT control feed contract differs')
        # Export can prune unused declared inputs (e.g. valid with no preemphasis).
        # Only omit those; missing required or unexpected caller feeds still fail.
        return session.run(None, {name: feeds[name] for name in required})[0], {
            'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(), 'graph_bytes': len(graph_bytes),
            'providers': session.get_providers(), 'inputs': sorted(required)}

    module = PreemphasisBoundary().eval()
    with torch.no_grad():
        native_signal = module(waveform, length).numpy()
    native_metric = compare_tensors(oracle_signal, native_signal, atol=1e-5, rtol=1e-4)
    native_metric['bytes_exact'] = oracle_signal.tobytes() == native_signal.tobytes()
    if not native_metric['bytes_exact']:
        raise ValueError('Native eval preemphasis does not reproduce captured STFT input')
    actual_signal, input_graph = execute(module,
        (waveform[:, :1600], torch.tensor([1600], dtype=torch.long)),
        {'pcm': waveform.numpy(), 'valid': length.numpy()}, ['pcm', 'valid'],
        {'pcm': {1: 'samples'}, 'output': {1: 'samples'}})
    ort_metric = compare_tensors(oracle_signal, actual_signal, atol=1e-5, rtol=1e-4)
    ort_metric['bytes_exact'] = oracle_signal.tobytes() == actual_signal.tobytes()
    packed_oracle = np.stack((oracle_spectrum.real, oracle_spectrum.imag), axis=-1)
    stft = StftBoundary().eval()
    with torch.no_grad():
        native_spectrum = stft(torch.from_numpy(oracle_signal)).numpy()
    spectrum_metric = compare_tensors(packed_oracle, native_spectrum, atol=1e-5, rtol=1e-4)
    spectrum_metric['bytes_exact'] = packed_oracle.tobytes() == native_spectrum.tobytes()
    if not spectrum_metric['bytes_exact']:
        raise ValueError('Native identical-input STFT does not reproduce captured spectrum')
    actual_spectrum, stft_graph = execute(stft, (torch.from_numpy(oracle_signal[:, :1600]),),
        {'signal': oracle_signal}, ['signal'], {'signal': {1: 'samples'}, 'output': {2: 'frames'}})
    ort_spectrum_metric = compare_tensors(packed_oracle, actual_spectrum, atol=1e-5, rtol=1e-4)
    ort_spectrum_metric['bytes_exact'] = packed_oracle.tobytes() == actual_spectrum.tobytes()
    # Split the STFT boundary without changing the full processor candidate.
    class WindowedFrames(torch.nn.Module):
        def forward(self, signal):
            padded = torch.nn.functional.pad(signal, (f.n_fft // 2, f.n_fft // 2))
            starts = torch.arange(signal.size(1) // f.hop_length + 1) * f.hop_length
            indices = starts[:, None] + torch.arange(f.n_fft)[None, :]
            left = (f.n_fft - f.win_length) // 2
            window = torch.nn.functional.pad(f.window.to(dtype=torch.float),
                (left, f.n_fft - f.win_length - left))
            return padded[:, indices] * window

    framing = WindowedFrames().eval()
    with torch.no_grad():
        native_frames = framing(torch.from_numpy(oracle_signal)).numpy()
        native_fft = torch.view_as_real(torch.fft.rfft(torch.from_numpy(native_frames), dim=-1)).numpy()
    expected_frames = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(
        np.pad(oracle_signal, ((0, 0), (f.n_fft // 2, f.n_fft // 2))), f.n_fft, axis=-1)
        [:, ::f.hop_length, :] * np.pad(f.window.numpy(),
            ((f.n_fft - f.win_length) // 2, (f.n_fft - f.win_length + 1) // 2)), dtype=np.float32)
    if expected_frames.tobytes() != native_frames.tobytes():
        raise ValueError('Native windowed frames differ from independent reconstruction')
    fft_oracle = packed_oracle.transpose(0, 2, 1, 3).copy()
    if fft_oracle.tobytes() != native_fft.tobytes():
        raise ValueError('Native FFT on windowed frames does not reproduce captured STFT')
    ort_frames, frame_graph = execute(framing, (torch.from_numpy(oracle_signal[:, :1600]),),
        {'signal': oracle_signal}, ['signal'], {'signal': {1: 'samples'}, 'output': {1: 'frames'}})
    frame_metric = compare_tensors(native_frames, ort_frames, atol=1e-5, rtol=1e-4)
    frame_metric['bytes_exact'] = native_frames.tobytes() == ort_frames.tobytes()
    from onnx import helper, TensorProto
    graph = helper.make_graph([
        helper.make_node('DFT', ['frames'], ['spectrum'], axis=2, inverse=0, onesided=1),
    ], 'identical-windowed-frame-fft-control', [
        helper.make_tensor_value_info('frames', TensorProto.FLOAT, [1, 'frames', f.n_fft, 1]),
    ], [helper.make_tensor_value_info('spectrum', TensorProto.FLOAT, [1, 'frames', f.n_fft // 2 + 1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    fft_bytes = model.SerializeToString()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(fft_bytes, options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Unexpected identical-frame FFT provider')
    ort_fft = session.run(None, {'frames': native_frames[..., None]})[0]
    fft_metric = compare_tensors(fft_oracle, ort_fft, atol=1e-5, rtol=1e-4)
    fft_metric['bytes_exact'] = fft_oracle.tobytes() == ort_fft.tobytes()
    stft_vs_fft = compare_tensors(actual_spectrum.transpose(0, 2, 1, 3).copy(), ort_fft,
        atol=1e-5, rtol=1e-4)
    stft_vs_fft['bytes_exact'] = actual_spectrum.transpose(0, 2, 1, 3).copy().tobytes() == ort_fft.tobytes()
    return {'native_preemphasis': native_metric, 'onnx_preemphasis': ort_metric,
        'native_identical_input_stft': spectrum_metric, 'onnx_identical_input_stft': ort_spectrum_metric,
        'input_graph': input_graph, 'stft_graph': stft_graph,
        'window_fft_control': {'native_frames_independent_bytes_exact': True,
            'native_fft_oracle_bytes_exact': True, 'onnx_windowed_frames': frame_metric,
            'onnx_identical_frame_fft': fft_metric, 'onnx_stft_vs_identical_frame_fft': stft_vs_fft,
            'frames_sha256_f32le': hashlib.sha256(native_frames.astype('<f4').tobytes()).hexdigest(),
            'frame_graph': frame_graph, 'fft_graph_sha256': hashlib.sha256(fft_bytes).hexdigest(),
            'fft_graph_bytes': len(fft_bytes), 'providers': session.get_providers(),
            'scope': 'Separate framing/window multiply and identical-input FFT; not full feature acceptance'},
        'stft_input_sha256_f32le': hashlib.sha256(oracle_signal.astype('<f4').tobytes()).hexdigest(),
        'scope': 'Separate eval preemphasis and identical-input real-pair STFT on CPU; no full processor acceptance'}


def compare_stages(reference, candidate):
    if len(reference) != len(NAMES) or len(candidate) != len(NAMES):
        raise ValueError('Processor stage count differs')
    metrics = {}
    for name, left, right in zip(NAMES, reference, candidate):
        metrics[name] = compare_tensors(left, right, atol=1e-5, rtol=1e-4)
        metrics[name]['bytes_exact'] = left.tobytes() == right.tobytes()
    return metrics


def run_normalization_control(processor, native, *, pairwise=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from nemo.collections.asr.parts.preprocessing.features import normalize_batch

    class NormalizationBoundary(torch.nn.Module):
        def forward(self, log_mel, valid):
            features, mean, std = normalize_batch(log_mel, valid, normalize_type='per_feature')
            mask = torch.arange(features.size(-1), device=features.device).repeat(features.size(0), 1) >= valid.unsqueeze(1)
            return features.masked_fill(mask.unsqueeze(1), processor.featurizer.pad_value), mean, std

    module = NormalizationBoundary().eval()
    x, length = torch.from_numpy(native[5]), torch.from_numpy(native[1])
    with torch.no_grad():
        expected = [value.numpy() for value in module(x, length)]
    for name, left, right in zip(('features', 'mean', 'std'), (native[0], native[6], native[7]), expected):
        compare_tensors(left, right, atol=1e-5, rtol=1e-4)
        if left.tobytes() != right.tobytes():
            raise ValueError('Isolated native normalization changed ' + name)
    buffer = io.BytesIO()
    torch.onnx.export(module, (x[:, :, :11], torch.tensor([10], dtype=torch.long)), buffer,
        opset_version=17, dynamo=False, input_names=['log_mel', 'valid'],
        output_names=['features', 'mean', 'std'],
        dynamic_axes={'log_mel': {2: 'frames'}, 'features': {2: 'frames'}})
    graph_bytes = buffer.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(graph_bytes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Unexpected normalization execution provider')
    actual = session.run(None, {'log_mel': native[5], 'valid': native[1]})
    metrics = {name: compare_tensors(left, right, atol=1e-5, rtol=1e-4)
               for name, left, right in zip(('features', 'mean', 'std'), expected, actual)}
    receipt = {'status': 'PASS' if all(value['allclose'] for value in metrics.values()) else 'FAIL',
        'comparisons': metrics, 'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(),
        'graph_bytes': len(graph_bytes), 'input_log_mel_sha256_f32le': hashlib.sha256(native[5].astype('<f4').tobytes()).hexdigest(),
        'scope': 'Identical native log-mel/length input, CPU normalization only; excludes FFT/mel/log differences'}
    if pairwise:
        from processor_pairwise_normalization import PairwiseNormalization
        if (x.dtype != torch.float32 or x.ndim != 3 or not torch.isfinite(x).all()
                or length.dtype != torch.int64 or length.shape != (x.shape[0],)
                or not torch.all((length >= 1) & (length <= x.shape[-1]))):
            raise ValueError('Invalid pairwise normalization diagnostic input')
        candidate = PairwiseNormalization(processor.featurizer.pad_value).eval()
        with torch.no_grad():
            candidate_native = [value.numpy() for value in candidate(x, length)]
        buffer = io.BytesIO()
        torch.onnx.export(candidate, (x[:, :, :11], torch.tensor([10], dtype=torch.long)), buffer,
            opset_version=17, dynamo=False, input_names=['log_mel', 'valid'],
            output_names=['features', 'mean', 'std'],
            dynamic_axes={'log_mel': {2: 'frames'}, 'features': {2: 'frames'}})
        graph_bytes = buffer.getvalue()
        onnx.checker.check_model(onnx.load_model_from_string(graph_bytes))
        session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Unexpected pairwise execution provider')
        actual = session.run(None, {'log_mel': native[5], 'valid': native[1]})
        comparisons = {name: compare_tensors(left, right, atol=1e-5, rtol=1e-4)
                       for name, left, right in zip(('features', 'mean', 'std'), expected, actual)}
        receipt['pairwise_candidate'] = {
            'status': 'PASS' if all(value['allclose'] for value in comparisons.values()) else 'FAIL',
            'official_vs_onnx': comparisons,
            'official_vs_candidate_native': {name: compare_tensors(left, right, atol=1e-5, rtol=1e-4)
                for name, left, right in zip(('features', 'mean', 'std'), expected, candidate_native)},
            'candidate_native_vs_onnx': {name: compare_tensors(left, right, atol=1e-5, rtol=1e-4)
                for name, left, right in zip(('features', 'mean', 'std'), candidate_native, actual)},
            'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(), 'graph_bytes': len(graph_bytes),
            'providers': session.get_providers(),
            'scope': 'Same-input normalization diagnostic only; no full processor or WebGPU acceptance'}
    return receipt


def run_pairwise_processor_control(processor, waveform, length, native):
    """Full candidate graph against the unchanged native processor, not promotion."""
    import onnx
    import onnxruntime as ort
    import torch
    from processor_real_pair import RealPairProcessor
    from processor_pairwise_normalization import PairwiseNormalization

    adapter = RealPairProcessor(processor, include_stages=True,
        diagnostic_normalizer=PairwiseNormalization(processor.featurizer.pad_value)).eval()
    with torch.no_grad():
        candidate_native = [value.numpy() for value in adapter(waveform, length)]
    native_metrics = compare_stages(native, candidate_native)
    buffer = io.BytesIO()
    axes = {'waveforms': {1: 'samples'}}
    axes.update({name: {2: 'frames'} for name in ('features', 'spectrum', 'power', 'mel', 'log_mel')})
    torch.onnx.export(adapter, (waveform[:, :1600], torch.tensor([1600], dtype=torch.long)),
        buffer, opset_version=17, dynamo=False, input_names=['waveforms', 'waveforms_lens'],
        output_names=list(NAMES), dynamic_axes=axes)
    graph_bytes = buffer.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(graph_bytes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Unexpected full pairwise processor execution provider')
    actual = session.run(list(NAMES), {'waveforms': waveform.numpy(), 'waveforms_lens': length.numpy()})
    metrics = compare_stages(native, actual)
    passed = all(value['allclose'] for value in metrics.values())
    return {'status': 'PASS' if passed else 'FAIL',
        'official_native_vs_candidate_native': native_metrics,
        'official_native_vs_onnx': metrics,
        'candidate_native_vs_onnx': compare_stages(candidate_native, actual),
        'first_tolerance_failure': next((name for name in ORDER if not metrics[name]['allclose']), None),
        'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(), 'graph_bytes': len(graph_bytes),
        'providers': session.get_providers(),
        'scope': 'Full fp32 processor on captured PCM, CPU only; excludes encoder/tokens/text/WASM/WebGPU'}


def run_stage_control(processor, waveform, length, oracle_spectrum, expected_features,
                      baseline_features, baseline_length, *, pairwise=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from processor_real_pair import RealPairProcessor
    adapter = RealPairProcessor(processor, include_stages=True).eval()
    with torch.no_grad():
        native = [value.numpy() for value in adapter(waveform, length)]
    native_final = compare_tensors(expected_features, native[0], atol=1e-5, rtol=1e-4)
    if not native_final['allclose'] or expected_features.tobytes() != native[0].tobytes():
        raise ValueError('Native stage taps changed captured feature output')
    packed_oracle = np.stack((oracle_spectrum.real, oracle_spectrum.imag), axis=-1)
    spectrum_control = compare_tensors(packed_oracle, native[2], atol=1e-5, rtol=1e-4)
    spectrum_control['bytes_exact'] = packed_oracle.tobytes() == native[2].tobytes()
    if not spectrum_control['allclose']:
        raise ValueError('Native tapped spectrum differs from official capture')
    buffer = io.BytesIO()
    axes = {'waveforms': {1: 'samples'}}
    axes.update({name: {2: 'frames'} for name in ('features', 'spectrum', 'power', 'mel', 'log_mel')})
    torch.onnx.export(adapter, (waveform[:, :1600], torch.tensor([1600], dtype=torch.long)),
        buffer, opset_version=17, dynamo=False, input_names=['waveforms', 'waveforms_lens'],
        output_names=list(NAMES), dynamic_axes=axes)
    graph_bytes = buffer.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(graph_bytes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Unexpected processor tap provider')
    candidate = session.run(list(NAMES), {'waveforms': waveform.numpy(), 'waveforms_lens': length.numpy()})
    output_control = compare_stages(native, candidate)
    unchanged_features = compare_tensors(baseline_features, candidate[0], atol=1e-5, rtol=1e-4)
    unchanged_lengths = compare_tensors(baseline_length, candidate[1], atol=0, rtol=0)
    unchanged = (baseline_features.tobytes() == candidate[0].tobytes()
                 and baseline_length.tobytes() == candidate[1].tobytes())
    receipt = {'status': 'DIAGNOSTIC_TAPS_UNCHANGED' if unchanged else 'DIAGNOSTIC_TAPS_CHANGED_OUTPUT',
        'native_spectrum_oracle': spectrum_control,
        'stage_comparisons': output_control,
        'first_tolerance_failure': next((name for name in ORDER if not output_control[name]['allclose']), None),
        'first_nonidentical_stage': next((name for name in ORDER if not output_control[name]['bytes_exact']), None),
        'baseline_feature_comparison': unchanged_features, 'baseline_length_comparison': unchanged_lengths,
        'final_output_bytes_unchanged': unchanged,
        'identical_input_normalization': run_normalization_control(processor, native, pairwise=pairwise),
        'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(), 'graph_bytes': len(graph_bytes),
        'scope': 'Native adapter versus tapped ONNX CPU, not isolated identical-input operator or browser acceptance'}
    if pairwise:
        receipt['full_pairwise_processor'] = run_pairwise_processor_control(processor, waveform, length, native)
    return receipt
