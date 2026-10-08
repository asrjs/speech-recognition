"""Frozen input contracts; no model restore or original-engine acceptance here."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('nemo_inspect_pcm_test', Path(__file__).with_name('python-nemo-inspect.py'))
runner = importlib.util.module_from_spec(spec)
asr_models = ModuleType('nemo.collections.asr.models')
asr_models.ASRModel = object
with patch.dict(sys.modules, {'torch': ModuleType('torch'), 'nemo.collections.asr.models': asr_models}):
    spec.loader.exec_module(runner)


class FrozenPcmTests(unittest.TestCase):
    def test_float32_bytes_and_signed_zero_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm.f32le'
            raw = struct.pack('<4f', 0, -0.0, 0.25, -0.5)
            path.write_bytes(raw)
            pcm, contract = runner.read_frozen_pcm(path, 4, runner.hashlib.sha256(raw).hexdigest())
            self.assertEqual(pcm.dtype.name, 'float32')
            self.assertEqual(pcm.ndim, 1)
            self.assertTrue(pcm.flags.c_contiguous)
            self.assertEqual(pcm.tobytes(), raw)
            self.assertEqual(contract['samples'], 4)

    def test_invalid_frozen_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm.f32le'
            for raw, rate, digest in [(b'', 4, None), (b'bad', 4, None),
                    (struct.pack('<f', float('nan')), 1, None), (struct.pack('<f', 0), 2, None),
                    (struct.pack('<f', 0), 1, 'a' * 64), (struct.pack('<f', 0), True, None)]:
                with self.subTest(raw=raw, rate=rate, digest=digest):
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):
                        runner.read_frozen_pcm(path, rate, digest or runner.hashlib.sha256(raw).hexdigest())

    def test_actual_boundary_must_match_every_input_contract_field(self):
        contract = dict(sha256='a' * 64, sample_rate=16000, samples=32000)
        actual = dict(sha256='a' * 64, sample_rate=16000, samples=32000, valid_samples=32000,
                      right_padding_samples=0, dtype='float32', encoding='f32le', channels=1, shape=[1, 32000])
        runner.verify_frozen_model_input(actual, contract)
        for key in actual:
            changed = dict(actual)
            changed.pop(key)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                runner.verify_frozen_model_input(changed, contract)

    def test_slice_preserves_bytes_and_separate_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm.f32le'
            raw = struct.pack('<6f', 1, 0, -0.0, 0.25, -0.5, 2)
            path.write_bytes(raw)
            source = runner.hashlib.sha256(raw).hexdigest()
            selected = raw[4:20]
            digest = runner.hashlib.sha256(selected).hexdigest()
            pcm, contract = runner.read_frozen_pcm(path, 4, digest,
                start_frame=1, end_frame=5, source_sha256=source)
            self.assertEqual(pcm.tobytes(), selected)
            self.assertEqual(contract['sha256'], digest)
            self.assertEqual(contract['samples'], 4)
            self.assertEqual(contract['source_slice'], dict(sha256=source, samples=6,
                start_frame=1, end_frame=5))
            pcm[0] = 99
            self.assertEqual(path.read_bytes(), raw)

    def test_slice_rejects_missing_invalid_bounds_and_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm.f32le'
            raw = struct.pack('<6f', 1, 0, -0.0, 0.25, -0.5, 2)
            path.write_bytes(raw)
            source = runner.hashlib.sha256(raw).hexdigest()
            selected = runner.hashlib.sha256(raw[4:20]).hexdigest()
            cases = [(None, 5, source, selected), (1, None, source, selected),
                (1, 5, None, selected), (True, 5, source, selected),
                (1.0, 5, source, selected), (-1, 5, source, selected),
                (1, 7, source, selected), (5, 1, source, selected),
                (1, 1, source, selected), (1, 5, 'a' * 64, selected),
                (1, 5, source.upper(), selected), (1, 5, source, source)]
            for start, end, source_hash, window_hash in cases:
                with self.subTest(start=start, end=end, source_hash=source_hash, window_hash=window_hash):
                    with self.assertRaises(ValueError):
                        runner.read_frozen_pcm(path, 4, window_hash, start_frame=start,
                            end_frame=end, source_sha256=source_hash)
            self.assertEqual(path.read_bytes(), raw)

    def test_slice_retains_finite_and_one_second_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm.f32le'
            for values, end in [([0, 1, 2, 3, 4, 5], 3),
                    ([0, float('nan'), 2, 3, 4, 5], 5)]:
                raw = struct.pack('<6f', *values)
                path.write_bytes(raw)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    runner.read_frozen_pcm(path, 4, runner.hashlib.sha256(raw[4:end * 4]).hexdigest(),
                        start_frame=1, end_frame=end, source_sha256=runner.hashlib.sha256(raw).hexdigest())

    def test_official_transcribe_receives_owned_array_not_audio_path(self):
        import numpy as np
        pcm = np.array([0.0, -0.0, 0.25], dtype=np.float32)
        before = pcm.tobytes()
        class Model:
            def transcribe(self, audio, batch_size, return_hypotheses, verbose, timestamps):
                self.received = audio
                audio[0][0] = 1
                return ['result']
        model = Model()
        args = SimpleNamespace(batch_size=1, timestamps=False)
        self.assertEqual(runner.call_transcribe(model, Path('unused.wav'), args, pcm), ['result'])
        self.assertEqual(pcm.tobytes(), before)
        self.assertEqual(model.received[0].dtype, pcm.dtype)

    def test_existing_audio_file_route_is_unchanged(self):
        class Model:
            def transcribe(self, audio, batch_size, return_hypotheses, verbose, timestamps):
                self.audio = audio
                return ['file-result']
        model = Model()
        args = SimpleNamespace(batch_size=1, timestamps=False)
        self.assertEqual(runner.call_transcribe(model, Path('fixture.wav'), args), ['file-result'])
        self.assertEqual(model.audio, ['fixture.wav'])


if __name__ == '__main__':
    unittest.main()
