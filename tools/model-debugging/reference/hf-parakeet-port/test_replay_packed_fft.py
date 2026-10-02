"""Weight-free contracts/sanity tests, not original-model parity evidence."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import replay_packed_fft as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        arrays = {"waveform": np.zeros((1, 160), np.float32),
                  "feature_stft/input": np.zeros((1, 160), np.float32),
                  "feature_stft/window": np.ones(400, np.float32),
                  "feature_stft/real": np.zeros((1, 257, 2), np.float32),
                  "feature_stft/imag": np.zeros((1, 257, 2), np.float32),
                  "features": np.zeros((1, 128, 2), np.float32),
                  "features_length": np.array([1], np.int64)}
        self.document = {"meta": {"capture_feature_stft": True}, "instrumentation_control": {
            name: True for name in ("baseline_tokens_match", "baseline_text_match", "baseline_frames_match",
                                    "after_tokens_match", "after_text_match", "after_durations_match")},
            "samples": [{"sample_id": "synthetic", "feature_stft_contract": {
                "n_fft": 512, "hop_length": 160, "win_length": 400, "exact_pad": False},
                "model_input": {"sample_rate": 16000, "channels": 1, "dtype": "float32",
                    "encoding": "f32le", "shape": [1, 160], "samples": 160, "valid_samples": 160,
                    "right_padding_samples": 0, "sha256": replay.digest(arrays["waveform"].tobytes())},
                "stages": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "data": value.ravel().tolist()} for name, value in arrays.items()}}]}

    def read(self, document=None, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            encoded = json.dumps(document or self.document).encode()
            replay.write_new_text(path, encoded.decode())
            return replay.read_capture(path, expected or replay.digest(encoded), "synthetic")

    def test_synthetic_capture_preserves_valid_physical_distinction(self):
        _, _, _, spectrum, frames, features, valid = self.read()
        self.assertEqual((spectrum.shape, frames.shape, features.shape, valid),
                         ((1, 257, 2), (1, 2, 512), (1, 128, 2), 1))

    def test_bad_hash_and_missing_instrumentation_fail(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.read(expected="0" * 64)
        for value in (False, 1, "true", None):
            document = copy.deepcopy(self.document)
            document["instrumentation_control"]["after_text_match"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(document)

    def test_duplicate_sample_wrong_pcm_hash_contract_and_padding_fail(self):
        for mode in ("duplicate", "pcm", "contract", "padding", "length"):
            document = copy.deepcopy(self.document)
            sample = document["samples"][0]
            if mode == "duplicate": document["samples"].append(copy.deepcopy(sample))
            if mode == "pcm": sample["model_input"]["sha256"] = "0" * 64
            if mode == "contract": sample["feature_stft_contract"]["exact_pad"] = 0
            if mode == "padding": sample["stages"]["features"]["data"][1] = 1
            if mode == "length": sample["stages"]["features_length"]["data"] = [3]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.read(document)

    def test_malformed_tensor_data_fail_before_rounding(self):
        for data in ([True], ["1"], [float("nan")], [1.00000001], [1.5]):
            sample = {"stages": {"x": {"shape": [1], "dtype": "float32", "data": data}}}
            if data == [1.5]: sample["stages"]["x"]["dtype"] = "int64"
            with self.subTest(data=data), self.assertRaises(ValueError):
                replay.read_stage(sample, "x", sample["stages"]["x"]["dtype"])

    def test_recombination_matches_independent_fft_sanity_without_mutation(self):
        frames = np.random.default_rng(17).normal(size=(1, 2, 512)).astype(np.float32)
        packed = np.empty((1, 2, 256), np.complex64)
        packed.real, packed.imag = frames[..., ::2], frames[..., 1::2]
        fft = np.fft.fft(packed, axis=-1).astype(np.complex64)
        before = fft.tobytes()
        actual = replay.recombine_packed(fft)
        reference = np.fft.rfft(frames, axis=-1).transpose(0, 2, 1)
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
        self.assertEqual(fft.tobytes(), before)

    def test_recombination_rejects_invalid_contracts(self):
        for values in (np.zeros((1, 1, 255), np.complex64), np.zeros((1, 1, 256), np.complex128),
                       np.zeros((1, 0, 256), np.complex64),
                       np.full((1, 1, 256), complex(np.inf, 0), np.complex64)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                replay.recombine_packed(values)

    def test_probe_is_deterministic_and_parses_without_running_inference(self):
        source = replay.build_probe()
        self.assertEqual(source, replay.build_probe())
        process = subprocess.run(["node", "--input-type=module", "--check"],
                                 input=source, text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_probe_composition_rejects_missing_or_repeated_marker(self):
        for source in ("", replay.NODE_PROBE + "const experiments = {"):
            with patch.object(replay, "NODE_PROBE", source), self.assertRaises(ValueError):
                replay.build_probe()

    def test_existing_output_fails_before_capture_or_heavy_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            replay.write_new_text(output, "preserve")
            arguments = ["replay", "--capture", str(Path(directory) / "missing.json"),
                         "--expected-capture-sha256", "0" * 64, "--sample-id", "synthetic",
                         "--output", str(output), "--probe-output", str(Path(directory) / "probe.mjs")]
            with patch.object(replay.sys, "argv", arguments), patch.object(replay, "read_capture") as read:
                with self.assertRaises(FileExistsError):
                    replay.main()
                read.assert_not_called()
            self.assertEqual(output.read_text(), "preserve")
            self.assertFalse((Path(directory) / "probe.mjs").exists())

    def test_probe_output_requires_complete_exact_result_schema(self):
        reference = np.zeros((1, 128, 2), np.float32)
        result = {"features": reference.ravel().tolist(), "mean": [0.0] * 128, "std": [1.0] * 128}
        js = {"physicalFrames": 2, "validLength": 1,
              "results": {name: copy.deepcopy(result) for name in ("control", "candidate")}}
        comparisons = replay.validate_probe_output(js, {"control", "candidate"}, reference, 1, None)
        self.assertEqual(set(comparisons), {"control", "candidate"})
        for mode in ("missing", "extra", "float64", "bool", "short", "mean", "std", "length"):
            broken = copy.deepcopy(js)
            row = broken["results"]["candidate"]
            if mode == "missing": del broken["results"]["candidate"]
            if mode == "extra": broken["results"]["unexpected"] = result
            if mode == "float64": row["features"][0] = 1.00000001
            if mode == "bool": row["features"][0] = True
            if mode == "short": row["features"].pop()
            if mode == "mean": row["mean"] = []
            if mode == "std": row["std"][0] = 0
            if mode == "length": broken["validLength"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                replay.validate_probe_output(broken, {"control", "candidate"}, reference, 1, None)

    def test_probe_point_requires_finite_complete_consistent_fields(self):
        reference = np.zeros((1, 128, 2), np.float32)
        row = {"features": reference.ravel().tolist(), "mean": [0.0]*128, "std": [1.0]*128,
               "point": {"raw": 0.0, "linearMel": 0.0, "logInput": 1.0,
                         "normalized": 0.0, "mean": 0.0, "std": 1.0}}
        js = {"physicalFrames": 2, "validLength": 1, "results": {"candidate": row}}
        replay.validate_probe_output(js, {"candidate"}, reference, 1, [0, 0])
        for mode in ("missing", "nonfinite", "bool", "inconsistent"):
            broken = copy.deepcopy(js)
            point = broken["results"]["candidate"]["point"]
            if mode == "missing": del point["raw"]
            if mode == "nonfinite": point["raw"] = float("inf")
            if mode == "bool": point["std"] = True
            if mode == "inconsistent": point["normalized"] = 1.0
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                replay.validate_probe_output(broken, {"candidate"}, reference, 1, [0, 0])

    def test_probe_writer_preserves_execution_bytes_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.mjs"
            source = "const a = 1;\n// UTF-8: ç\n"
            replay.write_probe(path, source)
            self.assertEqual(path.read_bytes(), source.encode("utf-8"))
            with self.assertRaises(FileExistsError):
                replay.write_probe(path, "changed")
            self.assertEqual(path.read_bytes(), source.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
