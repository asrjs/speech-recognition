"""Small, weight-free checks for the diagnostic's actual JS boundary injection."""

import hashlib
import base64
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import gzip
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("compare_js_feature_boundaries.py")
spec = importlib.util.spec_from_file_location("feature_boundary_probe", MODULE_PATH)
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


def inputs():
    # Synthetic PCM and boundaries test the probe, not NeMo model parity.
    import numpy as np
    real = np.asarray([((index * 17) % 83 - 41) / 32 for index in range(257 * 11)], dtype="<f4")
    imag = np.asarray([((index * 13) % 71 - 35) / 64 for index in range(257 * 11)], dtype="<f4")
    return {"pcm": [((index * 37) % 101 - 50) / 100 for index in range(1600)],
        "bins": 8, "referenceRaw": [0] * (8 * 11), "referenceWindow": [1] * 400,
        "referenceFilterbank": [0] * (8 * 257), "referencePower": [0] * (257 * 11),
        "referenceComplex": {"shape": [1, 257, 11], "dtype": "complex64",
            "layout": "batch-bin-frame", "encoding": "base64-f32le-components",
            "realF32LE": base64.b64encode(real.tobytes()).decode("ascii"),
            "imagF32LE": base64.b64encode(imag.tobytes()).decode("ascii")}}


class FeatureBoundaryProbeTests(unittest.TestCase):
    def test_official_log_input_control_preserves_inputs_and_uses_js_log(self):
        import numpy as np
        data = inputs()
        data['referenceLogInput'] = [1, 2, 2 ** -24, 0.5] * 22
        completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
            cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
            capture_output=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        control = result['officialLogInputControl']
        self.assertEqual(control['input'], data['referenceLogInput'])
        expected_raw = np.asarray([math.log(value) for value in data['referenceLogInput']], dtype=np.float32)
        np.testing.assert_array_equal(control['raw'], expected_raw)
        from normalization_reduction import normalize_candidate
        expected = normalize_candidate(expected_raw.reshape(8, 11), 10, 8)
        for key, values in zip(('features', 'mean', 'std'), expected):
            self.assertEqual(np.asarray(control[key], dtype='<f4').tobytes(), values.astype('<f4').tobytes())
        self.assertEqual(result['productionCapture']['features'], self.result['productionCapture']['features'])

    def test_official_log_input_control_rejects_malformed_inputs(self):
        for invalid in ('short', None, True, 0, -1, 0.1):
            with self.subTest(invalid=invalid):
                data = inputs()
                data['referenceLogInput'] = [1] * 88
                if invalid == 'short':
                    data['referenceLogInput'].pop()
                else:
                    data['referenceLogInput'][0] = invalid
                completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
                    cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
                    capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn('Official log input', completed.stderr)

    def test_reference_power_capture_records_actual_mel_and_log_boundaries(self):
        import numpy as np
        data = inputs()
        data['referenceFilterbank'][1:3] = [0.25, 0.5]
        data['referencePower'][11:22] = [3] * 11
        data['referencePower'][22:33] = [7] * 11
        completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
            cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
            capture_output=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        capture = result['referencePowerCapture']
        self.assertEqual(capture['power'], data['referencePower'])
        self.assertEqual(capture['features'], result['experiments']['reference_power'])
        for index in range(88):
            linear = 4.25 if index < 11 else 0
            guarded = linear + result['logZeroGuard']
            self.assertEqual(capture['linearMel'][index], linear)
            self.assertEqual(capture['logInput'][index], guarded)
            self.assertEqual(capture['raw'][index], float(np.float32(math.log(guarded))))
            self.assertEqual(capture['f32LogInputRaw'][index],
                float(np.float32(math.log(float(np.float32(guarded))))))
        # Guard addition is deliberately double in the observed JS path;
        # the separate f32-input control must not round the original capture.
        self.assertNotEqual(capture['logInput'][0], float(np.float32(capture['logInput'][0])))

    def test_reference_power_capture_rejects_invalid_spectrum(self):
        for invalid in ('short', 'negative', 'null', 'non-f32'):
            with self.subTest(invalid=invalid):
                data = inputs()
                if invalid == 'short':
                    data['referencePower'].pop()
                else:
                    data['referencePower'][0] = {'negative': -1, 'null': None, 'non-f32': 0.1}[invalid]
                completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
                    cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
                    capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn('Reference power must be', completed.stderr)

    def test_selected_power_override_is_local_and_recomputes_normalization(self):
        import numpy as np
        data = inputs()
        data['referenceFilterbank'][1] = 1
        data['referencePower'] = [9] * (257 * 11)
        data['powerOverrides'] = [[3, 1]]
        completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
            cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
            capture_output=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        baseline, candidate = result['complexPowerCapture'], result['powerOverrideCapture']
        self.assertEqual(candidate['power'], baseline['power'])
        self.assertEqual(candidate['frameBytesF32LE'], baseline['frameBytesF32LE'])
        changed = [index for index, (before, after) in enumerate(zip(
            baseline['power'], candidate['effectivePower'])) if before != after]
        self.assertEqual(changed, [1 * 11 + 3])
        changed_raw = [index for index, (before, after) in enumerate(zip(
            baseline['raw'], candidate['raw'])) if before != after]
        self.assertEqual(changed_raw, [3])
        self.assertEqual(candidate['raw'][3], float(np.float32(math.log(9 + result['logZeroGuard']))))
        report = probe_module.selected_power_intervention(result,
            np.asarray(data['referencePower'], dtype=np.float32).reshape(1, 257, 11),
            np.zeros((1, 8, 11), dtype=np.float32), np.zeros((1, 8, 11), dtype=np.float32), 10)
        self.assertTrue(report['unselected_power_bits_unchanged'])
        self.assertEqual(report['changed_raw_points'][0]['coordinate'], [0, 0, 3])
        self.assertNotEqual(report['mean_before'], report['mean_after'])
        candidate['effectivePower'][4] += 1
        with self.assertRaisesRegex(ValueError, 'outside the exact selected mask'):
            probe_module.selected_power_intervention(result,
                np.asarray(data['referencePower'], dtype=np.float32).reshape(1, 257, 11),
                np.zeros((1, 8, 11), dtype=np.float32), np.zeros((1, 8, 11), dtype=np.float32), 10)

    def test_selected_power_override_rejects_invalid_coordinates(self):
        for coordinates in ([[3, 1], [3, 1]], [[10, 1]], [[3, 257]], [[True, 1]], [[3, -1]]):
            with self.subTest(coordinates=coordinates):
                data = inputs()
                data['powerOverrides'] = coordinates
                completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
                    cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
                    capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn('Selected power coordinate', completed.stderr)

    def test_selected_power_override_rejects_malformed_reference(self):
        for invalid in ('short', 'negative', 'null'):
            with self.subTest(invalid=invalid):
                data = inputs()
                data['powerOverrides'] = [[3, 1]]
                if invalid == 'short':
                    data['referencePower'].pop()
                else:
                    data['referencePower'][0] = -1 if invalid == 'negative' else None
                completed = subprocess.run(['node', '--input-type=module', '-e', probe_module.NODE_PROBE],
                    cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding='utf-8',
                    capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn('Selected power reference', completed.stderr)

    def test_packed_f32_control_preserves_aligned_input_bytes_and_f32_components(self):
        import numpy as np
        capture = self.result['packedF32Capture']
        self.assertEqual(capture['frameBytesF32LE'], self.result['complexPowerCapture']['frameBytesF32LE'])
        for component in ('real', 'imag'):
            values = np.asarray(capture[component], dtype=np.float64)
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_array_equal(values, values.astype(np.float32).astype(np.float64))

    def test_named_raw_normalization_controls_match_python_candidate(self):
        import numpy as np
        from normalization_reduction import normalize_candidate
        sources = {'officialRaw': inputs()['referenceRaw'],
            'productionRaw': self.result['productionCapture']['raw'],
            'alignedWindowProducts': self.result['frameArithmeticCaptures']['windowProduct']['raw'],
            'alignedComplexPower': self.result['complexPowerCapture']['raw'],
            'packedF32': self.result['packedF32Capture']['raw'],
            'magnitudeRoundTrip': self.result['powerCapture']['raw'],
            'fullFftF32': self.result['fullFftF32Capture']['raw'],
            'fullFftF64': self.result['fullFftF64Capture']['raw'],
            'officialComplexLibrary': self.result['referenceComplexLibraryCapture']['raw'],
            'officialComplexRounded': self.result['referenceComplexRoundedCapture']['raw']}
        self.assertEqual(set(sources), set(self.result['normalizationControls']))
        for name, values in sources.items():
            with self.subTest(boundary=name):
                features, mean, std = normalize_candidate(np.asarray(values, dtype=np.float32).reshape(8, 11), 10, 8)
                observed = self.result['normalizationControls'][name]
                for key, expected in (('features', features.ravel()), ('mean', mean), ('std', std)):
                    self.assertEqual(np.asarray(observed[key], dtype='<f4').tobytes(), expected.astype('<f4').tobytes())

    @classmethod
    def setUpClass(cls):
        processor = probe_module.ROOT / "dist/audio/js-mel.js"
        cls.before_sha = hashlib.sha256(processor.read_bytes()).hexdigest()
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", probe_module.NODE_PROBE +
                "\nconsole.error(JSON.stringify({ features: Array.from(processor.process(pcm).features), "
                "raw: Array.from(processor.computeRawMel(pcm).rawMel) }));"],
            cwd=probe_module.ROOT, input=json.dumps(inputs()), text=True, encoding="utf-8",
            capture_output=True, timeout=30, check=True,
        )
        cls.result = json.loads(completed.stdout)
        after = json.loads(completed.stderr)
        cls.after_features, cls.unmasked_raw = after["features"], after["raw"]
        cls.after_sha = hashlib.sha256(processor.read_bytes()).hexdigest()

    def test_experiments_keep_full_feature_shape_and_masked_tail(self):
        self.assertEqual((self.result["frameCount"], self.result["validLength"]), (11, 10))
        self.assertEqual(len(self.result["experiments"]), 6)
        for values in self.result["experiments"].values():
            self.assertEqual(len(values), 88)
            self.assertTrue(all(values[bin_index * 11 + 10] == 0 for bin_index in range(8)))
        self.assertEqual(self.result["experiments"]["reference_power"], [0] * 88)

    def test_experiments_do_not_mutate_compiled_source_or_baseline_caches(self):
        self.assertEqual(self.before_sha, self.after_sha)
        self.assertEqual(self.result["normalized"], self.after_features)

    def test_power_capture_preserves_full_bin_major_layout(self):
        capture = self.result["powerCapture"]
        self.assertEqual(capture["shape"], [1, 257, 11])
        self.assertEqual(len(capture["power"]), 257 * 11)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in capture["power"]))
        self.assertGreater(max(capture["power"]), 0)
        self.assertEqual(capture["features"], self.result["experiments"]["magnitude_round_trip"])

    def test_power_capture_keeps_distinct_frame_columns(self):
        # DC bins have independently calculated sums of the exact framed input.
        import numpy as np
        pcm = np.asarray(inputs()["pcm"], dtype=np.float32)
        preemphasis = np.empty_like(pcm)
        preemphasis[0] = pcm[0]
        preemphasis[1:] = pcm[1:] - np.float32(0.97) * pcm[:-1]
        padded = np.pad(preemphasis, (256, 256))
        dc = []
        for frame in range(11):
            windowed = padded[frame * 160 + 56:frame * 160 + 456]
            squared = np.float32(float(windowed.sum(dtype=np.float64)) ** 2)
            magnitude = np.float32(math.sqrt(float(squared)))
            dc.append(np.float32(magnitude * magnitude))
        np.testing.assert_allclose(self.result["powerCapture"]["power"][:11], dc,
            atol=1e-6, rtol=1e-6)

    def test_complex_rounding_experiment_keeps_shape_and_finite_power(self):
        capture = self.result["complexPowerCapture"]
        self.assertEqual(capture["shape"], self.result["powerCapture"]["shape"])
        self.assertEqual(len(capture["features"]), 88)
        self.assertEqual(len(capture["power"]), 257 * 11)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in capture["power"]))

    def test_packed_complex_capture_matches_full_fft_on_identical_frame_bytes(self):
        import numpy as np
        packed = self.result["complexPowerCapture"]
        full = self.result["fullFftF64Capture"]
        self.assertEqual(packed["frameBytesF32LE"], full["frameBytesF32LE"])
        for component in ("real", "imag"):
            np.testing.assert_allclose(packed[component], full[component], atol=1e-12, rtol=1e-12)
        # Upper paired bins and N/4 need the conjugate imaginary sign;
        # equal powers alone would conceal a sign error in instrumentation.
        self.assertGreater(max(abs(value) for value in packed["imag"][129 * 11:256 * 11]), 0)

    def test_packed_complex_capture_reconstructs_unchanged_rounded_power(self):
        import numpy as np
        capture = self.result["complexPowerCapture"]
        real = np.asarray(capture["real"], dtype=np.float32)
        imag = np.asarray(capture["imag"], dtype=np.float32)
        squared = np.float32(np.float32(real * real) + np.float32(imag * imag))
        magnitude = np.asarray([np.float32(math.sqrt(float(value))) for value in squared])
        np.testing.assert_array_equal(capture["power"], np.float32(magnitude * magnitude))

    def test_production_capture_preserves_actual_raw_and_normalized_features(self):
        capture = self.result["productionCapture"]
        self.assertEqual(capture["features"], self.result["normalized"])
        self.assertEqual(capture["raw"], self.unmasked_raw)
        self.assertEqual(capture["shape"], [1, 257, 11])

    def test_production_capture_preserves_double_frames_and_original_power(self):
        import numpy as np
        capture = self.result["productionCapture"]
        frames = probe_module.read_production_frames(capture, 512, 11)
        self.assertEqual(frames.dtype, np.dtype("<f8"))
        np.testing.assert_array_equal(frames.ravel(), capture["frames"])
        self.assertTrue(np.any(frames != frames.astype(np.float32).astype(np.float64)))
        real = np.asarray(capture["real"], dtype=np.float64)
        imag = np.asarray(capture["imag"], dtype=np.float64)
        np.testing.assert_array_equal(capture["power"], np.float32(real * real + imag * imag))
        spectrum = np.fft.rfft(frames, axis=-1).transpose(0, 2, 1)
        np.testing.assert_allclose(real.reshape(1, 257, 11), spectrum.real, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(imag.reshape(1, 257, 11), spectrum.imag, atol=1e-12, rtol=1e-12)

    def test_production_frame_reader_rejects_float32_and_corrupt_bytes(self):
        capture = self.result["productionCapture"]
        for field, value in (("frameDtype", "float32"), ("frameShape", [1, 512, 11]),
                ("frameShape", [True, 11, 512]), ("frameShape", [1.0, 11, 512]),
                ("frameLayout", "batch-bin-frame"), ("frameBytesEncoding", "base64-f32le"),
                ("frameBytesF64LE", "!")):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    probe_module.read_production_frames({**capture, field: value}, 512, 11)

    def test_production_diagnostic_separates_changed_inputs_from_fft_error(self):
        import numpy as np
        capture = self.result["productionCapture"]
        aligned = probe_module.read_windowed_frames(self.result["complexPowerCapture"], 512, 11)
        report = probe_module.production_frame_diagnostic(capture, aligned)
        self.assertFalse(report["original_frame_values"]["allclose"])
        self.assertGreater(report["explicit_f32_cast_frame_bits"]["mismatch_count"], 0)
        self.assertTrue(report["same_production_frames_numpy_fft"]["real"]["allclose"])
        self.assertTrue(report["same_production_frames_numpy_fft"]["imag"]["allclose"])
        self.assertEqual(report["original_power_reconstruction"]["max_abs_error"], 0)
        frames = probe_module.read_production_frames(capture, 512, 11)
        self.assertEqual(report["production_frame_sha256_f64le"], hashlib.sha256(frames.tobytes()).hexdigest())
        self.assertEqual(report["status"], "DIAGNOSTIC_NOT_PORT_ACCEPTANCE")

    def test_production_frame_reader_rejects_coerced_mirrors_and_empty_shapes(self):
        capture = self.result["productionCapture"]
        for mirror in ([str(value) for value in capture["frames"]], [False] * len(capture["frames"])):
            with self.subTest(kind=type(mirror[0]).__name__), self.assertRaises(ValueError):
                probe_module.read_production_frames({**capture, "frames": mirror}, 512, 11)
        empty = {**capture, "frames": [], "frameBytesF64LE": "", "frameShape": [1, 0, 512]}
        with self.assertRaises(ValueError):
            probe_module.read_production_frames(empty, 512, 0)

    def test_frame_arithmetic_controls_preserve_existing_feature_experiments(self):
        import numpy as np
        for name, experiment in (("preemphasis", "fp32_preemphasis"),
                ("window", "reference_window"), ("windowProduct", "fp32_window_product")):
            with self.subTest(stage=name):
                capture = self.result["frameArithmeticCaptures"][name]
                self.assertEqual(capture["features"], self.result["experiments"][experiment])
                frames = probe_module.read_production_frames(capture, 512, 11)
                self.assertTrue(np.isfinite(frames).all())
                real = np.asarray(capture["real"], dtype=np.float64)
                imag = np.asarray(capture["imag"], dtype=np.float64)
                np.testing.assert_array_equal(capture["power"], np.float32(real * real + imag * imag))

    def test_final_frame_arithmetic_control_reaches_canonical_aligned_input(self):
        import numpy as np
        final = probe_module.read_production_frames(self.result["frameArithmeticCaptures"]["windowProduct"], 512, 11)
        aligned = probe_module.read_windowed_frames(self.result["complexPowerCapture"], 512, 11)
        self.assertEqual(final.astype("<f4").tobytes(), aligned.astype("<f4").tobytes())
        np.testing.assert_array_equal(final, aligned.astype(np.float64))

    def test_official_complex_injection_preserves_layout_and_library_power_arithmetic(self):
        import numpy as np
        boundary = inputs()["referenceComplex"]
        real = np.frombuffer(base64.b64decode(boundary["realF32LE"]), dtype="<f4")
        imag = np.frombuffer(base64.b64decode(boundary["imagF32LE"]), dtype="<f4")
        # Production JS uses double intermediates and one Float32Array assignment.
        expected = np.float32(real.astype(np.float64) ** 2 + imag.astype(np.float64) ** 2)
        capture = self.result["referenceComplexLibraryCapture"]
        self.assertEqual(capture["shape"], [1, 257, 11])
        np.testing.assert_array_equal(capture["power"], expected)
        np.testing.assert_array_equal(capture["real"], real)
        np.testing.assert_array_equal(capture["imag"], imag)
        self.assertEqual(capture["features"], self.result["experiments"]["reference_power"])

    def test_official_complex_injection_preserves_distinct_fp32_power_control(self):
        import numpy as np
        boundary = inputs()["referenceComplex"]
        real = np.frombuffer(base64.b64decode(boundary["realF32LE"]), dtype="<f4")
        imag = np.frombuffer(base64.b64decode(boundary["imagF32LE"]), dtype="<f4")
        squared = np.float32(np.float32(real * real) + np.float32(imag * imag))
        magnitude = np.asarray([np.float32(math.sqrt(float(value))) for value in squared])
        expected = np.float32(magnitude * magnitude)
        capture = self.result["referenceComplexRoundedCapture"]
        self.assertEqual(capture["shape"], [1, 257, 11])
        np.testing.assert_array_equal(capture["power"], expected)
        self.assertEqual(capture["features"], self.result["experiments"]["reference_power"])

    def test_official_complex_encoder_preserves_float32_signed_zero_and_rejects_wrong_input(self):
        import numpy as np
        real = np.asarray([0, -0.0, 1, -1], dtype=np.float32).reshape(1, 2, 2)
        imag = np.asarray([-0.0, 0, 0.5, -0.5], dtype=np.float32).reshape(1, 2, 2)
        values = np.empty(real.shape, dtype=np.complex64)
        values.real, values.imag = real, imag
        encoded = probe_module.encode_complex_boundary(values)
        self.assertEqual(encoded["shape"], [1, 2, 2])
        self.assertEqual(base64.b64decode(encoded["realF32LE"]), real.astype("<f4").tobytes())
        self.assertEqual(base64.b64decode(encoded["imagF32LE"]), imag.astype("<f4").tobytes())
        for invalid in (values.astype(np.complex128), real, values[0],
                np.full((1, 2, 2), complex(float("nan"), 0), dtype=np.complex64)):
            with self.subTest(dtype=invalid.dtype, shape=invalid.shape):
                with self.assertRaises(ValueError):
                    probe_module.encode_complex_boundary(invalid)

    def test_official_complex_injection_rejects_corrupt_metadata_or_components(self):
        import numpy as np
        for field, invalid in (("shape", [1, 11, 257]), ("dtype", "complex128"),
                ("layout", "batch-frame-bin"), ("encoding", "json"),
                ("realF32LE", "!invalid!"), ("imagF32LE", ""),
                ("realF32LE", base64.b64encode(np.full(257 * 11, np.nan, dtype="<f4").tobytes()).decode("ascii"))):
            with self.subTest(field=field):
                data = inputs()
                data["referenceComplex"][field] = invalid
                completed = subprocess.run(["node", "--input-type=module", "-e", probe_module.NODE_PROBE],
                    cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding="utf-8",
                    capture_output=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Reference complex", completed.stderr)

    def test_full_fft_controls_preserve_shape_mask_and_finite_complex_bins(self):
        for key in ("fullFftF32Capture", "fullFftF64Capture"):
            with self.subTest(capture=key):
                capture = self.result[key]
                self.assertEqual(capture["shape"], [1, 257, 11])
                self.assertEqual(len(capture["features"]), 88)
                for field in ("power", "real", "imag"):
                    self.assertEqual(len(capture[field]), 257 * 11)
                    self.assertTrue(all(math.isfinite(value) for value in capture[field]))
                self.assertTrue(all(value >= 0 for value in capture["power"]))
                self.assertTrue(all(capture["features"][index * 11 + 10] == 0
                    for index in range(8)))

    def test_power_controls_capture_raw_mel_before_normalization_and_masking(self):
        import numpy as np
        for key in ("powerCapture", "fullFftF32Capture", "fullFftF64Capture",
                "referenceComplexLibraryCapture", "referenceComplexRoundedCapture"):
            with self.subTest(capture=key):
                capture = self.result[key]
                raw = np.asarray(capture["raw"], dtype=np.float64).reshape(8, 11)
                self.assertTrue(np.isfinite(raw).all())
                # Zero filterbank: log(epsilon) is present even in the masked tail.
                self.assertTrue(np.all(raw[:, 10] < 0))
                mean = raw[:, :10].mean(axis=1, keepdims=True)
                std = raw[:, :10].std(axis=1, ddof=1, keepdims=True) + 1e-5
                expected = np.zeros_like(raw, dtype=np.float32)
                expected[:, :10] = (raw[:, :10] - mean) / std
                np.testing.assert_allclose(np.asarray(capture["features"]).reshape(8, 11),
                    expected, atol=1e-6, rtol=1e-6)

    def test_raw_capture_reconstructs_nonconstant_real_js_normalization(self):
        import numpy as np
        data = inputs()
        data["referenceFilterbank"] = [1 / (mel + 1) if mel + 1 <= bin < mel + 12 else 0
            for mel in range(8) for bin in range(257)]
        completed = subprocess.run(["node", "--input-type=module", "-e", probe_module.NODE_PROBE],
            cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding="utf-8",
            capture_output=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        for key in ("fullFftF32Capture", "fullFftF64Capture"):
            capture = result[key]
            raw = np.asarray(capture["raw"], dtype=np.float64).reshape(8, 11)
            self.assertGreater(float(raw[:, :10].std()), 0.1)
            expected = np.zeros_like(raw, dtype=np.float32)
            expected[:, :10] = (raw[:, :10] - raw[:, :10].mean(axis=1, keepdims=True)) / (
                raw[:, :10].std(axis=1, ddof=1, keepdims=True) + 1e-5)
            np.testing.assert_array_equal(np.asarray(capture["features"], dtype=np.float32).reshape(8, 11), expected)

    def test_full_fft_controls_match_independent_direct_dft(self):
        import numpy as np
        pcm = np.asarray(inputs()["pcm"], dtype=np.float32)
        preemphasis = np.empty_like(pcm)
        preemphasis[0] = pcm[0]
        preemphasis[1:] = pcm[1:] - np.float32(0.97) * pcm[:-1]
        padded = np.pad(preemphasis, (256, 256))
        window = np.pad(np.ones(400, dtype=np.float32), (56, 56))
        # Direct O(N^2) DFT is independent of both the packed and full FFT code.
        kernel = np.exp(-2j * np.pi * np.outer(np.arange(257), np.arange(512)) / 512)
        for frame in (0, 1, 5, 10):
            samples = np.float32(padded[frame * 160:frame * 160 + 512] * window)
            expected = kernel @ samples.astype(np.float64)
            for key, tolerance in (("fullFftF32Capture", 2e-5), ("fullFftF64Capture", 1e-10)):
                with self.subTest(capture=key, frame=frame):
                    capture = self.result[key]
                    real = np.asarray(capture["real"])[frame::11]
                    imag = np.asarray(capture["imag"])[frame::11]
                    # Functional FFT/layout check, not NeMo feature acceptance.
                    self.assertLess(float(np.max(np.abs(real - expected.real))), tolerance)
                    self.assertLess(float(np.max(np.abs(imag - expected.imag))), tolerance)

    def test_full_fft_captures_exact_float32_frame_major_inputs(self):
        import numpy as np
        pcm = np.asarray(inputs()["pcm"], dtype=np.float32)
        preemphasis = np.empty_like(pcm)
        preemphasis[0] = pcm[0]
        preemphasis[1:] = pcm[1:] - np.float32(0.97) * pcm[:-1]
        padded = np.pad(preemphasis, (256, 256))
        window = np.pad(np.ones(400, dtype=np.float32), (56, 56))
        expected = np.stack([np.float32(padded[frame * 160:frame * 160 + 512] * window)
            for frame in range(11)]).reshape(1, 11, 512)
        for key in ("fullFftF32Capture", "fullFftF64Capture"):
            with self.subTest(capture=key):
                capture = self.result[key]
                self.assertEqual(capture["frameShape"], [1, 11, 512])
                self.assertEqual(capture["frameDtype"], "float32")
                self.assertEqual(capture["frameLayout"], "batch-frame-sample")
                frames = probe_module.read_windowed_frames(capture, 512, 11)
                np.testing.assert_array_equal(frames, expected)
                self.assertEqual(frames.astype("<f4").tobytes(), expected.astype("<f4").tobytes())
        self.assertEqual(self.result["fullFftF32Capture"]["frames"],
            self.result["fullFftF64Capture"]["frames"])

    def test_frame_reader_rejects_wrong_shape_layout_count_or_non_float32_values(self):
        import numpy as np
        capture = {"frameShape": [1, 1, 4], "frameDtype": "float32",
            "frameLayout": "batch-frame-sample", "frames": [0, 1, -1, 0.5],
            "frameBytesEncoding": "base64-f32le", "frameBytesF32LE": base64.b64encode(
                np.asarray([0, 1, -1, 0.5], dtype="<f4").tobytes()).decode("ascii")}
        np.testing.assert_array_equal(probe_module.read_windowed_frames(capture, 4, 1),
            np.asarray([[[0, 1, -1, 0.5]]], dtype=np.float32))
        for field, invalid in (("frameShape", [1, 4, 1]), ("frameDtype", "float64"),
                ("frameLayout", "bin-major"), ("frames", [0, 1, -1]),
                ("frameBytesEncoding", "json"), ("frameBytesF32LE", "!invalid!"),
                ("frameBytesF32LE", ""),
                ("frames", [0, 1, float("nan"), 0.5]),
                ("frames", [0, 1, float("inf"), 0.5]),
                ("frames", [0, 1, -1, 0.1])):
            with self.subTest(field=field, invalid=invalid):
                with self.assertRaises(ValueError):
                    probe_module.read_windowed_frames({**capture, field: invalid}, 4, 1)

    def test_complex_comparison_preserves_raw_components_and_bin_frame_coordinates(self):
        import numpy as np
        capture = self.result["fullFftF64Capture"]
        reference = (np.asarray(capture["real"]) + 1j * np.asarray(capture["imag"])).reshape(1, 257, 11)
        metrics = probe_module.compare_fft_components(capture, reference, atol=1e-5, rtol=1e-4)
        self.assertTrue(all(metric["allclose"] for metric in metrics.values()))
        changed = {**capture, "real": capture["real"].copy(), "imag": capture["imag"].copy()}
        changed["real"][12] += 2
        changed["imag"][0] += 1  # DC must be reported, not silently forced to zero.
        metrics = probe_module.compare_fft_components(changed, reference, atol=1e-5, rtol=1e-4)
        self.assertEqual(metrics["real"]["first_divergent_coordinate"], [0, 1, 1])
        self.assertEqual(metrics["imag"]["first_divergent_coordinate"], [0, 0, 0])

    def test_complex_comparison_rejects_bad_shape_noncomplex_or_nonfinite_input(self):
        import numpy as np
        capture = self.result["fullFftF64Capture"]
        for reference in (np.zeros((1, 257, 11)), np.zeros((1, 11, 257), dtype=np.complex64),
                np.full((1, 257, 11), complex(float("nan"), 0))):
            with self.subTest(shape=reference.shape, dtype=reference.dtype):
                with self.assertRaises(ValueError):
                    probe_module.compare_fft_components(capture, reference, atol=1e-5, rtol=1e-4)

    def test_full_fft_power_rounding_matches_captured_complex_components(self):
        import numpy as np
        for key in ("fullFftF32Capture", "fullFftF64Capture"):
            capture = self.result[key]
            real = np.asarray(capture["real"], dtype=np.float32)
            imag = np.asarray(capture["imag"], dtype=np.float32)
            imag[:11] = 0
            imag[256 * 11:] = 0
            squared = np.float32(np.float32(real * real) + np.float32(imag * imag))
            magnitude = np.asarray([np.float32(math.sqrt(float(value)))
                for value in squared], dtype=np.float32)
            expected = np.float32(magnitude * magnitude)
            np.testing.assert_array_equal(capture["power"], expected)

    def test_changed_fft_scratch_contract_fails_without_editing_library(self):
        declaration = "let source = readFileSync('./dist/audio/js-mel.js', 'utf8');"
        self.assertEqual(probe_module.NODE_PROBE.count(declaration), 1)
        code = probe_module.NODE_PROBE.replace(declaration, declaration[:-1] +
            ".replace('this.fftRe = new Float64Array(N_FFT >> 1);', "
            "'this.fftRe = new Float64Array(256);');")
        completed = subprocess.run(["node", "--input-type=module", "-e", code],
            cwd=probe_module.ROOT, input=json.dumps(inputs()), text=True, encoding="utf-8",
            capture_output=True, timeout=30)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("FFT scratch contract changed", completed.stderr)
        processor = probe_module.ROOT / "dist/audio/js-mel.js"
        self.assertEqual(hashlib.sha256(processor.read_bytes()).hexdigest(), self.before_sha)

    def test_incompatible_reference_window_fails_closed(self):
        data = inputs()
        data["referenceWindow"].pop()
        completed = subprocess.run(["node", "--input-type=module", "-e", probe_module.NODE_PROBE],
            cwd=probe_module.ROOT, input=json.dumps(data), text=True, encoding="utf-8",
            capture_output=True, timeout=30)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Window contract differs", completed.stderr)


class SelectedDftTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        frames = np.zeros((1, 3, 8), dtype=np.float32)
        frames[0, 0, 2] = 1
        frames[0, 1, 0] = 2
        frames[0, 2, 0] = -0.0
        exact = np.fft.rfft(frames.astype(np.float64), axis=2).transpose(0, 2, 1)
        official = exact.astype(np.complex64)
        official[0, 1, 0] = 0.125 - 1j
        controls = {}
        for name, perturbation in (("f32", 0.25), ("f64", 0)):
            candidate = exact.copy()
            candidate[0, 1, 0] = perturbation - 1j
            controls[name] = {"shape": list(candidate.shape),
                "real": candidate.real.ravel().tolist(), "imag": candidate.imag.ravel().tolist(),
                "power": np.float32(abs(candidate) ** 2).ravel().tolist(),
                "frameShape": list(frames.shape), "frameDtype": "float32",
                "frameLayout": "batch-frame-sample", "frames": frames.ravel().tolist(),
                "frameBytesEncoding": "base64-f32le",
                "frameBytesF32LE": base64.b64encode(frames.astype("<f4").tobytes()).decode("ascii")}
        support = {name: [{"coordinate": [0, 0, 0], "active_bins": [1]},
                {"coordinate": [0, 1, 1], "active_bins": [2, 3]}] for name in controls}
        return frames, official, np.float32(abs(official) ** 2), controls, support

    def test_report_deduplicates_shared_support_and_keeps_engine_errors_separate(self):
        frames, official, power, controls, support = self.fixture()
        report = probe_module.fft_dft_diagnostic(frames, official, power, controls, support)
        self.assertEqual(report["selected_frame_bin_count"], 3)
        self.assertEqual([point["coordinate"] for point in report["bins"]], [[0, 1, 0], [0, 2, 1], [0, 3, 1]])
        first = report["bins"][0]
        self.assertAlmostEqual(first["engines"]["official_complex64"]["complex_abs_error"], 0.125, places=12)
        self.assertAlmostEqual(first["engines"]["f32"]["complex_abs_error"], 0.25, places=12)
        self.assertLess(first["engines"]["f64"]["complex_abs_error"], 1e-14)
        self.assertEqual(first["closest_to_dft"], ["f64"])
        self.assertEqual(first["engines"]["f32"]["boundary_power"], controls["f32"]["power"][3])
        self.assertEqual(report["frame_sha256_f32le"], hashlib.sha256(frames.astype("<f4").tobytes()).hexdigest())
        self.assertEqual(report["status"], "DIAGNOSTIC_NOT_PORT_ACCEPTANCE")

    def test_report_rejects_different_frame_bytes_even_when_numeric_zero_matches(self):
        import numpy as np
        frames, official, power, controls, support = self.fixture()
        other = frames.copy()
        other[0, 2, 0] = 0.0
        controls["f64"]["frameBytesF32LE"] = base64.b64encode(other.astype("<f4").tobytes()).decode("ascii")
        with self.assertRaisesRegex(ValueError, "identical canonical"):
            probe_module.fft_dft_diagnostic(frames, official, power, controls, support)

    def test_report_rejects_invalid_support_and_component_shapes(self):
        for field, invalid in (("coordinate", [0, 0, 3]), ("active_bins", [1, 1]), ("active_bins", [5]),
                ("active_bins", [True]), ("coordinate", [1, 0, 0])):
            frames, official, power, controls, support = self.fixture()
            support["f32"][0][field] = invalid
            with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                probe_module.fft_dft_diagnostic(frames, official, power, controls, support)
        frames, official, power, controls, support = self.fixture()
        controls["f64"]["real"].pop()
        with self.assertRaises(ValueError):
            probe_module.fft_dft_diagnostic(frames, official, power, controls, support)

    def test_empty_support_is_unknown_not_a_pass_claim(self):
        frames, official, power, controls, support = self.fixture()
        report = probe_module.fft_dft_diagnostic(frames, official, power, controls, {name: [] for name in support})
        self.assertEqual(report["bins"], [])
        self.assertEqual(report["selected_frame_bin_count"], 0)
        self.assertIsNone(report["engine_summaries"]["official_complex64"]["max_abs_component_error"])

    def test_impulse_has_analytic_sign_bin_and_frame_results(self):
        import numpy as np
        frames = np.zeros((1, 2, 8), dtype=np.float32)
        frames[0, 0, 2] = 1
        frames[0, 1, 0] = 3
        values = probe_module.direct_dft_bins(frames, [(0, 0), (0, 1), (0, 3), (0, 4), (1, 1)])
        np.testing.assert_allclose(values, [1, -1j, 1j, 1, 3], atol=1e-14, rtol=0)

    def test_dc_and_nyquist_do_not_lose_cancelling_terms(self):
        import numpy as np
        frames = np.asarray([[[32768, 1, -32768, 1]]], dtype=np.float32)
        values = probe_module.direct_dft_bins(frames, [(0, 0), (0, 2)])
        self.assertAlmostEqual(values[0].real, 2, places=12)
        self.assertAlmostEqual(values[1].real, -2, places=12)
        self.assertLess(abs(values[0].imag), 1e-12)
        self.assertLess(abs(values[1].imag), 1e-10)

    def test_dft_does_not_modify_canonical_input_or_signed_zero(self):
        import numpy as np
        frames = np.asarray([[[-0.0, 1, -1, 0.5]]], dtype=np.float32)
        before = frames.tobytes()
        frames.flags.writeable = False
        self.assertEqual(probe_module.direct_dft_bins(frames, []), [])
        probe_module.direct_dft_bins(frames, [(0, 1)])
        self.assertEqual(frames.tobytes(), before)

    def test_dft_rejects_noncanonical_nonfinite_and_wrong_shape_inputs(self):
        import numpy as np
        for frames in (np.ones((1, 2, 8), dtype=np.float64), np.ones((2, 8), dtype=np.float32),
                np.ones((2, 1, 8), dtype=np.float32), np.ones((1, 1, 7), dtype=np.float32),
                np.ones((1, 0, 8), dtype=np.float32), np.full((1, 1, 8), np.nan, dtype=np.float32)):
            with self.subTest(shape=frames.shape, dtype=frames.dtype):
                with self.assertRaises(ValueError):
                    probe_module.direct_dft_bins(frames, [])

    def test_dft_rejects_bad_duplicate_or_out_of_range_coordinates(self):
        import numpy as np
        frames = np.ones((1, 2, 8), dtype=np.float32)
        for coordinates in ([(0, 1), (0, 1)], [(2, 1)], [(0, 5)], [(-1, 0)], [(0, -1)],
                [(0, True)], [(0.5, 1)], [(0, 1, 2)], [None]):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    probe_module.direct_dft_bins(frames, coordinates)


class NormalizationDecompositionTests(unittest.TestCase):
    def observed_control(self, *, mean=1.25, std=2):
        import numpy as np
        raw = np.asarray([[[0, 1, 2, 999]]], dtype=np.float32)
        candidate = np.asarray([[[0.25, 1.25, 2.25, -999]]], dtype=np.float32)
        expected = np.asarray([[[-1, 0, 1, 0]]], dtype=np.float32)
        actual = np.zeros_like(candidate)
        actual[:, :, :3] = (candidate[:, :, :3] - np.float32(mean)) / np.float32(std)
        args = [raw, candidate, expected, actual, np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32)]
        stats = {"mean": np.asarray([[mean]], dtype=np.float32), "std": np.asarray([[std]], dtype=np.float32)}
        return args, stats

    def test_observed_statistics_separate_mean_std_and_pointwise_terms(self):
        args, stats = self.observed_control()
        before = [value.tobytes() for value in args] + [value.tobytes() for value in stats.values()]
        result = probe_module.normalization_decomposition(*args, observed_statistics=stats,
            valid_length=3, atol=1e-5, rtol=1e-4)
        self.assertEqual(result['candidate_statistics_source'], 'Observed f32 boundaries')
        self.assertEqual(result['per_mel'][0]['candidate_mean'], 1.25)
        self.assertEqual(result['per_mel'][0]['candidate_std_with_guard'], 2)
        for point in result['mismatches']:
            self.assertAlmostEqual(point['statistics_mean'] + point['statistics_std'], point['statistics'], places=12)
            self.assertAlmostEqual(sum(point[key] for key in ('pointwise_raw', 'statistics',
                'reference_rounding', 'candidate_rounding')), point['candidate_feature'] - point['reference_feature'], places=12)
        self.assertEqual(result['terms_max_abs']['candidate_rounding'], 0)
        self.assertEqual(before, [value.tobytes() for value in args] + [value.tobytes() for value in stats.values()])

    def test_observed_statistics_fail_closed_on_missing_wrong_or_nonfinite_values(self):
        import numpy as np
        args, stats = self.observed_control()
        for invalid in ({'mean': stats['mean']}, {'mean': stats['mean'], 'std': stats['std'], 'extra': 1},
                {'mean': stats['mean'].astype(np.float64), 'std': stats['std']},
                {'mean': stats['mean'], 'std': np.zeros_like(stats['std'])},
                {'mean': np.full_like(stats['mean'], np.nan), 'std': stats['std']},
                {'mean': stats['mean'][0], 'std': stats['std']}):
            with self.subTest(stats=invalid), self.assertRaises(ValueError):
                probe_module.normalization_decomposition(*args, observed_statistics=invalid,
                    valid_length=3, atol=1e-5, rtol=1e-4)

    def test_observed_singleton_preserves_guarded_inverse_and_zero_tail(self):
        import numpy as np
        raw = np.asarray([[[4, 999]]], dtype=np.float32)
        candidate = np.asarray([[[5, -999]]], dtype=np.float32)
        features = np.zeros_like(raw)
        mean, std = np.asarray([[4]], dtype=np.float32), np.asarray([[1e-5]], dtype=np.float32)
        result = probe_module.normalization_decomposition(raw, candidate, features, features, mean, std,
            observed_statistics={'mean': np.asarray([[5]], dtype=np.float32), 'std': std},
            valid_length=1, atol=1e-5, rtol=1e-4)
        self.assertEqual(result['mismatches'], [])
        self.assertEqual(result['per_mel'][0]['candidate_inverse_std'], 1 / float(std[0, 0]))
        self.assertEqual(result['terms_max_abs']['closure_residual'], 0)

    def run_control(self, raw, candidate, *, valid_length=3, atol=1e-6, rtol=1e-6):
        import numpy as np
        raw, candidate = np.asarray(raw, dtype=np.float32), np.asarray(candidate, dtype=np.float32)
        mean = raw[:, :, :valid_length].mean(axis=2, dtype=np.float32)
        std = np.float32(raw[:, :, :valid_length].std(axis=2, ddof=1) + np.float32(1e-5))
        reference_features = np.zeros_like(raw)
        reference_features[:, :, :valid_length] = (
            raw[:, :, :valid_length] - mean[:, :, None]) / std[:, :, None]
        candidate_features = np.zeros_like(candidate)
        values = candidate[:, :, :valid_length].astype(np.float64)
        if valid_length > 1:
            candidate_features[:, :, :valid_length] = (
                values - values.mean(axis=2, keepdims=True)) / (
                    values.std(axis=2, ddof=1, keepdims=True) + 1e-5)
        return probe_module.normalization_decomposition(raw, candidate, reference_features,
            candidate_features, mean, std, valid_length=valid_length, atol=atol, rtol=rtol)

    def test_uniform_raw_shift_cancels_through_statistics(self):
        result = self.run_control([[[0, 1, 2, 999]]], [[[0.25, 1.25, 2.25, -999]]])
        self.assertEqual(result["actual_features"]["mismatch_count"], 0)
        self.assertGreater(result["terms_max_abs"]["pointwise_raw"], 0.24)
        self.assertGreater(result["terms_max_abs"]["statistics"], 0.24)
        self.assertLess(result["terms_max_abs"]["closure_residual"], 1e-12)
        self.assertFalse(result["candidate_with_official_statistics"]["allclose"])
        self.assertEqual(result["per_mel"][0]["candidate_mean"], 1.25)

    def test_impulse_reports_all_mismatch_coordinates_and_additive_terms(self):
        result = self.run_control([[[0, 1, 2, 999]]], [[[0, 1, 2.01, -999]]])
        self.assertEqual([point["coordinate"] for point in result["mismatches"]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        for point in result["mismatches"]:
            total = sum(point[key] for key in ("pointwise_raw", "statistics",
                "reference_rounding", "candidate_rounding"))
            self.assertAlmostEqual(total, point["candidate_feature"] - point["reference_feature"], places=12)
        self.assertEqual(result["per_mel"][0]["mismatch_frames"], [0, 1, 2])

    def test_masked_tail_never_changes_statistics_or_coordinates(self):
        result = self.run_control([[[0, 1, 2, 999]]], [[[0, 1, 2, -999]]])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["per_mel"][0]["raw_max_abs_error"], 0)
        self.assertEqual(result["per_mel"][0]["candidate_mean"], 1)

    def test_multiple_mel_bands_keep_statistics_and_coordinates_independent(self):
        result = self.run_control([[[0, 1, 2], [10, 11, 12]]],
            [[[0, 1, 2], [10, 11, 12.01]]])
        self.assertEqual(result["per_mel"][0]["mismatch_frames"], [])
        self.assertEqual(result["per_mel"][0]["candidate_mean"], 1)
        self.assertTrue(all(point["coordinate"][1] == 1 for point in result["mismatches"]))

    def test_constant_band_uses_guard_without_nonfinite_terms(self):
        result = self.run_control([[[4, 4, 4]]], [[[5, 5, 5]]])
        self.assertEqual(result["mismatches"], [])
        self.assertTrue(all(math.isfinite(value) for value in result["terms_max_abs"].values()))
        self.assertAlmostEqual(result["per_mel"][0]["candidate_std_with_guard"], 1e-5)

    def test_single_valid_frame_uses_js_zero_inverse_std_and_masks_tail(self):
        import numpy as np
        raw = np.asarray([[[4, 999]]], dtype=np.float32)
        candidate = np.asarray([[[5, -999]]], dtype=np.float32)
        features = np.zeros_like(raw)
        result = probe_module.normalization_decomposition(raw, candidate, features, features,
            np.asarray([[4]], dtype=np.float32), np.asarray([[1e-5]], dtype=np.float32),
            valid_length=1, atol=1e-5, rtol=1e-4)
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["per_mel"][0]["candidate_inverse_std"], 0)
        self.assertEqual(result["terms_max_abs"]["closure_residual"], 0)

    def test_actual_boundary_rounding_remains_an_explicit_nonzero_residual(self):
        import numpy as np
        raw = np.asarray([[[0, 1, 2]]], dtype=np.float32)
        features = np.asarray([[[-1, 0, 1]]], dtype=np.float32)
        candidate_features = features.copy()
        candidate_features[0, 0, 1] = np.float32(0.001)
        result = probe_module.normalization_decomposition(raw, raw, features, candidate_features,
            np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32),
            valid_length=3, atol=1e-5, rtol=1e-4)
        self.assertFalse(result["candidate_reconstruction"]["allclose"])
        self.assertGreater(result["terms_max_abs"]["candidate_rounding"], 0.0009)
        self.assertEqual(result["mismatches"][0]["coordinate"], [0, 0, 1])
        self.assertLess(result["terms_max_abs"]["closure_residual"], 1e-12)

    def test_rejects_wrong_shape_dtype_nonfinite_stats_and_invalid_length(self):
        import numpy as np
        raw = np.asarray([[[0, 1, 2]]], dtype=np.float32)
        features = np.asarray([[[-1, 0, 1]]], dtype=np.float32)
        mean, std = np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
        original = [raw, raw.copy(), features, features.copy(), mean, std]
        for index, invalid in ((0, raw.astype(np.float64)), (1, raw[0]),
                (2, np.full_like(features, np.nan)), (4, mean.astype(np.float64)),
                (5, np.zeros_like(std)), (5, np.full_like(std, np.inf))):
            with self.subTest(index=index):
                args = original.copy()
                args[index] = invalid
                with self.assertRaises(ValueError):
                    probe_module.normalization_decomposition(*args, valid_length=3, atol=1e-5, rtol=1e-4)
        for length in (0, 4, 1.5, True):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    probe_module.normalization_decomposition(*original, valid_length=length, atol=1e-5, rtol=1e-4)


class PowerErrorAttributionTests(unittest.TestCase):
    def control(self, *, coordinates=None):
        import numpy as np
        reference = np.asarray([[[2, 2, 999], [4, 4, 999], [8, 8, 999], [16, 16, 999]]], dtype=np.float32)
        candidate = reference.copy()
        candidate[0, 0, 0] += 1
        candidate[0, 1, 0] -= 0.25
        candidate[0, 3, 1] += 2
        weights = np.asarray([[0.5, 1, 0, 0], [0, 0, 0.25, 0.5]], dtype=np.float32)
        guard = 2 ** -24
        reference_raw = np.float32(np.log(weights.astype(np.float64) @ reference[0] + guard))[None]
        candidate_raw = np.float32(np.log(weights.astype(np.float64) @ candidate[0] + guard))[None]
        args = [reference, candidate, weights, reference_raw, candidate_raw]
        result = probe_module.power_error_attribution(*args,
            mismatch_coordinates=[[0, 0, 0], [0, 1, 1]] if coordinates is None else coordinates,
            valid_length=2, log_zero_guard=guard)
        return result, args

    def test_signed_weights_and_support_preserve_independent_bin_frame_layout(self):
        result, _ = self.control()
        low, high = result["mismatches"]
        self.assertEqual(low["active_bins"], [0, 1])
        self.assertEqual(high["active_bins"], [2, 3])
        self.assertEqual(low["coordinate"], [0, 0, 0])
        self.assertEqual(high["coordinate"], [0, 1, 1])
        self.assertEqual(low["weighted_positive_error"], 0.5)
        self.assertEqual(low["weighted_negative_error"], -0.25)
        self.assertEqual(low["weighted_signed_error"], 0.25)
        self.assertEqual(low["weighted_absolute_error"], 0.75)
        self.assertAlmostEqual(low["signed_to_absolute_ratio"], 1 / 3)
        self.assertEqual(high["weighted_signed_error"], 1)
        self.assertEqual(low["largest_contributions"][0]["bin"], 0)
        self.assertEqual(high["largest_contributions"][0]["bin"], 3)

    def test_projection_rounding_terms_reconstruct_actual_raw_delta(self):
        result, _ = self.control()
        self.assertTrue(result["candidate_raw_reconstruction"]["allclose"])
        self.assertEqual(result["candidate_raw_reconstruction"]["max_abs_error"], 0)
        for point in result["mismatches"]:
            self.assertEqual(point["closure_residual"], 0)
            self.assertAlmostEqual(point["observed_raw_delta"], sum(point[key] for key in
                ("power_log_term", "reference_projection_term", "candidate_projection_term")), places=12)

    def test_reference_engine_projection_difference_is_not_called_power_error(self):
        import numpy as np
        _, args = self.control()
        args[3] = args[3].copy()
        args[3][0, 0, 0] += np.float32(0.01)
        result = probe_module.power_error_attribution(*args, mismatch_coordinates=[[0, 0, 0]],
            valid_length=2, log_zero_guard=2 ** -24)
        point = result["mismatches"][0]
        self.assertLess(point["reference_projection_term"], -0.009)
        self.assertEqual(point["weighted_signed_error"], 0.25)
        self.assertEqual(point["closure_residual"], 0)

    def test_matching_powers_and_empty_support_have_finite_zero_contributions(self):
        import numpy as np
        powers = np.ones((1, 3, 2), dtype=np.float32)
        weights = np.zeros((2, 3), dtype=np.float32)
        raw = np.full((1, 2, 2), np.float32(math.log(2 ** -24)), dtype=np.float32)
        result = probe_module.power_error_attribution(powers, powers, weights, raw, raw,
            mismatch_coordinates=[[0, 1, 0]], valid_length=1, log_zero_guard=2 ** -24)
        point = result["mismatches"][0]
        self.assertEqual(point["active_bins"], [])
        self.assertEqual(point["weighted_signed_error"], 0)
        self.assertIsNone(point["signed_to_absolute_ratio"])
        self.assertEqual(point["largest_contributions"], [])
        self.assertTrue(result["candidate_raw_reconstruction"]["allclose"])
        empty, _ = self.control(coordinates=[])
        self.assertEqual(empty["mismatches"], [])
        self.assertEqual(len(empty["per_mel"]), 2)

    def test_rejects_wrong_boundaries_negative_values_and_invalid_coordinates(self):
        import numpy as np
        _, original = self.control()
        for index, invalid in ((0, original[0].astype(np.float64)), (1, original[1][0]),
                (1, np.full_like(original[1], -1)), (2, original[2][:, :3]),
                (2, np.full_like(original[2], -1)), (3, original[3].astype(np.float64)),
                (4, np.full_like(original[4], np.nan))):
            with self.subTest(index=index):
                args = original.copy()
                args[index] = invalid
                with self.assertRaises(ValueError):
                    probe_module.power_error_attribution(*args, mismatch_coordinates=[],
                        valid_length=2, log_zero_guard=2 ** -24)
        for coordinates in ([[0, 0, 2]], [[1, 0, 0]], [[0, 2, 0]], [[0, 0, 0.5]],
                [[0, 0, True]], [[0, 0, 0], [0, 0, 0]]):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    probe_module.power_error_attribution(*original, mismatch_coordinates=coordinates,
                        valid_length=2, log_zero_guard=2 ** -24)
        for length, guard in ((0, 2 ** -24), (4, 2 ** -24), (True, 2 ** -24),
                (2, 0), (2, float("nan")), (2, True)):
            with self.subTest(length=length, guard=guard):
                with self.assertRaises(ValueError):
                    probe_module.power_error_attribution(*original, mismatch_coordinates=[],
                        valid_length=length, log_zero_guard=guard)


class DiagnosticOutputTests(unittest.TestCase):
    def test_normalized_support_uses_candidate_features_and_fixed_tolerance(self):
        import numpy as np
        reference = np.zeros((1, 2, 3), dtype=np.float32)
        candidate = reference.copy()
        candidate[0, 0, 0] = 9e-6
        candidate[0, 1, 1] = 2e-5
        before = candidate.tobytes()
        self.assertEqual(probe_module.normalized_failure_coordinates(reference, candidate, 2), [[0, 1, 1]])
        self.assertEqual(candidate.tobytes(), before)
        self.assertEqual(probe_module.normalized_failure_coordinates(reference, reference, 2), [])

    def test_normalized_support_refuses_unmasked_tail_and_invalid_contract(self):
        import numpy as np
        reference = np.zeros((1, 2, 3), dtype=np.float32)
        invalid = reference.copy()
        invalid[0, 0, 2] = 1
        for candidate, length in ((invalid, 2), (reference, True), (reference, 0),
                (reference, 4), (reference.astype(np.float64), 2), (reference[:, :, :2], 2),
                (np.full_like(reference, np.nan), 2)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                probe_module.normalized_failure_coordinates(reference, candidate, length)

    def test_raw_prefix_comparator_excludes_only_invalid_tail(self):
        import numpy as np
        reference = np.asarray([[1, 2, 0]], dtype=np.float32)
        candidate = np.asarray([[1, 2, 999]], dtype=np.float32)
        before = candidate.tobytes()
        self.assertTrue(probe_module.compare_raw_valid_prefix(reference, candidate, 2)['exact']['allclose'])
        self.assertEqual(candidate.tobytes(), before)
        candidate[0, 1] = 3
        self.assertEqual(probe_module.compare_raw_valid_prefix(reference, candidate, 2)['exact']['mismatch_count'], 1)

    def test_raw_prefix_comparator_refuses_invalid_length_or_shape(self):
        import numpy as np
        reference = np.ones((1, 3), dtype=np.float32)
        for length in (True, 0, 4, 1.5):
            with self.assertRaises(ValueError): probe_module.compare_raw_valid_prefix(reference, reference, length)
        with self.assertRaises(ValueError): probe_module.compare_raw_valid_prefix(reference, reference[:, :2], 2)

    def test_gzip_report_preserves_complete_payload_and_refuses_existing_output(self):
        report = {"status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE", "values": [0, -0.0, 1.5], "text": "Türkçe"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub" / "receipt.json.gz"
            probe_module.write_diagnostic_report(path, report)
            before = path.read_bytes()
            with gzip.open(path, "rt", encoding="utf-8") as file:
                self.assertEqual(json.load(file), report)
            with self.assertRaises(FileExistsError):
                probe_module.write_diagnostic_report(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_plain_report_keeps_existing_json_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            report = {"values": [1, 2, 3]}
            probe_module.write_diagnostic_report(path, report)
            self.assertEqual(path.read_text(encoding="utf-8"), json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
