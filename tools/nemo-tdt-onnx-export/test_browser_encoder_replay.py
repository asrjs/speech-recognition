"""Synthetic result gates only; real browser evidence is separate."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import browser_encoder_replay as replay
from browser_encoder_replay import evaluate


class EncoderResultTests(unittest.TestCase):
    def arguments(self, output):
        return ['probe', '--capture', 'missing.gz', '--encoder', 'missing.onnx',
            '--capture-sha256', '1' * 64, '--encoder-sha256', '1' * 64,
            '--encoder-data-sha256', '1' * 64, '--sample-id', 'fixture',
            '--backend', 'webgpu', '--output', str(output)]

    def test_existing_output_preserved_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            output.write_text('preserve', encoding='utf-8')
            with patch('sys.argv', self.arguments(output)), patch.object(replay, 'sha256') as hashing:
                with self.assertRaises(FileExistsError):
                    replay.main()
                hashing.assert_not_called()
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')

    def test_wrong_hash_rejected_before_graph_or_server(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            with patch('sys.argv', self.arguments(output)), patch.object(replay, 'sha256', return_value='0' * 64), \
                    patch.object(replay, 'audit_graph') as graph, patch.object(replay, 'ThreadingHTTPServer') as server:
                with self.assertRaisesRegex(ValueError, 'Pinned source differs'):
                    replay.main()
                graph.assert_not_called()
                server.assert_not_called()
            self.assertFalse(output.exists())

    def fixture(self):
        values = {'dtype': 'float32', 'shape': [1, 1, 2], 'data': [1.0, 2.0]}
        length = {'dtype': 'int64', 'shape': [1], 'data': [2]}
        sample = {'stages': {'encoder': values, 'encoder_length': length}}
        result = {'backendRequested': 'webgpu', 'crossOriginIsolated': True,
            'cleanupErrors': [], 'runs': [{'repeat': repeat, 'gpuDispatches': 1,
                'outputs': {'outputs': copy.deepcopy(values), 'encoded_lengths': copy.deepcopy(length)}}
                for repeat in range(2)]}
        return result, sample

    def test_exact_repeated_outputs_pass(self):
        result, sample = self.fixture()
        self.assertTrue(evaluate(result, sample, 'webgpu'))

    def test_numeric_divergence_fails(self):
        result, sample = self.fixture()
        result['runs'][1]['outputs']['outputs']['data'][0] = 1.1
        self.assertFalse(evaluate(result, sample, 'webgpu'))

    def test_length_must_be_exact(self):
        result, sample = self.fixture()
        result['runs'][1]['outputs']['encoded_lengths']['data'][0] = 3
        self.assertFalse(evaluate(result, sample, 'webgpu'))

    def test_missing_dispatch_or_repeat_fails(self):
        result, sample = self.fixture()
        result['runs'][0]['gpuDispatches'] = 0
        self.assertFalse(evaluate(result, sample, 'webgpu'))
        result['runs'].pop()
        self.assertFalse(evaluate(result, sample, 'webgpu'))

    def test_error_cleanup_or_wrong_backend_fails(self):
        for key, value in [('error', 'failed'), ('cleanupErrors', ['failed']),
                           ('backendRequested', 'wasm'), ('crossOriginIsolated', False)]:
            result, sample = self.fixture()
            result[key] = value
            self.assertFalse(evaluate(result, sample, 'webgpu'))

    def sensitivity_fixture(self):
        result, sample = self.fixture()
        sample['model_input'] = {'sha256': 'a' * 64}
        sample['stages']['features'] = copy.deepcopy(sample['stages']['encoder'])
        sample['stages']['features_length'] = copy.deepcopy(sample['stages']['encoder_length'])
        result.update(pcmSha256='a' * 64, productionLength=2,
                      productionFeatures=copy.deepcopy(sample['stages']['features']))
        official = result['runs']
        production = copy.deepcopy(official)
        for run in official:
            run['featureSource'] = 'official'
        for run in production:
            run['featureSource'] = 'production-js'
        result['runs'] = official + production
        return result, sample

    def test_failed_features_remain_failed_in_successful_measurement(self):
        result, sample = self.sensitivity_fixture()
        result['productionFeatures']['data'][0] = 1.1
        self.assertTrue(replay.evaluate_sensitivity(result, sample, 'webgpu'))
        self.assertFalse(result['feature_comparison']['allclose'])
        self.assertTrue(result['official_encoder_pass'])
        self.assertTrue(result['production_encoder_pass'])

    def test_failed_production_encoder_is_reported_not_promoted(self):
        result, sample = self.sensitivity_fixture()
        result['runs'][2]['outputs']['outputs']['data'][0] = 1.1
        self.assertTrue(replay.evaluate_sensitivity(result, sample, 'webgpu'))
        self.assertFalse(result['production_encoder_pass'])

    def test_failed_official_control_invalidates_measurement(self):
        result, sample = self.sensitivity_fixture()
        result['runs'][0]['outputs']['outputs']['data'][0] = 1.1
        self.assertFalse(replay.evaluate_sensitivity(result, sample, 'webgpu'))

    def test_missing_or_reordered_sensitivity_calls_rejected(self):
        for missing in (True, False):
            result, sample = self.sensitivity_fixture()
            if missing:
                result['runs'].pop()
            else:
                result['runs'].reverse()
            with self.assertRaisesRegex(ValueError, 'missing or reordered'):
                replay.evaluate_sensitivity(result, sample, 'webgpu')

    def test_pcm_identity_and_production_length_rejected(self):
        for key, value in [('pcmSha256', 'b' * 64), ('productionLength', 3),
                           ('productionLength', 2.0), ('productionLength', True)]:
            result, sample = self.sensitivity_fixture()
            result[key] = value
            with self.assertRaises(ValueError):
                replay.evaluate_sensitivity(result, sample, 'webgpu')

    def test_requested_missing_production_features_rejected(self):
        result, sample = self.sensitivity_fixture()
        del result['productionFeatures']
        with self.assertRaises(KeyError):
            replay.evaluate_sensitivity(result, sample, 'webgpu')


class OrtRuntimeBindingTests(unittest.TestCase):
    """The runtime-binding guard must fail, not only pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = 'ort.webgpu.min.mjs'
        (self.root / self.module).write_bytes(b'probe runtime bytes')
        self.sha = replay.sha256(self.root / self.module)
        self.size = (self.root / self.module).stat().st_size
        self.addCleanup(self.tmp.cleanup)

    def event(self, path=None, **overrides):
        row = {'path': path or f'/ort-dist/{self.module}', 'status': 200,
               'bytes_written': self.size, 'body_sha256': self.sha, 'body_complete': True}
        row.update(overrides)
        return row

    def test_matching_runtime_binds_and_verifies(self):
        result = replay.evaluate_ort_runtime([self.event()], self.root, f'/ort-dist/{self.module}')
        self.assertTrue(result['ort_runtime_bound'])
        self.assertTrue(result['ort_module_verified'])
        self.assertTrue(result['ort_runtime'][0]['matches_disk'])

    def test_wrong_selected_module_is_not_verified(self):
        result = replay.evaluate_ort_runtime(
            [self.event()], self.root, '/ort-dist/ort.all.bundle.min.mjs')
        self.assertTrue(result['ort_runtime_bound'])
        self.assertFalse(result['ort_module_verified'])

    def test_truncated_body_fails_the_binding(self):
        result = replay.evaluate_ort_runtime(
            [self.event(bytes_written=self.size - 1)], self.root, f'/ort-dist/{self.module}')
        self.assertFalse(result['ort_runtime_bound'])
        self.assertFalse(result['ort_runtime'][0]['matches_disk'])

    def test_mismatched_body_hash_fails_the_binding(self):
        result = replay.evaluate_ort_runtime(
            [self.event(body_sha256='0' * 64)], self.root, f'/ort-dist/{self.module}')
        self.assertFalse(result['ort_runtime_bound'])
        self.assertFalse(result['ort_module_verified'])

    def test_incomplete_or_failed_delivery_fails_the_binding(self):
        for overrides in ({'body_complete': False}, {'status': 404}):
            with self.subTest(overrides=overrides):
                result = replay.evaluate_ort_runtime(
                    [self.event(**overrides)], self.root, f'/ort-dist/{self.module}')
                self.assertFalse(result['ort_runtime_bound'])

    def test_missing_runtime_events_fail_the_binding(self):
        for events in ([], [{'path': '/graph.onnx', 'status': 200}]):
            with self.subTest(events=events):
                result = replay.evaluate_ort_runtime(events, self.root, f'/ort-dist/{self.module}')
                self.assertFalse(result['ort_runtime_bound'])
                self.assertFalse(result['ort_module_verified'])
                self.assertEqual(result['ort_runtime'], [])

    def test_unknown_sidecar_on_disk_fails_the_binding(self):
        result = replay.evaluate_ort_runtime(
            [self.event(), self.event(path='/ort-dist/absent.wasm')],
            self.root, f'/ort-dist/{self.module}')
        self.assertFalse(result['ort_runtime_bound'])


if __name__ == '__main__':
    unittest.main()
