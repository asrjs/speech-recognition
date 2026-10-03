"""Fast preflight/probe checks; no model or browser inference claims."""
import importlib
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
runner = importlib.import_module('local-parakeet-smoke')


class LocalParakeetResultTests(unittest.TestCase):
    def window_result(self):
        value = self.result()
        value['windowing'] = {'windowSeconds': 20, 'durationSeconds': 40}
        for item in value['results']:
            item['meta'].update(detailLevel='detailed', durationSeconds=40)
            item['meta']['metrics']['windowCount'] = 2
            item['tokens'] = [{'index': 0, 'id': 1, 'text': 'Test', 'startTime': 1, 'endTime': 2},
                {'index': 1, 'id': 1, 'text': 'Test', 'startTime': 21, 'endTime': 22}]
            item['words'] = [{'index': 0, 'text': 'Test', 'startTime': 1, 'endTime': 2, 'tokenIndices': [0]},
                {'index': 1, 'text': 'Test', 'startTime': 21, 'endTime': 22, 'tokenIndices': [1]}]
        return value

    def test_requested_window_tokens_requires_real_structure(self):
        runner.validate_result(self.window_result(), False, True)
        for change in ['absent', 'local-index', 'local-time', 'bad-reference', 'one-window']:
            value = self.window_result()
            item = value['results'][0]
            if change == 'absent': item.pop('tokens')
            elif change == 'local-index': item['tokens'][1]['index'] = 0
            elif change == 'local-time': item['tokens'][1]['startTime'] = 1
            elif change == 'bad-reference': item['words'][1]['tokenIndices'] = [99]
            else: item['meta']['metrics']['windowCount'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_result(value, False, True)

    def test_in_range_reference_to_wrong_window_rejects(self):
        value = self.window_result()
        for item in value['results']:
            item['words'][1]['tokenIndices'] = [0]
        with self.assertRaisesRegex(ValueError, 'timing'):
            runner.validate_result(value, False, True)

    def result(self):
        item = {'text': 'Test phrase.', 'meta': {'metrics': {
            'encoderBackend': 'webgpu', 'decoderBackend': 'wasm'}}}
        return {'status': 'PASS_LOCAL_HYBRID_SMOKE', 'disposed': True,
            'inferencePassed': True, 'results': [item, copy.deepcopy(item)],
            'cancellation': {'requested': True, 'rejected': True,
                'completeEmitted': False, 'errorName': 'PipelineAbortedError',
                'stageId': 'decode', 'completedUnits': 1, 'totalUnits': 10}}

    def test_complete_result_accepts_with_or_without_requested_cancel(self):
        runner.validate_result(self.result(), True)
        result = self.result()
        del result['cancellation']
        runner.validate_result(result, False)

    def test_pass_label_does_not_replace_lifecycle_evidence(self):
        for key in ['disposed', 'inferencePassed']:
            with self.subTest(key=key):
                result = self.result()
                result[key] = False
                with self.assertRaises(ValueError):
                    runner.validate_result(result, True)

    def test_missing_repeated_result_or_changed_transcript_rejects(self):
        for change in ['missing', 'changed', 'empty']:
            with self.subTest(change=change):
                result = self.result()
                if change == 'missing':
                    result['results'].pop()
                else:
                    result['results'][1]['text'] = '' if change == 'empty' else 'Different'
                with self.assertRaises(ValueError):
                    runner.validate_result(result, True)

    def test_wrong_provider_or_cleanup_error_rejects(self):
        result = self.result()
        result['results'][1]['meta']['metrics']['encoderBackend'] = 'wasm'
        with self.assertRaises(ValueError):
            runner.validate_result(result, True)
        for key in ['error', 'cleanupError', 'audioCleanupError']:
            result = self.result()
            result[key] = 'failure'
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_result(result, True)

    def test_requested_cancel_requires_decode_rejection_and_partial_progress(self):
        for key, value in [('requested', False), ('rejected', False),
                ('completeEmitted', True), ('errorName', 'Error'),
                ('stageId', 'encoder'), ('completedUnits', 0),
                ('completedUnits', 10), ('totalUnits', True)]:
            with self.subTest(key=key, value=value):
                result = self.result()
                result['cancellation'][key] = value
                with self.assertRaises(ValueError):
                    runner.validate_result(result, True)


class LocalParakeetPreflightTests(unittest.TestCase):
    def arguments(self, root, phrase='test'):
        return ['local-parakeet-smoke.py', '--consumer-root', str(root / 'consumer'),
            '--model-root', str(root / 'model'), '--audio', str(root / 'audio.wav'),
            '--expected-phrase', phrase, '--output', str(root / 'receipt.json')]

    def test_empty_expected_phrase_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, 'argv', self.arguments(root, ' ')), patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()
            self.assertFalse((root / 'receipt.json').exists())

    def test_missing_requested_model_fails_without_receipt_or_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, 'argv', self.arguments(root)), patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'Missing/empty/symlink'):
                    runner.main()
                launch.assert_not_called()
            self.assertFalse((root / 'receipt.json').exists())

    def test_existing_receipt_is_preserved_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model').mkdir()
            (root / 'consumer').mkdir()
            for name in ['encoder-model.onnx', 'decoder_joint-model.onnx', 'vocab.txt']:
                (root / 'model' / name).write_bytes(b'preflight-only, not model data')
            for name in ['vite.config.js', 'package.json']:
                (root / 'consumer' / name).write_bytes(b'preflight-only')
            (root / 'audio.wav').write_bytes(b'preflight-only')
            output = root / 'receipt.json'
            output.write_bytes(b'keep me')
            with patch.object(sys, 'argv', self.arguments(root)), patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(FileExistsError):
                    runner.main()
                launch.assert_not_called()
            self.assertEqual(output.read_bytes(), b'keep me')

    def test_invalid_url_is_not_reported_as_closed(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Existing Node required for actual closure-probe control')
        result = runner.probe_preview_closed(node, 'invalid-url')
        self.assertIs(result['closed'], False)
        self.assertEqual(result['code'], 'ERR_INVALID_URL')

    def test_probe_timeout_is_not_silently_a_closed_port(self):
        with patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 10)):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.probe_preview_closed('node', 'http://127.0.0.1:1')


if __name__ == '__main__':
    unittest.main()
