"""Fast preflight/probe checks; no model or browser inference claims."""
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
runner = importlib.import_module('local-parakeet-smoke')


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
