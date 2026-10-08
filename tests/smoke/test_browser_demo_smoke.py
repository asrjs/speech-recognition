"""Runner preflight and receipt parsing; no browser/build/dependency load."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).with_name('browser-demo-smoke.py')
spec = importlib.util.spec_from_file_location('companion_smoke', TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompanionSmokeTests(unittest.TestCase):
    def test_file_text_rejects_partial_normalized_empty_or_wrong_results(self):
        module.check_file_text('Exact sentence.', 'Exact sentence.')
        for actual, expected in [('exact sentence.', 'Exact sentence.'),
                                 ('Exact sentence', 'Exact sentence.'),
                                 ('Exact', 'Exact sentence.'), ('', ''), (' ', ' '), (None, 'Exact')]:
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                module.check_file_text(actual, expected)

    def test_file_workflow_requires_model_and_exact_nonempty_text(self):
        cases = [['--audio', 'missing.wav'],
                 ['--audio', 'missing.wav', '--local-model-root', 'missing', '--expected-text', ' '],
                 ['--expected-text', 'unexpected']]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            for extra in cases:
                with self.subTest(extra=extra):
                    result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', directory,
                        '--output', str(output)] + extra, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('requires', result.stderr)
                    self.assertFalse(output.exists())

    def test_preview_closure_requires_explicit_refusal_not_unknown_failure(self):
        for code, closed in [('ECONNREFUSED', True), ('REACHABLE', False),
                             ('TimeoutError', False), ('WSAEWOULDBLOCK', False)]:
            completed = subprocess.CompletedProcess([], 0, json.dumps({'closed': closed, 'code': code}))
            with self.subTest(code=code), patch.object(module.subprocess, 'run', return_value=completed):
                self.assertEqual(module.probe_preview_closed('node', 'http://127.0.0.1:12345')['closed'], closed)

    def test_contradictory_or_malformed_closure_evidence_rejects(self):
        for data in ({'closed': True, 'code': 'TimeoutError'}, {'closed': False, 'code': 'ECONNREFUSED'},
                     {'closed': 1, 'code': 'ECONNREFUSED'}, {}, []):
            completed = subprocess.CompletedProcess([], 0, json.dumps(data))
            with self.subTest(data=data), patch.object(module.subprocess, 'run', return_value=completed):
                with self.assertRaises(ValueError):
                    module.probe_preview_closed('node', 'http://127.0.0.1:12345')

    def test_local_model_requires_nonempty_selected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                module.local_model_inputs(root, 'fp16', 'nemo-tdt')
            for name in ('encoder-model.fp16.onnx', 'decoder_joint-model.onnx', 'vocab.txt'):
                (root / name).write_bytes(b'fixture-not-a-model')
            self.assertEqual(len(module.local_model_inputs(root, 'fp16', 'nemo-tdt')), 3)
            (root / 'encoder-model.fp16.onnx.data').write_bytes(b'')
            with self.assertRaises(ValueError):
                module.local_model_inputs(root, 'fp16', 'nemo-tdt')

    def test_tokenizer_requires_exact_nonempty_layout_file(self):
        for layout, (encoder, decoder, tokenizer) in module.ARTIFACT_LAYOUTS.items():
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in (encoder, decoder):
                    if name:
                        (root / (name + '.onnx')).write_bytes(b'fixture-not-a-model')
                backup = root / (tokenizer.rsplit('.', 1)[0] + '.backup')
                backup.write_bytes(b'not-the-tokenizer')
                with self.assertRaises(ValueError):
                    module.local_model_inputs(root, 'fp32', layout)
                with self.assertRaises(ValueError):
                    module.detect_layout(root)
                token_path = root / tokenizer
                token_path.write_bytes(b'')
                with self.assertRaises(ValueError):
                    module.local_model_inputs(root, 'fp32', layout)
                token_path.write_bytes(b'exact-layout-tokenizer')
                paths = module.local_model_inputs(root, 'fp32', layout)
                self.assertIn(token_path, paths)
                self.assertNotIn(backup, paths)

    def test_streamed_digest_matches_full_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            data = b'abcd' * (1024 * 1024)
            path.write_bytes(data)
            self.assertEqual(module.digest(path), hashlib.sha256(data).hexdigest())

    def test_build_receipt_parses_amid_logs(self):
        receipt = {'status': 'PASS_IN_MEMORY_COMPANION_BUILD', 'outputs': [{'fileName': 'index.html'}]}
        self.assertEqual(module.parse_build('vite log\n' + json.dumps(receipt, indent=2) + '\nready'), receipt)

    def test_missing_duplicate_empty_and_malformed_receipts_fail(self):
        for value in ('log only', module.BUILD_MARKER, module.BUILD_MARKER * 2,
                      json.dumps({'status': 'PASS_IN_MEMORY_COMPANION_BUILD', 'outputs': []}, indent=2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.parse_build(value)

    def test_existing_output_is_preserved_before_build_or_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'existing.json'
            output.write_bytes(b'original')
            result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', str(root / 'missing'),
                '--output', str(output)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_requested_missing_consumer_saves_preflight_failure_without_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', str(root / 'missing'),
                '--output', str(output)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            receipt = json.loads(output.read_text())
            self.assertIn('vite.config.js', receipt['error'])
            self.assertEqual(receipt['status'], 'FAIL_COMPANION_INPUT_PREFLIGHT')
            self.assertIs(receipt['browser_started'], False)

    def test_requested_missing_model_or_audio_saves_failure_before_browser(self):
        for missing in ('model', 'audio'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                consumer = root / 'consumer'
                (consumer / 'src').mkdir(parents=True)
                for name in ('vite.config.js', 'package.json', 'src/App.jsx'):
                    (consumer / name).write_text('fixture-not-executed')
                model = root / 'model'
                if missing == 'audio':
                    model.mkdir()
                    for name in ('encoder-model.onnx', 'decoder_joint-model.onnx', 'vocab.txt'):
                        (model / name).write_bytes(b'fixture-not-executed')
                output = root / 'receipt.json'
                result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', str(consumer),
                    '--local-model-root', str(model), '--audio', str(root / 'missing.wav'),
                    '--expected-text', 'Exact.', '--output', str(output)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                receipt = json.loads(output.read_text())
                self.assertEqual(receipt['status'], 'FAIL_COMPANION_INPUT_PREFLIGHT')
                self.assertIs(receipt['browser_started'], False)
                self.assertIn('error', receipt)


if __name__ == '__main__':
    unittest.main()
