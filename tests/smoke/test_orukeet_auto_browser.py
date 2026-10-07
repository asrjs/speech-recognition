"""No-browser regression checks for automatic receipt failure handling."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('auto_browser', Path(__file__).with_name('orukeet-auto-browser.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AutoReceiptTests(unittest.TestCase):
    def invoke(self, root, output):
        argv = ['auto', '--model-root', str(root / 'model'), '--audio', str(root / 'audio.wav'),
                '--oracle', str(root / 'oracle.json'), '--output', str(output)]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
            return runner.main()

    def test_missing_model_writes_fail_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            self.assertEqual(self.invoke(root, output), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'FAIL_AUTO_BROWSER')
            self.assertIn('Missing/empty/symlink', result['error'])

    def test_invalid_oracle_writes_fail_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            runner.write_new_text(root / 'oracle.json', 'not JSON')
            with patch.object(runner.helpers, 'local_model_inputs', return_value=[]):
                self.assertEqual(self.invoke(root, output), 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'FAIL_AUTO_BROWSER')

    def test_existing_receipt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            runner.write_new_text(output, 'original')
            with self.assertRaises(FileExistsError):
                self.invoke(root, output)
            self.assertEqual(output.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
