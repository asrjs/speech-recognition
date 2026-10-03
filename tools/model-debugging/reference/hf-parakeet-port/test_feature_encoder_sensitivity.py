"""Weight-free preflight guards, not encoder numerical acceptance."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import probe_feature_encoder_sensitivity as probe


class SensitivityPreflightTests(unittest.TestCase):
    def arguments(self, output):
        return ['probe', '--capture', 'missing-capture.gz', '--encoder', 'missing.onnx',
            '--encoder-data', 'missing.onnx.data', '--capture-sha256', '1' * 64,
            '--encoder-sha256', '1' * 64, '--encoder-data-sha256', '1' * 64,
            '--sample-id', 'fixture', '--output', str(output)]

    def test_existing_output_preserved_before_loading_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            output.write_text('preserve', encoding='utf-8')
            with patch('sys.argv', self.arguments(output)), patch.object(probe, 'range_hash') as hashing:
                with self.assertRaises(FileExistsError):
                    probe.main()
                hashing.assert_not_called()
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')

    def test_hash_mismatch_stops_before_capture_node_or_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            with patch('sys.argv', self.arguments(output)), \
                    patch.object(probe, 'range_hash', return_value='0' * 64), \
                    patch.object(probe, 'read_capture') as capture, \
                    patch.object(probe.subprocess, 'run') as node, \
                    patch.object(probe.ort, 'InferenceSession') as encoder:
                with self.assertRaisesRegex(ValueError, 'Pinned source hash differs'):
                    probe.main()
                capture.assert_not_called()
                node.assert_not_called()
                encoder.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
