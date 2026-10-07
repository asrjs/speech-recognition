"""Regressions for the port-manifest harness: it must fail closed, not pass silently.

The artifact set is a tiny synthetic bundle with the real file names, including
an external-data sidecar, so digest-binding behaviour is exercised without
hashing multi-gigabyte graphs on every test.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'model-debugging' / 'scripts'))

import port_manifest  # noqa: E402


STUB_MATCH = (
    "import json;"
    "print(json.dumps({'host': {'runtime': 'node'},"
    " 'backend': {'versions': {'node': 'stub'}},"
    " 'transcript': {'canonical': {'text': 'hello world', 'tokens': [{'id': 7}],"
    " 'warnings': [], 'meta': {'metrics': {'totalMs': 1.0}}}}}))"
)
STUB_MISMATCH = STUB_MATCH.replace("'hello world'", "'wrong text'").replace("{'id': 7}", "{'id': 9}")


BUNDLE_FILES = ('encoder-model.onnx', 'encoder-model.onnx.data', 'decoder_joint-model.onnx', 'vocab.txt')
BUNDLE = Path(tempfile.mkdtemp(prefix='port-manifest-bundle-'))
for _name in BUNDLE_FILES:
    (BUNDLE / _name).write_bytes(f'synthetic {_name}'.encode())
(BUNDLE / 'orukeet-v0.1.0.nemo').write_bytes(b'synthetic original weights')


def manifest_with(command, compare=('token_ids', 'text'), fixture='jfk-short'):
    fixtures = ROOT / 'tools/data/fixtures/audio/jfk-short.wav'
    oracle = ROOT / 'tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json'
    weights = BUNDLE / 'orukeet-v0.1.0.nemo'
    return {
        'schemaVersion': 1,
        'modelId': 'stub/model',
        'family': 'nemo-tdt',
        'originalWeights': {
            'path': str(weights),
            'sha256': port_manifest.digest(weights),
        },
        'localArtifacts': {
            'root': str(BUNDLE),
            'layout': 'nemo-tdt',
            'files': [
                {'path': str(BUNDLE / name), 'sha256': port_manifest.digest(BUNDLE / name)}
                for name in BUNDLE_FILES
            ],
        },
        'fixtures': [{
            'id': fixture,
            'path': str(fixtures.relative_to(ROOT)).replace('\\', '/'),
            'sha256': port_manifest.digest(fixtures),
            'oracle': {
                'path': str(oracle.relative_to(ROOT)).replace('\\', '/'),
                'sha256': port_manifest.digest(oracle),
                'field': 'result',
            },
        }],
        'runs': [{
            'id': 'stub',
            'runner': 'node',
            'backend': 'native-cpu',
            'fixture': fixture,
            'compare': list(compare) if compare is not None else [],
            'command': list(command),
        }],
    }


def oracle_tokens():
    document = json.loads(
        (ROOT / 'tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json')
        .read_text(encoding='utf-8')
    )['result']
    return document['token_ids'], document['text']


class PortManifestDigestContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='manifest-digest-')
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / 'input.bin'
        self.input.write_bytes(b'bound input')

    def test_unbound_digest_fails_before_reading_file(self):
        for expected in (None, '', ' ', 42, False, 'f' * 63, 'f' * 65, 'z' * 64):
            with self.subTest(expected=expected), patch.object(port_manifest, 'digest') as digest:
                with self.assertRaisesRegex(ValueError, 'Missing or invalid SHA-256'):
                    port_manifest.require_file(str(self.input), expected, 'local artifact')
                digest.assert_not_called()

    def test_valid_uppercase_digest_binds_the_same_bytes(self):
        expected = port_manifest.digest(self.input)
        bound = port_manifest.require_file(str(self.input), expected.upper(), 'local artifact')
        self.assertEqual(bound['sha256'], expected)

    def test_each_local_manifest_role_requires_a_recorded_digest(self):
        for role in ('weights', 'artifact', 'config', 'fixture', 'oracle'):
            with self.subTest(role=role):
                manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
                manifest['runtimeConfig'] = {
                    'path': str(self.input), 'sha256': port_manifest.digest(self.input),
                }
                target = {
                    'weights': manifest['originalWeights'],
                    'artifact': manifest['localArtifacts']['files'][0],
                    'config': manifest['runtimeConfig'],
                    'fixture': manifest['fixtures'][0],
                    'oracle': manifest['fixtures'][0]['oracle'],
                }[role]
                del target['sha256']
                with self.assertRaisesRegex(ValueError, 'Missing or invalid SHA-256'):
                    port_manifest.check_manifest(manifest)

    def test_missing_artifact_digest_stops_cli_before_runner_or_receipt(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        del manifest['localArtifacts']['files'][0]['sha256']
        source = Path(self.directory.name) / 'manifest.json'
        source.write_text(json.dumps(manifest), encoding='utf-8')
        output = Path(self.directory.name) / 'receipt.json'
        argv = ['port_manifest.py', '--manifest', str(source), '--run', 'stub',
                '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(port_manifest.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Missing or invalid SHA-256'):
                port_manifest.main()
            run.assert_not_called()
        self.assertFalse(output.exists())


class PortManifestComparisonContract(unittest.TestCase):
    def run_case(self, manifest):
        return port_manifest.run_case(manifest, manifest['runs'][0], Path('unused.json'))

    def test_matching_output_passes_with_declared_comparison(self):
        tokens, text = oracle_tokens()
        stub = (
            'import json;'
            f'print(json.dumps({{"host": {{"runtime": "node"}},'
            f' "backend": {{"versions": {{}}}},'
            f' "transcript": {{"canonical": {{"text": {json.dumps(text)}, '
            f'"tokens": {[{"id": t} for t in tokens]},'
            f' "warnings": [], "meta": {{"metrics": {{"totalMs": 1.0}}}}}}}}}}))'
        )
        result = self.run_case(manifest_with([sys.executable, '-c', stub]))
        self.assertEqual(result['status'], 'PASS_ORACLE_MATCH', result)
        self.assertEqual(result['checks'], {'token_ids': True, 'text': True})

    def test_mismatched_output_fails(self):
        result = self.run_case(manifest_with([sys.executable, '-c', STUB_MISMATCH]))
        self.assertEqual(result['status'], 'FAILED_ORACLE', result)
        self.assertEqual(result['checks'], {'token_ids': False, 'text': False})

    def test_missing_comparison_field_fails_before_running(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        del manifest['runs'][0]['compare']
        with self.assertRaises(ValueError) as error:
            self.run_case(manifest)
        self.assertIn('declares no comparison fields', str(error.exception))

    def test_empty_comparison_field_fails_before_running(self):
        with self.assertRaises(ValueError) as error:
            self.run_case(manifest_with([sys.executable, '-c', STUB_MATCH], compare=[]))
        self.assertIn('declares no comparison fields', str(error.exception))

    def test_unknown_comparison_field_fails_before_running(self):
        with self.assertRaises(ValueError) as error:
            self.run_case(manifest_with([sys.executable, '-c', STUB_MATCH], compare=['confidence']))
        self.assertIn('unknown comparison fields', str(error.exception))

    def test_native_run_without_fixture_fails_before_running(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        del manifest['runs'][0]['fixture']
        with self.assertRaises(ValueError) as error:
            self.run_case(manifest)
        self.assertIn('declares no fixture', str(error.exception))

    def test_identity_binding_rejects_a_changed_sidecar(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        sidecar = next(
            item for item in manifest['localArtifacts']['files']
            if item['path'].endswith('encoder-model.onnx.data')
        )
        sidecar['sha256'] = '1' * 64
        with self.assertRaises(ValueError) as error:
            port_manifest.check_manifest(manifest)
        self.assertIn('encoder-model.onnx.data', str(error.exception))
        self.assertIn('SHA-256 mismatch', str(error.exception))

    def test_identity_binding_rejects_a_wrong_graph_digest(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        graph = next(
            item for item in manifest['localArtifacts']['files']
            if item['path'].endswith('decoder_joint-model.onnx')
        )
        graph['sha256'] = '2' * 64
        with self.assertRaises(ValueError) as error:
            port_manifest.check_manifest(manifest)
        self.assertIn('decoder_joint-model.onnx', str(error.exception))

    def test_identity_binding_rejects_a_changed_oracle(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        manifest['fixtures'][0]['oracle']['sha256'] = '3' * 64
        with self.assertRaises(ValueError) as error:
            port_manifest.check_manifest(manifest)
        self.assertIn('oracle', str(error.exception))

    def test_identity_binding_rejects_a_wrong_digest(self):
        manifest = manifest_with([sys.executable, '-c', STUB_MATCH])
        manifest['fixtures'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError) as error:
            port_manifest.check_manifest(manifest)
        self.assertIn('SHA-256 mismatch', str(error.exception))


if __name__ == '__main__':
    unittest.main()
