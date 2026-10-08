"""Regressions for the port-manifest harness: it must fail closed, not pass silently.

The artifact set is a tiny synthetic bundle with the real file names, including
an external-data sidecar, so digest-binding behaviour is exercised without
hashing multi-gigabyte graphs on every test.
"""
import json
import os
import subprocess
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


class PortManifestOutputPreflight(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='manifest-output-', dir=ROOT / 'tests')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'manifest.json'
        self.output = self.root / 'receipt.json'
        self.manifest = manifest_with([sys.executable, '-c', STUB_MATCH])

    def assert_rejected_before_runner(self, output, exception, message, run_id='stub'):
        self.source.write_text(json.dumps(self.manifest), encoding='utf-8')
        argv = ['port_manifest.py', '--manifest', str(self.source), '--output', str(output)]
        if run_id is not None:
            argv += ['--run', run_id]
        with patch.object(sys, 'argv', argv), patch.object(
                port_manifest.subprocess, 'run', side_effect=AssertionError('Runner started before preflight')) as run:
            with self.assertRaisesRegex(exception, message):
                port_manifest.main()
            run.assert_not_called()

    def test_existing_receipt_rejected_before_native_inference(self):
        self.output.write_text('preserved receipt', encoding='utf-8')
        self.assert_rejected_before_runner(self.output, FileExistsError, 'existing output')
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'preserved receipt')

    def test_new_output_under_model_root_rejected_before_inference(self):
        output = BUNDLE / 'never-created-receipt.json'
        self.assert_rejected_before_runner(output, ValueError, 'protected input or source directory')
        self.assertFalse(output.exists())

    def test_manifest_itself_is_protected(self):
        self.assert_rejected_before_runner(self.source, ValueError, 'protected input')
        self.assertEqual(json.loads(self.source.read_text(encoding='utf-8')), self.manifest)

    def test_fixture_and_oracle_are_protected(self):
        fixture = self.manifest['fixtures'][0]
        for role in (fixture, fixture['oracle']):
            with self.subTest(path=role['path']):
                output = ROOT / role['path']
                before = port_manifest.digest(output)
                self.assert_rejected_before_runner(output, ValueError, 'protected input')
                self.assertEqual(port_manifest.digest(output), before)

    def test_weights_external_sidecar_and_runtime_config_are_protected(self):
        config = self.root / 'runtime-config.json'
        config.write_text('{}', encoding='utf-8')
        self.manifest['runtimeConfig'] = {'path': str(config), 'sha256': port_manifest.digest(config)}
        sidecar = self.manifest['localArtifacts']['files'][1]
        for role in (self.manifest['originalWeights'], sidecar, self.manifest['runtimeConfig']):
            with self.subTest(path=role['path']):
                output = Path(role['path'])
                before = port_manifest.digest(output)
                self.assert_rejected_before_runner(output, ValueError, 'protected input')
                self.assertEqual(port_manifest.digest(output), before)

    def test_existing_browser_child_receipt_rejected_before_launch(self):
        self.manifest['runs'][0]['runner'] = 'browser'
        child = Path(f'{self.output}.browser.json')
        child.write_text('preserved child', encoding='utf-8')
        self.assert_rejected_before_runner(self.output, FileExistsError, 'existing output')
        self.assertEqual(child.read_text(encoding='utf-8'), 'preserved child')
        self.assertFalse(self.output.exists())

    def test_multiple_browser_runs_cannot_share_one_child_receipt(self):
        first = self.manifest['runs'][0]
        first['runner'] = 'browser'
        self.manifest['runs'].append(dict(first, id='second'))
        self.assert_rejected_before_runner(self.output, ValueError, 'distinct', run_id=None)

    def test_listing_does_not_need_new_output(self):
        self.source.write_text(json.dumps(self.manifest), encoding='utf-8')
        with patch.object(sys, 'argv', ['port_manifest.py', '--manifest', str(self.source)]), \
                patch.object(port_manifest.subprocess, 'run') as run, patch('builtins.print'):
            self.assertEqual(port_manifest.main(), 0)
            run.assert_not_called()

    def test_fresh_output_runs_and_preserves_failed_oracle_receipt(self):
        self.source.write_text(json.dumps(self.manifest), encoding='utf-8')
        argv = ['port_manifest.py', '--manifest', str(self.source), '--run', 'stub',
                '--output', str(self.output)]
        # Real tiny child process, no model/runtime/download: exercises the
        # allowed path through comparison and exclusive final receipt write.
        with patch.object(sys, 'argv', argv), patch('builtins.print'):
            self.assertEqual(port_manifest.main(), 1)
        receipt = json.loads(self.output.read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'FAIL')
        self.assertEqual(receipt['results'][0]['status'], 'FAILED_ORACLE')

    def test_browser_child_output_is_absolute_from_callers_working_directory(self):
        self.manifest['runs'][0]['runner'] = 'browser'
        self.manifest['runs'][0]['command'] += ['--output', '{browserOutput}']
        self.source.write_text(json.dumps(self.manifest), encoding='utf-8')
        argv = ['port_manifest.py', '--manifest', str(self.source), '--run', 'stub',
                '--output', 'receipt.json']
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch.object(sys, 'argv', argv), patch('builtins.print'), patch.object(
                    port_manifest.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 1, '', 'controlled failure')) as run:
                self.assertEqual(port_manifest.main(), 1)
                self.assertEqual(Path(run.call_args.args[0][-1]),
                                 self.root.resolve() / 'receipt.json.browser.json')
                self.assertEqual(run.call_args.kwargs['cwd'], str(ROOT))
        finally:
            os.chdir(previous)
        self.assertTrue(self.output.is_file())


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
