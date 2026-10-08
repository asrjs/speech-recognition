"""Orukeet browser adapter controls: simulated receipts, never GPU acceptance."""
import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

from test_port_manifest import ROOT, manifest_with, oracle_tokens, port_manifest

spec = importlib.util.spec_from_file_location('orukeet_browser_adapter_test',
    ROOT / 'tools/model-debugging/reference/orukeet/browser_manifest.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

PCM = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'


class OrukeetBrowserManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='browser-manifest-')
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'receipt.json'
        self.manifest = manifest_with([])
        self.manifest['modelId'] = 'oruk/orukeet'
        self.fixture = self.manifest['fixtures'][0]
        self.fixture['effectivePcmSha256'] = PCM
        self.run = dict(id='browser-jfk', runner='browser-oracle', adapter='orukeet-auto-browser',
                        backend='webgpu-hybrid', precision='fp32', repeat=2, fixture='jfk-short',
                        compare=['token_ids', 'text'], url='https://127.0.0.1:8767/orukeet-smoke.html',
                        runnerSha256=adapter.smoke.helpers.digest(adapter.RUNNER))
        self.consumed = {str(Path(item['path']).resolve()): item['sha256']
                         for item in self.manifest['localArtifacts']['files']}
        tokens, text = oracle_tokens()
        row = dict(text=text, tokenIds=tokens,
                   metrics=dict(encoderBackend='webgpu', decoderBackend='wasm',
                                jointBackend='wasm', preprocessorBackend='js'))
        self.receipt = dict(status='PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE',
            inputs=[dict(path=path, sha256=sha) for path, sha in self.consumed.items()],
            audio_sha256=self.fixture['sha256'], oracle_sha256=self.fixture['oracle']['sha256'],
            runner_sha256=self.run['runnerSha256'], browser_closed=True, browser_version='test',
            url=self.run['url'] + '?' + urlencode(dict(auto=1, backend='webgpu-hybrid', precision='fp32',
                repeat=2, audioStrategy='wav', cancel=0, controller=0)),
            page_errors=[], blocked_requests=[], code_capture_errors=[],
            served_code=[dict(url='https://127.0.0.1:8767/src/orukeet-smoke.js', bytes=1, status=200,
                             sha256='a' * 64, capture='route-fulfilled-body'),
                         dict(url='https://127.0.0.1:8767/orukeet-smoke.html', bytes=1, status=200,
                             sha256='d' * 64, capture='route-fulfilled-body'),
                         dict(url='https://127.0.0.1:8767/onnxruntime-web/dist/ort.wasm', bytes=1, status=200,
                             sha256='b' * 64, capture='route-fulfilled-body')],
            result=dict(status='PASS_ORUKEET_SMOKE', disposed=True, audioSamples=176000,
                effectivePcm=dict(sha256=PCM, sampleRate=16000, channels=1, samples=176000,
                                  validSamples=176000, rightPaddingSamples=0, encoding='f32le'),
                selection=dict(encoderQuant='fp32', decoderQuant='fp32'), transcripts=[row, copy.deepcopy(row)]))
        self.contract_path = Path(self.directory.name) / 'code-contract.json'
        self.contract_path.write_text(json.dumps(dict(schemaVersion=1, files=[
            dict(path=adapter.urlparse(row['url']).path, bytes=row['bytes'], sha256=row['sha256'])
            for row in self.receipt['served_code']])), encoding='utf-8')
        self.manifest['browserCodeContract'] = dict(path=str(self.contract_path),
                                                   sha256=port_manifest.digest(self.contract_path))
        self.command, self.consumed = adapter.prepare(self.manifest, self.run, self.fixture,
                                                     {'browserOutput': str(self.output)})

    def compare(self):
        self.output.write_text(json.dumps(self.receipt), encoding='utf-8')
        return adapter.compare(self.manifest, self.run, self.fixture, self.output, self.consumed)

    def test_valid_simulated_receipt_matches_all_gates(self):
        result = self.compare()
        self.assertEqual(result['status'], 'PASS_BROWSER_ORACLE_MATCH')
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['browserReceiptSha256'], port_manifest.digest(self.output))

    def test_explicit_empty_silence_outputs_still_require_every_gate(self):
        oracle = json.loads((ROOT / self.fixture['oracle']['path']).read_text(encoding='utf-8'))
        oracle['result'].update(token_ids=[], text='')
        silence_oracle = Path(self.directory.name) / 'silence-oracle.json'
        silence_oracle.write_text(json.dumps(oracle), encoding='utf-8')
        self.fixture['oracle'].update(path=str(silence_oracle),
                                     sha256=port_manifest.digest(silence_oracle))
        self.receipt['oracle_sha256'] = self.fixture['oracle']['sha256']
        for row in self.receipt['result']['transcripts']:
            row.update(tokenIds=[], text='')
        self.assertTrue(all(self.compare()['checks'].values()))

        pristine = copy.deepcopy(self.receipt)
        mutations = [
            lambda r: r['result']['transcripts'].clear(),
            lambda r: r['result']['transcripts'][1].pop('tokenIds'),
            lambda r: r['result']['transcripts'][1].update(tokenIds=None),
            lambda r: r['result']['transcripts'][1].update(tokenIds=[1]),
            lambda r: r['result']['transcripts'][1].update(text='hallucination'),
            lambda r: r.update(served_code=[]),
            lambda r: r['served_code'][0].update(sha256='c' * 64),
            lambda r: r['result'].update(disposed=False),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.receipt = copy.deepcopy(pristine)
                mutate(self.receipt)
                self.assertEqual(self.compare()['status'], 'FAILED_BROWSER_ORACLE')

    def test_pass_label_cannot_hide_mismatches(self):
        pristine = copy.deepcopy(self.receipt)
        mutations = [
            lambda r: r.update(audio_sha256='0' * 64),
            lambda r: r.update(oracle_sha256='0' * 64),
            lambda r: r.update(runner_sha256='0' * 64),
            lambda r: r['inputs'].pop(),
            lambda r: r['inputs'].append(r['inputs'][0]),
            lambda r: r['inputs'][0].update(sha256='0' * 64),
            lambda r: r.update(browser_closed=False),
            lambda r: r['result'].update(disposed=False),
            lambda r: r['result']['transcripts'].pop(),
            lambda r: r['result']['transcripts'][1].update(text='different'),
            lambda r: r['result']['transcripts'][1].update(tokenIds=[True]),
            lambda r: r['result']['transcripts'][1]['metrics'].update(encoderBackend='wasm'),
            lambda r: r['result']['selection'].update(encoderQuant='fp16'),
            lambda r: r['result']['effectivePcm'].update(sha256='0' * 64),
            lambda r: r.update(page_errors=['error']),
            lambda r: r.update(blocked_requests=['https://external.invalid']),
            lambda r: r.update(code_capture_errors=['error']),
            lambda r: r.update(served_code=[]),
            lambda r: r['served_code'][0].update(sha256='c' * 64),
            lambda r: r['served_code'][1].update(sha256='c' * 64),
            lambda r: r.update(url=r['url'].replace('repeat=2', 'repeat=1')),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.receipt = copy.deepcopy(pristine)
                mutate(self.receipt)
                self.assertEqual(self.compare()['status'], 'FAILED_BROWSER_ORACLE')

    def test_requested_invalid_contract_fails_before_subprocess(self):
        for key, value in (('adapter', 'unknown'), ('precision', 'fp16'), ('repeat', 1),
                           ('runnerSha256', '0' * 64), ('fixture', 'missing'),
                           ('compare', ['text']), ('url', 'https://example.com/orukeet-smoke.html')):
            with self.subTest(key=key), patch.object(port_manifest.subprocess, 'run') as child:
                run = {**self.run, key: value}
                with self.assertRaises(ValueError):
                    port_manifest.run_case(self.manifest, run, self.output)
                child.assert_not_called()

    def test_manifest_dispatch_compares_new_child_receipt(self):
        child_output = Path(f'{self.output}.browser.json')
        def child(command, **kwargs):
            self.assertEqual(command[0], adapter.sys.executable)
            self.assertEqual(Path(command[1]), adapter.RUNNER)
            self.assertIn('--expected-pcm-sha256', command)
            child_output.write_text(json.dumps(self.receipt), encoding='utf-8')
            return subprocess.CompletedProcess(command, 0, '{}', '')
        with patch.object(port_manifest.subprocess, 'run', side_effect=child):
            result = port_manifest.run_case(self.manifest, self.run, self.output)
        self.assertEqual(result['status'], 'PASS_BROWSER_ORACLE_MATCH')
        self.assertTrue(result['comparesOracle'])

    def test_changed_code_contract_stops_before_browser_launch(self):
        self.contract_path.write_text('{}', encoding='utf-8')
        with patch.object(port_manifest.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'browser code contract'):
                port_manifest.run_case(self.manifest, self.run, self.output)
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
