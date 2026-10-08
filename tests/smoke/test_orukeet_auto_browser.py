"""No-browser regression checks for automatic receipt failure handling."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('auto_browser', Path(__file__).with_name('orukeet-auto-browser.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AutoReceiptTests(unittest.TestCase):
    def test_controller_requires_coverage_final_reset_and_actual_backends(self):
        metrics = dict(encoderBackend='webgpu', decoderBackend='wasm', jointBackend='wasm', preprocessorBackend='js')
        controller = dict(scope='chunk-fed-windowed-offline-asr', finalized=True, pushedSamples=64000,
            finalText='hello', resetState=dict(currentFrame=0, isFinalized=False, text=''),
            chunkRanges=[dict(startFrame=0, endFrame=32000), dict(startFrame=32000, endFrame=64000)],
            updates=[dict(kind='partial', text='hello'), dict(kind='final', text='hello')],
            requests=[dict(startFrame=0, endFrame=end, samples=end, sampleRate=16000,
                pcmSha256='a' * 64, text='hello', tokenIds=[1], metrics=metrics) for end in (32000, 64000)])
        result = dict(audioSamples=64000, controller=controller)
        self.assertTrue(runner.check_controller(result, True, 'webgpu-hybrid', 'hello'))
        self.assertTrue(runner.check_controller({}, False, 'webgpu-hybrid', 'hello'))
        self.assertFalse(runner.check_controller({}, True, 'webgpu-hybrid', 'hello'))
        for field, value in [('finalText', 'wrong'), ('finalized', False), ('pushedSamples', 1),
                             ('updates', []), ('requests', []), ('resetState', {})]:
            changed = copy.deepcopy(result)
            changed['controller'][field] = value
            self.assertFalse(runner.check_controller(changed, True, 'webgpu-hybrid', 'hello'), field)
        changed = copy.deepcopy(result)
        changed['controller']['chunkRanges'][1]['startFrame'] += 1
        self.assertFalse(runner.check_controller(changed, True, 'webgpu-hybrid', 'hello'))
        changed = copy.deepcopy(result)
        changed['controller']['requests'][0]['metrics'] = {}
        self.assertFalse(runner.check_controller(changed, True, 'webgpu-hybrid', 'hello'))
        changed = copy.deepcopy(result)
        changed['controller']['updates'][-1]['text'] = 'wrong'
        self.assertFalse(runner.check_controller(changed, True, 'webgpu-hybrid', 'hello'))

    def test_cancellation_requires_partial_decode_rejection_without_complete(self):
        cancellation = dict(requested=True, rejected=True, completeEmitted=False,
                            errorName='PipelineAbortedError', stageId='decode', completedUnits=1, totalUnits=10)
        self.assertTrue(runner.check_cancellation(dict(cancellation=cancellation), True))
        for key, value in [('requested', False), ('rejected', False), ('completeEmitted', True),
                           ('errorName', 'OtherError'), ('stageId', 'encoder'),
                           ('completedUnits', 0), ('completedUnits', 10), ('completedUnits', True),
                           ('totalUnits', 0)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(runner.check_cancellation(dict(cancellation=dict(cancellation, **{key: value})), True))
        self.assertFalse(runner.check_cancellation({}, True))
        self.assertTrue(runner.check_cancellation({}, False))

    def test_cancel_option_requires_successful_reuse_repeat(self):
        argv = ['auto', '--model-root', 'unused', '--audio', 'unused.wav', '--oracle', 'unused.json',
                '--output', 'unused-output.json', '--verify-cancellation', '--repeat', '1']
        stderr = io.StringIO()
        with patch('sys.argv', argv), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            runner.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn('requires at least two successful repeats', stderr.getvalue())

    def test_code_binding_allows_empty_type_module_not_empty_runtime(self):
        code = [dict(url=url, status=200, bytes=size, sha256='a' * 64, capture='route-fulfilled-body')
                for url, size in [('https://127.0.0.1/src/orukeet-smoke.js', 10),
                                  ('https://127.0.0.1/onnxruntime-web/dist/runtime.wasm', 10),
                                  ('https://127.0.0.1/types.ts', 0)]]
        receipt = dict(served_code=code, code_capture_errors=[])
        self.assertTrue(runner.check_code_bindings(receipt))
        for index in (0, 1):
            with self.subTest(index=index):
                changed = [dict(item) for item in code]
                changed[index]['bytes'] = 0
                self.assertFalse(runner.check_code_bindings(dict(receipt, served_code=changed)))
        self.assertFalse(runner.check_code_bindings(dict(receipt, code_capture_errors=['missing body'])))
        changed = [dict(item) for item in code]
        changed[2]['status'] = 500
        self.assertFalse(runner.check_code_bindings(dict(receipt, served_code=changed)))

    def test_code_capture_binds_fulfilled_bytes_and_releases_response(self):
        route = Mock()
        route.request.url = 'https://127.0.0.1/runtime.wasm'
        response = route.fetch.return_value
        response.status = 200
        response.body.return_value = b'actual runtime bytes'
        receipt = dict(served_code=[], code_capture_errors=[])
        runner.capture_code_route(route, receipt)
        route.fulfill.assert_called_once_with(response=response, body=b'actual runtime bytes')
        route.fetch.assert_called_once_with(max_redirects=0, timeout=30000)
        response.dispose.assert_called_once_with()
        self.assertEqual(receipt['served_code'][0]['sha256'], runner.hashlib.sha256(b'actual runtime bytes').hexdigest())
        self.assertEqual(receipt['code_capture_errors'], [])

    def test_failed_code_capture_aborts_and_records_failure(self):
        route = Mock()
        route.request.url = 'https://127.0.0.1/runtime.wasm'
        route.fetch.side_effect = RuntimeError('network failed')
        receipt = dict(served_code=[], code_capture_errors=[])
        runner.capture_code_route(route, receipt)
        route.abort.assert_called_once_with()
        route.fulfill.assert_not_called()
        self.assertEqual(receipt['served_code'], [])
        self.assertIn('network failed', receipt['code_capture_errors'][0]['error'])

    def test_execution_requires_requested_repeats_and_component_backends(self):
        item = dict(metrics=dict(encoderBackend='webgpu', decoderBackend='wasm',
                                 jointBackend='wasm', preprocessorBackend='js'))
        result = dict(transcripts=[item, item])
        self.assertTrue(runner.check_execution(result, 'webgpu-hybrid', 2))
        self.assertFalse(runner.check_execution(result, 'webgpu-hybrid', 1))
        self.assertFalse(runner.check_execution(result, 'wasm', 2))
        for key in item['metrics']:
            with self.subTest(key=key):
                changed = dict(metrics=dict(item['metrics'], **{key: 'wrong'}))
                self.assertFalse(runner.check_execution(dict(transcripts=[changed]), 'webgpu-hybrid', 1))
        self.assertFalse(runner.check_execution(dict(transcripts=[{}]), 'webgpu-hybrid', 1))

    def test_pcm_identity_requires_exact_hash_and_metadata(self):
        result = dict(audioSamples=3, effectivePcm=dict(sha256='a' * 64, sampleRate=16000,
            channels=1, samples=3, validSamples=3, rightPaddingSamples=0, encoding='f32le'))
        self.assertTrue(runner.check_pcm(result, 'a' * 64))
        self.assertFalse(runner.check_pcm(result, 'b' * 64))
        for key, value in [('sha256', 'invalid'), ('sampleRate', 8000), ('channels', 2),
                           ('samples', 4), ('validSamples', 2), ('rightPaddingSamples', 1),
                           ('encoding', 'pcm16')]:
            with self.subTest(key=key):
                changed = dict(result, effectivePcm=dict(result['effectivePcm'], **{key: value}))
                self.assertFalse(runner.check_pcm(changed, 'a' * 64))

    def test_missing_pcm_identity_fails_closed(self):
        self.assertFalse(runner.check_pcm(dict(audioSamples=3), None))

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
