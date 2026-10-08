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
    def timing_result(self):
        sources, requests, updates = [], [], []
        for index, start in enumerate((0, 32000)):
            item = dict(index=0, text='hello', startTime=0.25, endTime=1.0)
            local = dict(words=[dict(item)], segments=[dict(item)], sentences=[dict(item)],
                         tokens=[dict(item, id=1), dict(index=1, text='untimed', id=2)])
            shifted = copy.deepcopy(local)
            for items in shifted.values():
                for value in items:
                    for key in ('startTime', 'endTime'):
                        if key in value:
                            value[key] += start / 16000
            sources.append(dict(startFrame=start, endFrame=start + 32000,
                                sampleRate=16000, localTiming=local))
            requests.append(dict(startFrame=start, endFrame=start + 32000, sampleRate=16000))
            updates.append(dict(kind='partial', inferenceIndex=index,
                window=dict(startFrame=start, endFrame=start + 32000),
                canonicalTiming=shifted, partialTiming=copy.deepcopy(shifted)))
        empty = dict(words=None, segments=None, sentences=None, tokens=None)
        updates.append(dict(kind='final', inferenceIndex=None,
                            canonicalTiming=empty, partialTiming=copy.deepcopy(empty)))
        return dict(controller=dict(inferenceResults=sources, requests=requests, updates=updates))

    def test_timing_preserves_fields_and_nonzero_offsets_without_final_inference(self):
        self.assertTrue(runner.check_controller_timing(self.timing_result(), True, True))
        self.assertTrue(runner.check_controller_timing({}, False))
        self.assertFalse(runner.check_controller_timing({}, True))

    def test_timing_rejects_missing_corrupt_and_misattributed_records(self):
        mutations = [
            lambda c: c.pop('inferenceResults'),
            lambda c: c['inferenceResults'].pop(),
            lambda c: c['updates'][0].pop('inferenceIndex'),
            lambda c: c['updates'][1].update(inferenceIndex=True),
            lambda c: c['updates'][1].update(inferenceIndex=0),
            lambda c: c['updates'][1]['window'].update(startFrame=0),
            lambda c: c['updates'][0]['window'].update(startFrame=False),
            lambda c: c['updates'][1]['window'].update(endFrame=64000.0),
            lambda c: c['inferenceResults'][1].update(sampleRate=8000),
            lambda c: c['updates'][-1].update(inferenceIndex=1),
            lambda c: c['updates'][-1].update(kind='partial'),
            lambda c: c['updates'][0]['canonicalTiming'].pop('words'),
            lambda c: c['inferenceResults'][0]['localTiming']['tokens'][1].update(startTime=0),
            lambda c: c['updates'][1]['canonicalTiming']['words'][0].update(startTime=4.25),
            lambda c: c['updates'][1]['partialTiming']['sentences'][0].update(endTime=1),
            lambda c: c['updates'][1]['canonicalTiming']['tokens'][1].update(startTime=2, endTime=2),
            lambda c: c['updates'][1]['canonicalTiming']['tokens'][0].update(id=99),
            lambda c: c['updates'][1]['partialTiming']['tokens'][0].update(id=True),
            lambda c: c['inferenceResults'][0]['localTiming']['segments'][0].update(endTime=float('nan')),
            lambda c: c['inferenceResults'][0]['localTiming']['segments'][0].update(startTime=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                result = self.timing_result()
                mutate(result['controller'])
                self.assertFalse(runner.check_controller_timing(result, True, True))

    def test_timing_requires_observed_detail_and_optional_nonzero_origin(self):
        result = self.timing_result()
        for source in result['controller']['inferenceResults']:
            source['localTiming'] = dict(words=[], segments=[], sentences=[], tokens=[])
        for update in result['controller']['updates'][:-1]:
            update['canonicalTiming'] = update['partialTiming'] = dict(words=[], segments=[], sentences=[], tokens=[])
        self.assertFalse(runner.check_controller_timing(result, True))
        result = self.timing_result()
        result['controller']['inferenceResults'] = result['controller']['inferenceResults'][:1]
        result['controller']['requests'] = result['controller']['requests'][:1]
        result['controller']['updates'] = [result['controller']['updates'][0], result['controller']['updates'][-1]]
        self.assertTrue(runner.check_controller_timing(result, True))
        self.assertFalse(runner.check_controller_timing(result, True, True))

    def test_timing_accepts_json_integer_float_spelling_not_bool_substitution(self):
        result = self.timing_result()
        for key in ('canonicalTiming', 'partialTiming'):
            result['controller']['updates'][0][key]['words'][0]['endTime'] = 1
        self.assertTrue(runner.check_controller_timing(result, True))
        for key in ('canonicalTiming', 'partialTiming'):
            result['controller']['updates'][0][key]['tokens'][0]['id'] = True
        self.assertFalse(runner.check_controller_timing(result, True))

    def test_nonzero_timing_gate_rejects_untimed_nonzero_window(self):
        result = self.timing_result()
        empty = dict(words=[], segments=[], sentences=[], tokens=[])
        result['controller']['inferenceResults'][1]['localTiming'] = copy.deepcopy(empty)
        for key in ('canonicalTiming', 'partialTiming'):
            result['controller']['updates'][1][key] = copy.deepcopy(empty)
        self.assertTrue(runner.check_controller_timing(result, True))
        self.assertFalse(runner.check_controller_timing(result, True, True))

    def test_nonzero_option_requires_controller(self):
        argv = ['auto', '--model-root', 'unused', '--audio', 'unused.wav', '--oracle', 'unused.json',
                '--output', 'unused-output.json', '--require-controller-nonzero-origin']
        stderr = io.StringIO()
        with patch('sys.argv', argv), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            runner.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn('requires --verify-controller', stderr.getvalue())

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
        code.append(dict(url='https://127.0.0.1/orukeet-smoke.html', status=200,
                         bytes=10, sha256='a' * 64, capture='route-fulfilled-body'))
        receipt = dict(served_code=code, code_capture_errors=[])
        self.assertTrue(runner.check_code_bindings(receipt))
        for index in (0, 1, 3):
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

    def test_html_is_captured_even_with_auto_query(self):
        self.assertTrue(runner.is_code_response('https://127.0.0.1/orukeet-smoke.html?auto=1'))
        self.assertFalse(runner.is_code_response('https://127.0.0.1/favicon.ico'))

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
