"""Orukeet-owned adapter: existing Chrome inference runner, not a new decoder."""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / 'tests/smoke/orukeet-auto-browser.py'
ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('orukeet_auto_manifest_runner', RUNNER)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def compare_capture_oracle(run, oracle, original_weights_sha256):
    """One saved capture session vs original NeMo output on that exact PCM.

    This is output-only evidence, not stage numerical parity. File/receipt
    digests and the parent capture lifecycle must be checked by the caller.
    """
    meta, output, capture = (oracle.get(key, {}) for key in ('meta', 'result', 'stage_capture'))
    actual, frozen = capture.get('model_input', {}), meta.get('frozen_pcm') or {}
    samples, digest = run.get('capturedSamples'), run.get('capturedPcmSha256')
    tokens = output.get('token_ids')
    controls = capture.get('instrumentation_control', {})
    control_keys = ('baseline_tokens_match', 'baseline_text_match', 'baseline_frames_match',
                    'after_tokens_match', 'after_text_match', 'after_durations_match')
    checks = {
        'original_model': meta.get('model') == 'oruk/orukeet' and meta.get('device') == 'cpu'
            and meta.get('batch_size') == 1 and meta.get('input_mode') == 'frozen-pcm',
        'original_weights': isinstance(original_weights_sha256, str)
            and re.fullmatch('[0-9a-f]{64}', original_weights_sha256) is not None
            and meta.get('weight_sha256') == original_weights_sha256,
        'frozen_input': frozen.get('sha256') == digest and frozen.get('samples') == samples
            and frozen.get('sample_rate') == 16000 and frozen.get('channels') == 1
            and frozen.get('encoding') == 'f32le' and frozen.get('dtype') == 'float32',
        'effective_pcm': type(samples) is int and samples >= 16000
            and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None
            and actual.get('boundary') == 'preprocessor_input_signal'
            and actual.get('sha256') == digest and actual.get('sample_rate') == 16000
            and actual.get('channels') == 1 and actual.get('dtype') == 'float32'
            and actual.get('encoding') == 'f32le' and actual.get('shape') == [1, samples]
            and actual.get('samples') == samples and actual.get('valid_samples') == samples
            and actual.get('right_padding_samples') == 0,
        'instrumentation': all(controls.get(key) is True for key in control_keys),
        'token_ids': isinstance(tokens, list) and bool(tokens)
            and all(type(token) is int and token >= 0 for token in tokens)
            and tokens == run.get('tokenIds'),
        'text': isinstance(output.get('text'), str) and bool(output['text'].strip())
            and output['text'] == run.get('text'),
    }
    return dict(status='PASS_CAPTURE_ORACLE_OUTPUT' if all(checks.values()) else 'FAIL_CAPTURE_ORACLE_OUTPUT',
                scope='exact-captured-PCM original-engine token/text; not numerical parity', checks=checks,
                tokenCount=len(tokens) if isinstance(tokens, list) else None)


def code_contract(manifest):
    binding = manifest.get('browserCodeContract', {})
    path = (ROOT / binding.get('path', '')).resolve()
    if not path.is_file() or smoke.helpers.digest(path) != binding.get('sha256'):
        raise ValueError('Missing or changed manifest-bound browser code contract')
    contract = json.loads(path.read_text(encoding='utf-8'))
    rows = contract.get('files')
    if contract.get('schemaVersion') != 1 or not isinstance(rows, list) or not rows:
        raise ValueError('Invalid browser code contract')
    expected = {}
    for row in rows:
        path = row.get('path')
        if (not isinstance(path, str) or not path.startswith('/') or path in expected
                or type(row.get('bytes')) is not int or row['bytes'] < 0
                or not isinstance(row.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise ValueError('Invalid or duplicate browser code contract entry')
        expected[path] = (row['bytes'], row['sha256'])
    return expected


def prepare(manifest, run, fixture, context):
    """Validate explicit model scope and bind the existing runner before launch."""
    if (manifest['modelId'] != 'oruk/orukeet' or manifest['family'] != 'nemo-tdt'
            or manifest['localArtifacts']['layout'] != 'nemo-tdt'
            or run.get('precision') != 'fp32' or fixture['oracle']['field'] != 'result'):
        raise ValueError('Orukeet browser adapter requires its fp32 NeMo TDT/result contract')
    if set(run['compare']) != {'token_ids', 'text'}:
        raise ValueError('Browser oracle requires both token_ids and text')
    pcm = fixture.get('effectivePcmSha256')
    if not isinstance(pcm, str) or re.fullmatch('[0-9a-f]{64}', pcm) is None:
        raise ValueError('Browser oracle requires recorded effectivePcmSha256')
    runner_hash = run.get('runnerSha256')
    if runner_hash != smoke.helpers.digest(RUNNER):
        raise ValueError('Browser runner SHA-256 mismatch or missing binding')
    code_contract(manifest)
    oracle = json.loads((ROOT / fixture['oracle']['path']).read_text(encoding='utf-8'))
    if (oracle.get('meta', {}).get('model') != manifest['modelId']
            or oracle.get('meta', {}).get('audio_sha256') != fixture['sha256'].lower()):
        raise ValueError('Official oracle model/audio binding mismatch')
    repeat = run.get('repeat')
    if type(repeat) is not int or not 2 <= repeat <= 5:
        raise ValueError('Browser oracle requires two to five repeats')
    if run['backend'] not in ('webgpu-hybrid', 'wasm'):
        raise ValueError('Unsupported explicit Orukeet browser backend')
    parsed = urlparse(run.get('url', ''))
    if (parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.query or parsed.fragment or not parsed.path.endswith('/orukeet-smoke.html')):
        raise ValueError('Browser oracle requires a plain loopback Orukeet page URL')
    root = (ROOT / manifest['localArtifacts']['root']).resolve()
    declared = {(ROOT / item['path']).resolve(): item['sha256'].lower()
                for item in manifest['localArtifacts']['files']}
    consumed = smoke.helpers.local_model_inputs(root, 'fp32', 'nemo-tdt')
    if any(path.resolve() not in declared for path in consumed):
        raise ValueError('Browser-consumed artifact is not manifest-bound')
    command = [sys.executable, str(RUNNER), '--url', run['url'], '--model-root', str(root),
               '--audio', str((ROOT / fixture['path']).resolve()),
               '--oracle', str((ROOT / fixture['oracle']['path']).resolve()),
               '--backend', run['backend'], '--precision', 'fp32', '--repeat', str(repeat),
               '--expected-pcm-sha256', pcm, '--output', context['browserOutput']]
    return command, {str(path.resolve()): declared[path.resolve()] for path in consumed}


def compare(manifest, run, fixture, receipt_path, consumed):
    """Recheck full outputs and identities; never trust a PASS label alone."""
    receipt_bytes = receipt_path.read_bytes()
    browser = json.loads(receipt_bytes)
    result = browser.get('result') or {}
    oracle = json.loads((ROOT / fixture['oracle']['path']).read_text(encoding='utf-8'))
    transcripts = result.get('transcripts') or []
    rows = browser.get('inputs') or []
    actual = {str(Path(item['path']).resolve()): item['sha256'] for item in rows}
    expected_code = code_contract(manifest)
    served = browser.get('served_code') or []
    served_paths = {urlparse(item['url']).path for item in served}
    query = dict(auto=1, backend=run['backend'], precision='fp32', repeat=run['repeat'],
                 audioStrategy='wav', cancel=0, controller=0)
    token_rows_valid = (isinstance(transcripts, list) and len(transcripts) == run['repeat']
                        and all(isinstance(item, dict) and isinstance(item.get('tokenIds'), list)
                                and all(type(token) is int and token >= 0 for token in item['tokenIds'])
                                for item in transcripts))
    checks = {
        'harness_status': browser.get('status') == 'PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE',
        'page_status': result.get('status') == 'PASS_ORUKEET_SMOKE',
        'artifact_identity': len(rows) == len(actual) and actual == consumed,
        'audio_identity': browser.get('audio_sha256') == fixture['sha256'].lower(),
        'oracle_identity': browser.get('oracle_sha256') == fixture['oracle']['sha256'].lower(),
        'runner_identity': browser.get('runner_sha256') == run['runnerSha256'] == smoke.helpers.digest(RUNNER),
        'adapter_identity': hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == ADAPTER_SHA256,
        'url_identity': browser.get('url') == run['url'] + '?' + urlencode(query),
        'selection': result.get('selection', {}).get('encoderQuant') == 'fp32'
                     and result.get('selection', {}).get('decoderQuant') == 'fp32',
        'pcm_identity': smoke.check_pcm(result, fixture['effectivePcmSha256']),
        'execution': token_rows_valid and smoke.check_execution(result, run['backend'], run['repeat']),
        'token_ids': token_rows_valid and all(item['tokenIds'] == oracle['result']['token_ids'] for item in transcripts),
        'text': token_rows_valid and all(item.get('text') == oracle['result']['text'] for item in transcripts),
        'code_binding': smoke.check_code_bindings(browser),
        'expected_code': served_paths == set(expected_code) and all(
            (item['bytes'], item['sha256']) == expected_code.get(urlparse(item['url']).path)
            for item in served),
        'no_page_errors': browser.get('page_errors') == [],
        'offline': browser.get('blocked_requests') == [],
        'disposed': result.get('disposed') is True,
        'browser_closed': browser.get('browser_closed') is True,
    }
    return {
        'id': run['id'], 'status': 'PASS_BROWSER_ORACLE_MATCH' if all(checks.values()) else 'FAILED_BROWSER_ORACLE',
        'scope': 'same-PCM repeated token/text browser smoke; not stage numerical parity',
        'comparesOracle': True, 'checks': checks, 'runner': 'browser-oracle',
        'adapter': run['adapter'], 'backend': run['backend'], 'fixture': fixture['id'],
        'browserVersion': browser.get('browser_version'), 'gpuAdapter': result.get('adapter'),
        'browserReceipt': str(receipt_path.resolve()),
        'browserReceiptSha256': hashlib.sha256(receipt_bytes).hexdigest(),
        'adapterSha256': ADAPTER_SHA256,
        'metrics': [item.get('metrics') for item in transcripts if isinstance(item, dict)],
    }
