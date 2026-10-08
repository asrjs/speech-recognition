"""Drive the real local-folder demo in installed Chrome and save a fresh receipt.

Requires an already-running examples/demo Vite server; never downloads models.
Smoke/text checks are not stage numerical parity.
"""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text

spec = importlib.util.spec_from_file_location('browser_smoke', Path(__file__).with_name('browser-demo-smoke.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def check_pcm(result, expected_hash):
    """Missing or inconsistent waveform metadata never proves same-input parity."""
    pcm = result.get('effectivePcm', {})
    return (isinstance(pcm.get('sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', pcm['sha256']) is not None
            and pcm.get('sampleRate') == 16000 and pcm.get('channels') == 1
            and type(pcm.get('samples')) is int and pcm['samples'] > 0
            and pcm['samples'] == result.get('audioSamples') == pcm.get('validSamples')
            and pcm.get('rightPaddingSamples') == 0 and pcm.get('encoding') == 'f32le'
            and (expected_hash is None or pcm['sha256'] == expected_hash))


def check_execution(result, backend, repeat):
    transcripts = result.get('transcripts')
    if not isinstance(transcripts, list) or len(transcripts) != repeat:
        return False
    encoder = 'webgpu' if backend == 'webgpu-hybrid' else 'wasm'
    return all(item.get('metrics', {}).get('encoderBackend') == encoder
               and item.get('metrics', {}).get('decoderBackend') == 'wasm'
               and item.get('metrics', {}).get('jointBackend') == 'wasm'
               and item.get('metrics', {}).get('preprocessorBackend') == 'js'
               for item in transcripts)


def check_cancellation(result, requested):
    if not requested:
        return True
    event = result.get('cancellation')
    if not isinstance(event, dict):
        return False
    done, total = event.get('completedUnits'), event.get('totalUnits')
    return (event.get('requested') is True and event.get('rejected') is True
            and event.get('completeEmitted') is False and event.get('errorName') == 'PipelineAbortedError'
            and event.get('stageId') == 'decode' and type(done) is int and type(total) is int
            and 0 < done < total)


def check_controller(result, requested, backend, expected_text):
    """Windowed offline coverage/lifecycle gate, not native streaming parity."""
    if not requested:
        return True
    event = result.get('controller')
    if not isinstance(event, dict):
        return False
    total = result.get('audioSamples')
    if (type(total) is not int or total <= 0 or event.get('scope') != 'chunk-fed-windowed-offline-asr'
            or event.get('finalized') is not True or event.get('pushedSamples') != total
            or not expected_text or event.get('finalText') != expected_text
            or event.get('resetState') != dict(currentFrame=0, isFinalized=False, text='')):
        return False
    chunks, updates, requests = (event.get(key) for key in ('chunkRanges', 'updates', 'requests'))
    if not all(isinstance(items, list) and len(items) >= 2 for items in (chunks, updates, requests)):
        return False
    cursor = 0
    for chunk in chunks:
        if (not isinstance(chunk, dict) or type(chunk.get('startFrame')) is not int
                or type(chunk.get('endFrame')) is not int or chunk['startFrame'] != cursor
                or not cursor < chunk['endFrame'] <= total):
            return False
        cursor = chunk['endFrame']
    if (cursor != total or not all(isinstance(item, dict) for item in updates)
            or not any(item.get('kind') == 'partial' for item in updates)
            or updates[-1].get('kind') != 'final' or updates[-1].get('text') != expected_text):
        return False
    for request in requests:
        if not isinstance(request, dict):
            return False
        start, end = request.get('startFrame'), request.get('endFrame')
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= total
                or request.get('samples') != end - start or request.get('sampleRate') != 16000
                or not isinstance(request.get('pcmSha256'), str)
                or re.fullmatch('[0-9a-f]{64}', request['pcmSha256']) is None
                or not isinstance(request.get('text'), str)
                or not isinstance(request.get('tokenIds'), list)
                or not all(type(token) is int and token >= 0 for token in request['tokenIds'])
                or not check_execution(dict(transcripts=[request]), backend, 1)):
            return False
    return True


def capture_code_route(route, receipt):
    """Bind the bytes actually fulfilled, without relying on Chrome inspector caches."""
    response = None
    try:
        response = route.fetch(max_redirects=0, timeout=30000)
        body = response.body()
        receipt['served_code'].append(dict(url=route.request.url, status=response.status,
            bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), capture='route-fulfilled-body'))
        route.fulfill(response=response, body=body)
    except Exception as error:
        receipt['code_capture_errors'].append(dict(url=route.request.url, error=str(error)))
        route.abort()
    finally:
        if response is not None:
            response.dispose()


def check_code_bindings(receipt):
    code = receipt.get('served_code', [])
    # Type-only TS modules legitimately compile to empty ESM. Bind their empty
    # bytes, but require actual executable entrypoint and ORT WASM bodies.
    return (not receipt.get('code_capture_errors')
            and any(urlparse(item['url']).path.endswith('/src/orukeet-smoke.js') and item['bytes'] > 0 for item in code)
            and any('/onnxruntime-web/dist/' in urlparse(item['url']).path
                    and urlparse(item['url']).path.endswith('.wasm') and item['bytes'] > 0 for item in code)
            and all(item['status'] == 200 and type(item['bytes']) is int and item['bytes'] >= 0
                    and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
                    and item['capture'] == 'route-fulfilled-body' for item in code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://127.0.0.1:8765/orukeet-smoke.html')
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['webgpu-hybrid', 'wasm'], default='webgpu-hybrid')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--repeat', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--timeout-ms', type=int, default=600000)
    parser.add_argument('--expected-pcm-sha256', help='Require this official effective f32le PCM digest; no WAV-hash substitution')
    parser.add_argument('--audio-strategy', choices=['wav', 'audiocontext'], default='wav')
    parser.add_argument('--verify-cancellation', action='store_true', help='Abort partial decode between two oracle-matched runs')
    parser.add_argument('--verify-controller', action='store_true', help='Verify chunk-fed windowed offline ASR coverage, final text and reset')
    args = parser.parse_args()
    if args.verify_cancellation and args.repeat < 2:
        parser.error('--verify-cancellation requires at least two successful repeats to prove reuse')
    parsed = urlparse(args.url)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.query or parsed.fragment:
        parser.error('--url must be a plain loopback HTTP(S) page URL')
    if args.expected_pcm_sha256 is not None and re.fullmatch('[0-9a-f]{64}', args.expected_pcm_sha256) is None:
        parser.error('--expected-pcm-sha256 must be a lowercase SHA-256')
    require_new_outputs([args.output], protected_inputs=[args.audio, args.oracle],
                        protected_directories=[args.model_root])
    url = args.url + '?' + urlencode(dict(auto=1, backend=args.backend, precision=args.precision, repeat=args.repeat, audioStrategy=args.audio_strategy, cancel=int(args.verify_cancellation), controller=int(args.verify_controller)))
    receipt = dict(status='FAIL_AUTO_BROWSER', url=url, scope='real-browser-smoke-and-oracle-output-only',
                   page_errors=[], console=[], blocked_requests=[], expected_pcm_sha256=args.expected_pcm_sha256,
                   cancellation_requested=args.verify_cancellation, controller_requested=args.verify_controller)
    try:
        inputs = helpers.local_model_inputs(args.model_root, args.precision, 'nemo-tdt')
        oracle = json.loads(args.oracle.read_text(encoding='utf-8'))
        audio_hash = helpers.digest(args.audio)
        if oracle['meta']['model'] != 'oruk/orukeet' or oracle['meta']['audio_sha256'] != audio_hash:
            raise ValueError('Oracle model/audio binding mismatch')
        receipt.update(inputs=[dict(path=str(p.resolve()), sha256=helpers.digest(p)) for p in inputs],
                       audio_sha256=audio_hash, oracle_sha256=helpers.digest(args.oracle),
                       runner_sha256=helpers.digest(Path(__file__)))
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True,
                args=['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'])
            try:
                receipt['browser_version'] = browser.version
                # Only loopback URLs are allowed; the demo uses a local dev certificate.
                context = browser.new_context(service_workers='block', ignore_https_errors=True)
                origin = f'{parsed.scheme}://{parsed.netloc}'
                receipt['served_code'] = []
                receipt['code_capture_errors'] = []
                def route_request(route):
                    if route.request.url.startswith(origin + '/'):
                        if urlparse(route.request.url).path.endswith(('.js', '.ts', '.mjs', '.wasm')):
                            capture_code_route(route, receipt)
                        else:
                            route.continue_()
                    else:
                        receipt['blocked_requests'].append(route.request.url)
                        route.abort()
                context.route('**/*', route_request)
                page = context.new_page()
                page.on('pageerror', lambda error: receipt['page_errors'].append(str(error)))
                page.on('console', lambda message: receipt['console'].append(message.text[:2000]) if len(receipt['console']) < 100 else None)
                page.goto(url, wait_until='networkidle', timeout=60000)
                if receipt['page_errors'] or receipt['code_capture_errors'] or any(item['status'] != 200 for item in receipt['served_code']):
                    raise ValueError('Browser code failed to load; refusing to wait for nonexistent inference')
                page.locator('#model').set_input_files(str(args.model_root.resolve()), timeout=60000)
                page.locator('#audio').set_input_files(str(args.audio.resolve()))
                page.wait_for_function("document.getElementById('result').textContent.trim().startsWith('{')", timeout=args.timeout_ms)
                result = json.loads(page.locator('#result').inner_text())
                receipt['result'] = result
                receipt['execution_valid'] = check_execution(result, args.backend, args.repeat)
                receipt['cancellation_valid'] = check_cancellation(result, args.verify_cancellation)
                receipt['controller_valid'] = check_controller(result, args.verify_controller, args.backend, oracle['result']['text'])
                receipt['controller_scope'] = 'chunk-fed-windowed-offline-asr' if args.verify_controller else 'SKIP_NOT_REQUESTED'
                receipt['pcm_identity_valid'] = check_pcm(result, args.expected_pcm_sha256)
                receipt['same_oracle_pcm'] = bool(args.expected_pcm_sha256) and receipt['pcm_identity_valid']
                receipt['code_bound'] = check_code_bindings(receipt)
                receipt['text_exact'] = all(t['text'] == oracle['result']['text'] for t in result['transcripts']) and bool(result['transcripts'])
                receipt['tokens_exact'] = all(t['tokenIds'] == oracle['result']['token_ids'] for t in result['transcripts']) and bool(result['transcripts'])
                if result['status'] == 'PASS_ORUKEET_SMOKE' and result['disposed'] is True and receipt['execution_valid'] and receipt['cancellation_valid'] and receipt['controller_valid'] and receipt['text_exact'] and receipt['tokens_exact'] and receipt['pcm_identity_valid'] and receipt['code_bound'] and not receipt['page_errors'] and not receipt['blocked_requests']:
                    receipt['status'] = ('PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE' if receipt['same_oracle_pcm']
                                         else 'PASS_AUTO_BROWSER_TOKEN_TEXT_SMOKE_PCM_UNBOUND')
            finally:
                browser.close()
                receipt['browser_closed'] = True
    except Exception as error:
        receipt['error'] = str(error)
    finally:
        if 'runner_sha256' in receipt and helpers.digest(Path(__file__)) != receipt['runner_sha256']:
            receipt['status'] = 'FAIL_AUTO_BROWSER'
            receipt['error'] = 'Runner source changed during execution'
        write_new_text(args.output, json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=receipt['status'], output=str(args.output.resolve()), error=receipt.get('error'))))
    return 0 if receipt['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
