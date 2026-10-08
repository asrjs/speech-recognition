"""Drive the real local-folder demo in installed Chrome and save a fresh receipt.

Requires an already-running examples/demo Vite server; never downloads models.
Smoke/text checks are not stage numerical parity.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
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


def check_controller_timing(result, requested, require_nonzero=False):
    """Coordinate/payload transfer only; never an original-engine timing oracle."""
    def json_value(value):
        # JSON has one numeric type; preserve bool and reject nonfinite values.
        if type(value) is float and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_value(item) for item in value]
        return value

    if not requested:
        return not require_nonzero
    event = result.get('controller')
    if not isinstance(event, dict):
        return False
    sources, requests, updates = (event.get(key) for key in ('inferenceResults', 'requests', 'updates'))
    if (not all(isinstance(items, list) and items for items in (sources, requests, updates))
            or len(sources) != len(requests)):
        return False
    fields = ('words', 'segments', 'sentences', 'tokens')
    observed = dict.fromkeys(fields, False)
    observed_nonzero = dict.fromkeys(fields, False)
    next_index, nonzero = 0, False
    for position, update in enumerate(updates):
        if not isinstance(update, dict) or 'inferenceIndex' not in update:
            return False
        canonical, partial = update.get('canonicalTiming'), update.get('partialTiming')
        if (not isinstance(canonical, dict) or not isinstance(partial, dict)
                or set(canonical) != set(fields) or set(partial) != set(fields)):
            return False
        try:
            if json.dumps(json_value(canonical), sort_keys=True, allow_nan=False) != json.dumps(json_value(partial), sort_keys=True, allow_nan=False):
                return False
        except (TypeError, ValueError):
            return False
        index = update['inferenceIndex']
        if index is None:
            if (position != len(updates) - 1 or update.get('kind') != 'final'
                    or any(value is not None for value in canonical.values())):
                return False
            continue
        if type(index) is not int or index != next_index or index >= len(sources):
            return False
        next_index += 1
        source, request, window = sources[index], requests[index], update.get('window')
        if not all(isinstance(value, dict) for value in (source, request, window)):
            return False
        start, end = source.get('startFrame'), source.get('endFrame')
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or source.get('sampleRate') != 16000 or request.get('sampleRate') != 16000
                or any(type(value.get('startFrame')) is not int or type(value.get('endFrame')) is not int
                       or value.get('startFrame') != start or value.get('endFrame') != end
                       for value in (request, window))):
            return False
        nonzero |= start > 0
        local = source.get('localTiming')
        if not isinstance(local, dict) or set(local) != set(fields):
            return False
        for field in fields:
            before, after = local[field], canonical[field]
            if before is None:
                if after is not None:
                    return False
                continue
            if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
                return False
            for original, shifted in zip(before, after):
                if not isinstance(original, dict) or not isinstance(shifted, dict):
                    return False
                has_times = 'startTime' in original or 'endTime' in original
                if has_times:
                    times = [original.get(key) for key in ('startTime', 'endTime')]
                    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
                            or not 0 <= times[0] <= times[1]):
                        return False
                    observed[field] = True
                    observed_nonzero[field] |= start > 0
                elif field != 'tokens':
                    return False
                expected = dict(original)
                if has_times:
                    expected.update(startTime=times[0] + start / 16000, endTime=times[1] + start / 16000)
                try:
                    # JSON comparison preserves bool/int distinctions and rejects NaN.
                    if json.dumps(json_value(expected), sort_keys=True, allow_nan=False) != json.dumps(json_value(shifted), sort_keys=True, allow_nan=False):
                        return False
                except (TypeError, ValueError):
                    return False
    return (next_index == len(sources) and all(observed.values())
            and (not require_nonzero or (nonzero and all(observed_nonzero.values()))))


def check_capture(result, requested, backend):
    """Actual fake-device manual capture/ASR, not the file oracle or native streaming."""
    if not requested:
        return True
    capture = result.get('capture')
    if (not isinstance(capture, dict) or capture.get('scope') != 'fake-device-manual-capture-utterance-asr'
            or capture.get('disposed') is not True or capture.get('restartRejected') is not True
            or capture.get('modelDisposed') is not True or capture.get('error') or capture.get('cleanupError')):
        return False
    runs = capture.get('runs')
    if not isinstance(runs, list) or len(runs) != 2:
        return False
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            return False
        samples, chunks = run.get('capturedSamples'), run.get('chunkCount')
        tokens, tracks = run.get('tokenIds'), run.get('trackStates')
        replay_tokens = run.get('replayTokenIds')
        if (type(run.get('iteration')) is not int or run['iteration'] != index
                or type(samples) is not int or samples < 32000 or type(chunks) is not int or samples != chunks * 256
                or run.get('sampleRate') != 16000 or run.get('utteranceSamples') != samples
                or run.get('pcmExact') is not True
                or not isinstance(run.get('capturedPcmSha256'), str)
                or re.fullmatch('[0-9a-f]{64}', run['capturedPcmSha256']) is None
                or run.get('utterancePcmSha256') != run['capturedPcmSha256']
                or not isinstance(run.get('text'), str) or not run['text'].strip()
                or not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens)
                or not isinstance(replay_tokens, list) or any(type(token) is not int or token < 0 for token in replay_tokens)
                or run.get('replayText') != run['text'] or run.get('replayTokenIds') != tokens
                or run.get('contextState') != 'closed' or not isinstance(tracks, list) or not tracks
                or any(state != 'ended' for state in tracks) or run.get('workletLoads') != 1 or run.get('stopCalls') != 1
                or not isinstance(run.get('metrics'), dict) or not isinstance(run.get('replayMetrics'), dict)
                or not check_execution(dict(transcripts=[run]), backend, 1)
                or not check_execution(dict(transcripts=[dict(metrics=run.get('replayMetrics'))]), backend, 1)):
            return False
    return True


def capture_pcm_paths(output):
    return [Path(str(output) + f'.capture-{index}.f32le') for index in range(2)]


def save_capture_pcm(result, payloads, paths):
    """Preserve exact browser f32le bytes for an independent original-engine run."""
    runs = result.get('capture', {}).get('runs')
    if not isinstance(payloads, list) or len(payloads) != 2 or not isinstance(runs, list) or len(runs) != 2 or len(paths) != 2:
        raise ValueError('Capture PCM requires both sessions')
    require_new_outputs(paths)
    prepared = []
    for run, payload, path in zip(runs, payloads, paths):
        if (not isinstance(payload, list) or any(type(value) is not int or not 0 <= value <= 255 for value in payload)
                or type(run.get('capturedSamples')) is not int or run['capturedSamples'] <= 0
                or run.get('sampleRate') != 16000 or len(payload) != run['capturedSamples'] * 4):
            raise ValueError('Invalid captured f32le payload')
        raw = bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != run.get('capturedPcmSha256') or any(not math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)):
            raise ValueError('Captured PCM digest/finite-value mismatch')
        prepared.append((path, raw, dict(path=str(path.resolve()), bytes=len(raw), sha256=digest,
            encoding='f32le', sampleRate=16000, channels=1, samples=run['capturedSamples'],
            validSamples=run['capturedSamples'], rightPaddingSamples=0)))
    # Validate every payload before creating any sidecar; retain partial output on write failure.
    for path, raw, artifact in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    return [artifact for _, _, artifact in prepared]


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
            and any(urlparse(item['url']).path.endswith('/orukeet-smoke.html') and item['bytes'] > 0 for item in code)
            and any(urlparse(item['url']).path.endswith('/src/orukeet-smoke.js') and item['bytes'] > 0 for item in code)
            and any('/onnxruntime-web/dist/' in urlparse(item['url']).path
                    and urlparse(item['url']).path.endswith('.wasm') and item['bytes'] > 0 for item in code)
            and all(item['status'] == 200 and type(item['bytes']) is int and item['bytes'] >= 0
                    and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None
                    and item['capture'] == 'route-fulfilled-body' for item in code))


def is_code_response(url):
    # HTML can execute inline scripts too; bind it alongside external modules.
    return urlparse(url).path.endswith(('.html', '.js', '.ts', '.mjs', '.wasm'))


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
    parser.add_argument('--verify-capture', action='store_true', help='Also verify two real fake-device manual capture -> utterance -> ASR sessions; not original-engine capture parity')
    parser.add_argument('--save-capture-pcm', action='store_true', help='Save exact captured f32le sidecars for later original-engine comparison; requires --verify-capture')
    parser.add_argument('--require-controller-nonzero-origin', action='store_true',
                        help='Also require timestamped detail in controller windows starting after frame zero')
    args = parser.parse_args()
    if args.save_capture_pcm and not args.verify_capture:
        parser.error('--save-capture-pcm requires --verify-capture')
    if args.verify_cancellation and args.repeat < 2:
        parser.error('--verify-cancellation requires at least two successful repeats to prove reuse')
    if args.require_controller_nonzero_origin and not args.verify_controller:
        parser.error('--require-controller-nonzero-origin requires --verify-controller')
    parsed = urlparse(args.url)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.query or parsed.fragment:
        parser.error('--url must be a plain loopback HTTP(S) page URL')
    if args.expected_pcm_sha256 is not None and re.fullmatch('[0-9a-f]{64}', args.expected_pcm_sha256) is None:
        parser.error('--expected-pcm-sha256 must be a lowercase SHA-256')
    pcm_outputs = capture_pcm_paths(args.output) if args.save_capture_pcm else []
    require_new_outputs([args.output, *pcm_outputs], protected_inputs=[args.audio, args.oracle],
                        protected_directories=[args.model_root])
    url = args.url + '?' + urlencode(dict(auto=1, backend=args.backend, precision=args.precision, repeat=args.repeat, audioStrategy=args.audio_strategy, cancel=int(args.verify_cancellation), controller=int(args.verify_controller)))
    receipt = dict(status='FAIL_AUTO_BROWSER', url=url, scope='real-browser-smoke-and-oracle-output-only',
                   page_errors=[], console=[], blocked_requests=[], expected_pcm_sha256=args.expected_pcm_sha256,
                   cancellation_requested=args.verify_cancellation, controller_requested=args.verify_controller,
                   capture_requested=args.verify_capture,
                   capture_pcm_requested=args.save_capture_pcm,
                   controller_nonzero_required=args.require_controller_nonzero_origin)
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
            launch_args = ['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required']
            if args.verify_capture:
                launch_args.extend(['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
                                    f'--use-file-for-fake-audio-capture={args.audio.resolve()}'])
            browser = playwright.chromium.launch(channel='chrome', headless=True, args=launch_args)
            try:
                receipt['browser_version'] = browser.version
                # Only loopback URLs are allowed; the demo uses a local dev certificate.
                context = browser.new_context(service_workers='block', ignore_https_errors=True,
                                              permissions=['microphone'] if args.verify_capture else [])
                origin = f'{parsed.scheme}://{parsed.netloc}'
                receipt['served_code'] = []
                receipt['code_capture_errors'] = []
                def route_request(route):
                    if route.request.url.startswith(origin + '/'):
                        if is_code_response(route.request.url):
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
                if args.verify_capture and result.get('status') == 'PASS_ORUKEET_SMOKE':
                    page.evaluate("""({libraryUrl, wasmPaths, backend, precision, savePcm}) => {
                        window.captureResult = null; window.captureError = null;
                        window.capturePcm = [];
                        void (async () => {
                        const { runCapturedController } = await import('/src/capture-controller.js');
                        const { createSpeechModelLocalEntries, loadSpeechModelFromLocalEntries } = await import(libraryUrl);
                        const receipt = { modelDisposed: false };
                        let model;
                        try {
                            model = await loadSpeechModelFromLocalEntries({ modelId: 'oruk/orukeet',
                                entries: createSpeechModelLocalEntries([...document.getElementById('model').files]),
                                backend, encoderQuant: precision, preprocessorBackend: 'js', wasmPaths });
                            await runCapturedController({ receipt,
                              onCapturedPcm: savePcm ? (pcm) => {
                                const bytes = new ArrayBuffer(pcm.length * 4), view = new DataView(bytes);
                                pcm.forEach((sample, index) => view.setFloat32(index * 4, sample, true));
                                window.capturePcm.push(Array.from(new Uint8Array(bytes)));
                              } : undefined,
                              async transcribe(pcm, sampleRate) {
                                const envelope = await model.transcribeMonoPcm(pcm, sampleRate, {
                                    returnTokenIds: true, responseFlavor: 'canonical+native' });
                                return { text: envelope.canonical.text, tokenIds: envelope.native.debug.tokenIds,
                                         metrics: envelope.canonical.meta.metrics };
                            } });
                        } catch (error) { receipt.error = String(error); }
                        finally {
                            if (model) {
                                try { await model.dispose(); receipt.modelDisposed = true; }
                                catch (error) { receipt.cleanupError = String(error); }
                            }
                        }
                        return receipt;
                        })().then(value => window.captureResult = value).catch(error => window.captureError = String(error));
                    }""", dict(libraryUrl='/@fs/' + ROOT.as_posix() + '/src/browser.ts',
                               wasmPaths='/@fs/' + ROOT.as_posix() + '/node_modules/onnxruntime-web/dist/',
                               backend=args.backend, precision=args.precision, savePcm=args.save_capture_pcm))
                    page.wait_for_function('window.captureResult !== null || window.captureError !== null', timeout=args.timeout_ms)
                    if page.evaluate('window.captureError'):
                        raise ValueError(page.evaluate('window.captureError'))
                    result['capture'] = page.evaluate('window.captureResult')
                receipt['capture_valid'] = check_capture(result, args.verify_capture, args.backend)
                if args.save_capture_pcm:
                    if not receipt['capture_valid']:
                        raise ValueError('Capture failed; refusing unverified PCM sidecars')
                    receipt['capture_pcm_artifacts'] = save_capture_pcm(result, page.evaluate('window.capturePcm'), pcm_outputs)
                receipt['capture_scope'] = 'fake-device-manual-capture-utterance-asr' if args.verify_capture else 'SKIP_NOT_REQUESTED'
                receipt['execution_valid'] = check_execution(result, args.backend, args.repeat)
                receipt['cancellation_valid'] = check_cancellation(result, args.verify_cancellation)
                receipt['controller_valid'] = check_controller(result, args.verify_controller, args.backend, oracle['result']['text'])
                receipt['controller_timing_valid'] = check_controller_timing(result, args.verify_controller, args.require_controller_nonzero_origin)
                receipt['controller_timing_scope'] = 'local-to-global-coordinate-transfer' if args.verify_controller else 'SKIP_NOT_REQUESTED'
                receipt['controller_scope'] = 'chunk-fed-windowed-offline-asr' if args.verify_controller else 'SKIP_NOT_REQUESTED'
                receipt['pcm_identity_valid'] = check_pcm(result, args.expected_pcm_sha256)
                receipt['same_oracle_pcm'] = bool(args.expected_pcm_sha256) and receipt['pcm_identity_valid']
                receipt['code_bound'] = check_code_bindings(receipt)
                receipt['text_exact'] = all(t['text'] == oracle['result']['text'] for t in result['transcripts']) and bool(result['transcripts'])
                receipt['tokens_exact'] = all(t['tokenIds'] == oracle['result']['token_ids'] for t in result['transcripts']) and bool(result['transcripts'])
                if result['status'] == 'PASS_ORUKEET_SMOKE' and result['disposed'] is True and receipt['execution_valid'] and receipt['cancellation_valid'] and receipt['controller_valid'] and receipt['controller_timing_valid'] and receipt['capture_valid'] and receipt['text_exact'] and receipt['tokens_exact'] and receipt['pcm_identity_valid'] and receipt['code_bound'] and not receipt['page_errors'] and not receipt['blocked_requests']:
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
