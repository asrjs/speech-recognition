"""Real stock-v3 public local-loader smoke; no download, model parity or promotion."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/smoke'))
shared = importlib.import_module('browser-demo-smoke')
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text


def validate_result(result, verify_cancellation, verify_window_tokens=False, window_token_overlap=False):
    """Validate lifecycle evidence, not just the page's PASS label or ASR parity."""
    if not isinstance(result, dict) or result.get('status') != 'PASS_LOCAL_HYBRID_SMOKE':
        raise ValueError('Public API smoke failed')
    if result.get('disposed') is not True or result.get('inferencePassed') is not True:
        raise ValueError('Inference/disposal evidence missing')
    if any(result.get(key) for key in ['error', 'cleanupError', 'audioCleanupError']):
        raise ValueError('Public API smoke reports execution/cleanup errors')
    results = result.get('results')
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError('Exactly two successful repeated transcriptions required')
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError('Nonempty transcript evidence required')
        meta = item.get('meta')
        metrics = meta.get('metrics') if isinstance(meta, dict) else None
        if not isinstance(metrics, dict) or metrics.get('encoderBackend') != 'webgpu' or metrics.get('decoderBackend') != 'wasm':
            raise ValueError('Actual hybrid provider evidence differs')
    if results[0]['text'] != results[1]['text']:
        raise ValueError('Repeated transcript differs')
    if verify_window_tokens:
        policy = result.get('windowing', {})
        boundary, duration = policy.get('windowSeconds'), policy.get('durationSeconds')
        expected_duration = 34 if window_token_overlap else 2 * boundary if type(boundary) in (int, float) else None
        if type(boundary) not in (int, float) or not math.isfinite(boundary) or boundary < 20 or duration != expected_duration:
            raise ValueError('Two-window input evidence missing')
        if window_token_overlap and (boundary != 20 or policy.get('overlapSeconds') != 6):
            raise ValueError('Shared-speech overlap policy missing')
        for item in results:
            if item['meta'].get('detailLevel') != 'detailed' or item['meta']['metrics'].get('windowCount') != 2:
                raise ValueError('Detailed two-window execution required')
            if window_token_overlap:
                # This diagnostic intentionally exercises the captured ambiguous
                # overlap, not a general gate for all future reconciled outputs.
                warnings = item.get('warnings')
                if not isinstance(warnings, list) or not any(isinstance(warning, dict) and
                        warning.get('code') == 'pipeline.windowing.token-overlap-unresolved' for warning in warnings):
                    raise ValueError('Expected ambiguous overlap processing evidence missing')
            tokens, words = item.get('tokens'), item.get('words')
            if not isinstance(tokens, list) or not tokens or not isinstance(words, list) or not words:
                raise ValueError('Real token and word detail required')
            for collection in (tokens, words):
                halves = set()
                for index, entry in enumerate(collection):
                    if not isinstance(entry, dict) or type(entry.get('index')) is not int or entry['index'] != index:
                        raise ValueError('Global contiguous indices required')
                    start, end = entry.get('startTime'), entry.get('endTime')
                    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end)) or not 0 <= start <= end <= duration:
                        raise ValueError('Finite global timestamps required')
                    if not window_token_overlap and start < boundary < end:
                        raise ValueError('Window-local timestamp leaked across boundary')
                    halves.add(start >= boundary)
                if not window_token_overlap and halves != {False, True}:
                    raise ValueError('Detail from both windows required')
            for word in words:
                refs = word.get('tokenIndices')
                if not isinstance(refs, list) or not refs or any(type(index) is not int or not 0 <= index < len(tokens) for index in refs):
                    raise ValueError('Valid global word-token references required')
                for index in refs:
                    token = tokens[index]
                    if token['startTime'] < word['startTime'] - 1e-6 or token['endTime'] > word['endTime'] + 1e-6:
                        raise ValueError('Word-token timing association differs')
        if any(results[0][key] != results[1][key] for key in ('tokens', 'words')):
            raise ValueError('Repeated canonical detail differs')
    if verify_cancellation:
        cancellation = result.get('cancellation')
        if not isinstance(cancellation, dict) or any(cancellation.get(key) is not value
                for key, value in [('requested', True), ('rejected', True), ('completeEmitted', False)]):
            raise ValueError('Requested cancellation evidence missing or failed')
        done, total = cancellation.get('completedUnits'), cancellation.get('totalUnits')
        if cancellation.get('errorName') != 'PipelineAbortedError' or cancellation.get('stageId') != 'decode' or type(done) is not int or type(total) is not int or not 0 < done < total:
            raise ValueError('Cancellation must reject during partially completed decode')


def probe_preview_closed(node, url):
    return shared.probe_preview_closed(node, url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--expected-phrase', required=True)
    parser.add_argument('--encoder-precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--verify-cancellation', action='store_true')
    parser.add_argument('--verify-window-tokens', action='store_true')
    parser.add_argument('--window-token-overlap', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.window_token_overlap and not args.verify_window_tokens:
        parser.error('--window-token-overlap requires --verify-window-tokens')
    if not args.expected_phrase.strip():
        parser.error('Expected smoke phrase must not be empty')
    consumer, model_root = args.consumer_root.resolve(), args.model_root.resolve()
    graphs = shared.local_model_inputs(model_root, args.encoder_precision)
    preview = Path(__file__).with_name('companion-build.mjs')
    inputs = graphs + [args.audio.resolve(), Path(__file__), Path(shared.__file__), preview,
        preview.with_name('local-ort-assets.mjs'), preview.with_name('local-parakeet.mjs'),
        preview.with_name('local-parakeet.html'), ROOT / 'tools/artifact_outputs.py',
        preview.with_name('smoke-transcript.mjs'), preview.with_name('window-token-fixture.mjs'),
        consumer / 'vite.config.js', consumer / 'package.json',
        ROOT / 'src/models/nemo-tdt/executor.ts', ROOT / 'src/models/nemo-tdt/model.ts',
        ROOT / 'src/models/nemo-common/types.ts']
    for path in inputs:
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError('Required regular nonempty input: ' + str(path))
    require_new_outputs([args.output], protected_inputs=inputs,
        protected_directories=[consumer, model_root, ROOT / 'src', ROOT / 'models'])
    hashes = {str(path.resolve()): shared.digest(path) for path in inputs}
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Existing Node required')
    from playwright.sync_api import sync_playwright, expect
    receipt = {'status': 'FAIL_LOCAL_PUBLIC_API_SMOKE', 'sources_sha256': hashes,
        'scope': 'Stock-v3 repeated public local inference, selected session providers and awaited disposal; optional cooperative decode cancellation/reuse, not GPU kernel interruption, per-node GPU placement, numerical parity, Orukeet or soak acceptance',
        'cancellation_requested': args.verify_cancellation,
        'window_tokens_requested': args.verify_window_tokens,
        'window_token_overlap_requested': args.window_token_overlap,
        'expected_phrase': args.expected_phrase, 'encoder_precision': args.encoder_precision,
        'blocked_requests': [], 'page_errors': [], 'console': []}
    process, ready = None, None
    try:
        process = subprocess.Popen([node, str(preview), str(consumer), '--serve-local-ort'],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        lines = queue.Queue()
        def read_lines():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=read_lines, daemon=True).start()
        ready, build = shared.wait_preview(process, lines)
        if Path(build['root']).resolve() != consumer or not ready.get('engine'):
            raise ValueError('Unexpected preview identity/engine')
        receipt.update(preview=ready, build=build, preview_pid=process.pid)
        origin = ready['url']
        known = {'/' + item['fileName']: item for item in build['outputs'] + ready['publicAssets']}
        responses = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True,
                args=['--enable-unsafe-webgpu'])
            try:
                context = browser.new_context(service_workers='block')
                try:
                    def route_request(route):
                        request = route.request
                        if request.method == 'GET' and request.url.startswith(origin + '/') and urlparse(request.url).path in known:
                            route.continue_()
                        else:
                            receipt['blocked_requests'].append(request.url)
                            route.abort()
                    context.route('**/*', route_request)
                    page = context.new_page()
                    page.on('pageerror', lambda error: receipt['page_errors'].append(str(error)))
                    page.on('response', lambda response: responses.append(response))
                    def log(message):
                        if len(receipt['console']) < 100:
                            receipt['console'].append({'type': message.type, 'text': message.text[:2000]})
                    page.on('console', log)
                    page.goto(origin + '/harness/local-parakeet.html', wait_until='networkidle', timeout=20000)
                    expect(page).to_have_title('Local Parakeet public API smoke')
                    page.get_by_label('Model files', exact=True).set_input_files([str(path) for path in graphs])
                    page.get_by_label('Audio fixture', exact=True).set_input_files(str(args.audio.resolve()))
                    page.get_by_label('Encoder precision', exact=True).select_option(args.encoder_precision)
                    page.get_by_label('Expected smoke phrase', exact=True).fill(args.expected_phrase)
                    page.get_by_label('Verify decode cancellation and reuse', exact=True).set_checked(args.verify_cancellation)
                    page.get_by_label('Verify windowed token detail', exact=True).set_checked(args.verify_window_tokens)
                    page.get_by_label('Use shared-speech overlap fixture', exact=True).set_checked(args.window_token_overlap)
                    page.get_by_role('button').click()
                    expect(page.get_by_role('status')).to_have_text(re.compile('^(PASS_|FAIL_)'), timeout=240000)
                    receipt['result'] = json.loads(page.locator('#result').inner_text())
                    validate_result(receipt['result'], args.verify_cancellation, args.verify_window_tokens, args.window_token_overlap)
                    served = set()
                    for response in tuple(responses):
                        path = urlparse(response.url).path
                        if not response.url.startswith(origin + '/') or path not in known:
                            raise ValueError('Unknown HTTP response: ' + response.url)
                        if response.status != 200:
                            raise ValueError('Served status differs: ' + path)
                        served.add(path)
                    if any('/' + item['fileName'] not in served for item in ready['publicAssets'][:2]):
                        raise ValueError('Both pinned engine assets must be observed')
                    # Emscripten pthread responses can lack a page CDP body ID.
                    # Re-fetch each observed path from the immutable snapshot map
                    # in the page, retaining byte/hash checks rather than skipping workers.
                    verified = page.evaluate('''async paths => {
                      const output = [];
                      for (const path of paths) {
                        const response = await fetch(path);
                        const bytes = await response.arrayBuffer();
                        const digest = await crypto.subtle.digest('SHA-256', bytes);
                        output.push({path, status:response.status, bytes:bytes.byteLength,
                          sha256:Array.from(new Uint8Array(digest), b=>b.toString(16).padStart(2,'0')).join('')});
                      }
                      return output;
                    }''', sorted(served))
                    receipt['verified_resources'] = verified
                    for item in verified:
                        expected = known[item['path']]
                        if item['status'] != 200 or item['sha256'] != expected['sha256'] or item['bytes'] != expected['bytes']:
                            raise ValueError('Served bytes differ: ' + item['path'])
                    if receipt['blocked_requests'] or receipt['page_errors']:
                        raise ValueError('Blocked requests or browser errors')
                    receipt.update(browser_version=browser.version, served_resources=sorted(served))
                finally:
                    context.close()
            finally:
                browser.close()
        receipt['status'] = 'PASS_LOCAL_PUBLIC_API_SMOKE'
    except Exception as error:
        receipt['error'] = str(error)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            receipt['preview_process_stopped'] = process.poll() is not None
            if ready:
                try:
                    receipt['preview_closure_probe'] = probe_preview_closed(node, ready['url'])
                    receipt['preview_port_closed'] = receipt['preview_closure_probe']['closed'] is True
                except Exception as error:
                    receipt['preview_port_closed'] = False
                    receipt['cleanup_error'] = str(error)
                if not receipt['preview_port_closed']:
                    receipt['status'] = 'FAIL_LOCAL_PUBLIC_API_SMOKE'
                    receipt.setdefault('error', 'Preview closure unproven')
        receipt['inputs_unchanged'] = hashes == {str(path.resolve()): shared.digest(path) for path in inputs}
        if not receipt['inputs_unchanged']:
            receipt.update(status='FAIL_LOCAL_PUBLIC_API_SMOKE', error='Inputs changed during execution')
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))
    if receipt['status'] != 'PASS_LOCAL_PUBLIC_API_SMOKE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
