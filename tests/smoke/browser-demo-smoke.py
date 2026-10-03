"""No-download companion UI smoke with owned preview/browser cleanup.

Requires existing Node, Python Playwright and Chrome; never installs dependencies.
Default is UI/build only. Explicit stock-v3 opt-in attempts app warm-up;
neither mode establishes numerical parity or cancellation/disposal acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text

BUILD_MARKER = '{\n  "status": "PASS_IN_MEMORY_COMPANION_BUILD"'


def parse_build(text):
    if text.count(BUILD_MARKER) != 1:
        raise ValueError('Expected exactly one companion build receipt')
    receipt, _ = json.JSONDecoder().raw_decode(text[text.index(BUILD_MARKER):])
    if not receipt.get('outputs'):
        raise ValueError('Missing companion build outputs')
    return receipt


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def probe_preview_closed(node, url):
    """Unknown connection errors are not evidence that an owned port closed."""
    script = """try {
      await fetch(process.argv[1], {signal: AbortSignal.timeout(2000)});
      console.log(JSON.stringify({closed:false, code:'REACHABLE'}));
    } catch (error) {
      const code = error.cause?.code ?? error.name;
      console.log(JSON.stringify({closed:code === 'ECONNREFUSED', code}));
    }"""
    result = subprocess.run([node, '-e', script, url], capture_output=True,
        text=True, encoding='utf-8', timeout=10, check=True)
    proof = json.loads(result.stdout)
    if (not isinstance(proof, dict) or type(proof.get('closed')) is not bool
            or not isinstance(proof.get('code'), str)
            or proof['closed'] != (proof['code'] == 'ECONNREFUSED')):
        raise ValueError('Invalid preview closure evidence')
    return proof


def local_model_inputs(root, precision):
    suffix = '' if precision == 'fp32' else '.' + precision
    paths = [root / ('encoder-model' + suffix + '.onnx'),
             root / 'decoder_joint-model.onnx', root / 'vocab.txt']
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing/empty/symlink local model input: ' + str(path))
        external = Path(str(path) + '.data')
        if external.exists():
            if external.is_symlink() or not external.is_file() or not external.stat().st_size:
                raise ValueError('Invalid external data: ' + str(external))
    return paths + [Path(str(path) + '.data') for path in paths if Path(str(path) + '.data').exists()]


def wait_preview(process, lines, timeout=120):
    output = []
    deadline = time.monotonic() + timeout
    size = 0
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if process.poll() is not None:
                raise RuntimeError('Preview exited before readiness')
            continue
        if line is None:
            raise RuntimeError('Preview output ended before readiness')
        output.append(line)
        size += len(line)
        if size > 4 * 1024 * 1024:
            raise ValueError('Preview output exceeds bound')
        if line.startswith('{"status":"PREVIEW_READY"'):
            ready = json.loads(line)
            url = urlparse(ready['url'])
            if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port:
                raise ValueError('Preview must use an explicit loopback port')
            return ready, parse_build(''.join(output))
    raise TimeoutError('Preview readiness timeout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-model-root', type=Path,
                        help='Opt in to real stock-v3 local hybrid loading and app warm-up')
    parser.add_argument('--encoder-precision', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()
    consumer = args.consumer_root.resolve()
    preview = Path(__file__).with_name('companion-build.mjs')
    inputs = [Path(__file__), preview, ROOT / 'tools/artifact_outputs.py',
              consumer / 'vite.config.js', consumer / 'package.json', consumer / 'src/App.jsx']
    model_root = args.local_model_root.resolve() if args.local_model_root else None
    if model_root:
        inputs += local_model_inputs(model_root, args.encoder_precision)
    require_new_outputs([args.output], protected_inputs=inputs,
                        protected_directories=[consumer, ROOT / 'src', ROOT / 'models'] + ([model_root] if model_root else []))
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Existing Node executable required')
    from playwright.sync_api import sync_playwright, expect
    receipt = {'status': 'FAIL_COMPANION_UI_SMOKE', 'sources_sha256': hashes,
               'scope': 'Fresh-context no-download UI/build smoke; not model loading, inference or GPU acceptance'}
    if model_root:
        receipt['scope'] = 'Stock-v3 local hybrid load and existing app warm-up; not Orukeet parity, numerical oracle, cancellation or disposal acceptance'
        receipt['local_model_root'] = str(model_root)
        receipt['encoder_precision'] = args.encoder_precision
    process = None
    ready = None
    try:
        process = subprocess.Popen([node, str(preview), str(consumer), '--serve'], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
            errors='replace', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        lines = queue.Queue()
        def read_lines():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        ready, build = wait_preview(process, lines)
        if Path(build['root']).resolve() != consumer:
            raise ValueError('Build receipt consumer differs')
        receipt.update(build=build, preview=ready, preview_pid=process.pid)
        known = {'/' + item['fileName']: item for item in build['outputs'] + ready['publicAssets']}
        origin = ready['url']
        blocked, errors, responses = [], [], []
        receipt.update(blocked_requests=blocked, page_errors=errors)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True,
                args=['--enable-unsafe-webgpu'] if model_root else [])
            try:
                context = browser.new_context(service_workers='block')
                def route_request(route):
                    request = route.request
                    parsed = urlparse(request.url)
                    path = '/index.html' if parsed.path == '/' else parsed.path
                    if request.url.startswith(origin + '/') and path in known and request.method == 'GET':
                        route.continue_()
                    else:
                        blocked.append(request.url)
                        route.abort()
                context.route('**/*', route_request)
                page = context.new_page()
                page.set_default_timeout(10000)
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('response', lambda response: responses.append(response))
                page.goto(origin, wait_until='networkidle', timeout=20000)
                expect(page).to_have_title('ASRJS Browser Demo')
                load = page.get_by_role('button', name='Load Model', exact=False)
                expect(load).to_be_enabled()
                def select(label):
                    # Existing labels are not associated via for/id; scope by exact rendered label.
                    return page.locator('label').filter(has_text=re.compile('^' + re.escape(label) + '$')).locator('..').locator('select')
                select('Model').select_option('parakeet-tdt-0.6b-v3')
                expect(select('Encoder Backend')).to_have_value('webgpu')
                expect(select('Decoder Backend')).to_have_value('wasm')
                select('Runtime Backend').select_option('wasm')
                expect(select('Runtime Backend')).to_have_value('wasm')
                select('Model Source').select_option('local')
                expect(page.get_by_text('No folder selected', exact=True)).to_be_visible()
                load.click()
                expect(page.get_by_text(re.compile('Pick a local model folder first'))).to_be_visible()
                expect(load).to_be_enabled()
                if model_root:
                    receipt['local_load'] = {'status': 'STARTED', 'console': []}
                    def record_console(message):
                        logs = receipt['local_load']['console']
                        if len(logs) < 200:
                            logs.append({'type': message.type, 'text': message.text[:4000]})
                    page.on('console', record_console)
                    adapter = page.evaluate('async () => { const a = await navigator.gpu?.requestAdapter(); return a ? {vendor:a.info?.vendor, architecture:a.info?.architecture, device:a.info?.device, description:a.info?.description, fallback:a.info?.isFallbackAdapter} : null; }')
                    if not adapter:
                        raise ValueError('Requested local hybrid run has no WebGPU adapter')
                    receipt['local_load']['adapter'] = adapter
                    select('Runtime Backend').select_option('webgpu-hybrid')
                    select('Encoder Backend').select_option('webgpu')
                    page.locator('input[webkitdirectory]').set_input_files(str(model_root), timeout=30000)
                    select('Encoder').select_option(args.encoder_precision)
                    select('Decoder').select_option('fp32')
                    load.click()
                    terminal = page.get_by_text(re.compile(r'^(Model ready$|Failed:|Verification failed$|Warm-up failed$)')).first
                    expect(terminal).to_be_visible(timeout=180000)
                    if terminal.inner_text() != 'Model ready':
                        raise ValueError('App local load rejected: ' + terminal.inner_text())
                    expect(page.get_by_role('button', name='Model Loaded', exact=False)).to_be_visible(timeout=180000)
                    expect(page.get_by_text('Model ready', exact=True)).to_be_visible()
                    receipt['local_load']['status'] = 'PASS_APP_LOCAL_LOAD_AND_WARMUP'
                served = []
                for response in responses:
                    parsed = urlparse(response.url)
                    path = '/index.html' if parsed.path == '/' else parsed.path
                    if not response.url.startswith(origin + '/') or path not in known:
                        raise ValueError('Unexpected served browser resource')
                    body = response.body()
                    if response.status != 200 or hashlib.sha256(body).hexdigest() != known[path]['sha256']:
                        raise ValueError('Browser bytes differ from build receipt: ' + path)
                    served.append(path)
                if '/index.html' not in served or not any(path.endswith('.js') and path.startswith('/assets/') for path in served):
                    raise ValueError('HTML and application bundle were not observed')
                if errors:
                    raise ValueError('Browser page errors: ' + repr(errors))
                if any(re.search(r'\.(onnx|nemo|safetensors|bin)(?:[?#]|$)', url) for url in blocked):
                    raise ValueError('UI smoke attempted a model-weight request')
                receipt.update(browser_version=browser.version, served_resources=sorted(set(served)),
                    blocked_requests=blocked, page_errors=errors,
                    checks=['boot', 'v3-hybrid-selection', 'wasm-selection', 'local-source-panel', 'missing-local-folder-error'])
                context.close()
            finally:
                browser.close()
        receipt['status'] = 'PASS_COMPANION_LOCAL_WARMUP' if model_root else 'PASS_COMPANION_UI_SMOKE'
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
                    receipt['status'] = 'FAIL_COMPANION_UI_SMOKE'
                    receipt['error'] = 'Owned preview port remains reachable'
        if hashes != {str(path.resolve()): digest(path) for path in inputs}:
            receipt['status'] = 'FAIL_COMPANION_UI_SMOKE'
            receipt['error'] = 'Diagnostic or consumer source changed during execution'
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))
    if receipt['status'] not in ('PASS_COMPANION_UI_SMOKE', 'PASS_COMPANION_LOCAL_WARMUP'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
