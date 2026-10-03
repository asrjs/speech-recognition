"""No-download companion UI smoke with owned preview/browser cleanup.

Requires existing Node, Python Playwright and Chrome; never installs dependencies.
Build/UI validation only: no inference, GPU execution or model parity claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socket
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    args = parser.parse_args()
    consumer = args.consumer_root.resolve()
    preview = Path(__file__).with_name('companion-build.mjs')
    inputs = [Path(__file__), preview, ROOT / 'tools/artifact_outputs.py',
              consumer / 'vite.config.js', consumer / 'package.json', consumer / 'src/App.jsx']
    require_new_outputs([args.output], protected_inputs=inputs,
                        protected_directories=[consumer, ROOT / 'src', ROOT / 'models'])
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Existing Node executable required')
    from playwright.sync_api import sync_playwright, expect
    receipt = {'status': 'FAIL_COMPANION_UI_SMOKE', 'sources_sha256': hashes,
               'scope': 'Fresh-context no-download UI/build smoke; not model loading, inference or GPU acceptance'}
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
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True)
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
        receipt['status'] = 'PASS_COMPANION_UI_SMOKE'
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
                address = urlparse(ready['url'])
                with socket.socket() as probe:
                    probe.settimeout(2)
                    receipt['preview_port_closed'] = probe.connect_ex((address.hostname, address.port)) != 0
                if not receipt['preview_port_closed']:
                    receipt['status'] = 'FAIL_COMPANION_UI_SMOKE'
                    receipt['error'] = 'Owned preview port remains reachable'
        if hashes != {str(path.resolve()): digest(path) for path in inputs}:
            receipt['status'] = 'FAIL_COMPANION_UI_SMOKE'
            receipt['error'] = 'Diagnostic or consumer source changed during execution'
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))
    if receipt['status'] != 'PASS_COMPANION_UI_SMOKE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
