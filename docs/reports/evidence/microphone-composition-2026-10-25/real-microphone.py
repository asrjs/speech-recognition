"""Real local Parakeet + controlled Chrome capture, no weight downloads."""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import re
import threading
from urllib.parse import unquote, urlsplit
import wave
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--core', required=True, type=Path)
parser.add_argument('--model-dir', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--url', default='http://127.0.0.1:4341/')
parser.add_argument('--asset-origin')
args = parser.parse_args()
core, model = args.core.resolve(), args.model_dir.resolve()
args.output.mkdir(parents=True, exist_ok=False)
fixture = core / 'tests/fixtures/jfk2.en.wav'
expected_sha = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected_sha
with wave.open(str(fixture), 'rb') as source:
    params, pcm = source.getparams(), source.readframes(source.getnframes())
padded = args.output.resolve() / 'jfk2-padded.wav'
with wave.open(str(padded), 'wb') as target:
    target.setparams(params)
    target.writeframes(pcm + bytes(5 * params.framerate * params.nchannels * params.sampwidth))
oracle = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'
def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', '', str(text).lower()).split())
requests = []
roots = {'model': model, 'ort': core / 'node_modules/onnxruntime-web/dist', 'fixtures': core / 'tests/fixtures'}
class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.mjs': 'application/javascript', '.wasm': 'application/wasm'}
    def translate_path(self, path):
        pieces = unquote(urlsplit(path).path).strip('/').split('/')
        base = roots.get(pieces[0])
        if base is None:
            return str(args.output.resolve() / 'not-found')
        result = base.joinpath(*pieces[1:]).resolve()
        return str(result if result.is_relative_to(base.resolve()) else args.output.resolve() / 'not-found')
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()
    def do_GET(self):
        requests.append(self.path)
        super().do_GET()
    def log_message(self, *_):
        pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
origin = args.asset_origin or f'http://127.0.0.1:{server.server_port}'
receipt = {'status': 'UNVERIFIED', 'physicalMicrophone': 'UNVERIFIED', 'fixtureSha256': expected_sha,
           'modelDir': str(model), 'backend': 'WebGPU FP16 encoder + WASM INT8 decoder', 'console': [], 'pageErrors': []}
browser = None
browser_requests = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True, args=[
            '--enable-unsafe-webgpu', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
            f'--use-file-for-fake-audio-capture={padded}%noloop',
        ])
        context = browser.new_context(permissions=['microphone'])
        context.route('https://huggingface.co/**', lambda route: route.abort())
        context.route('https://cdn.jsdelivr.net/**', lambda route: route.abort())
        page = context.new_page()
        page.on('request', lambda request: browser_requests.append(request.url))
        page.on('console', lambda msg: receipt['console'].append({'type': msg.type, 'text': msg.text}))
        page.on('pageerror', lambda error: receipt['pageErrors'].append(str(error)))
        page.goto(args.url, wait_until='networkidle')
        page.wait_for_function('window.micAcceptanceReady === true')
        page.set_default_timeout(180000)
        page.evaluate('(origin) => window.micAcceptance.load(origin)', origin)
        receipt['state'] = page.evaluate('window.micAcceptance.receipt')
        assert normalize(receipt['state']['baseline']['text']) == oracle, receipt['state']['baseline']
        print('File baseline matched', flush=True)
        for mode in ('manual', 'ten-vad'):
            page.evaluate('(mode) => { window.micAcceptance.mode = mode; }', mode)
            page.locator('#start').click()
            page.wait_for_function('window.micAcceptance.receipt.captures.at(-1).started || window.micAcceptance.receipt.errors.length')
            assert not page.evaluate('window.micAcceptance.receipt.errors'), page.evaluate('window.micAcceptance.receipt.errors')
            page.wait_for_timeout(14000)
            page.locator('#stop').click()
            page.wait_for_function('window.micAcceptance.receipt.captures.at(-1).status === "EXECUTED" || window.micAcceptance.receipt.errors.length')
            receipt['state'] = page.evaluate('window.micAcceptance.receipt')
            assert not receipt['state']['errors'], receipt['state']['errors']
            capture = receipt['state']['captures'][-1]
            assert normalize(capture['final']['text']) == oracle, capture
            assert capture['final']['status'] == 'ready' and not capture['final']['mic']['isMicActive'], capture
            assert capture['updates'] and all(state == 'ended' for state in capture['trackStates']), capture
            if mode == 'ten-vad':
                assert capture['startVadState']['state'] == 'ready', capture['startVadState']
                assert any(update.get('context') for update in capture['updates']), capture
            capture['status'] = 'PASS'
            (args.output / f'{mode}.json').write_text(json.dumps(capture, indent=2) + '\n', encoding='utf-8', newline='\n')
            print(f'{mode} capture matched', flush=True)
        page.evaluate('(origin) => window.micAcceptance.verifyReuse(origin)', origin)
        receipt['state'] = page.evaluate('window.micAcceptance.receipt')
        assert normalize(receipt['state']['afterCapture']['text']) == oracle
        receipt['state'] = page.evaluate('window.micAcceptance.dispose()')
        assert receipt['state']['disposal']['status'] == 'disposed'
        assert all(state == 'ended' for state in receipt['state']['disposal']['tracks'])
        assert sum('encoder-model.fp16.onnx' in path for path in browser_requests) == 1, browser_requests
        assert sum('decoder_joint-model.int8.onnx' in path for path in browser_requests) == 1, browser_requests
        assert not receipt['pageErrors'], receipt['pageErrors']
        receipt['status'] = 'PASS'
        page.screenshot(path=str(args.output / 'final.png'), full_page=True)
        browser.close()
except Exception as error:
    receipt['error'] = repr(error)
    receipt['status'] = 'FAIL'
finally:
    receipt['requests'] = requests
    receipt['browserRequests'] = browser_requests
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    server.shutdown()
    server.server_close()
print(json.dumps({key:receipt.get(key) for key in ('status','error','physicalMicrophone')}, indent=2))
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
