"""Drive the real local-folder demo in installed Chrome and save a fresh receipt.

Requires an already-running examples/demo Vite server; never downloads models.
Smoke/text checks are not stage numerical parity.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text

spec = importlib.util.spec_from_file_location('browser_smoke', Path(__file__).with_name('browser-demo-smoke.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


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
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.query or parsed.fragment:
        parser.error('--url must be a plain loopback HTTP(S) page URL')
    require_new_outputs([args.output], protected_inputs=[args.audio, args.oracle],
                        protected_directories=[args.model_root])
    url = args.url + '?' + urlencode(dict(auto=1, backend=args.backend, precision=args.precision, repeat=args.repeat))
    receipt = dict(status='FAIL_AUTO_BROWSER', url=url, scope='real-browser-smoke-and-oracle-output-only',
                   page_errors=[], console=[], blocked_requests=[])
    try:
        inputs = helpers.local_model_inputs(args.model_root, args.precision, 'nemo-tdt')
        oracle = json.loads(args.oracle.read_text(encoding='utf-8'))
        audio_hash = helpers.digest(args.audio)
        if oracle['meta']['model'] != 'oruk/orukeet' or oracle['meta']['audio_sha256'] != audio_hash:
            raise ValueError('Oracle model/audio binding mismatch')
        receipt.update(inputs=[dict(path=str(p.resolve()), sha256=helpers.digest(p)) for p in inputs],
                       audio_sha256=audio_hash, oracle_sha256=helpers.digest(args.oracle))
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True,
                args=['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'])
            try:
                receipt['browser_version'] = browser.version
                # Only loopback URLs are allowed; the demo uses a local dev certificate.
                context = browser.new_context(service_workers='block', ignore_https_errors=True)
                origin = f'{parsed.scheme}://{parsed.netloc}'
                def route_request(route):
                    if route.request.url.startswith(origin + '/'):
                        route.continue_()
                    else:
                        receipt['blocked_requests'].append(route.request.url)
                        route.abort()
                context.route('**/*', route_request)
                page = context.new_page()
                page.on('pageerror', lambda error: receipt['page_errors'].append(str(error)))
                page.on('console', lambda message: receipt['console'].append(message.text[:2000]) if len(receipt['console']) < 100 else None)
                page.goto(url, wait_until='networkidle', timeout=60000)
                page.locator('#model').set_input_files(str(args.model_root.resolve()), timeout=60000)
                page.locator('#audio').set_input_files(str(args.audio.resolve()))
                page.wait_for_function("document.getElementById('result').textContent.trim().startsWith('{')", timeout=args.timeout_ms)
                result = json.loads(page.locator('#result').inner_text())
                receipt['result'] = result
                receipt['text_exact'] = all(t['text'] == oracle['result']['text'] for t in result['transcripts']) and bool(result['transcripts'])
                receipt['tokens_exact'] = all(t['tokenIds'] == oracle['result']['token_ids'] for t in result['transcripts']) and bool(result['transcripts'])
                if result['status'] == 'PASS_ORUKEET_SMOKE' and result['disposed'] and receipt['text_exact'] and receipt['tokens_exact'] and not receipt['page_errors'] and not receipt['blocked_requests']:
                    receipt['status'] = 'PASS_AUTO_BROWSER_TOKEN_TEXT_SMOKE'
            finally:
                browser.close()
                receipt['browser_closed'] = True
    except Exception as error:
        receipt['error'] = str(error)
    finally:
        write_new_text(args.output, json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=receipt['status'], output=str(args.output.resolve()), error=receipt.get('error'))))
    return 0 if receipt['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
