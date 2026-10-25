import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[2] / 'speech-recognition'
output = root / 'docs/reports/evidence/model-loading-2026-10-25'
output.mkdir(parents=True, exist_ok=True)
rows = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    for framework in ('react', 'svelte', 'vue'):
        for backend, quant in (('webgpu-hybrid', 'fp16'), ('wasm', 'int8')):
            context = browser.new_context()
            page = context.new_page()
            seen = []
            def route(request):
                url = request.request.url
                if 'huggingface.co/' not in url:
                    request.continue_()
                else:
                    seen.append(url)
                    if '/api/' in url:
                        request.fulfill(status=200, content_type='application/json', body='[]')
                    else:
                        request.abort()
            page.route('**/*', route)
            page.goto(f'http://127.0.0.1:4190/examples/{framework}/', wait_until='networkidle')
            page.get_by_label('Backend', exact=True).select_option(backend)
            page.get_by_role('button', name='Load model', exact=True).click()
            page.get_by_role('alert').wait_for(timeout=30000)
            models = [url for url in seen if '/encoder-model.' in url]
            expected = f'/encoder-model.{quant}.onnx'
            assert models and models[0].endswith(expected), (framework, backend, seen)
            rows.append({'framework': framework, 'backend': backend, 'firstEncoderRequest': models[0],
                         'scope': 'request selection only; all model asset traffic aborted',
                         'error': page.get_by_role('alert').inner_text()})
            context.close()
    mounts = []
    for mount in ('showcase', 'playground', 'realtime', 'benchmark', 'examples/react', 'examples/svelte', 'examples/vue'):
        page = browser.new_page()
        response = page.goto(f'http://127.0.0.1:4190/{mount}/', wait_until='networkidle')
        assert response.status == 200, (mount, response.status)
        page.locator('main').first.wait_for()
        mounts.append({'mount': mount, 'status': response.status})
        page.close()
    browser.close()
result = {'loadRequests': rows, 'pageSmoke': mounts, 'realWeightInference': 'NOT RUN'}
(output / 'browser-request-selection.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
