"""Real Chrome IndexedDB migration smoke; tiny intercepted fixture, no models.

Exercises internal built-in HF helpers, not the published public import surface.
"""
import argparse
import base64
import hashlib
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text
probe_preview_closed = importlib.import_module('browser-demo-smoke').probe_preview_closed


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python disables acceptance assertions; run without -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_new_outputs([args.output], protected_directories=[ROOT / 'src', ROOT / 'models', ROOT / 'tests'])
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Existing Node required')
    build = r"""
      import {build} from 'esbuild';
      import {readFile} from 'node:fs/promises';
      import {createHash} from 'node:crypto';
      import {extname} from 'node:path';
      const hash=b=>createHash('sha256').update(b).digest('hex');
      const sources={};
      const result=await build({stdin:{contents:
        "import {getModelText} from './src/runtime/huggingface.ts'; import {IndexedDbAssetCache} from './src/io/cache.ts'; globalThis.cacheProbe={getModelText,IndexedDbAssetCache};",
        resolveDir:process.cwd(),sourcefile:'cache-probe.ts'},
        bundle:true,platform:'browser',format:'esm',write:false,
        plugins:[{name:'identity',setup(b){b.onLoad({filter:/\.(ts|js|mjs|cjs)$/},async({path})=>{
          const bytes=await readFile(path); sources[path]=hash(bytes);
          return {contents:bytes.toString('utf8'),loader:extname(path)==='.ts'?'ts':'js'};
        });}}]});
      if(result.outputFiles.length!==1)throw Error('Expected single bundle');
      for(const [path,h] of Object.entries(sources))if(hash(await readFile(path))!==h)throw Error('Changed source');
      console.log(JSON.stringify({sources,bundle:Buffer.from(result.outputFiles[0].contents).toString('base64')}));
    """
    result = subprocess.run([node, '--input-type=module', '-e', build], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', timeout=30, check=True)
    built = json.loads(result.stdout)
    bundle = base64.b64decode(built['bundle'], validate=True)
    assets = {'/': b'<title>HF cache probe</title><script type="module" src="/probe.js"></script>',
              '/probe.js': bundle}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in assets:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript' if self.path.endswith('.js') else 'text/html')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(assets[self.path])
        def log_message(self, *values):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = 'http://127.0.0.1:' + str(server.server_port)
    inputs = [Path(__file__), Path(__file__).with_name('browser-demo-smoke.py'), ROOT / 'tools/artifact_outputs.py']
    tool_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    receipt = {'status': 'FAIL_REAL_CHROME_HF_CACHE', 'sources_sha256': built['sources'],
               'tools_sha256': tool_hashes,
               'bundle_sha256': hashlib.sha256(bundle).hexdigest(),
               'scope': __doc__, 'fixture_requests': [], 'blocked_requests': [], 'page_errors': []}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome', headless=True)
            try:
                context = browser.new_context(service_workers='block')
                fixture = 'https://huggingface.co/asrjs-fixture/cache/resolve/pinned/vocab.txt'
                allow_fixture = True
                def route_request(route):
                    url = route.request.url
                    if url in [origin + '/', origin + '/probe.js'] and route.request.method == 'GET':
                        route.continue_()
                    elif url == fixture and allow_fixture and route.request.method == 'GET':
                        receipt['fixture_requests'].append(url)
                        route.fulfill(status=200, body='pinned fixture bytes', headers={
                            'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'})
                    else:
                        receipt['blocked_requests'].append(url)
                        route.abort()
                context.route('**/*', route_request)
                page = context.new_page()
                page.on('pageerror', lambda error: receipt['page_errors'].append(str(error)))
                page.set_default_timeout(10000)
                page.goto(origin, wait_until='networkidle')
                page.wait_for_function('!!globalThis.cacheProbe')
                page.evaluate("""async()=>{
                  const cache=new cacheProbe.IndexedDbAssetCache();
                  for(const [key,text] of [['hf-asrjs-fixture/cache-pinned--vocab.txt','legacy wrong bytes'],
                     ['hf-asrjs-fixture/cache-main--vocab.txt','main fixture bytes']])
                    await cache.set(key,{bytes:new TextEncoder().encode(text)});
                }""")
                read = "() => cacheProbe.getModelText('asrjs-fixture/cache','vocab.txt',{revision:'pinned'})"
                assert page.evaluate(read) == 'pinned fixture bytes'
                assert len(receipt['fixture_requests']) == 1
                allow_fixture = False
                page.reload(wait_until='networkidle')
                page.wait_for_function('!!globalThis.cacheProbe')
                assert page.evaluate(read) == 'pinned fixture bytes'
                assert page.evaluate("() => cacheProbe.getModelText('asrjs-fixture/cache','vocab.txt')") == 'main fixture bytes'
                receipt['entries'] = page.evaluate("""async()=>{
                  const cache=new cacheProbe.IndexedDbAssetCache(); const entries={};
                  for(const key of ['hf-asrjs-fixture/cache-pinned--vocab.txt','hf-asrjs-fixture/cache-main--vocab.txt',
                    'hf-strict-revision-v1:'+JSON.stringify(['asrjs-fixture/cache','pinned','','vocab.txt'])]) {
                    const value=await cache.get(key); entries[key]=value?new TextDecoder().decode(value.bytes):null;
                  } return entries;
                }""")
                assert list(receipt['entries'].values()) == ['legacy wrong bytes', 'main fixture bytes', 'pinned fixture bytes']
                assert len(receipt['fixture_requests']) == 1
                assert not receipt['blocked_requests'] and not receipt['page_errors']
                receipt['browser_version'] = browser.version
            finally:
                browser.close()
        for path, expected in built['sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in tool_hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        receipt['sources_unchanged'] = True
        receipt['status'] = 'PASS_REAL_CHROME_HF_CACHE'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        receipt['preview_closed'] = probe_preview_closed(node, origin)
        if not receipt['preview_closed']['closed']:
            receipt['status'] = 'FAIL_REAL_CHROME_HF_CACHE'
        write_new_text(args.output, json.dumps(receipt, indent=2))
    if receipt['status'] != 'PASS_REAL_CHROME_HF_CACHE':
        raise RuntimeError('Cache smoke failed')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
