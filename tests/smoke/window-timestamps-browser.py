"""Real Chrome, synthetic window results: no model/GPU/app acceptance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text

PROBE = r"""async () => {
  const { transcribeWithWindowing } = await import('/index.js');
  const equal = (actual, expected) => {
    if (JSON.stringify(actual) !== JSON.stringify(expected))
      throw new Error(JSON.stringify({actual, expected}));
  };
  const word = (index, text, startTime, endTime) => ({index,text,startTime,endTime});
  const result = words => ({text:words.map(w=>w.text).join(' '), words, warnings:[],
    meta:{detailLevel:'words',isFinal:true}});
  const inference = {sampleRate:16000,maxInputDurationSec:2,recommendedWindowDurationSec:2,
    minWindowDurationSec:1,maxWindowDurationSec:2,autoWindowThresholdSec:2,
    defaultOverlapSec:0.5,supportsWordTimestamps:true,supportsSegmentTimestamps:true,
    defaultSegmentationStrategy:'word-punctuation',defaultMergeStrategy:'word-dedupe'};
  const cases = [];
  for (const count of [2,3]) {
    const locals = [result([word(0,'alpha',.1,.4),word(1,'shared',1.6,1.9)]),
      result([word(0,'shared',.1,.5),word(1,'omega',.8,1.2)]),
      result([word(0,'tail',.1,.5)])];
    let calls=0;
    const out=await transcribeWithWindowing({input:new Float32Array((count===2?3.5:5)*16000),
      inference,options:{detail:'words'},transcribeWindow:async()=>locals[calls++]});
    const times=out.words.map(w=>[w.text,w.startTime,w.endTime]);
    equal(calls,count);
    equal(times,[['alpha',.1,.4],['shared',1.6,2],['omega',2.3,2.7],
      ...(count===3?[['tail',3.1,3.5]]:[])]);
    equal(locals[1].words[0].startTime,.1);
    cases.push({kind:'overlap',count,times});
  }
  for (const detail of ['words','segments']) {
    const locals=[];
    const out=await transcribeWithWindowing({input:new Float32Array(4*16000),
      inference:{...inference,defaultOverlapSec:0},options:{detail},
      transcribeWindow:async()=>{
        const text=locals.length?'Second.':'First.';
        const value=result(detail==='words'?[word(0,text,.25,1.25)]:[]);
        value.text=text; value.segments=[{index:0,text,startTime:.25,endTime:1.25}];
        locals.push(value); return value;
      }});
    equal(locals.length,2);
    equal(out.segments.map(s=>[s.startTime,s.endTime]),[[.25,1.25],[2.25,3.25]]);
    equal(locals[1].segments[0].startTime,.25);
    if(detail==='words') equal(out.words.map(w=>[w.startTime,w.endTime]),[[.25,1.25],[2.25,3.25]]);
    else equal(out.words,undefined);
    cases.push({kind:'offset',detail,segments:out.segments});
  }
  return cases;
}"""


def build_identity(directory):
    files = sorted(directory.rglob('*.js'))
    if not files or not (directory / 'index.js').is_file():
        raise ValueError('Requested compiled build is missing')
    manifest = [(p.relative_to(directory).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                for p in files]
    return {'files': len(files), 'sha256': hashlib.sha256(json.dumps(manifest).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if urlparse(args.url).hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('Only a loopback test server is allowed')
    require_new_outputs([args.output], protected_inputs=[Path(__file__)],
                        protected_directories=[ROOT / 'src', ROOT / 'models', args.build])
    identity = build_identity(args.build)
    tool_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='chrome', headless=True)
        try:
            page = browser.new_page()
            errors = []
            loaded = set()
            responses = []
            origin = urlparse(args.url)
            build = args.build.resolve()
            def verified_module(response):
                if not urlparse(response.url).path.endswith('.js'):
                    return
                try:
                    url = urlparse(response.url)
                    if (url.scheme, url.netloc) != (origin.scheme, origin.netloc):
                        raise ValueError('Module escaped the test origin')
                    local = (build / unquote(url.path).lstrip('/')).resolve()
                    local.relative_to(build)
                    body = response.body()
                    if response.status != 200 or hashlib.sha256(body).digest() != hashlib.sha256(local.read_bytes()).digest():
                        raise ValueError(f'Served module differs from requested build: {url.path}')
                    loaded.add(local.relative_to(build).as_posix())
                except Exception as error:
                    errors.append(str(error))
            page.on('response', lambda response: responses.append(response))
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(args.url, wait_until='networkidle')
            if 'Directory listing' not in page.content():
                raise ValueError('Expected isolated build directory server')
            try:
                cases = page.evaluate('() => Promise.race([(' + PROBE + ')(), new Promise((_, reject) => setTimeout(() => reject(new Error("Window probe exceeded 60 seconds")), 60000))])')
            except Exception as error:
                raise RuntimeError(f'Window probe failed; module/page errors: {errors}') from error
            for response in responses:
                verified_module(response)
            if errors:
                raise ValueError(f'Browser errors: {errors}')
            if 'index.js' not in loaded or len(cases) != 4:
                raise ValueError('Missing root module or expected probe cases')
            version = browser.version
        finally:
            browser.close()
    if identity != build_identity(args.build) or tool_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Diagnostic artifacts changed during run')
    receipt = {'status': 'PASS_BROWSER_SYNTHETIC_WINDOWING_ONLY', 'chrome': version,
               'build': str(args.build.resolve()), 'build_identity': identity,
               'tool_sha256': tool_hash, 'cases': cases,
               'verified_loaded_modules': sorted(loaded),
               'scope': 'Public compiled root API in Chrome; synthetic results, no ASR/GPU/companion acceptance'}
    write_new_text(args.output, json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
