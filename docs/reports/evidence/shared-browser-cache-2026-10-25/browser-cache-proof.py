import hashlib,json,mimetypes,os
from pathlib import Path
from urllib.parse import urlsplit,unquote
from playwright.sync_api import sync_playwright

PORT=int(os.environ.get('ASRJS_TEST_PORT','4191'))
BASE=f'http://127.0.0.1:{PORT}'
SDK=Path('../speech-recognition/dist').resolve()
OUT=Path(f'tmp/shared-cache-browser-proof-{PORT}');OUT.mkdir(exist_ok=True)
DATA=bytes(range(256))*256
SHA=hashlib.sha256(DATA).hexdigest()
URL='https://huggingface.co/asrjs/cache-fixture/resolve/main/onnx/encoder.onnx'
counts={'fixtureNetworkRequests':0};receipts=[]
with sync_playwright() as p:
 browser=p.chromium.launch(channel='chrome',headless=True)
 context=browser.new_context()
 def route_request(route):
  url=route.request.url;parts=urlsplit(url)
  if '/__asrjs-test-sdk/' in parts.path:
   relative=unquote(parts.path.split('/__asrjs-test-sdk/',1)[1]);target=(SDK/relative).resolve()
   if not target.is_relative_to(SDK) or not target.is_file():route.fulfill(status=404,body='missing');return
   route.fulfill(status=200,content_type='text/javascript',body=target.read_bytes());return
  if url==URL:
   counts['fixtureNetworkRequests']+=1
   route.fulfill(status=200,content_type='application/octet-stream',headers={'Access-Control-Allow-Origin':'*','Cross-Origin-Resource-Policy':'cross-origin'},body=DATA);return
  if parts.hostname not in ('127.0.0.1','localhost',None):route.abort();return
  route.continue_()
 context.route('**/*',route_request)
 mounts=['showcase','playground','realtime','benchmark','examples/react','examples/svelte','examples/vue']
 for index,mount in enumerate(mounts):
  page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
  response=page.goto(f'{BASE}/{mount}/',wait_until='networkidle');assert response.status==200
  result=page.evaluate('''async ({index,url}) => {
   const io=await import('/__asrjs-test-sdk/io.js');
   const cache=io.getDefaultIndexedDbAssetCache();
   const mode=index%3;
   const request=mode===0
    ? {id:'family',provider:'huggingface',repoId:'asrjs/cache-fixture',filename:'onnx/encoder.onnx',cacheKey:'huggingface:asrjs/cache-fixture:main:onnx/encoder.onnx'}
    : mode===1
    ? {id:'helper',provider:'huggingface',repoId:'asrjs/cache-fixture',subfolder:'onnx',filename:'encoder.onnx',cacheKey:'hf-asrjs/cache-fixture-main-onnx-encoder.onnx'}
    : {id:'url',provider:'url',url,cacheKey:`url:${url}`,preferBlobUrl:true};
   const progress=[];request.onProgress=(event)=>progress.push(event.source);
   const provider=mode===2?io.createBrowserFetchAssetProvider({cache}):io.createHuggingFaceAssetProvider({cache});
   const handle=await provider.resolve(request);
   let bytes;
   try {
    if(mode===2){const locator=await handle.getLocator('url');bytes=new Uint8Array(await (await fetch(locator)).arrayBuffer());}
    else bytes=await handle.readBytes();
   } finally {await handle.dispose();}
   const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
   const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('asrjs-cache-db');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
   const keys=await new Promise((resolve,reject)=>{const r=db.transaction('asset-cache').objectStore('asset-cache').getAllKeys();r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});db.close();
   return {origin:location.origin,bytes:bytes.length,sha256:hash,source:progress.at(-1),keys};
  }''',{'index':index,'url':URL})
  assert result['sha256']==SHA,result
  assert result['source']==('network' if index==0 else 'cache'),result
  assert len(result['keys'])==1,result
  assert not errors,errors
  receipts.append({'app':mount,**result,'pageErrors':errors})
  page.close()
 assert counts['fixtureNetworkRequests']==1,counts
 page=context.new_page();page.goto(BASE+'/',wait_until='networkidle');page.screenshot(path=str(OUT/'portal.png'),full_page=True)
 legacy=page.evaluate('''async () => {
  const io=await import('/__asrjs-test-sdk/io.js');const cache=io.getDefaultIndexedDbAssetCache();
  const old='hf-asrjs/cache-fixture-main--legacy.bin';await cache.set(old,{bytes:new Uint8Array([11,22,33])});
  const p=io.createHuggingFaceAssetProvider({cache});const h=await p.resolve({id:'old',provider:'huggingface',repoId:'asrjs/cache-fixture',filename:'legacy.bin',cacheKey:'huggingface:asrjs/cache-fixture:main:legacy.bin'});
  const bytes=Array.from(await h.readBytes());await h.dispose();
  const canonical=await cache.get('hf-asset-v1:["asrjs/cache-fixture","main","legacy.bin"]');
  const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('firered-vad-web-cache',2);r.onupgradeneeded=()=>r.result.createObjectStore('asset-cache');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  const key='firered:https://example.invalid/existing-vad.onnx';
  await new Promise((resolve,reject)=>{const t=db.transaction('asset-cache','readwrite');t.objectStore('asset-cache').put(new Blob([new Uint8Array([44,55])]),key);t.oncomplete=resolve;t.onerror=()=>reject(t.error);});db.close();
  const fire=await import('/__asrjs-test-sdk/runtime/firered-vad/core/asset-cache.js');
  const reused=await fire.getDefaultAssetCache().get(key);const copied=await cache.get(key);
  return {legacyBytes:bytes,legacyCopied:canonical!==null,fireRedBytes:Array.from(reused.bytes),fireRedCopied:copied!==null};
 }''')
 assert legacy=={'legacyBytes':[11,22,33],'legacyCopied':False,'fireRedBytes':[44,55],'fireRedCopied':False},legacy
 other=context.new_page();other.goto(f'http://127.0.0.1:{4190 if PORT==4191 else 4191}/',wait_until='networkidle')
 isolated=other.evaluate('''async () => {const io=await import('/__asrjs-test-sdk/io.js');const c=io.getDefaultIndexedDbAssetCache();return (await c.get('hf-asset-v1:["asrjs/cache-fixture","main","onnx/encoder.onnx"]'))===null;}''')
 assert isolated,'Different port unexpectedly shared IndexedDB'
 browser.close()
proof={'scope':'Actual Chrome IndexedDB and public asset provider, deterministic 64 KiB fixture; not model inference or cross-site sharing','payloadSha256':SHA,**counts,'apps':receipts,'legacy':legacy,'differentPortIsolated':isolated}
(OUT/'results.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps({'networkRequests':counts['fixtureNetworkRequests'],'cacheReads':len(receipts)-1,'legacy':legacy,'apps':len(receipts)}))
