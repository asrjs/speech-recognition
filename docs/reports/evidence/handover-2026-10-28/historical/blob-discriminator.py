import json
from pathlib import Path
from playwright.sync_api import sync_playwright
results=[]
with sync_playwright() as p:
 for mode in ['chunks','native']:
  browser=p.chromium.launch(channel='chrome',headless=True)
  page=browser.new_page()
  page.goto('http://127.0.0.1:4343/')
  try:
   result=page.evaluate('''async mode=>{
    const response=await fetch('/model/encoder-model.fp16.onnx'); let blob;
    if(mode==='native') blob=await response.blob();
    else { const chunks=[]; const reader=response.body.getReader(); while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value.slice().buffer);} blob=new Blob(chunks); }
    const url=URL.createObjectURL(blob);
    try { const r=await fetch(url,{headers:{Range:'bytes=0-31'}}); const b=await r.arrayBuffer(); return {mode,size:blob.size,fetchStatus:r.status,readBytes:b.byteLength}; }
    finally {URL.revokeObjectURL(url);}
   }''',mode)
  except Exception as e: result={'mode':mode,'error':str(e)}
  results.append(result); print(json.dumps(result),flush=True);browser.close()
Path('tmp/approved-model-profiles/blob-discriminator.json').write_text(json.dumps(results,indent=2))
