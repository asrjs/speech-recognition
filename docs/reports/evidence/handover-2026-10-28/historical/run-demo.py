"""Real-weight demo acceptance with local replay of hash-verified HF graph bytes.

Only the exact pinned Parakeet resolve URLs are replayed. Hub API/catalog calls,
model loading, ONNX inference and UI operations are real. This does not certify
Hub weight-download transport. The ordinary runner retains strict ASR acceptance.
"""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('real_demo',Path(__file__).resolve().parents[3]/'demos/scripts/e2e/remaining-model-flows.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_probe=runner.add_gpu_probe
original_mark=runner.mark_gpu
original_receipt=runner.new_receipt

def install(page):
 original_probe(page)
 page.add_init_script('''(() => {
  const prefix='https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/f88260fa0777fe0868dda6df85d1a98f012a4a7a/';
  const allowed=new Set(['encoder-model.fp16.onnx','decoder_joint-model.int8.onnx','vocab.txt','config.json']);
  const original=globalThis.fetch.bind(globalThis);
  globalThis.__approvedAssetReplay=[];
  globalThis.fetch=(input,init)=>{
   const url=input instanceof Request?input.url:String(input);
   if(url.startsWith(prefix)){
    const name=decodeURIComponent(url.slice(prefix.length).split('?')[0]);
    if(!allowed.has(name)) throw Error('Unexpected profile artifact: '+name);
    globalThis.__approvedAssetReplay.push({requestedUrl:url,localPath:'/model/'+name});
    return original('/model/'+encodeURIComponent(name),init);
   }
   return original(input,init);
  };
 })();''')

def mark(page,receipt,key):
 original_mark(page,receipt,key)
 receipt['artifactReplay']=page.evaluate('globalThis.__approvedAssetReplay || []')

def receipt(*args,**kwargs):
 r=original_receipt(*args,**kwargs)
 r['weightTransport']='LOCAL_REPLAY_OF_HASH_VERIFIED_HF_ARTIFACTS'
 r['hubWeightDownloadTransport']='UNVERIFIED'
 return r

runner.add_gpu_probe=install
runner.mark_gpu=mark
runner.new_receipt=receipt
if __name__=='__main__': raise SystemExit(runner.main())
