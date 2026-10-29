import hashlib,json
from pathlib import Path
root=Path(r'N:/github/asrjs/demos/tmp/real-e2e/qwen-loading-20261023/model-fp16')
files=['audio-encoder-dynamic.onnx','decoder-prefill-fp16.onnx','decoder-step-fp16.onnx','decoder-fp16.onnx.data','tokenizer/tokenizer.json']
rows=[]
for name in files:
 p=root/name; digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 rows.append({'path':name,'sizeBytes':p.stat().st_size,'sha256':digest.hexdigest()})
manifest={'schemaVersion':1,'modelId':'Qwen/Qwen3-ASR-0.6B-hf','artifactSetId':'qwen3-asr-0.6b-dynamic-encoder-fp16-decoder','status':'experimental','source':{'repo':'Qwen/Qwen3-ASR-0.6B','revision':'5eb144179a02acc5e5ba31e748d22b0cf3e303b0','license':'Apache-2.0'},'graphs':{'encoder':'audio-encoder-dynamic.onnx','decoderPrefill':'decoder-prefill-fp16.onnx','decoderStep':'decoder-step-fp16.onnx','decoderExternalData':'decoder-fp16.onnx.data','tokenizer':'tokenizer/tokenizer.json'},'artifacts':rows,'acceptance':{'status':'PARTIAL_PASS','scope':'The real-weight four-demo matrix recorded 16/18 exact fixture flows; native streaming was unsupported. See the 2026-10-25 demos report.','physicalMicrophone':'UNVERIFIED','broadAccuracy':'UNVERIFIED','allBackendQuantizationCombinations':'UNVERIFIED'}}
Path(r'N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/qwen3-asr/asrjs-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'bytes':sum(x['sizeBytes'] for x in rows),'artifacts':rows},indent=2))
