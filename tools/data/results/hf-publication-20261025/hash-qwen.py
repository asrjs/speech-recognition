import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2])
files=['audio-encoder-dynamic.onnx','decoder-prefill-fp16.onnx','decoder-step-fp16.onnx','decoder-fp16.onnx.data']+[f'tokenizer/{n}' for n in ['tokenizer.json','tokenizer_config.json','added_tokens.json','special_tokens_map.json','merges.txt','vocab.json']]
rows=[]
for n in files:
 p=root/n; h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 rows.append({'path':n,'sizeBytes':p.stat().st_size,'sha256':h.hexdigest()})
manifest={'schemaVersion':1,'modelId':'Qwen/Qwen3-ASR-0.6B-hf','artifactSetId':'qwen3-asr-0.6b-local-dynamic-fp16-decoder-fp16','source':{'repo':'Qwen/Qwen3-ASR-0.6B','revision':'5eb144179a02acc5e5ba31e748d22b0cf3e303b0','license':'Apache-2.0'},'license':'Apache-2.0','executionProfiles':{'status':'experimental','note':'This bundle identity is hashed; it is not an approved general backend profile. See real-weight scope in repository README.','graphs':{'encoder':'audio-encoder-dynamic.onnx','decoderPrefill':'decoder-prefill-fp16.onnx','decoderStep':'decoder-step-fp16.onnx','decoderExternalData':'decoder-fp16.onnx.data'},'precision':{'encoder':'source graph metadata; see runtime inspection','decoder':'fp16'}},'artifacts':rows,'acceptance':{'status':'PARTIAL_PASS','scope':'Qwen v16 real-weight four-demo matrix: 16/18 exact requested flows; native streaming unsupported. Latest per-flow receipts are in asrjs/demos docs/reports/newer-models-2026-10-25.md.','accuracy':'Limited to recorded English JFK and Russian fixture cases; not an all-language accuracy claim.','physicalMicrophone':'UNVERIFIED'}}
(out/'qwen3-asr'/'asrjs-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(rows),'bytes':sum(x['sizeBytes'] for x in rows),'manifest':str(out/'qwen3-asr'/'asrjs-manifest.json')},indent=2))
