import hashlib,json
from pathlib import Path
root=Path(r'N:/github/asrjs/demos/tmp/real-e2e/remaining-models-20261024/native-moonshine-streaming-tiny')
out=Path(r'N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/moonshine-streaming-tiny')
reference=json.loads(Path(r'N:/github/asrjs/demos/docs/reports/evidence/context-preserving-capture-2026-10-25/parent-model-artifacts.json').read_text(encoding='utf-8'))
expected={Path(x['path']).name:x for x in reference['files'] if str(root).lower() in x['path'].lower()}
files=['encoder_model.onnx','encoder_model.onnx.data','encoder_chunked_model.onnx','encoder_chunked_model.onnx.data','decoder_prefill_model.onnx','decoder_prefill_model.onnx.data','decoder_step_model.onnx','decoder_step_model.onnx.data','tokenizer.json','config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json']
rows=[]
for name in files:
 p=root/name; h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 digest=h.hexdigest(); exp=expected.get(name)
 if not exp or exp['sha256']!=digest or exp['sizeBytes']!=p.stat().st_size: raise SystemExit(f'No exact independently recorded hash for {name}')
 rows.append({'path':name,'sizeBytes':p.stat().st_size,'sha256':digest,'matchesIndependentModelArtifactReceipt':True})
manifest={'schemaVersion':1,'modelId':'moonshine-streaming-tiny','artifactSetId':'moonshine-streaming-tiny-remapped-4graph-fp32','status':'experimental','source':{'repo':'moonshine-ai/moonshine-streaming-tiny','revision':'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a','license':'MIT','variant':'streaming tiny; custom remapped 4-graph bundle'},'graphs':{'encoder':'encoder_model.onnx','chunkedEncoder':'encoder_chunked_model.onnx','decoderPrefill':'decoder_prefill_model.onnx','decoderStep':'decoder_step_model.onnx','externalData':'all matching *.onnx.data files'},'processor':'config.json + preprocessor_config.json + processor_config.json + tokenizer_config.json + generation_config.json + tokenizer.json','artifacts':rows,'acceptance':{'status':'PARTIAL_PASS','scope':'The 2026-10-24 real-weight app matrix recorded 13/18 requested flows. Manual captures and Playground recordings were empty on padded input; WebGPU FireRed missed one word. Native-stream flow accepted only in tested scenarios.','physicalMicrophone':'UNVERIFIED','broadAccuracy':'UNVERIFIED','allBackendQuantizationCombinations':'UNVERIFIED'}}
(out/'asrjs-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
source=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
for row in source.get('files',[]): row.pop('source',None)
source['sourceRepository']='moonshine-ai/moonshine-streaming-tiny'; source['sourceRevision']='f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a'; source['license']='MIT'; source['status']='Artifact hashes cross-checked against independent parent model-artifacts receipt; this is not an accuracy claim.'
(out/'asrjs-provenance.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(rows),'bytes':sum(x['sizeBytes'] for x in rows),'hashesMatchParentReceipt':all(x['matchesIndependentModelArtifactReceipt'] for x in rows)},indent=2))
