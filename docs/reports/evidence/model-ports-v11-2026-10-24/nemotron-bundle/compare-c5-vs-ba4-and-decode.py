from __future__ import annotations
import datetime, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT=Path(r'N:/github/asrjs/speech-recognition')
TMP=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-conditional-drop-cli-real-20261024')
FEATURES=ROOT/'tools/data/results/nemotron/nemotron-3.5-nemo-features-jfk.json'
NATIVE=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-native-cpu-reference-20261024/native-cache-stream-jfk-560ms-c5-vs-ba4.json')
ORIGINAL=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-staticdrop-surgery-20261024/encoder.dynamic-drop.onnx')
CANDIDATE=TMP/'encoder.onnx'
MODEL=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-conditional-drop-cli-real-20261024')
OUT=TMP/'c5-vs-ba4-and-decoder-sdk-greedy.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def session(path):
 o=ort.SessionOptions();o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.intra_op_num_threads=2;o.inter_op_num_threads=2;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
 s=ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
 if s.get_providers()!=['CPUExecutionProvider']:raise RuntimeError(f'non-CPU provider {path}: {s.get_providers()}')
 return s

def feed_encoder(features, ch, tm, ln):
 return {'audio_signal':features.astype(np.float32,copy=False),'length':np.asarray([features.shape[1]],dtype=np.int64),'cache_last_channel':ch,'cache_last_time':tm,'cache_last_channel_len':ln,'lang_id':np.asarray([0],dtype=np.int64)}

def metrics(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return {'sameShape':a.shape==b.shape,'shapeA':list(a.shape),'shapeB':list(b.shape),'arrayEqual':bool(np.array_equal(a,b)) if a.shape==b.shape else False,'maxAbs':float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max(initial=0)) if a.shape==b.shape else None}

start=time.monotonic(); raw=FEATURES.read_bytes(); fjson=json.loads(raw)
feature_bins=np.asarray(fjson['feature'],dtype=np.float32)
if feature_bins.shape[0]!=128:raise ValueError(f'expected [128,T], got {feature_bins.shape}')
mel=feature_bins.T.copy(); total=mel.shape[0]
native=json.loads(NATIVE.read_text(encoding='utf-8'))
if native.get('fixtureSha256')!='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e':raise RuntimeError('official native reference fixture mismatch')
if len(native.get('tokenIds',[]))!=48:raise RuntimeError('expected frozen native 48-token IDs')
# Native buffer contract observed: first 49; continuation window has 9 left context +56 new.
chunks=[];cursor=0
chunks.append(mel[0:49][None,:,:].copy());cursor=49
while cursor<total:
 lo=max(0,cursor-9);hi=min(total,cursor+56)
 chunks.append(mel[lo:hi][None,:,:].copy());cursor+=56
if [x.shape[1] for x in chunks][0]!=49 or [x.shape[1] for x in chunks][1]!=65 or [x.shape[1] for x in chunks][-1]!=53:raise RuntimeError(f'unexpected chunks {[x.shape[1] for x in chunks]} totalFrames={total}')
# Both graphs run CPU-only with optimizations disabled. Candidate owns the cache chain.
candidate=session(CANDIDATE); original=session(ORIGINAL)
ch=np.zeros((1,24,56,1024),np.float32);tm=np.zeros((1,24,1024,8),np.float32);ln=np.zeros((1,),np.int64)
cache_branch=[];encoded=[];out_names=[x.name for x in candidate.get_outputs()]
for i,feat in enumerate(chunks):
 fd=feed_encoder(feat,ch,tm,ln)
 cand=dict(zip(out_names,candidate.run(None,fd)))
 if True:
  base_names=[x.name for x in original.get_outputs()]
  base=dict(zip(base_names,original.run(None,fd)))
  comp={k:metrics(base[k],cand[k]) for k in base}
  cache_branch.append({'step':i,'inputFrames':int(feat.shape[1]),'cacheLengthInput':ln.tolist(),'dropBranch':'continuation-2','outputs':comp,'allArrayEqual':all(v['arrayEqual'] for v in comp.values())})
 encoded.append(cand['outputs'])
 ch=cand['cache_last_channel_next'];tm=cand['cache_last_time_next'];ln=cand['cache_last_channel_len_next']
del original;gc.collect()
enc=np.concatenate(encoded,axis=1)
# FP32 decoder + joint greedy RNNT decode over the exact candidate 20-chunk encoder sequence.
dec=session(MODEL/'decoder.onnx');joint=session(MODEL/'joint.onnx')
h=np.zeros((2,1,640),np.float32);c=np.zeros_like(h);target=13087;tokens=[];token_frames=[];max_steps=1024
last_t=0
for _step in range(max_steps):
 if last_t>=enc.shape[1]:break
 d=dict(zip([x.name for x in dec.get_outputs()],dec.run(None,{'targets':np.asarray([[target]],np.int64),'h_in':h,'c_in':c})))
 dec_out=d['decoder_output'].transpose(0,2,1) # [1,1,640]
 logits=joint.run(None,{'encoder_output':enc[:,last_t:,:].astype(np.float32,copy=False),'decoder_output':dec_out})[0]
 per_frame=np.argmax(logits[0,:,0,:],axis=-1)
 nonblank=np.flatnonzero(per_frame!=13087)
 if not len(nonblank):break
 emitted_frame=last_t+int(nonblank[0]);emitted=int(per_frame[nonblank[0]])
 tokens.append(emitted);token_frames.append(emitted_frame)
 # Exact SDK contract: preserve predictor state when all intervening frames are blank; commit only when a label is emitted.
 h=d['h_out'];c=d['c_out'];target=emitted;last_t=emitted_frame
comparison={'nativeTokenCount':len(native['tokenIds']),'candidateTokenCount':len(tokens),'exactTokenIdMatch':tokens==native['tokenIds'],'firstMismatchIndex':next((i for i,(a,b) in enumerate(zip(native['tokenIds'],tokens)) if a!=b),None),'nativePrefix':native['tokenIds'][:64],'candidatePrefix':tokens[:64],'candidateExtraOrMissingAfterCommonPrefix':{'nativeTail':native['tokenIds'][next((i for i,(a,b) in enumerate(zip(native['tokenIds'],tokens)) if a!=b),min(len(tokens),len(native['tokenIds']))):],'candidateTail':tokens[next((i for i,(a,b) in enumerate(zip(native['tokenIds'],tokens)) if a!=b),min(len(tokens),len(native['tokenIds']))):]},'tokenFrameIndices':token_frames,'maxDecodeSteps':max_steps}
branch_fail=[r for r in cache_branch if not r['allArrayEqual']]
record={'schemaVersion':1,'generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':{'continuationDrop2BitExact':'PASS' if not branch_fail and len(cache_branch)==len(chunks) else 'FAIL','candidateEncoderFp32DecoderJointRawIds':'PASS' if comparison['exactTokenIdMatch'] else 'FAIL'},'providers':['CPUExecutionProvider'],'threads':2,'optimization':'ORT_DISABLE_ALL','featureSource':str(FEATURES),'featureSourceSha256':hashlib.sha256(raw).hexdigest(),'featureMeta':{k:fjson.get(k) for k in ('fixture','sampleRate','nFft','winLength','hopLength','nMels','normalize','preemphasis','dither','logZeroGuard','featureShape')},'nativeFixtureSha256':native['fixtureSha256'],'nativeCheckpointSha256':native['checkpointSha256'],'candidateGraphSha256':sha(CANDIDATE),'originalGraphSha256':sha(ORIGINAL),'candidateWeightsSidecarSha256':sha(TMP/'encoder.onnx.data'),'originalWeightsSidecarSha256':sha(ROOT/'tmp/nemotron-fp32-repack-20261024/encoder-packed/encoder.onnx.data'),'chunkFrames':[int(x.shape[1]) for x in chunks],'chunkCount':len(chunks),'continuationDrop2Comparison':{'stepCount':len(cache_branch),'allArrayEqual':not branch_fail,'failSteps':[r for r in branch_fail],'proofScope':'same features and cache tensors generated by candidate first49 frame, then fed to original and candidate in each continuation step; validates branch remains identical; not native-cache baseline'},'encodedSequenceShape':list(enc.shape),'decoderGreedyProtocol':'faithful port of src/models/nemotron-rnnt/executor.ts runGreedyDecoder: predictor sees current target, scan remaining frames for first nonblank, preserve state across blank, commit h/c only on emitted token', 'decodedRawIdComparison':comparison,'elapsedSeconds':round(time.monotonic()-start,3),'scope':'CPU component/control only; does not validate browser/WebGPU/WASM or model frontend against fresh stream'}
OUT.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(record,indent=2,ensure_ascii=False))


