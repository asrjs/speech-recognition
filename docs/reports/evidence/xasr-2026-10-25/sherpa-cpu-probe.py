import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import sherpa_onnx
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--audio',required=True);p.add_argument('--capture-dir',required=True);p.add_argument('--flow-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--chunk-ms',type=int,default=160);p.add_argument('--sample-rate',type=int,default=16000);a=p.parse_args()
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
model_dir=Path(a.model_dir); rate=a.sample_rate; chunk=int(rate*a.chunk_ms/1000)
files={'encoder':model_dir/f'encoder-{a.chunk_ms}ms.onnx','decoder':model_dir/f'decoder-{a.chunk_ms}ms.onnx','joiner':model_dir/f'joiner-{a.chunk_ms}ms.onnx','tokens':model_dir/'tokens.txt'}
for n,f in files.items():
 if not f.is_file():raise FileNotFoundError(f'{n}: {f}')
recognizer=sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(files['tokens']),encoder=str(files['encoder']),decoder=str(files['decoder']),joiner=str(files['joiner']),num_threads=1,sample_rate=rate,feature_dim=80,decoding_method='greedy_search',provider='cpu',model_type='zipformer2',enable_endpoint_detection=False)
def decode_one(pcm):
 x=np.asarray(pcm,dtype=np.float32).reshape(-1)
 stream=recognizer.create_stream();partials=[];offset=0;started=time.perf_counter()
 while offset<x.size:
  end=min(x.size,offset+chunk);stream.accept_waveform(rate,x[offset:end]);offset=end
  while recognizer.is_ready(stream):
   recognizer.decode_stream(stream);text=recognizer.get_result(stream)
   if not partials or partials[-1]['text']!=text:partials.append({'offsetSamples':offset,'text':text})
 stream.input_finished()
 while recognizer.is_ready(stream):recognizer.decode_stream(stream)
 result=recognizer.get_result(stream);elapsed=time.perf_counter()-started
 return {'text':result,'partials':partials,'decodeSeconds':round(elapsed,4)}
def stats(pcm):
 x=np.asarray(pcm,dtype=np.float32).reshape(-1)
 return {'frames':int(x.size),'durationSeconds':x.size/rate,'pcmFloat32LeSha256':sha(x.astype('<f4',copy=False).tobytes()),'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))) if x.size else 0.0,'peakAbs':float(np.max(np.abs(x))) if x.size else 0.0,'exactZeroFraction':float(np.mean(x==0)) if x.size else 1.0}
start=time.perf_counter();audio_path=Path(a.audio);wave,fs=sf.read(str(audio_path),dtype='float32',always_2d=True)
if fs!=rate:raise ValueError(f'input rate {fs} != {rate}; no implicit resampling in this control')
original=wave.mean(axis=1).astype(np.float32);original_result=decode_one(original)
original_entry={'path':str(audio_path.resolve()),'fileSha256':sha_file(audio_path),**stats(original),'sherpa':original_result}
metas=[]
for mp in Path(a.capture_dir).glob('*.json'):
 meta=json.loads(mp.read_text(encoding='utf-8'));raw=Path(meta['pcmPath']).read_bytes()
 if sha(raw)!=meta['pcmSha256'] or len(raw)!=meta['frames']*4:raise ValueError('capture metadata mismatch: '+str(mp))
 metas.append((meta,raw,np.frombuffer(raw,dtype='<f4').copy()))
flows=[]
for fp in sorted(Path(a.flow_dir).glob('mic-*.json')):
 flow=json.loads(fp.read_text(encoding='utf-8'));label=flow.get('flow','')
 if label=='mic-manual-manual-none':
  captured=[entry for entry in metas if entry[0]['flow']==label]
  if len(captured)!=1:raise ValueError(f'expected exactly one manual mic capture, got {len(captured)}')
  meta,raw,pcm=captured[0];flows.append({'flow':label,'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'fullCapture':{'requestId':meta['requestId'],**stats(pcm),'sherpa':decode_one(pcm)}})
 elif label.startswith('mic-speech-detect-'):
  relevant=sorted((entry for entry in metas if entry[0]['flow']==label),key=lambda entry:entry[0]['requestId']);segments=[]
  for meta,raw,pcm in relevant:
   segments.append({'requestId':meta['requestId'],**stats(pcm),'sherpa':decode_one(pcm)})
  joined=' '.join(s['sherpa']['text'].strip() for s in segments if s['sherpa']['text'].strip())
  flows.append({'flow':label,'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'segments':segments,'sherpaJoinedInRequestOrder':joined,'sherpaJoinedMatchesBrowserTranscript':joined==flow.get('transcript')})
receipt={'schemaVersion':1,'status':'OFFICIAL_SHERPA_CPU_CONTROL_COMPLETED_NOT_RUNTIME_PASS','scope':'Project-documented sherpa-onnx OnlineRecognizer.from_transducer CPU path on exact captured X-ASR float32 PCM. Each VAD segment uses a fresh stateful stream, receives 160ms chunks, then input_finished() and ready-drain exactly as the official reference runner; results joined in ascending requestId. This is a reference behavior control for the official graphs, not a numerical WebGPU parity test or microphone-flow PASS.','source':{'modelDir':str(model_dir.resolve()),'sherpaOnnxVersion':getattr(sherpa_onnx,'__version__','unknown'),'sherpaOnnxModule':str(Path(sherpa_onnx.__file__).resolve()),'python':sys.version,'platform':platform.platform(),'provider':'cpu','modelType':'zipformer2','decodingMethod':'greedy_search','enableEndpointDetection':False,'sampleRate':rate,'featureDim':80,'chunkMs':a.chunk_ms,'chunkSamples':chunk,'files':{name:{'path':str(path.resolve()),'sizeBytes':path.stat().st_size,'sha256':sha_file(path)} for name,path in files.items()}},'originalFixture':original_entry,'flows':flows,'elapsedSeconds':round(time.perf_counter()-start,3)}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'source':receipt['source'],'original':{'text':original_result['text'],'frames':original_entry['frames']},'flows':[{'flow':f['flow'],'browserTranscript':f.get('browserTranscript'),'sherpaJoined':f.get('sherpaJoinedInRequestOrder',f.get('fullCapture',{}).get('sherpa',{}).get('text')),'segments':[{'requestId':s['requestId'],'durationSeconds':s['durationSeconds'],'text':s['sherpa']['text']} for s in f.get('segments',[])]} for f in flows],'elapsedSeconds':receipt['elapsedSeconds'],'receipt':str(out.resolve())},ensure_ascii=False))
