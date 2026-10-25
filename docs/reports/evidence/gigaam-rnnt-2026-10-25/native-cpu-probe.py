import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import gigaam
from gigaam.preprocess import load_audio
p=argparse.ArgumentParser(); p.add_argument('--audio',required=True); p.add_argument('--capture-dir',required=True); p.add_argument('--flow-dir',required=True); p.add_argument('--download-root',required=True); p.add_argument('--output',required=True); a=p.parse_args()
def sha(b): return hashlib.sha256(b).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
torch.set_num_threads(min(4,torch.get_num_threads()))
root=Path(a.download_root); ckpt=root/'v3_e2e_rnnt.ckpt'; tokenizer=root/'v3_e2e_rnnt_tokenizer.model'
if not ckpt.is_file() or not tokenizer.is_file(): raise FileNotFoundError('Expected existing official checkpoint and tokenizer under '+str(root))
ckpt_md5=__import__('hashlib').md5(ckpt.read_bytes()).hexdigest()
if ckpt_md5!='2730de7545ac43ad256485a462b0a27a': raise ValueError('Official checkpoint MD5 mismatch: '+ckpt_md5)
model=gigaam.load_model('v3_e2e_rnnt',fp16_encoder=False,use_flash=False,device='cpu',download_root=str(root)); model.eval()
def infer(pcm):
 x=np.asarray(pcm,dtype=np.float32).reshape(-1).copy(); wav=torch.from_numpy(x).unsqueeze(0); lengths=torch.tensor([x.size],dtype=torch.long)
 with torch.inference_mode():
  encoded,encoded_len=model(wav,lengths); decoded=model.decoding.decode(model.head,encoded,encoded_len)[0]
 text,token_ids,token_frames=decoded
 return {'text':str(text),'tokenIds':[int(v) for v in token_ids],'tokenFrames':[int(v) for v in token_frames],'encodedShape':list(encoded.shape),'encodedLengths':[int(v) for v in encoded_len.tolist()]}
def stats(pcm):
 x=np.asarray(pcm,dtype=np.float32).reshape(-1)
 return {'frames':int(x.size),'durationSeconds':x.size/16000,'pcmFloat32LeSha256':sha(x.astype('<f4',copy=False).tobytes()),'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))) if x.size else 0.0,'peakAbs':float(np.max(np.abs(x))) if x.size else 0.0,'exactZeroFraction':float(np.mean(x==0)) if x.size else 1.0}
start=time.perf_counter(); original_path=Path(a.audio); original=load_audio(str(original_path)).cpu().numpy().astype(np.float32); original_infer=infer(original)
# Confirm the official path-based API uses the same reference text on the original example.
with torch.inference_mode(): wrapper_text=str(model.transcribe(str(original_path)))
original_entry={'path':str(original_path.resolve()),'wavSha256':sha_file(original_path),**stats(original),'nativeDirectPcm':original_infer,'officialTranscribeText':wrapper_text,'directMatchesOfficialTranscribe':original_infer['text']==wrapper_text}
metas=[]
for mp in Path(a.capture_dir).glob('*.json'):
 meta=json.loads(mp.read_text(encoding='utf-8')); raw=Path(meta['pcmPath']).read_bytes()
 if sha(raw)!=meta['pcmSha256'] or len(raw)!=meta['frames']*4: raise ValueError('Capture metadata mismatch: '+str(mp))
 metas.append((meta,raw,np.frombuffer(raw,dtype='<f4').copy()))
flows=[]
for fp in sorted(Path(a.flow_dir).glob('mic-*.json')):
 flow=json.loads(fp.read_text(encoding='utf-8')); label=flow.get('flow','')
 if label=='mic-manual-manual-none':
  cap=[v for v in metas if v[0]['flow']==label]
  if len(cap)!=1: raise ValueError(f'Expected one manual capture, got {len(cap)}')
  meta,raw,pcm=cap[0]; flows.append({'flow':label,'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'fullCapture':{'requestId':meta['requestId'],**stats(pcm),'native':infer(pcm)}})
 elif label.startswith('mic-speech-detect-'):
  relevant=sorted((v for v in metas if v[0]['flow']==label),key=lambda v:v[0]['requestId']); segments=[]
  for meta,raw,pcm in relevant:
   t=time.perf_counter(); result=infer(pcm); segments.append({'requestId':meta['requestId'],**stats(pcm),'native':result,'cpuSeconds':round(time.perf_counter()-t,4)})
  joined=' '.join(s['native']['text'].strip() for s in segments if s['native']['text'].strip())
  flows.append({'flow':label,'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'segments':segments,'nativeJoinedInRequestOrder':joined,'nativeJoinedMatchesBrowserTranscript':joined==flow.get('transcript')})
repo=Path(r'N:\github\salute-developers\GigaAM')
try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()
except Exception: revision=None
receipt={'schemaVersion':1,'status':'OFFICIAL_CPU_REFERENCE_COMPLETED_NOT_RUNTIME_PASS','scope':'Official GigaAM v3 E2E RNN-T PyTorch CPU decode on the exact captured float32 PCM, separately per VAD request and joined by ascending requestId. Full manual capture is decoded once; official Pushkin example is the source-model control. This establishes neither browser/ONNX parity nor VAD acceptance.','source':{'modelName':'v3_e2e_rnnt','officialRepo':str(repo),'gigaamGitRevision':revision,'checkpointPath':str(ckpt.resolve()),'checkpointMd5':ckpt_md5,'checkpointSha256':sha_file(ckpt),'tokenizerPath':str(tokenizer.resolve()),'tokenizerSha256':sha_file(tokenizer),'gigaamModule':str(Path(gigaam.__file__).resolve()),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','encoderFp16':False,'flashAttention':False,'localCheckpointOnly':True},'officialExample':original_entry,'flows':flows,'elapsedSeconds':round(time.perf_counter()-start,3)}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'source':receipt['source'],'original':{'direct':original_infer['text'],'wrapper':wrapper_text,'match':original_entry['directMatchesOfficialTranscribe']},'flows':[{'flow':f['flow'],'browserTranscript':f.get('browserTranscript'),'nativeJoined':f.get('nativeJoinedInRequestOrder',f.get('fullCapture',{}).get('native',{}).get('text')),'segments':[{'requestId':s['requestId'],'durationSeconds':s['durationSeconds'],'text':s['native']['text']} for s in f.get('segments',[])]} for f in flows],'elapsedSeconds':receipt['elapsedSeconds'],'receipt':str(out.resolve())},ensure_ascii=False))
