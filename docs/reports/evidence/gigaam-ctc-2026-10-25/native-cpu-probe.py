import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import gigaam
from gigaam.preprocess import load_audio

p=argparse.ArgumentParser()
p.add_argument('--audio',required=True); p.add_argument('--capture-dir',required=True); p.add_argument('--flow-dir',required=True)
p.add_argument('--download-root',required=True); p.add_argument('--output',required=True)
a=p.parse_args()
def sha(b): return hashlib.sha256(b).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
torch.set_num_threads(min(4,torch.get_num_threads()))
ckpt=Path(a.download_root)/'multilingual_ctc.ckpt'
if not ckpt.is_file(): raise FileNotFoundError(f'Expected existing official checkpoint: {ckpt}')
model=gigaam.load_model('multilingual_ctc',fp16_encoder=False,use_flash=False,device='cpu',download_root=str(Path(a.download_root)))
model.eval()
def infer(pcm):
    pcm=np.asarray(pcm,dtype=np.float32).reshape(-1)
    wav=torch.from_numpy(pcm.copy()).unsqueeze(0)
    length=torch.tensor([pcm.size],dtype=torch.long)
    with torch.inference_mode():
        encoded,encoded_len=model(wav,length)
        decoded=model.decoding.decode(model.head,encoded,encoded_len)[0]
    text,token_ids,token_frames=decoded
    return {'text':str(text),'tokenIds':[int(v) for v in token_ids],'tokenFrames':[int(v) for v in token_frames], 'encodedShape':list(encoded.shape),'encodedLengths':[int(v) for v in encoded_len.tolist()]}
def stats(pcm):
    x=np.asarray(pcm,dtype=np.float32).reshape(-1)
    return {'frames':int(x.size),'durationSeconds':x.size/16000,'pcmFloat32LeSha256':sha(x.astype('<f4',copy=False).tobytes()),'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))) if x.size else 0.0,'peakAbs':float(np.max(np.abs(x))) if x.size else 0.0,'exactZeroFraction':float(np.mean(x==0)) if x.size else 1.0}
start=time.perf_counter()
original_path=Path(a.audio)
original=torch.from_numpy(load_audio(str(original_path)).cpu().numpy().astype(np.float32)).numpy()
original_result=infer(original)
original_entry={'path':str(original_path.resolve()),'wavSha256':sha_file(original_path),**stats(original),'native':original_result}
metas=[]
for mp in Path(a.capture_dir).glob('*.json'):
    meta=json.loads(mp.read_text(encoding='utf-8')); raw=Path(meta['pcmPath']).read_bytes()
    if sha(raw)!=meta['pcmSha256'] or len(raw)!=meta['frames']*4: raise ValueError(f'capture metadata mismatch: {mp}')
    pcm=np.frombuffer(raw,dtype='<f4').copy()
    metas.append((meta,raw,pcm))
flows=[]
for fp in sorted(Path(a.flow_dir).glob('mic-*.json')):
    flow=json.loads(fp.read_text(encoding='utf-8'))
    if flow.get('flow')=='mic-manual-manual-none':
        pair=[(m,r,x) for m,r,x in metas if m['flow']==flow['flow']]
        if len(pair)!=1: raise ValueError(f'Expected one manual PCM capture for {flow["flow"]}, got {len(pair)}')
        m,raw,pcm=pair[0]; native=infer(pcm)
        flows.append({'flow':flow['flow'],'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'fullCapture':{'requestId':m['requestId'],**stats(pcm),'native':native}})
    elif flow.get('flow','').startswith('mic-speech-detect-'):
        relevant=sorted(((m,r,x) for m,r,x in metas if m['flow']==flow['flow']),key=lambda it:it[0]['requestId'])
        segments=[]
        for m,raw,pcm in relevant:
            t=time.perf_counter(); native=infer(pcm)
            segments.append({'requestId':m['requestId'],**stats(pcm),'native':native,'cpuSeconds':round(time.perf_counter()-t,4)})
        joined=' '.join(x['native']['text'].strip() for x in segments if x['native']['text'].strip())
        flows.append({'flow':flow['flow'],'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'segments':segments,'nativeJoinedInRequestOrder':joined,'nativeJoinedMatchesBrowserTranscript':joined==flow.get('transcript')})
repo=Path(r'N:\github\salute-developers\GigaAM')
try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()
except Exception: revision=None
receipt={'schemaVersion':1,'status':'OFFICIAL_CPU_REFERENCE_COMPLETED_NOT_RUNTIME_PASS','scope':'Official GigaAM PyTorch CPU inference on exact captured float32 PCM. Whole-clip manual mic is one inference; each VAD capture is one independent inference joined by requestId. The native model path receives the exact float32 samples directly (no PCM16 roundtrip). This does not establish ONNX/WebGPU numerical parity or microphone flow acceptance.','source':{'modelName':'multilingual_ctc','officialRepo':str(repo),'gigaamGitRevision':revision,'checkpointPath':str(ckpt.resolve()),'checkpointSha256':sha_file(ckpt),'checkpointMd5':__import__('hashlib').md5(ckpt.read_bytes()).hexdigest(),'gigaamModule':str(Path(gigaam.__file__).resolve()),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','encoderFp16':False,'flashAttention':False,'localCheckpointOnly':True},'originalFixture':original_entry,'flows':flows,'elapsedSeconds':round(time.perf_counter()-start,3)}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'source':receipt['source'],'original':{'text':original_result['text'],'tokens':original_result['tokenIds']},'flows':[{'flow':f['flow'],'browserTranscript':f.get('browserTranscript'),'nativeJoined':f.get('nativeJoinedInRequestOrder',f.get('fullCapture',{}).get('native',{}).get('text')),'segments':[{'requestId':s['requestId'],'durationSeconds':s['durationSeconds'],'text':s['native']['text']} for s in f.get('segments',[])]} for f in flows],'elapsedSeconds':receipt['elapsedSeconds'],'receipt':str(out.resolve())},ensure_ascii=False))
