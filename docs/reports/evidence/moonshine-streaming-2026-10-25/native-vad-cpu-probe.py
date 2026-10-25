import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch, transformers
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
p=argparse.ArgumentParser()
p.add_argument('--snapshot',required=True); p.add_argument('--original',required=True)
p.add_argument('--capture-dir',required=True); p.add_argument('--flow-dir',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); snapshot=Path(a.snapshot); torch.set_num_threads(min(4,torch.get_num_threads()))
def sha(b): return hashlib.sha256(b).hexdigest()
processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
model=MoonshineStreamingForConditionalGeneration.from_pretrained(str(snapshot),local_files_only=True); model.eval()
def infer(pcm):
    encoded=processor(pcm.tolist(),return_tensors='pt',sampling_rate=16000)
    with torch.no_grad():
        generated=model.generate(input_values=encoded.input_values,attention_mask=encoded.attention_mask,max_new_tokens=128,do_sample=False,num_beams=1)
    return {'tokens':generated[0].tolist(),'text':processor.batch_decode(generated,skip_special_tokens=True)[0], 'validSamples':int(encoded.attention_mask.sum().item()),'inputShape':list(encoded.input_values.shape)}
with __import__('wave').open(str(Path(a.original)), 'rb') as w:
    if (w.getframerate(),w.getnchannels(),w.getsampwidth()) != (16000,1,2): raise ValueError('original must be mono 16k PCM16')
    original=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768.0
start=time.perf_counter(); original_result=infer(original)
metas=[]
for mp in Path(a.capture_dir).glob('*.json'):
    meta=json.loads(mp.read_text(encoding='utf-8')); raw=Path(meta['pcmPath']).read_bytes()
    if sha(raw)!=meta['pcmSha256'] or len(raw)!=meta['frames']*4: raise ValueError(f'metadata mismatch: {mp}')
    pcm=np.frombuffer(raw,dtype='<f4').copy()
    metas.append((meta,raw,pcm))
flows=[]
for fp in sorted(Path(a.flow_dir).glob('mic-*.json')):
    flow=json.loads(fp.read_text(encoding='utf-8'))
    if not flow['flow'].startswith('mic-speech-detect-'): continue
    relevant=sorted(((m,r,x) for m,r,x in metas if m['flow']==flow['flow']),key=lambda it:it[0]['requestId'])
    outputs=[]
    for meta,raw,pcm in relevant:
        t=time.perf_counter(); out=infer(pcm)
        outputs.append({'requestId':meta['requestId'],'frames':meta['frames'],'durationSeconds':meta['frames']/16000,'pcmSha256':meta['pcmSha256'],'native':out,'generateSeconds':round(time.perf_counter()-t,3)})
    joined=' '.join(x['native']['text'].strip() for x in outputs if x['native']['text'].strip())
    flows.append({'flow':flow['flow'],'browserTranscript':flow.get('transcript'),'browserTranscriptBefore':flow.get('transcriptBefore'),'browserTranscriptAfter':flow.get('transcriptAfter'),'segments':outputs,'nativeJoinedInRequestOrder':joined,'nativeJoinedMatchesBrowserTranscript':joined==flow.get('transcript'),'nonemptySegmentCount':sum(bool(x['native']['text'].strip()) for x in outputs),'segmentCount':len(outputs)})
receipt={'schemaVersion':1,'status':'CPU_SEGMENT_REFERENCE_COMPLETED_NOT_RUNTIME_PASS','scope':'Official Transformers local-only CPU whole-utterance generation separately on each captured VAD PCM segment, joined by ascending worker requestId. This isolates model behavior on exact segments from browser runtime; no streaming-cache or WebGPU equivalence claim.','source':{'modelId':'moonshine-ai/moonshine-streaming-tiny','upstreamRevision':'f8e9dfd8c562','snapshot':str(snapshot.resolve()),'modelSha256':sha((snapshot/'model.safetensors').read_bytes()),'configSha256':sha((snapshot/'config.json').read_bytes()),'tokenizerSha256':sha((snapshot/'tokenizer.json').read_bytes()),'transformers':transformers.__version__,'torch':torch.__version__,'device':'cpu','localFilesOnly':True},'originalJfk':{'path':str(Path(a.original).resolve()),'pcmSha256':sha(original.astype('<f4').tobytes()),'durationSeconds':len(original)/16000,'native':original_result},'flows':flows,'elapsedSeconds':round(time.perf_counter()-start,3)}
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'originalText':original_result['text'],'flows':[{'flow':f['flow'],'browserTranscript':f['browserTranscript'],'nativeJoined':f['nativeJoinedInRequestOrder'],'segments':[{'requestId':s['requestId'],'durationSeconds':s['durationSeconds'],'text':s['native']['text']} for s in f['segments']]} for f in flows],'receipt':str(out.resolve()),'elapsedSeconds':receipt['elapsedSeconds']},ensure_ascii=False))
