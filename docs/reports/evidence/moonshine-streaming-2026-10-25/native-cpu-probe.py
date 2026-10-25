import argparse, hashlib, json, time, wave
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

p=argparse.ArgumentParser()
p.add_argument('--snapshot',required=True)
p.add_argument('--original',required=True)
p.add_argument('--padded',required=True)
p.add_argument('--capture',required=True)
p.add_argument('--capture-meta',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
snapshot=Path(a.snapshot)
torch.set_num_threads(min(4, torch.get_num_threads()))
def sha(b): return hashlib.sha256(b).hexdigest()
def readwav(path):
    path=Path(path)
    with wave.open(str(path),'rb') as w:
        rate,channels,width=w.getframerate(),w.getnchannels(),w.getsampwidth()
        raw=w.readframes(w.getnframes())
    if (rate,channels,width)!=(16000,1,2): raise ValueError(f'{path}: expected mono 16k PCM16; got {(rate,channels,width)}')
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768.0, path.read_bytes()
inputs={}
for label,path in [('original_jfk',a.original),('padded_21s_jfk',a.padded)]:
    pcm,wav=readwav(path)
    inputs[label]=(pcm,{'path':str(Path(path).resolve()),'wavSha256':sha(wav),'effectivePcmSha256':sha(pcm.astype('<f4').tobytes()),'durationSeconds':len(pcm)/16000})
capture_path=Path(a.capture)
capture_raw=capture_path.read_bytes()
capture=np.frombuffer(capture_raw,dtype='<f4').copy()
meta=json.loads(Path(a.capture_meta).read_text(encoding='utf-8'))
if sha(capture_raw)!=meta['pcmSha256'] or len(capture)!=meta['frames']:
    raise ValueError('captured PCM bytes/frame count do not match capture metadata')
inputs['captured_mic_21s']=(capture,{'path':str(capture_path.resolve()),'effectivePcmSha256':sha(capture_raw),'durationSeconds':len(capture)/16000,'captureMetadata':meta})
processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
model=MoonshineStreamingForConditionalGeneration.from_pretrained(str(snapshot),local_files_only=True)
model.eval()
results=[]
for label,(pcm,info) in inputs.items():
    started=time.perf_counter()
    encoded=processor(pcm.tolist(),return_tensors='pt',sampling_rate=16000)
    with torch.no_grad():
        generated=model.generate(input_values=encoded.input_values,attention_mask=encoded.attention_mask,max_new_tokens=128,do_sample=False,num_beams=1)
    tokens=generated[0].tolist()
    text=processor.batch_decode(generated,skip_special_tokens=True)[0]
    mask=encoded.attention_mask.detach().cpu().numpy()
    results.append({'inputId':label,**info,'processorInputShape':list(encoded.input_values.shape),'attentionMaskShape':list(mask.shape),'attentionMaskValidSamples':int(mask.sum()),'energy':{'rms':float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),'peakAbs':float(np.max(np.abs(pcm))) if pcm.size else 0.0,'exactZeroFraction':float(np.mean(pcm==0)) if pcm.size else 1.0,'windowRms1s':[float(np.sqrt(np.mean(pcm[i:i+16000].astype(np.float64)**2))) for i in range(0,len(pcm),16000)]},'tokens':tokens,'tokenCount':len(tokens),'text':text,'generateSeconds':round(time.perf_counter()-started,3)})
receipt={'schemaVersion':1,'status':'CPU_REFERENCE_COMPLETED_NOT_RUNTIME_PASS','scope':'Official Transformers 5.18.0 whole-utterance MoonshineStreaming output, local-only CPU; distinguishes model response from ONNX streaming/runtime behavior. Not an ONNX export, browser, microphone, streaming-cache, or quality PASS.','source':{'modelId':'moonshine-ai/moonshine-streaming-tiny','upstreamRevision':'f8e9dfd8c562','snapshot':str(snapshot.resolve()),'modelSha256':sha((snapshot/'model.safetensors').read_bytes()),'configSha256':sha((snapshot/'config.json').read_bytes()),'tokenizerSha256':sha((snapshot/'tokenizer.json').read_bytes()),'transformers':transformers.__version__,'torch':torch.__version__,'device':'cpu','localFilesOnly':True},'inputs':results,'comparison':{'paddedMatchesOriginal':results[1]['text']==results[0]['text'],'captureMatchesOriginal':results[2]['text']==results[0]['text'],'allTextsNonEmpty':all(x['text'].strip() for x in results)}}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'source':receipt['source'],'summary':[{'inputId':x['inputId'],'durationSeconds':x['durationSeconds'],'effectivePcmSha256':x['effectivePcmSha256'],'tokenCount':x['tokenCount'],'text':x['text'],'generateSeconds':x['generateSeconds']} for x in results],'paddedMatchesOriginal':receipt['comparison']['paddedMatchesOriginal'],'captureMatchesOriginal':receipt['comparison']['captureMatchesOriginal'],'receipt':str(out.resolve())},ensure_ascii=False))
