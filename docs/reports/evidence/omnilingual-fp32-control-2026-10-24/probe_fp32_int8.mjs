import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import path from 'node:path';
const repo=String.raw`N:\github\asrjs\speech-recognition`;
const require=createRequire(pathToFileURL(path.join(repo,'package.json')));
const ort=require('onnxruntime-node');
const {decodeWavToMonoPcm}=await import(pathToFileURL(path.join(repo,'dist/audio/wav.js')));
const {OmnilingualCtcTokenizer}=await import(pathToFileURL(path.join(repo,'dist/models/omnilingual-ctc/tokenizer.js')));
const {CtcDecoder}=await import(pathToFileURL(path.join(repo,'dist/ctc/index.js')));
const {normalizeWav2Vec2Waveform}=await import(pathToFileURL(path.join(repo,'dist/models/wav2vec2/executor.js')));
const root=String.raw`C:\Users\steam\AppData\Local\Temp\asrjs-omnilingual-fp32-control-20261024`;
const artifactDir=path.join(repo,'models/onnx/omnilingual-ctc-300m');
const fixture=path.join(repo,'tests/fixtures/jfk2.en.wav');
const oracle='And so my fellow Americans ask not what your country can do for you ask what you can do for your country';
const sha=b=>createHash('sha256').update(b).digest('hex');
const shaFile=p=>new Promise((resolve,reject)=>{const h=createHash('sha256');const s=createReadStream(p);s.on('data',c=>h.update(c));s.on('error',reject);s.on('end',()=>resolve(h.digest('hex')));});
const wav=await readFile(fixture);
const parsed=decodeWavToMonoPcm(wav);
const pcm=parsed.pcm;
const normalized=normalizeWav2Vec2Waveform(pcm);
const tokText=await readFile(path.join(artifactDir,'tokens.txt'),'utf8');
const tokenizer=OmnilingualCtcTokenizer.fromText(tokText);
const decoder=new CtcDecoder({blankId:tokenizer.blankId,vocabSize:tokenizer.vocabSize,tokenizer});
const variants={raw:pcm,normalized};
const graphs={fp32:path.join(root,'model.onnx'),int8:path.join(artifactDir,'model.int8.onnx')};
const results={};
const logitsByVariant={};
for(const [name,graph] of Object.entries(graphs)){
 const session=await ort.InferenceSession.create(graph,{executionProviders:['cpu'],intraOpNumThreads:2,interOpNumThreads:1});
 results[name]={graph:path.basename(graph),runs:{}};
 for(const [variant,audio] of Object.entries(variants)){
  const input=new ort.Tensor('float32',audio,[1,audio.length]);
  const t0=performance.now();
  const outputs=await session.run({x:input});
  const elapsed=performance.now()-t0;
  input.dispose?.();
  const tensor=outputs.logits??Object.values(outputs)[0];
  const dims=Array.from(tensor.dims,Number);
  const frames=dims[1], vocab=dims[2];
  const data=tensor.data instanceof Float32Array?tensor.data:Float32Array.from(tensor.data);
  const decoded=decoder.decodeFromLogits(data.subarray(0,frames*vocab),frames,{outFrames:frames});
  results[name].runs[variant]={inputSamples:audio.length,logitsShape:dims,elapsedMs:elapsed,transcript:decoded.text.trim(),frameIds:decoded.frameIds,collapsedIds:decoded.collapsedIds};
  logitsByVariant[variant]??={}; logitsByVariant[variant][name]=Float32Array.from(data.subarray(0,frames*vocab));
  for(const out of Object.values(outputs)) out.dispose?.();
 }
 await session.release?.();
}
const comparisons={};
for(const variant of Object.keys(variants)){
 const a=logitsByVariant[variant].fp32,b=logitsByVariant[variant].int8;
 if(a.length!==b.length) throw new Error(`logit size mismatch ${variant}: ${a.length} vs ${b.length}`);
 let maxAbs=0,sumAbs=0,dot=0,aa=0,bb=0,argmaxMatch=0,frames=results.fp32.runs[variant].logitsShape[1],vocab=results.fp32.runs[variant].logitsShape[2];
 for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]);if(d>maxAbs)maxAbs=d;sumAbs+=d;dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
 for(let f=0;f<frames;f++){let ia=0,ib=0,va=-Infinity,vb=-Infinity;for(let k=0;k<vocab;k++){const ix=f*vocab+k;if(a[ix]>va){va=a[ix];ia=k;}if(b[ix]>vb){vb=b[ix];ib=k;}}if(ia===ib)argmaxMatch++;}
 comparisons[variant]={logitsCount:a.length,maxAbsDiff:maxAbs,meanAbsDiff:sumAbs/a.length,cosineSimilarity:dot/Math.sqrt(aa*bb),frameArgmaxAgreement:argmaxMatch/frames,frames};
}
const gold=Buffer.from(oracle,'utf8');
const receipt={schemaVersion:1,status:'CPU_ORT_CONTROL_ONLY_NOT_NATIVE_OR_MODEL_ACCEPTANCE',source:{repo:'OpenVoiceOS/omnilingual-asr-ctc-300m-onnx',revision:'bc55e1de19af8a907b0a010d06fc4637a2d5bf0c',nativeMetaV2Identity:'UNVERIFIED'},environment:{runtime:'node',node:process.version,onnxruntimeNode:require('onnxruntime-node/package.json').version,provider:'cpu',intraOpThreads:2,interOpThreads:1},fixture:{path:fixture,sha256:sha(wav),bytes:wav.length,sampleRate:16000,samples:pcm.length,oracleSource:'flows-round3/omnilingual/playground/file.json immutable receipt',oracleSha256:gold.length?sha(gold):null,oracleText:gold.length?gold.toString('utf8').trim():null},artifacts:{fp32:{path:graphs.fp32,sha256:await shaFile(graphs.fp32)},int8:{path:graphs.int8,sha256:await shaFile(graphs.int8)},tokens:{path:path.join(artifactDir,'tokens.txt'),sha256:sha(Buffer.from(tokText))},vocabularySize:tokenizer.vocabSize,blankId:tokenizer.blankId},inputContract:{graphInput:'x float32 [1,samples]',rawControl:'decoded WAV float samples without per-utterance standardization',normalizedControl:'same samples transformed with library normalizeWav2Vec2Waveform',sampleRate:16000},results,fp32VsInt8:comparisons};
const out=path.join(root,'fp32-int8-jfk-controls.json');
await writeFile(out,JSON.stringify(receipt,null,2),'utf8');
console.log(JSON.stringify({status:receipt.status,fixture:receipt.fixture,artifacts:receipt.artifacts,results:Object.fromEntries(Object.entries(results).map(([k,v])=>[k,{runs:Object.fromEntries(Object.entries(v.runs).map(([n,x])=>[n,{transcript:x.transcript,logitsShape:x.logitsShape,elapsedMs:x.elapsedMs,frameIds:x.frameIds,collapsedIds:x.collapsedIds}]))}])),fp32VsInt8:comparisons,receipt:out},null,2));

