from __future__ import annotations
import datetime, faulthandler, hashlib, json, os, sys, threading, time
from pathlib import Path
ROOT=Path(r'N:\github\asrjs\speech-recognition')
CHECKPOINT=Path(r'N:\models\nemo\nemotron-3.5-asr-streaming-0.6b\nemotron-3.5-asr-streaming-0.6b.nemo')
FIXTURE=ROOT/'tests/fixtures/jfk2.en.wav'
OUT=Path(os.environ['TEMP'])/'nemotron-native-cpu-reference-20261024'/'native-cache-stream-jfk-560ms-c5-vs-ba4.json'
EXPECTED_CHECKPOINT='210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74'
EXPECTED_FIXTURE='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
ORACLE='And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['NUMBA_CACHE_DIR']=str(OUT.parent/'numba-cache')
faulthandler.enable(file=sys.stderr,all_threads=True); faulthandler.dump_traceback_later(60,repeat=True,file=sys.stderr)
def watchdog():
 print('HARD_TIMEOUT_600S: owned native streaming process',file=sys.stderr,flush=True); faulthandler.dump_traceback(file=sys.stderr,all_threads=True); os._exit(124)
timer=threading.Timer(600,watchdog); timer.daemon=True; timer.start()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
ckpt_sha=sha(CHECKPOINT); fixture_sha=sha(FIXTURE)
if ckpt_sha!=EXPECTED_CHECKPOINT or fixture_sha!=EXPECTED_FIXTURE: raise RuntimeError('checkpoint or fixture SHA mismatch')
import torch
torch.set_num_threads(2); torch.set_num_interop_threads(2)
orig_set=torch.set_num_threads; torch.set_num_threads=lambda _:orig_set(2)
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
print(json.dumps({'stage':'restore-start','time':datetime.datetime.now().astimezone().isoformat(),'checkpointSha256':ckpt_sha,'fixtureSha256':fixture_sha,'torch':torch.__version__,'torchThreads':torch.get_num_threads(),'torchInteropThreads':torch.get_num_interop_threads(),'device':'cpu','prompt':'en'}),flush=True)
start=time.monotonic(); model=EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=str(CHECKPOINT),map_location=torch.device('cpu')); model.eval()
model.encoder.set_default_att_context_size([56,6]); model.set_inference_prompt('en')
enc=model.encoder; cfg=enc.streaming_cfg
streamcfg={k:getattr(cfg,k,None) for k in ('chunk_size','shift_size','cache_drop_size','valid_out_len','pre_encode_cache_size','drop_extra_pre_encoded','last_channel_cache_size')}
print(json.dumps({'stage':'restore-complete','elapsedSeconds':round(time.monotonic()-start,3),'streamingCfg':streamcfg,'attentionContext':enc.att_context_size},default=str),flush=True)
# Python 3.10 compatibility shim for one NeMo module expression requiring PEP 646.
import importlib.util
_stream_name = "nemo.collections.asr.parts.utils.streaming_utils"
_stream_spec = importlib.util.find_spec(_stream_name)
_stream_source = Path(_stream_spec.origin).read_text(encoding="utf-8")
_old_index = "shifted_indices = shifted_indices[..., *[None for _ in range(len(self.dim_shape))]]"
_new_index = "shifted_indices = shifted_indices[(Ellipsis,) + (None,) * len(self.dim_shape)]"
if _old_index not in _stream_source: raise RuntimeError("expected NeMo 3.0 streaming index expression changed")
_stream_path = OUT.parent / "streaming_utils_py310_compat.py"
_stream_path.write_text(_stream_source.replace(_old_index, _new_index, 1), encoding="utf-8")
_stream_module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(_stream_name, _stream_path))
_stream_module.__spec__.loader.exec_module(_stream_module)
sys.modules[_stream_name] = _stream_module
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
# Compare the packed ONNX encoder on the exact native chunk features and cache state (CPU only).
import numpy as np
import onnxruntime as ort
ENCODER=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-conditional-drop-cli-real-20261024/encoder.onnx')
ort_options=ort.SessionOptions(); ort_options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
ort_options.intra_op_num_threads=2; ort_options.inter_op_num_threads=2
ort_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ort_session=ort.InferenceSession(str(ENCODER),sess_options=ort_options,providers=['CPUExecutionProvider'])
ORIGINAL_ENCODER=Path(r'C:/Users/steam/AppData/Local/Temp/nemotron-staticdrop-surgery-20261024/encoder.dynamic-drop.onnx')
original_session=ort.InferenceSession(str(ORIGINAL_ENCODER),sess_options=ort_options,providers=['CPUExecutionProvider'])
if ort_session.get_providers()!=['CPUExecutionProvider'] or original_session.get_providers()!=['CPUExecutionProvider']: raise RuntimeError(f'unexpected ONNX providers: {ort_session.get_providers()} {original_session.get_providers()}')
ort_output_names=[x.name for x in ort_session.get_outputs()]
original_output_names=[x.name for x in original_session.get_outputs()]
if ort_output_names!=original_output_names: raise RuntimeError('original/candidate output names differ')
onnx_step_records=[]
onnx_cache_state=None
native_encoder_step=enc.cache_aware_stream_step
def compare_encoder_step(*,processed_signal,processed_signal_length=None,cache_last_channel=None,cache_last_time=None,cache_last_channel_len=None,keep_all_outputs=True,drop_extra_pre_encoded=None,bypass_pre_encode=False):
    native=native_encoder_step(processed_signal=processed_signal,processed_signal_length=processed_signal_length,cache_last_channel=cache_last_channel,cache_last_time=cache_last_time,cache_last_channel_len=cache_last_channel_len,keep_all_outputs=keep_all_outputs,drop_extra_pre_encoded=drop_extra_pre_encoded,bypass_pre_encode=bypass_pre_encode)
    global onnx_cache_state
    native_encoded,native_len,native_ch_next,native_t_next,native_len_next=native
    def cpu(t): return t.detach().cpu().numpy()
    def maxabs(a,b):
        a=np.asarray(a); b=np.asarray(b)
        if a.shape!=b.shape: return {'sameShape':False,'nativeShape':list(a.shape),'onnxShape':list(b.shape)}
        d=np.abs(a.astype(np.float64)-b.astype(np.float64))
        return {'sameShape':True,'shape':list(a.shape),'maxAbs':float(d.max(initial=0)),'meanAbs':float(d.mean()) if d.size else 0.0,'allcloseAtol1e-5Rtol1e-5':bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)),'allcloseAtol1e-4Rtol1e-4':bool(np.allclose(a,b,atol=1e-4,rtol=1e-4))}
    if onnx_cache_state is None:
        onnx_cache_state={'cache_last_channel':cpu(cache_last_channel).transpose(1,0,2,3).copy(),'cache_last_time':cpu(cache_last_time).transpose(1,0,2,3).copy(),'cache_last_channel_len':cpu(cache_last_channel_len).astype(np.int64).copy()}
    feed={'audio_signal':cpu(processed_signal).transpose(0,2,1).copy(),'length':cpu(processed_signal_length).astype(np.int64),'cache_last_channel':onnx_cache_state['cache_last_channel'],'cache_last_time':onnx_cache_state['cache_last_time'],'cache_last_channel_len':onnx_cache_state['cache_last_channel_len'],'lang_id':np.asarray([0],dtype=np.int64)}
    onnx_values=dict(zip(ort_output_names,ort_session.run(None,feed)))
    onnx_cache_state={'cache_last_channel':onnx_values['cache_last_channel_next'],'cache_last_time':onnx_values['cache_last_time_next'],'cache_last_channel_len':onnx_values['cache_last_channel_len_next']}
    native_feed={'audio_signal':feed['audio_signal'],'length':feed['length'],'cache_last_channel':cpu(cache_last_channel).transpose(1,0,2,3).copy(),'cache_last_time':cpu(cache_last_time).transpose(1,0,2,3).copy(),'cache_last_channel_len':cpu(cache_last_channel_len).astype(np.int64).copy(),'lang_id':np.asarray([0],dtype=np.int64)}
    chain_previous_values=dict(zip(original_output_names,original_session.run(None,feed)))
    chain_previous_equal={k:bool(np.array_equal(chain_previous_values[k],onnx_values[k])) for k in original_output_names}
    chain_feed_hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in feed.items()}
    original_values=dict(zip(original_output_names,original_session.run(None,native_feed)))
    candidate_native_control_values=dict(zip(ort_output_names,ort_session.run(None,native_feed)))
    native_feed_hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in native_feed.items()}
    original_vs_candidate={k:bool(np.array_equal(original_values[k],candidate_native_control_values[k])) for k in original_output_names}
    native_prompted=model._apply_prompt_to_encoded(native_encoded)
    record={'inputFrames':int(feed['audio_signal'].shape[1]),'inputLength':int(feed['length'][0]),'dropExtraPreEncoded':drop_extra_pre_encoded,'keepAllOutputs':keep_all_outputs,'nativeMelShape':list(processed_signal.shape),'cacheChannelShapeNative':list(cache_last_channel.shape),'cacheTimeShapeNative':list(cache_last_time.shape),'cacheChannelLengthNative':cpu(cache_last_channel_len).tolist(),'cacheChannelLengthCandidateInput':feed['cache_last_channel_len'].tolist(),'previousVsCandidateWithSameCandidateChainInputCache':chain_previous_equal,'sameCandidateChainFeedSha256':chain_feed_hashes,'originalVsCandidateWithSameNativeInputCache':original_vs_candidate,'sameNativeFeedSha256':native_feed_hashes,'originalCandidateOutputEqual':all(original_vs_candidate.values()),'comparisons':{
      'encoded':maxabs(cpu(native_prompted).transpose(0,2,1),onnx_values['outputs']),
      'encodedLength':maxabs(cpu(native_len),onnx_values['encoded_lengths']),
      'cacheLastChannelNext':maxabs(cpu(native_ch_next).transpose(1,0,2,3),onnx_values['cache_last_channel_next']),
      'cacheLastTimeNext':maxabs(cpu(native_t_next).transpose(1,0,2,3),onnx_values['cache_last_time_next']),
      'cacheLastChannelLengthNext':maxabs(cpu(native_len_next),onnx_values['cache_last_channel_len_next'])}}
    onnx_step_records.append(record)
    print(json.dumps({'stage':'onnx-native-step','step':len(onnx_step_records)-1,'inputFrames':record['inputFrames'],'drop':drop_extra_pre_encoded,'encoded':record['comparisons']['encoded'],'encodedLength':record['comparisons']['encodedLength']},ensure_ascii=False),flush=True)
    return native
enc.cache_aware_stream_step=compare_encoder_step
print(json.dumps({'stage':'onnx-cpu-session','path':str(ENCODER),'providers':ort_session.get_providers(),'inputNames':[x.name for x in ort_session.get_inputs()],'outputNames':ort_output_names},ensure_ascii=False),flush=True)
buffer=CacheAwareStreamingAudioBuffer(model=model,online_normalization=False)
_,_,stream_id=buffer.append_audio_file(str(FIXTURE),stream_id=-1)
cache_ch,cache_t,cache_len=enc.get_initial_cache_state(batch_size=1,device=torch.device('cpu'))
previous_hyp=None; chunks=[]; step=0; inferred_start=time.monotonic()
while True:
    before=buffer.buffer_idx
    try: chunk_audio,chunk_lengths=next(iter(buffer)) if step==0 else next(iterator)
    except (StopIteration,NameError): break
    # The generator is persistent for a single stream.
    iterator=iter(buffer) if step==0 else iterator
    after=buffer.buffer_idx
    drop=cfg.drop_extra_pre_encoded if step!=0 else 0
    result=model.conformer_stream_step(processed_signal=chunk_audio,processed_signal_length=chunk_lengths,cache_last_channel=cache_ch,cache_last_time=cache_t,cache_last_channel_len=cache_len,keep_all_outputs=buffer.is_buffer_empty(),previous_hypotheses=previous_hyp,drop_extra_pre_encoded=drop,return_transcription=True,return_log_probs=False)
    preds,texts,cache_ch,cache_t,cache_len,previous_hyp=result
    hyp=previous_hyp[0] if previous_hyp else None
    chunks.append({'step':step,'startMelFrame':before,'advanceMelFrames':after-before,'inputFeatureFrames':int(chunk_audio.shape[-1]),'inputLength':int(chunk_lengths[0].item()),'dropExtraPreEncoded':int(drop),'bufferEmptyAfterYield':buffer.is_buffer_empty(),'partialText':getattr(hyp,'text',None),'tokenCount':len(hyp.y_sequence) if hyp is not None and getattr(hyp,'y_sequence',None) is not None else None})
    print(json.dumps({'stage':'chunk','step':step,'start':before,'advance':after-before,'inputFrames':int(chunk_audio.shape[-1]),'inputLength':int(chunk_lengths[0].item()),'drop':int(drop),'tokenCount':chunks[-1]['tokenCount']},ensure_ascii=False),flush=True)
    step+=1
    if buffer.is_buffer_empty(): break
if previous_hyp is None: raise RuntimeError('native cache-aware stream produced no hypotheses')
hyp=previous_hyp[0]
text=getattr(hyp,'text',None)
tokens=[int(x) for x in hyp.y_sequence.tolist()] if getattr(hyp,'y_sequence',None) is not None else None
record={'schemaVersion':1,'generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpointSha256':ckpt_sha,'fixtureSha256':fixture_sha,'fixtureFrames':176000,'fixtureSampleRate':16000,'prompt':'en','device':'cpu','pythonExecutable':sys.executable,'pythonPrefix':sys.prefix,'torchModule':torch.__file__,'onnxruntimeModule':ort.__file__,'numpyModule':np.__file__,'onnxruntimeVersion':ort.__version__,'torchVersion':torch.__version__,'torchThreads':torch.get_num_threads(),'torchInteropThreads':torch.get_num_interop_threads(),'numbaCacheDir':os.environ['NUMBA_CACHE_DIR'],'attentionContext':enc.att_context_size,'streamingCfg':streamcfg,'streamingChunks':chunks,'onnxVsNativeEncoderSteps':onnx_step_records,'onnxVsNativeAllClose':len(onnx_step_records)==20 and all(all(v.get('sameShape',False) and v.get('allcloseAtol1e-4Rtol1e-4',False) for v in r['comparisons'].values()) for r in onnx_step_records),'continuationDrop2BitExactOnSameNativeCaches':len(onnx_step_records)==20 and all(r['originalCandidateOutputEqual'] is True for r in onnx_step_records),'all20CandidateChainFeedsBitExact':len(onnx_step_records)==20 and all(all(r['previousVsCandidateWithSameCandidateChainInputCache'].values()) for r in onnx_step_records),'streamingText':text,'tokenIds':tokens,'tokenCount':len(tokens) if tokens is not None else None,'offlineOfficialReferenceText':'And so my fellow Americans ask not what your country can do for you. <en-US> Ask what you can do for your country. <en-US>','browserLegacyOnnxText':'And so my fellow Americans ask not what your country can do for you at what you can do for your country.','strictExactCanonicalOracleMatch':text==ORACLE,'elapsedSeconds':round(time.monotonic()-start,3),'scope':'fresh native checkpoint CPU stream; c5 candidate and prior ba4 candidate compare bitwise on both identical native feature/cache feeds and identical c5-chain feeds at all 20 windows; cache/native numeric parity remains separate; no SDK/browser changes'}
OUT.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'stage':'done','streamingText':text,'tokenCount':len(tokens) if tokens is not None else None,'elapsedSeconds':record['elapsedSeconds'],'output':str(OUT)},ensure_ascii=False),flush=True)
faulthandler.cancel_dump_traceback_later(); timer.cancel()


