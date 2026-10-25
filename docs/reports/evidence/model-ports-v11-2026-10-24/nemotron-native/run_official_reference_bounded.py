from __future__ import annotations
import datetime, faulthandler, hashlib, json, os, runpy, sys, threading, time
from pathlib import Path
ROOT = Path(r'N:\github\asrjs\speech-recognition')
RUNNER = ROOT / 'tools/model-debugging/reference/nemotron-3.5-asr-streaming/run_reference.py'
CHECKPOINT = Path(r'N:\models\nemo\nemotron-3.5-asr-streaming-0.6b\nemotron-3.5-asr-streaming-0.6b.nemo')
FIXTURE = ROOT / 'tests/fixtures/jfk2.en.wav'
OUTDIR = Path(os.environ['TEMP']) / 'nemotron-native-cpu-reference-20261024'
OUTPUT = OUTDIR / 'official-reference-jfk2-en-prompt-en.json'
EXPECTED_CHECKPOINT = '210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74'
EXPECTED_FIXTURE = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
ORACLE = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NUMBA_CACHE_DIR'] = str(OUTDIR / 'numba-cache')
faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
def watchdog():
    print('HARD_TIMEOUT_600S: captured stack then terminating only this owned reference process', file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    os._exit(124)
timer = threading.Timer(600, watchdog)
timer.daemon = True
timer.start()
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
print(json.dumps({'stage':'preflight-start','at':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(CHECKPOINT),'checkpointBytes':CHECKPOINT.stat().st_size,'fixture':str(FIXTURE),'fixtureBytes':FIXTURE.stat().st_size,'python':sys.executable},indent=2),flush=True)
checkpoint_sha=sha256(CHECKPOINT); fixture_sha=sha256(FIXTURE)
if checkpoint_sha != EXPECTED_CHECKPOINT: raise RuntimeError(f'checkpoint hash mismatch: {checkpoint_sha}')
if fixture_sha != EXPECTED_FIXTURE: raise RuntimeError(f'fixture hash mismatch: {fixture_sha}')
print(json.dumps({'stage':'preflight-complete','checkpointSha256':checkpoint_sha,'fixtureSha256':fixture_sha},indent=2),flush=True)
print('stage=torch-import-start',flush=True)
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
original_set_num_threads=torch.set_num_threads
def fixed_threads(count):
    return original_set_num_threads(2)
torch.set_num_threads=fixed_threads
print(json.dumps({'stage':'torch-import-complete','torch':torch.__version__,'torchThreads':torch.get_num_threads(),'torchInteropThreads':torch.get_num_interop_threads(),'cudaAvailable':torch.cuda.is_available()},indent=2),flush=True)
print('stage=nemo-class-import-start',flush=True)
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
print('stage=nemo-class-import-complete',flush=True)
original_restore=EncDecRNNTBPEModelWithPrompt.restore_from
def traced_restore(cls,*args,**kwargs):
    start=time.monotonic(); print(json.dumps({'stage':'restore-start','at':datetime.datetime.now().astimezone().isoformat(),'args':[str(x) for x in args[:1]],'restorePath':str(kwargs.get('restore_path')),'mapLocation':str(kwargs.get('map_location'))},indent=2),flush=True)
    result=original_restore(*args,**kwargs)
    print(json.dumps({'stage':'restore-complete','elapsedSeconds':round(time.monotonic()-start,3)},indent=2),flush=True)
    return result
EncDecRNNTBPEModelWithPrompt.restore_from=classmethod(traced_restore)
started=time.monotonic()
sys.argv=[str(RUNNER),'--nemo',str(CHECKPOINT),'--fixture',str(FIXTURE),'--output',str(OUTPUT),'--prompt','en','--device','cpu']
print(json.dumps({'stage':'reference-runner-start','prompt':'en','device':'cpu','maxDurationSeconds':600,'stackDumpSeconds':60},indent=2),flush=True)
runpy.run_path(str(RUNNER),run_name='__main__')
record=json.loads(OUTPUT.read_text(encoding='utf-8'))
result=next((item for item in record.get('results',[]) if item.get('fixtureName')==FIXTURE.name),{})
record['executionAudit']={'checkpointSha256Verified':checkpoint_sha,'fixtureSha256Verified':fixture_sha,'promptRequested':'en','device':'cpu','torchThreads':torch.get_num_threads(),'torchInteropThreads':torch.get_num_interop_threads(),'cudaVisibleDevices':os.environ['CUDA_VISIBLE_DEVICES'],'numbaCacheDir':os.environ['NUMBA_CACHE_DIR'],'elapsedSeconds':round(time.monotonic()-started,3) if 'started' in globals() else None,'provenance':'official run_reference.py; local wrapper added strict CPU limits, stage traces, and bounded watchdog'}
record['exactOracleComparison']={'expected':ORACLE,'actual':result.get('text'),'tokenIds':result.get('tokenIds'),'tokenCount':result.get('tokenCount'),'exactTextMatch':result.get('text')==ORACLE,'status':'PASS_EXACT_TEXT' if result.get('text')==ORACLE else 'FAIL_EXACT_TEXT'}
OUTPUT.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'stage':'reference-run-complete','exactOracleComparison':record['exactOracleComparison'],'output':str(OUTPUT)},indent=2,ensure_ascii=False),flush=True)
faulthandler.cancel_dump_traceback_later(); timer.cancel()



