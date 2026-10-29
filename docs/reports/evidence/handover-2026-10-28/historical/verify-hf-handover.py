import json
from pathlib import Path
from huggingface_hub import HfApi
api=HfApi(); rows=[]
for repo,rev in [('ysdede/parakeet-tdt-0.6b-v3-onnx','e5356bf6e24bdb7a269536b0e9e7869bc034b893'),('ysdede/qwen3-asr-0.6b-onnx','ac0bedcc0f1614c0b3982940c14bdb169ff2365d'),('ysdede/moonshine-streaming-tiny-onnx','cb347a0760c2d7d6cf9083eb770376ba108e5b31')]:
 try:
  pinned=api.model_info(repo,revision=rev,files_metadata=True); current=api.model_info(repo)
  rows.append({'repoId':repo,'pinnedRevision':pinned.sha,'currentRevision':current.sha,'pinnedFiles':[{'path':f.rfilename,'bytes':f.size,'sha256':f.lfs.sha256 if f.lfs else None} for f in pinned.siblings],'status':'PINNED_REVISION_EXISTS','scope':'Remote metadata only; no weights downloaded or inference'})
 except Exception as e: rows.append({'repoId':repo,'status':'UNVERIFIED','error':str(e)[:500]})
Path('docs/reports/evidence/handover-2026-10-28/hf-live-metadata.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:r.get(k) for k in ['repoId','pinnedRevision','currentRevision','status']} for r in rows],indent=2))
