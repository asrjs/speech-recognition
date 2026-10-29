import hashlib,json
from pathlib import Path
from urllib.request import urlopen
from huggingface_hub import HfApi
root=Path(r'N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx')
repo='ysdede/parakeet-tdt-0.6b-v3-onnx'
info=HfApi().model_info(repo,files_metadata=True)
rows=[]
for name in ['encoder-model.fp16.onnx','decoder_joint-model.int8.onnx','vocab.txt','config.json']:
 p=root/name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 remote=next(s for s in info.siblings if s.rfilename==name)
 remote_hash=remote.lfs.sha256 if remote.lfs else hashlib.sha256(urlopen(f'https://huggingface.co/{repo}/resolve/{info.sha}/{name}').read()).hexdigest()
 row=dict(path=name,bytes=p.stat().st_size,localSha256=h.hexdigest(),remoteSha256=remote_hash,match=h.hexdigest()==remote_hash)
 rows.append(row)
assert all(r['match'] for r in rows), rows
out=dict(repoId=repo,revision=info.sha,files=rows,status='ARTIFACT_IDENTITY_MATCH',accuracy='NOT_EVALUATED_BY_THIS_CHECK')
Path('tmp/approved-model-profiles/artifact-identity.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
