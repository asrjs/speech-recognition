import hashlib,json,subprocess
from pathlib import Path
base=Path.cwd(); out=base/'docs/reports/evidence/handover-2026-10-28'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''): h.update(c)
 return h.hexdigest()
def git(root,*args):
 return subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}','-C',str(root),*args],text=True).strip()
repos=[]
for name in ['speech-recognition','demos','frameworks']:
 root=base.parent/name
 repos.append(dict(name=name,path=str(root),branch=git(root,'branch','--show-current'),head=git(root,'rev-parse','HEAD'),status=git(root,'status','--porcelain=v1').splitlines(),recent=git(root,'log','-6','--format=%h %aI %s').splitlines()))
pkg=base.parent/'demos/vendor/asrjs-speech-recognition-0.2.0-approved-profiles-v3-20261025.tgz'
prov=json.loads(pkg.with_suffix('.provenance.json').read_text())
checks={}
for key,root in [('sourceFilesSha256',base),('distFilesSha256',base),('installedDist',base.parent/'demos/node_modules/@asrjs/speech-recognition')]:
 expected=prov['sourceFilesSha256' if key=='sourceFilesSha256' else 'distFilesSha256']
 mismatches=[]
 for rel,wanted in expected.items():
  p=root/rel
  if not p.is_file() or digest(p)!=wanted: mismatches.append(rel)
 checks[key]={'checked':len(expected),'mismatches':mismatches}
checks['archive']={'path':str(pkg),'actualSha256':digest(pkg),'expectedSha256':prov['sha256']}
flows=[]
for app in ['showcase','playground','benchmark','realtime']:
 p=base/'tmp/approved-model-profiles'/f'{app}-v3/file.json'; r=json.loads(p.read_text())
 flows.append({k:r.get(k) for k in ['app','status','transcriptStatus','fixtureSha256','requestedEncoderQuant','requestedDecoderQuant','actualEncoderQuant','actualDecoderQuant','actualGpuDispatches','artifactReplay','transcript','weightTransport','hubWeightDownloadTransport','physicalMicrophone','finishedAt']})
result={'auditDate':'2026-10-28','scope':'Current Git/hash/receipt inspection; no fresh inference or regression suite run','repositories':repos,'currentTreeVsHistoricalSdkPackage':checks,'historicalRealWeightFileFlows':flows}
(out/'workspace-snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'repos':[(x['name'],x['branch'],x['head'][:8],len(x['status'])) for x in repos],'checks':checks},indent=2))
