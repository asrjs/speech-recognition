import hashlib, json, wave
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

root = Path('/mnt/n/github/asrjs/speech-recognition/tmp/omnilingual-native-wsl-20261024')
raw = np.load(root / 'artifacts/jfk2-decoded-pcm-float32.npy', allow_pickle=False)
official = np.load(root / 'artifacts/native-v2-jfk-waveform.npy', allow_pickle=False)
graph = '/mnt/n/github/asrjs/speech-recognition/models/onnx/omnilingual-ctc-300m/model.int8.onnx'
def digest(x): return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()
def stats(x): return {'sha256Float32LE': digest(x), 'mean': float(x.mean()), 'stdPopulation': float(x.std()), 'min': float(x.min()), 'max': float(x.max())}
def ln(eps):
    return torch.nn.functional.layer_norm(torch.from_numpy(raw), raw.shape, eps=eps).numpy().astype('<f4')
waveforms = {'official_meta': official, 'torch_layer_norm_eps_1e-5': ln(1e-5), 'torch_layer_norm_eps_1e-7': ln(1e-7)}
for name, value in waveforms.items():
    print(name, stats(value), 'max_abs_to_meta', float(np.max(np.abs(value-official))), 'all_equal_to_meta', bool(np.array_equal(value, official)), flush=True)
so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
session=ort.InferenceSession(graph, sess_options=so, providers=['CPUExecutionProvider'])
logits={name: session.run(None, {'x': value[None, :]})[0][0] for name,value in waveforms.items()}
ids={name: np.argmax(value, axis=-1) for name,value in logits.items()}
result={'scope':'Same Linux CPU ONNX Runtime 1.30 session and INT8 graph; controlled Torch normalization counterfactual only. Does not isolate cross-runtime SDK-versus-native differences.','runtime':{'torch':torch.__version__,'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','intraOpThreads':2},'source':{'fixtureSha256':'59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e','graphSha256':'38a6cc569d46eb382f27d85b3208b883e834ff281b2057a606ec79e4ae501545'},'waveforms':{name:stats(value) for name,value in waveforms.items()},'comparisons':{}}
for left,right in [('official_meta','torch_layer_norm_eps_1e-5'),('official_meta','torch_layer_norm_eps_1e-7'),('torch_layer_norm_eps_1e-5','torch_layer_norm_eps_1e-7')]:
    changed=np.flatnonzero(ids[left]!=ids[right]).tolist()
    diff=np.abs(logits[left].astype(np.float64)-logits[right].astype(np.float64))
    result['comparisons'][left+'_vs_'+right]={'frameArgmaxAgreement':float(np.mean(ids[left]==ids[right])),'changedFrames':changed,'changedFrameCount':len(changed),'maxAbsLogitDiff':float(diff.max()),'meanAbsLogitDiff':float(diff.mean())}
out=root/'logs/epsilon-counterfactual.json'; out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('receipt',out,flush=True)
