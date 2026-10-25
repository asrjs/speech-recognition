import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

root = Path.cwd()
task = root / 'tmp/omnilingual-native-wsl-20261024'
artifacts = task / 'artifacts'
waveforms = {'meta': np.load(artifacts / 'native-v2-jfk-waveform.npy'),
             'sdk': np.fromfile(artifacts / 'parent-sdk-fixed.f32le',dtype='<f4')}
options = ort.SessionOptions()
options.intra_op_num_threads=2
options.inter_op_num_threads=1
result = {'scope':'same CPU runtime discriminating input control, not quality acceptance', 'runtime':ort.__version__, 'models':{}}
for name, path in [('int8',root/'models/onnx/omnilingual-ctc-300m/model.int8.onnx'),('fp32',artifacts/'model.onnx')]:
    session = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    outputs = {key:session.run(None,{'x':value[None,:]})[0][0] for key,value in waveforms.items()}
    a,b=outputs['meta'],outputs['sdk']
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
    changed=np.flatnonzero(a.argmax(-1)!=b.argmax(-1)).tolist()
    linux=np.load(artifacts/f'{name}-jfk-logits.npy')
    runtime_diff=np.abs(a.astype(np.float64)-linux.astype(np.float64))
    result['models'][name]={'inputDifference':{'maxAbs':float(diff.max()),'meanAbs':float(diff.mean()),'changedFrames':changed,'argmaxAgreement':float(np.mean(a.argmax(-1)==b.argmax(-1))),'atol':1e-3,'rtol':1e-4,'allclose':bool(np.allclose(a,b,atol=1e-3,rtol=1e-4))},'runtimeDifferenceOnExactMetaInput':{'maxAbs':float(runtime_diff.max()),'meanAbs':float(runtime_diff.mean()),'changedFrames':np.flatnonzero(a.argmax(-1)!=linux.argmax(-1)).tolist()}}
    del session
destination=task/'logs/parent-input-cross-control.json'
assert not destination.exists()
destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
