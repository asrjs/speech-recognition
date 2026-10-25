import hashlib
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

root = Path.cwd()
task = root / 'tmp/omnilingual-native-wsl-20261024'
artifacts = task / 'artifacts'
fixture = root / 'tests/fixtures/jfk2.en.wav'
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
with wave.open(str(fixture), 'rb') as wav:
    assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
    raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
assert np.array_equal(raw, np.load(artifacts / 'jfk2-decoded-pcm-float32.npy'))
raw.tofile(artifacts / 'parent-raw.f32le')
node = """
import fs from 'node:fs';
import { normalizeWav2Vec2Waveform } from './dist/models/wav2vec2/executor.js';
const b=fs.readFileSync('tmp/omnilingual-native-wsl-20261024/artifacts/parent-raw.f32le');
const pcm=new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
for (const [name,epsilon] of [['fixed',1e-5],['legacy',1e-7]]) {
  const out=normalizeWav2Vec2Waveform(pcm,epsilon);
  fs.writeFileSync(`tmp/omnilingual-native-wsl-20261024/artifacts/parent-sdk-${name}.f32le`,Buffer.from(out.buffer));
}
"""
subprocess.run(['node', '--input-type=module', '-e', node], check=True)
sdk = np.fromfile(artifacts / 'parent-sdk-fixed.f32le', dtype='<f4')
native_wave = np.load(artifacts / 'native-v2-jfk-waveform.npy')
assert hashlib.sha256(native_wave.tobytes()).hexdigest() == 'ed1689e6d2515827b26aeb57f837915a6b34018693ea6df00d43adc6c0d6733d'

def compare(a, b, atol, rtol):
    assert a.shape == b.shape
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.abs(x-y)
    return {'shape': list(a.shape), 'atol': atol, 'rtol': rtol,
            'allclose': bool(np.allclose(a,b,atol=atol,rtol=rtol)),
            'maxAbs': float(d.max()), 'meanAbs': float(d.mean()),
            'cosine': float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))}

results = {'scope': 'parent independent CPU replay; native JFK quality remains FAIL',
           'preprocessing': compare(native_wave,sdk,1e-6,1e-6), 'logits': {}, 'artifacts': {}}
assert results['preprocessing']['allclose']
native_logits = np.load(artifacts / 'native-v2-jfk-logits.npy')
for name in ['fp32', 'int8']:
    saved = np.load(artifacts / f'{name}-jfk-logits.npy')
    comparison = compare(native_logits,saved,1e-3,1e-4)
    comparison['argmaxAgreement'] = float(np.mean(native_logits.argmax(-1)==saved.argmax(-1)))
    results['logits'][name] = comparison
graph = root / 'models/onnx/omnilingual-ctc-300m/model.int8.onnx'
assert hashlib.sha256(graph.read_bytes()).hexdigest() == '38a6cc569d46eb382f27d85b3208b883e834ff281b2057a606ec79e4ae501545'
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
output = session.run(None, {'x': sdk[None,:]})[0][0]
saved_int8 = np.load(artifacts / 'int8-jfk-logits.npy')
results['sdkInt8Replay'] = compare(saved_int8,output,1e-3,1e-4)
results['sdkInt8Replay']['argmaxAgreement'] = float(np.mean(saved_int8.argmax(-1)==output.argmax(-1)))
results['runtime'] = {'onnxruntime': ort.__version__, 'providers': session.get_providers(), 'threads': 2}
for path in [fixture, graph, *sorted(artifacts.glob('*.npy'))]:
    results['artifacts'][str(path)] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
destination = task / 'logs/parent-array-verification.json'
assert not destination.exists()
destination.write_text(json.dumps(results, indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in results.items() if k!='artifacts'}))
assert results['logits']['fp32']['allclose']
assert results['sdkInt8Replay']['allclose'] and results['sdkInt8Replay']['argmaxAgreement'] == 1.0
