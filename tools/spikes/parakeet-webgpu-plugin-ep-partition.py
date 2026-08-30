import json, sys
from collections import Counter
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as wgpu_ep
MODEL = sys.argv[1]
ort.register_execution_provider_library('webgpu', wgpu_ep.get_library_path())
device = next(d for d in ort.get_ep_devices() if d.ep_name == wgpu_ep.get_ep_name())
so = ort.SessionOptions()
so.log_severity_level = 3
so.enable_profiling = True
so.add_provider_for_devices([device], {})
sess = ort.InferenceSession(MODEL, sess_options=so)
rng = np.random.default_rng(7)
inp = {'encoder_outputs': rng.standard_normal((1, 1024, 1), dtype=np.float32),
       'targets': np.array([[4]], dtype=np.int32),
       'target_length': np.array([1], dtype=np.int32),
       'input_states_1': rng.standard_normal((2, 1, 640), dtype=np.float32),
       'input_states_2': rng.standard_normal((2, 1, 640), dtype=np.float32)}
sess.run(None, inp)
prof = sess.end_profiling()
evs = json.load(open(prof, encoding='utf-8'))
provs = Counter()
cpu_nodes = []
for e in evs:
    if e.get('cat') != 'Node': continue
    p = str(e.get('args', {}).get('provider', '?'))
    provs[p] += 1
    if 'WebGpu' not in p:
        cpu_nodes.append((str(e.get('name')), p))
print('per-run node providers:', dict(provs))
print('non-webgpu nodes:')
for n, p in cpu_nodes[:20]:
    print('  ', n[:80], p)
