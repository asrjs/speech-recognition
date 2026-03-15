import onnx
from pathlib import Path
for name in ['model.onnx','model_fp16.onnx','model_int8.onnx']:
    p = Path('medasrjs/models/medasr')/name
    m=onnx.load(str(p), load_external_data=False)
    locs=[]
    for t in m.graph.initializer:
        if t.data_location==onnx.TensorProto.EXTERNAL:
            kv={e.key:e.value for e in t.external_data}
            locs.append(kv.get('location',''))
    uniq=sorted(set(locs))
    print(name, 'external_tensors=',len(locs),'unique_locations=',uniq[:3])
