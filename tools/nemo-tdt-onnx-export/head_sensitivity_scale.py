"""Second probe: is the encoder feed error amplified linearly?

Applies scaled copies of the recorded call-20 encoder delta on the official
reference inputs and measures the resulting normalized-score error. Read-only.
"""
import gzip, json, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, "tools/nemo-tdt-onnx-export")
from validate_stage_replay import log_softmax

CAP="tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz"
RUN="tools/data/results/orukeet/browser-real-public-hybrid-jfk-2026-10-03.json"
GRAPH="models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx"
CALL, FRAME, PRED = 20, 69, "predictor/0015"

cap=json.load(gzip.open(CAP)); run=json.load(open(RUN))
stages=cap["samples"][0]["stages"]; calls=run["result"]["runs"][0]["headCalls"]
opts=ort.SessionOptions(); opts.intra_op_num_threads=4
sess=ort.InferenceSession(GRAPH,sess_options=opts,providers=["CPUExecutionProvider"])
names=[o.name for o in sess.get_outputs()]
t=lambda v: np.asarray(v["data"],dtype=v["dtype"]).reshape(v["shape"])

official=t(stages[f"joint/{CALL:04d}/output"])
ref={"encoder_outputs":np.ascontiguousarray(t(stages["encoder"])[:,:,FRAME:FRAME+1]),
     "targets":t(stages[PRED+"/labels"]).astype(np.int32),
     "target_length":np.ones(1,dtype=np.int32),
     "input_states_1":t(stages[PRED+"/input_state_h"]),
     "input_states_2":t(stages[PRED+"/input_state_c"])}
rec={n:t(v) for n,v in calls[CALL]["feeds"].items()}
delta=(rec["encoder_outputs"]-ref["encoder_outputs"]).astype(np.float32)
print(f"recorded encoder delta maxAbs={np.abs(delta).max():.6e}\n")
print(" scale  scoreMaxAbsError  mismatch  first")
for scale in (0.0,0.25,0.5,1.0,2.0,4.0,8.0):
    f=dict(ref)
    f["encoder_outputs"]=np.ascontiguousarray(ref["encoder_outputs"]+delta*scale,dtype=np.float32)
    ls=log_softmax(dict(zip(names,sess.run(None,f)))["outputs"])
    err=np.abs(official-ls); bad=np.nonzero(err>1e-5+1e-4*np.abs(official))[0]
    print(f"{scale:5.2f}   {err.max():.6e}       {bad.size:3d}      {bad[0] if bad.size else '-'}")
