
"""Spike: Parakeet TDT v3 decoder_joint on ONNX Runtime WebGPU Plugin EP 0.3.0.

Measures session creation, first inference, steady-state step latency, and
fp32 parity vs CPU EP on synthetic single-step decoder inputs. Read-only probe;
no library changes.
"""
import json, statistics, sys, time
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as wgpu_ep

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
if not MODEL:
    print("usage: spike_webgpu_plugin_ep.py <decoder_joint.onnx>")
    sys.exit(2)

def make_inputs(seed=7):
    rng = np.random.default_rng(seed)
    # Shapes mirror a single decoder step: one target token, one frame.
    enc = rng.standard_normal((1, 1024, 1), dtype=np.float32)
    targets = np.array([[4]], dtype=np.int32)
    tlen = np.array([1], dtype=np.int32)
    st1 = rng.standard_normal((2, 1, 640), dtype=np.float32) * 0.1
    st2 = rng.standard_normal((2, 1, 640), dtype=np.float32) * 0.1
    return {"encoder_outputs": enc, "targets": targets, "target_length": tlen,
            "input_states_1": st1, "input_states_2": st2}

def run(eps):
    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if eps == "webgpu_plugin":
        # Plugin EP flow from the onnxruntime-ep-webgpu 0.3.0 README:
        # register the shared library, then attach via EP devices.
        ort.register_execution_provider_library("webgpu", wgpu_ep.get_library_path())
        device = next(
            (d for d in ort.get_ep_devices() if d.ep_name == wgpu_ep.get_ep_name()),
            None,
        )
        if device is None:
            raise RuntimeError("No WebGPU EP device found")
        so.add_provider_for_devices([device], {})
        sess = ort.InferenceSession(MODEL, sess_options=so)
    else:
        sess = ort.InferenceSession(MODEL, sess_options=so, providers=eps)
    create_ms = (time.perf_counter() - t0) * 1000
    inp = make_inputs()
    t1 = time.perf_counter()
    out = sess.run(None, inp)
    first_ms = (time.perf_counter() - t1) * 1000
    steps = []
    for i in range(20):
        t2 = time.perf_counter()
        out = sess.run(None, inp)
        steps.append((time.perf_counter() - t2) * 1000)
    return sess, {
        "providers_active": sess.get_providers(),
        "create_ms": round(create_ms, 1),
        "first_ms": round(first_ms, 1),
        "median_step_ms": round(statistics.median(steps), 3),
        "min_step_ms": round(min(steps), 3),
        "outputs_finite": all(np.isfinite(o).all() for o in out),
    }, out

res = {}
_, cpu_stats, cpu_out = run(["CPUExecutionProvider"])
res["cpu"] = cpu_stats

for label, eps in [
    ("plugin_webgpu", "webgpu_plugin"),
]:
    try:
        sess, stats, out = run(eps)
        stats["parities"] = {
            "outputs_max_abs_diff": float(np.max(np.abs(out[0].astype(np.float64) - cpu_out[0].astype(np.float64)))),
            "state1_max_abs_diff": float(np.max(np.abs(out[2].astype(np.float64) - cpu_out[2].astype(np.float64)))),
        }
        stats["active"] = sess.get_providers()
        res[label] = stats
        del sess
    except Exception as e:  # noqa: BLE001
        res[label] = {"error": str(e)[:400]}

print(json.dumps(res, indent=2))
