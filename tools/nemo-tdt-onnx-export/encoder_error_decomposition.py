"""Decompose the live WebGPU encoder feed error into its sources.

The hybrid browser run feeds the real WebGPU encoder production JS features.
Two recorded real-browser encoder runs exist for the same sample: one fed
official features and one fed production-JS features. Comparing those two
encoder outputs isolates how much of the decoder-input feed error comes from
the feature input rather than from the encoder kernel itself.

Then runs the fp32 encoder graph on native CPU with the recorded production-JS
features and length to separate kernel error from feature error without a
browser.

Read-only: it reads existing receipts and writes nothing.
"""
import gzip
import json

import numpy as np
import onnxruntime as ort

ENCODER_RECEIPT = "tools/data/results/orukeet/browser-production-feature-encoder-jfk-2026-10-03.json"
CAPTURE = "tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz"
ARTIFACTS = "models/orukeet/onnx/fp32-packed-2026-10-01"
ATOL, RTOL = 1e-5, 1e-4

receipt = json.load(open(ENCODER_RECEIPT))
cap = json.load(gzip.open(CAPTURE))
stages = cap["samples"][0]["stages"]

runs = {}
for run in receipt["result"]["runs"]:
    runs.setdefault(run["featureSource"], []).append(run["outputs"]["outputs"])

official_features = np.asarray(stages["features"]["data"], dtype=np.float32)
official_features = official_features.reshape(stages["features"]["shape"])
official_encoder = np.asarray(stages["encoder"]["data"], dtype=np.float32)
official_encoder = official_encoder.reshape(stages["encoder"]["shape"])


def as_array(value):
    return np.asarray(value["data"], dtype=np.float32).reshape(value["shape"])


def gate(err, ref):
    return int((err > ATOL + RTOL * np.abs(ref)).sum())


print("== source A: official features -> real WebGPU encoder ==")
a = as_array(runs["official"][0])
err_a = np.abs(a - official_encoder)
print(f"  vs official encoder: maxAbs={err_a.max():.6e} meanAbs={err_a.mean():.6e} "
      f"failing={gate(err_a, official_encoder)}/{err_a.size}")

print()
print("== source B: production-JS features -> real WebGPU encoder ==")
b = as_array(runs["production-js"][0])
err_b = np.abs(b - official_encoder)
print(f"  vs official encoder: maxAbs={err_b.max():.6e} meanAbs={err_b.mean():.6e} "
      f"failing={gate(err_b, official_encoder)}/{err_b.size}")

print()
print("== A vs B: encoder response to the feature change alone (same kernel) ==")
ab = np.abs(a - b)
print(f"  encoder output difference: maxAbs={ab.max():.6e} meanAbs={ab.mean():.6e}")
print(f"  ratio to kernel-only error: {ab.max() / err_a.max():.2f}x")

print()
print("== feature delta that produced it ==")
pf = as_array(receipt["result"]["productionFeatures"])
feature_delta = np.abs(pf - official_features)
print(f"  feature tensor shape={pf.shape} vs official={official_features.shape}")
if pf.shape == official_features.shape:
    print(f"  feature delta maxAbs={feature_delta.max():.6e} "
          f"meanAbs={feature_delta.mean():.6e} failing={gate(feature_delta, official_features)}")
    print(f"  encoder response / feature delta at max: "
          f"{ab.max() / max(feature_delta.max(), 1e-30):.4f}x")

print()
print("== native CPU encoder on the same production-JS features ==")
length = int(receipt["result"]["productionLength"])
print(f"  recorded productionLength={length}")
options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(
    f"{ARTIFACTS}/encoder-model.onnx", sess_options=options,
    providers=["CPUExecutionProvider"])


def run_encoder(session, features, valid_length):
    """Run the fp32 encoder on one session with an explicit int64 length."""
    outputs = dict(zip(
        [o.name for o in session.get_outputs()],
        session.run(None, {
            "audio_signal": np.ascontiguousarray(features),
            "length": np.asarray([valid_length], dtype=np.int64),
        })))
    return np.asarray(outputs["outputs"], dtype=np.float32)
native = run_encoder(session, pf, length)
err_native = np.abs(native - official_encoder)
print(f"  actual providers: {session.get_providers()}")
print(f"  vs official encoder: maxAbs={err_native.max():.6e} "
      f"meanAbs={err_native.mean():.6e} failing={gate(err_native, official_encoder)}")
err_webgpu = np.abs(b - native)
print(f"  vs real WebGPU encoder (identical features): maxAbs={err_webgpu.max():.6e} "
      f"meanAbs={err_webgpu.mean():.6e} failing={gate(err_webgpu, native)}")

print()
print("== native CPU on the official features (separates execution from feature) ==")
official_length = int(np.asarray(stages["features_length"]["data"]).reshape(-1)[0])
print(f"  recorded official features_length={official_length} "
      f"(production length={length})")
native_official = run_encoder(session, official_features, official_length)
err_native_official = np.abs(native_official - official_encoder)
print(f"  native(official) vs official encoder: maxAbs={err_native_official.max():.6e} "
      f"meanAbs={err_native_official.mean():.6e} "
      f"failing={gate(err_native_official, official_encoder)}")
print("  (this is the native-versus-official execution residual, with no feature change)")

print()
print("== separated effects, same session ==")
feature_effect = np.abs(native - native_official)
print(f"  feature effect   native(production) - native(official): maxAbs="
      f"{feature_effect.max():.6e} meanAbs={feature_effect.mean():.6e}")
print(f"  execution residual native(official) - official encoder: maxAbs="
      f"{err_native_official.max():.6e} meanAbs={err_native_official.mean():.6e}")
print(f"  ratio feature effect / execution residual: "
      f"{feature_effect.max() / max(err_native_official.max(), 1e-30):.2f}x")
print()
print("  => the earlier 'feature-only 4.917383e-7' figure mixed both effects;")
print("     it is native(production) versus the official encoder. The two")
print("     quantities above are the separated ones. Neither tolerance nor")
print("     normalization is altered by this probe.")