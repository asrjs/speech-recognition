"""Measure WebGPU encoder output variation across separate browser launches.

Runs of the same graph, same features, same options, in different browser
processes. Cross-session equality within one page does not answer this.

Reported for each pair: bitwise equality and max absolute difference, on both
feature sources. The earlier production-feature receipt is included as a fourth
observation so the sample is not only the three new launches.

Read-only: reads recorded receipts and writes nothing.
"""
import glob
import json

import numpy as np

RUNS = sorted(glob.glob("tools/data/results/orukeet/browser-run-variation-*-jfk-2026-10-03.json"))
BASELINE = "tools/data/results/orukeet/browser-production-feature-encoder-jfk-2026-10-03.json"
ATOL, RTOL = 1e-5, 1e-4

observations = {}
for path in RUNS:
    receipt = json.load(open(path))
    for run in receipt["result"]["runs"]:
        if run["repeat"] != 0:
            continue
        key = f"{path.split('variation-')[1].split('-jfk')[0]}:{run['featureSource']}"
        record = run["outputs"]["outputs"]
        observations[key] = np.asarray(record["data"], dtype=np.float32).reshape(record["shape"])

baseline = json.load(open(BASELINE))
for run in baseline["result"]["runs"]:
    if run["repeat"] != 0:
        continue
    key = f"prior:{run['featureSource']}"
    record = run["outputs"]["outputs"]
    observations[key] = np.asarray(record["data"], dtype=np.float32).reshape(record["shape"])

if not observations:
    raise SystemExit("No observations found; did the launches write receipts?")

print(f"observations: {len(observations)}")
for key, array in sorted(observations.items()):
    print(f"  {key:22s} shape={array.shape} "
          f"max={array.max():.8f} mean={array.mean():.10f}")

for source in ("official", "production-js"):
    keys = sorted(k for k in observations if k.endswith(source))
    if len(keys) < 2:
        continue
    print()
    print(f"== {source}: pairwise across separate browser launches ==")
    print(f"  {'pair':44s} {'bitwise':8s} {'maxAbs':>13s} {'meanAbs':>13s} {'mismatch':>9s}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            left, right = observations[keys[i]], observations[keys[j]]
            if left.shape != right.shape:
                print(f"  {keys[i]} vs {keys[j]}: SHAPE DIFFERS")
                continue
            err = np.abs(left - right)
            bad = int((err > ATOL + RTOL * np.abs(right)).sum())
            bitwise = left.tobytes() == right.tobytes()
            print(f"  {keys[i] + ' vs ' + keys[j]:44s} {str(bitwise):8s} "
                  f"{err.max():13.6e} {err.mean():13.6e} {bad:9d}")

print()
print("== interpretation key ==")
print("  bitwise True everywhere -> encoder is reproducible across launches,")
print("    and the 1.1316e-6 gap needs another explanation.")
print("  bitwise False with ~1e-6 max -> cross-run variation is confirmed and its")
print("    magnitude is what the hybrid-vs-replay comparison measured.")