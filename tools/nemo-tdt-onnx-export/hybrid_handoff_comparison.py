"""Compare the hybrid's own encoder inputs, encoder outputs, and head feed.

The observation the record said was missing: the hybrid browser run now records
what it actually fed the encoder and what the encoder actually produced, in the
same run as the head feed.

1. Did the hybrid feed the encoder the same features the encoder-sensitivity
   replay recorded? Previously inferred from a Node recomputation.
2. Do the hybrid's encoder outputs differ from the replay's encoder outputs?
3. Does the hybrid's recorded head feed equal its own encoder output column, or
   has something altered it in between?

Read-only: reads recorded receipts and writes nothing.
"""
import json
import sys

import numpy as np

HYBRID = "tools/data/results/orukeet/browser-hybrid-encoder-trace-jfk-2026-10-03.json"
ENCODER_RUN = "tools/data/results/orukeet/browser-production-feature-encoder-jfk-2026-10-03.json"
ATOL, RTOL = 1e-5, 1e-4


def array_of(record):
    arr = np.asarray(record["data"], dtype=np.float32).reshape(record["shape"])
    return arr


def report(label, left, right):
    err = np.abs(left - right)
    bad = np.flatnonzero((err > ATOL + RTOL * np.abs(right)).reshape(-1))
    print(f"  {label}: maxAbs={err.max():.6e} meanAbs={err.mean():.6e} "
          f"mismatch={bad.size} first={int(bad[0]) if bad.size else None}")
    return float(err.max())


hybrid = json.load(open(HYBRID))
encoder_run = json.load(open(ENCODER_RUN))
runs = hybrid["result"]["runs"]
traced = runs[0]
encoder_calls = traced.get("encoderCalls") or []
head_calls = traced.get("headCalls") or []
print(f"hybrid encoder calls captured: {len(encoder_calls)}")
print(f"hybrid head calls captured   : {len(head_calls)}")
if not encoder_calls:
    sys.exit("No encoder calls captured; nothing to compare.")

call = encoder_calls[0]
hybrid_features = array_of(call["feeds"]["audio_signal"])
hybrid_length = int(np.asarray(call["feeds"]["length"]["data"]).reshape(-1)[0])
hybrid_encoder = array_of(call["outputs"]["outputs"])
print(f"hybrid encoder input  shape={hybrid_features.shape} length={hybrid_length}")
print(f"hybrid encoder output shape={hybrid_encoder.shape}")

print()
print("== Q1: did the hybrid feed the encoder the same features as the replay? ==")
recorded = encoder_run["result"]["productionFeatures"]
recorded_features = np.asarray(recorded["data"], dtype=np.float32).reshape(recorded["shape"])
replay_length = int(encoder_run["result"]["productionLength"])
print(f"  replay recorded shape={recorded_features.shape} length={replay_length}")
print(f"  lengths agree: {hybrid_length == replay_length}")
if hybrid_features.shape == recorded_features.shape:
    print(f"  byte-identical: "
          f"{hybrid_features.tobytes() == recorded_features.tobytes()}")
    report("hybrid vs replay features", hybrid_features, recorded_features)
else:
    print("  SHAPE DIFFERS — the two runs fed different tensors")

print()
print("== Q2: hybrid encoder output vs replay encoder output ==")
replay_rows = [r for r in encoder_run["result"]["runs"]
               if r["featureSource"] == "production-js"]
replay_encoder = array_of(replay_rows[0]["outputs"]["outputs"])
print(f"  replay encoder output shape={replay_encoder.shape}")
if hybrid_encoder.shape == replay_encoder.shape:
    report("hybrid vs replay encoder output", hybrid_encoder, replay_encoder)
    print(f"  bitwise identical: "
          f"{hybrid_encoder.tobytes() == replay_encoder.tobytes()}")
else:
    print(f"  shape differs: hybrid={hybrid_encoder.shape} replay={replay_encoder.shape}")

print()
print("== Q3: does the hybrid head feed equal its own encoder output? ==")
call20 = next((c for c in head_calls if c["index"] == 20), None)
if call20 is None:
    print("  call 20 not captured")
else:
    head_feed = array_of(call20["feeds"]["encoder_outputs"])
    print(f"  head feed shape={head_feed.shape}")
    frames = hybrid_encoder.shape[2]
    deltas = [float(np.abs(hybrid_encoder[:, :, f:f + 1] - head_feed).max())
              for f in range(frames)]
    best = int(np.argmin(deltas))
    print(f"  closest encoder column: frame {best} at maxAbs={deltas[best]:.6e}")
    print(f"  (frame 69: {deltas[69]:.6e})")
    if deltas[best] == 0.0:
        print("  HEAD FEED IS BITWISE EQUAL to its own encoder output column.")
    else:
        print("  HEAD FEED DIFFERS from every encoder column.")