import array
import hashlib
import json
import math
import os
import sys
import wave

import numpy as np
import onnxruntime as ort

ROOT = os.getcwd()
WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "fixtures", "jfk2.en.wav")
MODEL_PATH = os.path.join(ROOT, "models", "onnx", "omnilingual-ctc-300m", "model.int8.onnx")
TOKENS_PATH = os.path.join(ROOT, "models", "onnx", "omnilingual-ctc-300m", "tokens.txt")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def stats(values):
    return {"mean": float(np.mean(values, dtype=np.float64)),
            "std": float(np.std(values, dtype=np.float64)),
            "min": float(np.min(values)), "max": float(np.max(values))}

with wave.open(WAV_PATH, "rb") as wav:
    channels, rate, width, frames = wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()
    assert channels == 1 and width == 2, (channels, width)
    raw = wav.readframes(frames)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)

# Mirrors normalizeWav2Vec2Waveform: JS-number mean/variance, epsilon 1e-7,
# each final sample stored as float32.
mean = sum(float(x) for x in pcm) / len(pcm)
variance = sum((float(x) - mean) ** 2 for x in pcm) / len(pcm)
std = math.sqrt(variance + 1e-7)
normalized = np.asarray([(float(x) - mean) / std for x in pcm], dtype=np.float32)

# Parse the same `<token> <id>` lines; preserve the real-space token at ID 4.
id_to_token = []
for raw_line in open(TOKENS_PATH, encoding="utf-8").read().splitlines():
    line = raw_line.rstrip()
    if not line:
        continue
    separator = line.rfind(" ")
    if separator <= 0:
        continue
    token, token_id = line[:separator], int(line[separator + 1:])
    while len(id_to_token) <= token_id:
        id_to_token.append("")
    id_to_token[token_id] = token
blank_id = id_to_token.index("<s>")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
logits = session.run(["logits"], {"x": normalized[None, :]})[0]
frame_ids = np.argmax(logits[0], axis=-1).tolist()
collapsed = []
previous = None
for token_id in frame_ids:
    if token_id != previous and token_id != blank_id:
        collapsed.append(token_id)
    previous = token_id
text = "".join(id_to_token[token_id] if id_to_token[token_id] not in {"<s>", "<pad>", "<unk>", "</s>"} else "" for token_id in collapsed)
first_frames = []
for frame in logits[0, :min(6, len(logits[0]))]:
    top_ids = np.argsort(frame)[-5:][::-1]
    first_frames.append({"argmax": int(top_ids[0]), "top": [{"id": int(i), "value": float(frame[i])} for i in top_ids]})

result = {
    "status": "CPU_TRACE_NOT_ACCEPTANCE",
    "runtime": {"python": __import__("sys").version, "onnxruntime": ort.__version__, "providers": session.get_providers()},
    "input": {"path": WAV_PATH, "wavSha256": sha(open(WAV_PATH, "rb").read()), "sampleRate": rate, "channels": channels, "samples": len(pcm)},
    "pcm": {"sha256Float32le": sha(pcm.tobytes()), **stats(pcm)},
    "normalizedPcm": {"sha256Float32le": sha(normalized.tobytes()), **stats(normalized)},
    "artifacts": {"modelPath": MODEL_PATH, "modelSha256": sha(open(MODEL_PATH, "rb").read()), "tokensSha256": sha(open(TOKENS_PATH, "rb").read())},
    "graph": {"input": [x.name for x in session.get_inputs()], "output": [x.name for x in session.get_outputs()], "frameCount": int(logits.shape[1]), "vocabSize": int(logits.shape[2])},
    "tokenizer": {"vocabSize": len(id_to_token), "blankId": blank_id},
    "firstFrames": first_frames,
    "collapsedIds": collapsed,
    "text": text,
}
out = os.path.join(ROOT, "tmp", "omnilingual-diagnosis", "python-cpu-trace-" + os.path.splitext(os.path.basename(WAV_PATH))[0] + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({**result, "collapsedIds": collapsed[:100]}, ensure_ascii=False, indent=2))
