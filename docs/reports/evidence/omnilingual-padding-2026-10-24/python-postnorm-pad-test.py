"""CPU-only test: does extra sequence length help when speech scale is held fixed?"""
import array
import hashlib
import json
import math
import os
import wave

import numpy as np
import onnxruntime as ort

root = os.getcwd()
wav = os.path.join(root, "tests", "fixtures", "jfk2.en.wav")
model = os.path.join(root, "models", "onnx", "omnilingual-ctc-300m", "model.int8.onnx")
tokens_path = os.path.join(root, "models", "onnx", "omnilingual-ctc-300m", "tokens.txt")
with wave.open(wav, "rb") as f:
    pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / np.float32(32768)
mean = sum(float(x) for x in pcm) / len(pcm)
variance = sum((float(x) - mean) ** 2 for x in pcm) / len(pcm)
std = math.sqrt(variance + 1e-7)
normalized = np.asarray([(float(x) - mean) / std for x in pcm], dtype=np.float32)
# Official Meta preprocessing uses torch.nn.functional.layer_norm with its
# default eps=1e-5; compare that contract with the library's eps=1e-7.
official_std = math.sqrt(variance + 1e-5)
official_normalized = np.asarray([(float(x) - mean) / official_std for x in pcm], dtype=np.float32)
# Hold the normalization of the original file fixed, then append one second
# of normalized silence. This separates sequence extension from re-normalizing.
inputs = {"library_eps1e-7_no_pad": normalized, "official_eps1e-5_no_pad": official_normalized, "fixed_scale_right1": np.pad(normalized, (0, 16000))}
id_to_token = []
for raw in open(tokens_path, encoding="utf-8").read().splitlines():
    line = raw.rstrip()
    if not line:
        continue
    cut = line.rfind(" ")
    if cut <= 0:
        continue
    token, token_id = line[:cut], int(line[cut + 1:])
    while len(id_to_token) <= token_id:
        id_to_token.append("")
    id_to_token[token_id] = token
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
out = {"runtime": {"ort": ort.__version__, "providers": session.get_providers()}, "normalization": {"sourceSha256": hashlib.sha256(normalized.tobytes()).hexdigest(), "mean": float(normalized.mean()), "std": float(normalized.std())}, "cases": {}}
logits_by_name = {}
for name, audio in inputs.items():
    logits = session.run(["logits"], {"x": audio[None, :]})[0]
    logits_by_name[name] = logits[0]
    ids = np.argmax(logits[0], axis=-1).tolist()
    collapsed, previous = [], None
    for token_id in ids:
        if token_id != previous and token_id != 0:
            collapsed.append(token_id)
        previous = token_id
    text = "".join(id_to_token[i] if id_to_token[i] not in {"<s>", "<pad>", "<unk>", "</s>"} else "" for i in collapsed)
    out["cases"][name] = {"samples": len(audio), "frames": int(logits.shape[1]), "collapsedIds": collapsed, "text": text}
short = np.argmax(logits_by_name["library_eps1e-7_no_pad"], axis=-1)
long = np.argmax(logits_by_name["fixed_scale_right1"][:len(short)], axis=-1)
different = np.flatnonzero(short != long)
out["same_content_frame_alignment"] = {
    "framesCompared": len(short),
    "argmaxFramesChanged": int(len(different)),
    "changedFraction": float(len(different) / len(short)),
    "firstChangedFrames": different[:40].tolist(),
}
eps7 = np.argmax(logits_by_name["library_eps1e-7_no_pad"], axis=-1)
eps5 = np.argmax(logits_by_name["official_eps1e-5_no_pad"], axis=-1)
eps_diff = np.flatnonzero(eps7 != eps5)
out["normalization_epsilon_comparison"] = {
    "framesCompared": len(eps7),
    "argmaxFramesChanged": int(len(eps_diff)),
    "changedFraction": float(len(eps_diff) / len(eps7)),
    "officialNormalizedSha256": hashlib.sha256(official_normalized.tobytes()).hexdigest(),
}
path = os.path.join(root, "tmp", "omnilingual-diagnosis", "python-postnorm-pad-test.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
