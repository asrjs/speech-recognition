"""Frozen NeMo FFT512 diagnostic: Torch complex256 + f32 recombination + JS mel.

No weight loading or production inference. Negative numerical results are
valid diagnostic receipts, not accepted ports. Requires built dist/audio.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from artifact_outputs import require_new_outputs, write_new_text
from tensor_parity import compare_tensors
from complex_intervention import validate_complex_intervention
from compare_js_feature_boundaries import NODE_PROBE, encode_complex_boundary


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_probe(path, source):
    """Save executed UTF-8 bytes exclusively, without Windows newline translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(source.encode("utf-8"))


def read_stage(sample, name, dtype):
    stage = sample["stages"][name]
    shape, data = stage.get("shape"), stage.get("data")
    if (stage.get("dtype") != dtype or not isinstance(shape, list) or not shape
            or any(type(size) is not int or size < 1 for size in shape)
            or not isinstance(data, list) or len(data) != int(np.prod(shape))
            or any(type(value) not in (int, float) for value in data)):
        raise ValueError(f"Invalid captured tensor contract: {name}")
    values = np.asarray(data)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite captured tensor: {name}")
    if dtype == "int64":
        if any(type(value) is not int for value in data):
            raise ValueError(f"Noninteger captured length: {name}")
    elif not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
        raise ValueError(f"Captured tensor is not exact float32: {name}")
    return values.astype(dtype).reshape(shape)


def read_capture(path, expected_hash, sample_id):
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)):
        raise ValueError("Expected a lowercase SHA-256 capture identity")
    captured_bytes = path.read_bytes()
    if digest(captured_bytes) != expected_hash:
        raise ValueError("Capture SHA-256 differs")
    document = json.loads(gzip.decompress(captured_bytes) if path.suffix == ".gz" else captured_bytes)
    meta = document["meta"]
    if meta.get("capture_feature_stft") is not True:
        raise ValueError("Actual STFT capture marker missing")
    control = document["instrumentation_control"]
    for key in ("baseline_tokens_match", "baseline_text_match", "baseline_frames_match",
                "after_tokens_match", "after_text_match", "after_durations_match"):
        if control.get(key) is not True:
            raise ValueError("Original instrumentation control failed or missing")
    matches = [sample for sample in document["samples"] if sample["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError("Exactly one requested sample is required")
    sample = matches[0]
    contract = sample["feature_stft_contract"]
    for key, value in {"n_fft": 512, "hop_length": 160, "win_length": 400, "exact_pad": False}.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError("Unsupported frozen STFT contract")
    pcm = read_stage(sample, "waveform", "float32")
    signal = read_stage(sample, "feature_stft/input", "float32")
    window = read_stage(sample, "feature_stft/window", "float32")
    real = read_stage(sample, "feature_stft/real", "float32")
    imag = read_stage(sample, "feature_stft/imag", "float32")
    features = read_stage(sample, "features", "float32")
    length = read_stage(sample, "features_length", "int64")
    if (pcm.ndim != 2 or pcm.shape[0] != 1 or signal.shape != pcm.shape
            or window.shape != (400,) or real.shape != imag.shape
            or real.shape != (1, 257, pcm.size // 160 + 1)
            or features.shape != (1, 128, real.shape[2]) or length.shape != (1,)
            or not 0 < length[0] <= real.shape[2] or np.any(features[:, :, length[0]:] != 0)):
        raise ValueError("Frozen boundary shape or valid-prefix contract differs")
    identity = sample["model_input"]
    if (identity.get("sample_rate") != 16000 or identity.get("channels") != 1
            or identity.get("dtype") != "float32" or identity.get("encoding") != "f32le"
            or identity.get("shape") != list(pcm.shape) or identity.get("samples") != pcm.size
            or identity.get("valid_samples") != pcm.size or identity.get("right_padding_samples") != 0
            or identity.get("sha256") != digest(pcm.astype("<f4").tobytes())):
        raise ValueError("Frozen PCM identity differs")
    spectrum = np.empty(real.shape, np.complex64)
    spectrum.real, spectrum.imag = real, imag
    frames = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(
        np.pad(signal, ((0, 0), (256, 256))), 512, axis=-1)[:, ::160, :]
        * np.pad(window, (56, 56)), dtype=np.float32)
    return meta, sample, pcm, spectrum, frames, features, int(length[0])


def recombine_packed(fft):
    if (not isinstance(fft, np.ndarray) or fft.dtype != np.complex64 or fft.ndim != 3
            or fft.shape[0] != 1 or fft.shape[2] != 256 or not fft.size or not np.isfinite(fft).all()):
        raise ValueError("Expected finite batch-frame-bin complex64 FFT256")
    real, imag = fft.real, fft.imag
    output = np.empty(fft.shape[:-1] + (257,), np.complex64)
    output.real[..., 0], output.imag[..., 0] = real[..., 0] + imag[..., 0], 0
    output.real[..., 256], output.imag[..., 256] = real[..., 0] - imag[..., 0], 0
    for k in range(1, 128):
        er = np.float32(np.float32(real[..., k] + real[..., 256-k]) * np.float32(.5))
        ei = np.float32(np.float32(imag[..., k] - imag[..., 256-k]) * np.float32(.5))
        oddr = np.float32(np.float32(imag[..., k] + imag[..., 256-k]) * np.float32(.5))
        oddi = np.float32(np.float32(real[..., k] - real[..., 256-k]) * np.float32(-.5))
        wc, ws = np.float32(np.cos(-2*np.pi*k/512)), np.float32(np.sin(-2*np.pi*k/512))
        tr = np.float32(oddr.astype(np.float64)*wc - oddi.astype(np.float64)*ws)
        ti = np.float32(oddr.astype(np.float64)*ws + oddi.astype(np.float64)*wc)
        output.real[..., k], output.imag[..., k] = er + tr, ei + ti
        output.real[..., 256-k], output.imag[..., 256-k] = er - tr, ti - ei
    output.real[..., 128], output.imag[..., 128] = real[..., 128], -imag[..., 128]
    return output.transpose(0, 2, 1)


def build_probe():
    marker = "const experiments = {"
    if NODE_PROBE.count(marker) != 1:
        raise ValueError("Re-inspect existing probe composition contract")
    return NODE_PROBE.split(marker)[0] + r'''
if (raw.frameCount !== data.physicalFrames || raw.length !== data.validLength)
  throw new Error('Probe physical/valid frame contract differs');
const results = {};
for (const [name, boundary] of Object.entries(data.spectra)) {
  data.referenceComplex = boundary;
  const c = await experiment(name, { window: true, windowProduct: true, filterbank: true,
    referenceComplexPower: 'fp32-rounded', melSpectrum: 'official-complex-rounded',
    melArithmetic: 'f64-product-f32-sum', roundLogGuard: true, capturePower: true });
  const n = normalizeCandidate(Float32Array.from(c.raw), raw.frameCount, raw.length, 8);
  results[name] = { features: Array.from(n.features), mean: Array.from(n.mean), std: Array.from(n.std) };
  if (data.coordinate) {
    const [mel, frame] = data.coordinate, index = mel * raw.frameCount + frame;
    results[name].point = { raw: c.raw[index], linearMel: c.linearMel[index], logInput: c.logInput[index],
      normalized: n.features[index], mean: n.mean[mel], std: n.std[mel] };
  }
}
console.log(JSON.stringify({ physicalFrames: raw.frameCount, validLength: raw.length, results }));
'''


def validate_probe_output(js, names, reference, valid, coordinate):
    if (not isinstance(js, dict) or set(js) != {"physicalFrames", "validLength", "results"}
            or type(js["physicalFrames"]) is not int or js["physicalFrames"] != reference.shape[2]
            or type(js["validLength"]) is not int or js["validLength"] != valid
            or not isinstance(js["results"], dict) or set(js["results"]) != set(names)):
        raise ValueError("Returned probe names or physical/valid contract differs")
    comparisons = {}
    for name, result in js["results"].items():
        fields = {"features", "mean", "std"} | ({"point"} if coordinate is not None else set())
        if not isinstance(result, dict) or set(result) != fields:
            raise ValueError("Returned probe result fields differ")
        values = {}
        for key, shape in (("features", list(reference.shape)),
                           ("mean", [reference.shape[1]]), ("std", [reference.shape[1]])):
            values[key] = read_stage({"stages": {key: {
                "shape": shape, "dtype": "float32", "data": result[key]}}}, key, "float32")
        if np.any(values["features"][:, :, valid:] != 0) or np.any(values["std"] <= 0):
            raise ValueError("Returned feature padding or standard deviation differs")
        comparisons[name] = compare_tensors(reference, values["features"], atol=1e-5, rtol=1e-4)
        if coordinate is not None:
            point = result["point"]
            if (not isinstance(point, dict)
                    or set(point) != {"raw", "linearMel", "logInput", "normalized", "mean", "std"}
                    or any(type(value) not in (int, float) or not np.isfinite(value)
                           for value in point.values())):
                raise ValueError("Returned probe point fields differ")
            mel, frame = coordinate
            if (point["normalized"] != values["features"][0, mel, frame]
                    or point["mean"] != values["mean"][mel] or point["std"] != values["std"][mel]):
                raise ValueError("Returned probe point does not match full-prefix output")
            comparisons[name]["point"] = point
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--coordinate", type=int, nargs=2, metavar=("MEL", "FRAME"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.timeout < 1:
        parser.error("Positive CPU threads and Node timeout required")
    sources = [Path(__file__), Path(__file__).with_name("compare_js_feature_boundaries.py"),
               Path(__file__).with_name("complex_intervention.py"),
               Path(__file__).with_name("normalization_reduction.py"),
               Path(__file__).with_name("normalization-reduction.mjs"),
               Path(__file__).with_name("packed-f32-source.mjs"), ROOT / "dist/audio/js-mel.js",
               ROOT / "tools/tensor_parity.py", ROOT / "tools/artifact_outputs.py"]
    require_new_outputs([args.output, args.probe_output], protected_inputs=[args.capture, *sources],
                        protected_directories=[ROOT / "models", ROOT / "src"])
    meta, sample, pcm, oracle, frames, reference, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    source_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    if args.coordinate is not None:
        mel, frame = args.coordinate
        if not 0 <= mel < 128 or not 0 <= frame < valid:
            raise ValueError("Intervention mel/frame is outside the valid prefix")
    # All identity/output checks precede heavyweight imports; no NeMo/model load.
    cache = ROOT / ".cache" / "nemo-reference" / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    import torch
    torch.set_num_threads(args.cpu_threads)
    import librosa
    real512 = torch.fft.rfft(torch.from_numpy(frames), dim=-1).numpy().transpose(0, 2, 1)
    if real512.tobytes() != oracle.tobytes():
        raise ValueError("Torch real512 does not reproduce actual captured complex bytes")
    packed = np.empty(frames.shape[:-1] + (256,), np.complex64)
    packed.real, packed.imag = frames[..., ::2], frames[..., 1::2]
    fft256 = torch.fft.fft(torch.from_numpy(packed), dim=-1).numpy()
    candidate = recombine_packed(fft256)
    filterbank = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney")
    spectra = {"official_complex_control": oracle, "packed256_candidate": candidate}
    intervention = None
    if args.coordinate is not None:
        active = np.flatnonzero(filterbank[mel])
        patched = candidate.copy()
        patched[0, active, frame] = oracle[0, active, frame]
        intervention = validate_complex_intervention(candidate, oracle, patched,
            [(int(bin_index), frame) for bin_index in active], valid)
        intervention["coordinate_mel_frame"] = args.coordinate
        spectra["patched_candidate"] = patched
    probe = build_probe()
    write_probe(args.probe_output, probe)
    payload = {"pcm": pcm.ravel().tolist(), "bins": 128, "referenceRaw": [0]*reference.size,
               "referenceWindow": read_stage(sample, "feature_stft/window", "float32").tolist(),
               "referenceFilterbank": filterbank.ravel().tolist(), "physicalFrames": oracle.shape[2],
               "validLength": valid, "coordinate": args.coordinate,
               "spectra": {name: encode_complex_boundary(value) for name, value in spectra.items()}}
    process = subprocess.run(["node", "--input-type=module", "-e", probe], cwd=ROOT,
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, timeout=args.timeout)
    if process.returncode:
        raise RuntimeError("Node probe failed: " + process.stderr[-1500:])
    js = json.loads(process.stdout)
    comparisons = validate_probe_output(js, set(spectra), reference, valid, args.coordinate)
    if not comparisons["official_complex_control"]["allclose"]:
        raise ValueError("Official-complex downstream control fails; candidate interpretation rejected")
    if source_hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise ValueError("Diagnostic sources changed during replay; receipt rejected")
    receipt = {"status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE", "sample_id": args.sample_id,
        "capture_sha256": args.expected_capture_sha256, "capture_meta": meta,
        "physical_frames": oracle.shape[2], "valid_length": valid,
        "windowed_frames_f32le_sha256": digest(frames.astype("<f4").tobytes()),
        "filterbank_f32le_sha256": digest(filterbank.astype("<f4").tobytes()),
        "probe_path": str(args.probe_output.resolve()), "probe_sha256": digest(probe.encode()),
        "sources_sha256": source_hashes, "torch_real512_actual_stft_bits_exact": True,
        "packed256_complex64_le_sha256": digest(fft256.astype("<c8").tobytes()),
        "spectra_complex64_le_sha256": {name: digest(value.astype("<c8").tobytes())
                                       for name, value in spectra.items()},
        "filterbank_recipe": {"sample_rate": 16000, "n_fft": 512, "n_mels": 128,
                              "fmin": 0, "fmax": 8000, "norm": "slaney"},
        "torch": torch.__version__, "torch_build_config": torch.__config__.show(),
        "cpu_threads": torch.get_num_threads(), "librosa": librosa.__version__,
        "numba_cache": str(cache),
        "comparisons": comparisons, "intervention": intervention,
        "scope": "Frozen original-engine FFT control and diagnostic mel/normalization only; no portable FFT, ASR or browser gate"}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": receipt["status"],
                      "candidate_mismatch_count": comparisons["packed256_candidate"]["mismatch_count"]}))


if __name__ == "__main__":
    main()
