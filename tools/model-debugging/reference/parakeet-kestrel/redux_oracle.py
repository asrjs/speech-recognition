"""Exact-weight official Redux oracle on the pinned JFK effective PCM.

Runs the official Kestrel engine (pinned venv) over moondream/parakeet-redux at
the local snapshot revision, in the dense/oracle resident form: this workstation
has no VNNI-class CPU path, so the packed int8 kernel cannot run (see the
pack-parity receipt). The input is the deterministic mono 16 kHz float32 decode
of tools/data/fixtures/audio/jfk-short.wav, byte-bound to the established
effective-PCM SHA-256 before inference. Output-only oracle evidence: text and
request/result identities; no stage captures, no packed-kernel or speed claim.
"""
import argparse
import asyncio
import ctypes
import hashlib
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import torch

import kestrel_kernels.ternary as kernels_ternary

_original_resident_form = kernels_ternary.resident_form


def _dense_oracle_resident_form(device):
    if torch.device(device).type == 'cpu':
        return 'dense'
    return _original_resident_form(device)


kernels_ternary.resident_form = _dense_oracle_resident_form

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1, 'expected mono 16 kHz wav'
        assert handle.getsampwidth() == 2, 'expected 16-bit PCM wav'
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def rss_mb() -> int | None:
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('pageFaultCount', ctypes.c_ulong),
                    ('peakWorkingSetSize', ctypes.c_size_t), ('workingSetSize', ctypes.c_size_t),
                    ('quotaPeakPagedPoolUsage', ctypes.c_size_t), ('quotaPagedPoolUsage', ctypes.c_size_t),
                    ('quotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('quotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('pagefileUsage', ctypes.c_size_t), ('peakPagefileUsage', ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.peakWorkingSetSize / 1e6) if ok and counters.peakWorkingSetSize else None


def to_jsonable(value):
    import dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    for attr in ('output', 'to_dict', 'asdict'):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except Exception:
                pass
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default='moondream/parakeet-redux',
                        choices=['moondream/parakeet-redux', 'moondream/parakeet-ultra'])
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/parakeet-redux-full'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    pcm = effective_pcm(args.audio)
    pcm_bytes = pcm.astype('<f4').tobytes()
    pcm_sha = hashlib.sha256(pcm_bytes).hexdigest()
    if pcm_sha != EXPECTED_PCM_SHA256:
        raise SystemExit(f'effective PCM digest mismatch: {pcm_sha}')

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    started = time.time()
    from kestrel.config import RuntimeConfig
    from kestrel.engine import InferenceEngine
    config = RuntimeConfig(model=args.model_id,
                           model_path=str(args.snapshot / 'model.safetensors'),
                           tokenizer_path=str(args.snapshot / 'tokenizer.json'),
                           device='cpu')

    async def run() -> dict:
        engine = await InferenceEngine.create(config)
        try:
            model = engine.model(args.model_id)
            return await model.transcribe(audio=pcm, sample_rate=16000)
        finally:
            shutdown = getattr(engine, 'shutdown', None) or getattr(engine, 'close', None)
            if shutdown is None:
                for name in ('aclose', 'stop'):
                    shutdown = getattr(engine, name, None)
                    if shutdown is not None:
                        break
            if shutdown is not None:
                value = shutdown()
                if value is not None and hasattr(value, '__await__'):
                    await value

    result = asyncio.run(run())
    result_json = to_jsonable(result)
    result_dict = result_json if isinstance(result_json, dict) else {}
    output = result_dict.get('output')
    text = output.get('text') if isinstance(output, dict) else result_dict.get('text')
    load_and_transcribe_s = time.time() - started

    versions = {}
    for package in ('torch', 'kestrel', 'kestrel-kernels', 'kestrel-native'):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = None

    receipt = dict(
        status='PASS_OFFICIAL_ORACLE_OUTPUT' if isinstance(text, str) and text.strip()
        else 'FAIL_OFFICIAL_ORACLE_OUTPUT',
        scope='exact-weight official Kestrel Redux oracle, dense-oracle resident form, output-only; '
              'not packed-kernel evidence, not stage parity, not a speed/memory benchmark',
        weights=dict(model_id=args.model_id,
                     revision={'moondream/parakeet-redux': '2bf128600aac4b16946f7ed8372e56117fe5e23b',
                               'moondream/parakeet-ultra': '73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b'}[args.model_id],
                     archive_sha256=digest(args.snapshot / 'model.safetensors'),
                     config_sha256=digest(args.snapshot / 'config.json'),
                     tokenizer_sha256=digest(args.snapshot / 'tokenizer.json'),
                     ternary_manifest_sha256=digest(args.snapshot / 'ternary.json')
                     if (args.snapshot / 'ternary.json').is_file() else None),
        input=dict(audio_wav_sha256=digest(args.audio), effectivePcmSha256=pcm_sha,
                   samples=int(pcm.size), sampleRate=16000, encoding='f32le',
                   preparation='python wave int16 -> float32 /32768, no resampling'),
        runtime=dict(packages=versions, residentForm='dense (oracle)',
                     nativeCpuProbe=kernels_ternary._cpu.ternary_gemm_isa(),
                     torchThreads=torch.get_num_threads()),
        timings=dict(loadAndTranscribeSeconds=round(load_and_transcribe_s, 2),
                     peakRssMb=rss_mb()),
        result=result_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + '\n',
                           encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], text=text,
        seconds=receipt['timings']['loadAndTranscribeSeconds'],
        peakRssMb=receipt['timings']['peakRssMb'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
