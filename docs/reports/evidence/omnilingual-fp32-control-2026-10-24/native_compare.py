from __future__ import annotations
import hashlib
import json
import os
import sys
import time
import faulthandler
import wave
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from fairseq2.nn.batch_layout import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

ROOT = Path('/mnt/n/github/asrjs/speech-recognition/tmp/omnilingual-native-wsl-20261024')
REPO = Path('/mnt/n/github/asrjs/speech-recognition')
FIXTURE = REPO / 'tests/fixtures/jfk2.en.wav'
FP32 = ROOT / 'artifacts/model.onnx'
INT8 = REPO / 'models/onnx/omnilingual-ctc-300m/model.int8.onnx'
OUTPUT = ROOT / 'logs/native-onnx-jfk.json'
STATE = ROOT / 'logs/native-run-state.json'
WAVEFORM_NPY = ROOT / 'artifacts/native-v2-jfk-waveform.npy'
RAW_PCM_NPY = ROOT / 'artifacts/jfk2-decoded-pcm-float32.npy'
LOGITS_NPY = ROOT / 'artifacts/native-v2-jfk-logits.npy'

faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)


def mark(stage: str, **details) -> None:
    STATE.write_text(json.dumps({'stage': stage, **details}, indent=2) + '\n', encoding='utf-8')
    print(f'[{stage}] {json.dumps(details, ensure_ascii=False)}', flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_summary(t: torch.Tensor) -> dict:
    x = t.detach().to(dtype=torch.float32, device='cpu').contiguous()
    return {
        'shape': list(x.shape),
        'sha256Float32LE': sha256_bytes(x.numpy().astype('<f4', copy=False).tobytes()),
        'mean': float(x.mean()),
        'stdPopulation': float(x.std(unbiased=False)),
        'min': float(x.min()),
        'max': float(x.max()),
    }


def collapse_ids(frame_ids: np.ndarray) -> list[int]:
    collapsed: list[int] = []
    last = None
    for value in frame_ids.tolist():
        value = int(value)
        if value != last:
            collapsed.append(value)
        last = value
    return collapsed


def compare_logits(native: np.ndarray, other: np.ndarray) -> dict:
    if native.shape != other.shape:
        return {'sameShape': False, 'nativeShape': list(native.shape), 'otherShape': list(other.shape)}
    a = native.astype(np.float64, copy=False).reshape(-1)
    b = other.astype(np.float64, copy=False).reshape(-1)
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    native_ids = native.argmax(axis=-1)
    other_ids = other.argmax(axis=-1)
    return {
        'sameShape': True,
        'logitsCount': int(a.size),
        'maxAbsDiff': float(diff.max(initial=0)),
        'meanAbsDiff': float(diff.mean()),
        'cosineSimilarity': float(np.dot(a, b) / denom) if denom else None,
        'frameArgmaxAgreement': float(np.mean(native_ids == other_ids)),
        'frames': int(native.shape[0]),
    }


def ctc_decode(pipe: ASRInferencePipeline, logits: torch.Tensor, lengths: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)[0]
    length = int(lengths[0])
    frame_ids = pred[:length].detach().cpu().numpy().astype(np.int64)
    ids = collapse_ids(frame_ids)
    decoded = pipe.token_decoder(torch.tensor(ids, dtype=torch.int64))
    return {
        'transcript': str(decoded).strip(),
        'frameCount': len(frame_ids),
        'collapsedIds': ids,
        'frameIds': frame_ids.tolist(),
    }


def main() -> None:
    mark('imports-ready', python=sys.version.split()[0], torch=torch.__version__, ort=ort.__version__)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    model_card = 'omniASR_CTC_300M_v2'
    mark('model-load-start', modelCard=model_card, device='cpu', dtype='float32')
    pipe = ASRInferencePipeline(model_card=model_card, device='cpu', dtype=torch.float32)
    mark('model-load-complete', modelType=type(pipe.model).__name__)
    asset_root = Path(os.environ.get('FAIRSEQ2_CACHE_DIR', ''))
    asset_files = []
    if asset_root.exists():
        for asset in sorted(p for p in asset_root.rglob('*') if p.is_file()):
            asset_files.append({
                'path': str(asset.relative_to(asset_root)),
                'bytes': asset.stat().st_size,
                'sha256': sha256_file(asset),
            })
    mark('assets-inventoried', cacheRoot=str(asset_root), files=asset_files)
    waveforms = list(pipe._build_audio_wavform_pipeline([FIXTURE]).and_return())
    if len(waveforms) != 1:
        raise RuntimeError(f'expected one preprocessed waveform, got {len(waveforms)}')
    waveform = waveforms[0].detach().to(device='cpu', dtype=torch.float32).contiguous()
    np.save(WAVEFORM_NPY, waveform.numpy().astype('<f4', copy=False), allow_pickle=False)
    with wave.open(str(FIXTURE), 'rb') as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise RuntimeError('fixture must remain mono 16 kHz PCM16')
        raw_pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype('<f4') / 32768.0
    np.save(RAW_PCM_NPY, raw_pcm.astype('<f4', copy=False), allow_pickle=False)
    batch = pipe._create_batch_simple([(waveform, None)])
    layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens, device=batch.source_seqs.device)

    native_started = time.perf_counter()
    mark('native-forward-start', waveform=tensor_summary(waveform))
    native_logits, native_layout = pipe.model(batch.source_seqs, layout)
    native_elapsed = time.perf_counter() - native_started
    native_output = ctc_decode(pipe, native_logits, native_layout.seq_lens)
    native_array = native_logits[0, :native_output['frameCount']].detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
    np.save(LOGITS_NPY, native_array.astype('<f4', copy=False), allow_pickle=False)
    mark('native-forward-complete', transcript=native_output['transcript'], logitsShape=list(native_logits.shape), sequenceLength=native_output['frameCount'])

    result = {
        'schemaVersion': 1,
        'status': 'IN_PROGRESS',
        'scope': 'official Meta v2 native CPU versus exact local ONNX artifacts on identical upstream-preprocessed waveform; not a broad quality certification',
        'source': {
            'repository': 'facebookresearch/omnilingual-asr',
            'revision': '81f51e224ce9e74b02cc2a3eaf21b2d91d743455',
            'card': model_card,
            'checkpoint': 'https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt',
            'tokenizer': 'https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model',
            'identity': 'official v2 card assets; hashes recorded after resolution/download',
        },
        'runtime': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torchaudio': __import__('torchaudio').__version__,
            'onnxruntime': ort.__version__,
            'device': 'cpu',
            'intraOpThreads': torch.get_num_threads(),
            'interOpThreads': torch.get_num_interop_threads(),
            'cudaVisibleDevices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        },
        'fixture': {
            'path': str(FIXTURE),
            'sha256': sha256_bytes(FIXTURE.read_bytes()),
            'bytes': FIXTURE.stat().st_size,
        },
        'preprocessing': {
            'source': 'Meta ASRInferencePipeline._build_audio_wavform_pipeline',
            'sampleRate': 16000,
            'waveform': tensor_summary(waveform),
        },
        'native': {
            'logitsShape': list(native_logits.shape),
            'sequenceLength': native_output['frameCount'],
            'elapsedSeconds': native_elapsed,
            **native_output,
        },
        'onnx': {},
    }

    for name, graph in [('fp32', FP32), ('int8', INT8)]:
        mark(f'onnx-{name}-start', path=str(graph))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(graph), sess_options=so, providers=['CPUExecutionProvider'])
        x = waveform.detach().cpu().numpy().astype(np.float32, copy=False)[None, :]
        onnx_started = time.perf_counter()
        outputs = session.run(None, {'x': x})
        onnx_elapsed = time.perf_counter() - onnx_started
        onnx_logits = np.asarray(outputs[0], dtype=np.float32)[0]
        np.save(ROOT / f'artifacts/{name}-jfk-logits.npy', onnx_logits.astype('<f4', copy=False), allow_pickle=False)
        frame_ids = onnx_logits.argmax(axis=-1).astype(np.int64)
        collapsed = collapse_ids(frame_ids)
        transcript = str(pipe.token_decoder(torch.tensor(collapsed, dtype=torch.int64))).strip()
        result['onnx'][name] = {
            'path': str(graph),
            'sha256': sha256_file(graph),
            'logitsShape': list(np.asarray(outputs[0]).shape),
            'elapsedSeconds': onnx_elapsed,
            'frameCount': int(frame_ids.size),
            'transcript': transcript,
            'frameIds': frame_ids.tolist(),
            'collapsedIds': collapsed,
            'nativeComparison': compare_logits(native_array, onnx_logits),
        }
        mark(f'onnx-{name}-complete', transcript=transcript, logitsShape=list(np.asarray(outputs[0]).shape), comparison=result['onnx'][name]['nativeComparison'])
        session = None
        del outputs, onnx_logits, frame_ids, collapsed

    result['status'] = 'ORACLE_RUN_COMPLETE_REVIEW_REQUIRED'
    result['elapsedTotalSeconds'] = time.perf_counter() - started
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    mark('oracle-run-complete', output=str(OUTPUT), elapsedTotalSeconds=result['elapsedTotalSeconds'])
    print(json.dumps({
        'status': result['status'],
        'fixture': result['fixture'],
        'preprocessing': result['preprocessing'],
        'native': {k: result['native'][k] for k in ('transcript', 'logitsShape', 'sequenceLength', 'elapsedSeconds')},
        'onnx': {k: {kk: vv for kk, vv in v.items() if kk not in ('frameIds', 'collapsedIds')} for k, v in result['onnx'].items()},
        'output': str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
