#!/usr/bin/env python3
"""
Real-time transcription simulator: like long-file transcriber but with a sliding
window that appends 0.48s of new audio at the end each step (configurable).

Each chunk = current window; next chunk = same window length shifted by step_sec
(e.g. 30s window, 0.48s step => overlapping chunks suitable for testing
real-time merger behavior). Output format matches long_file_transcribe (JSON with
full_text, words, segments, asr_results) so you can compare to long-file reference
or run realtime_merger_sim on the output.

Usage:
  python scripts/realtime_transcribe_sim.py --audio path/to.wav -o realtime_out.json
  python scripts/realtime_transcribe_sim.py --audio path/to.wav --step-sec 0.48 --window-sec 30 -o out.json --merge-with python
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from typing import Any

# Project root = parent of scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

from audio import SAMPLE_RATE, load_audio_wav
from stream_sim import _timestamped_to_words
from window_strategies import get_strategy


def _log(verbose: bool, step: str, message: str) -> None:
    if verbose:
        print(f"[{step}] {message}")


def transcribe_realtime_sliding(
    audio_path: str | Path,
    model_dir: str | Path,
    model_name: str = "nemo-parakeet-tdt-0.6b-v2",
    window_sec: float = 30.0,
    step_sec: float = 0.48,
    merge_with: str | None = None,
    verbose: bool = False,
    suppress_silence: bool = False,
    suppress_gain: float = 0.0,
    suppress_smooth_sec: float = 0.05,
    vad_mode: str = "energy",
) -> dict[str, Any]:
    """
    Real-time-style transcription: sliding window (append step_sec new audio at end
    each time). Runs ASR per chunk, collects words and segments. Returns same
    structure as long_file_transcribe (full_text, words, segments, asr_results, meta).
    """
    try:
        import onnx_asr
    except ImportError:
        raise ImportError("onnx-asr is required; pip install onnx-asr")

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    use_backend = bool(os.environ.get("ASR_BACKEND_URL", "").strip())
    adapter = None
    if use_backend:
        _log(verbose, "model", "Using ASR_BACKEND_URL (no local model load)")
    else:
        _log(verbose, "model", f"Loading ASR model {model_name} from {model_dir}")
        model = onnx_asr.load_model(model_name, path=model_dir)
        adapter = model.with_timestamps()
        _log(verbose, "model", "Model loaded")

    _log(verbose, "audio", f"Loading audio: {audio_path}")
    waveform = load_audio_wav(audio_path)
    duration_sec = len(waveform) / SAMPLE_RATE
    _log(verbose, "audio", f"Loaded {duration_sec:.1f}s ({len(waveform)} samples at {SAMPLE_RATE} Hz)")

    if suppress_silence:
        from long_file.vad import apply_silence_suppress
        _log(verbose, "suppress", f"Applying VAD-based silence suppression (gain={suppress_gain}, smooth={suppress_smooth_sec}s)")
        waveform = apply_silence_suppress(
            waveform, SAMPLE_RATE,
            vad_mode=vad_mode,
            suppress_gain=suppress_gain,
            smooth_sec=suppress_smooth_sec,
        )
        _log(verbose, "suppress", "Done")

    strategy = get_strategy(
        "realtime_sliding",
        window_sec=window_sec,
        step_sec=step_sec,
    )
    all_words: list[dict[str, Any]] = []
    segment_results: list[dict[str, Any]] = []
    full_parts: list[str] = []
    asr_results_for_merger: list[dict[str, Any]] = []
    segment_id = 0

    _log(verbose, "segment", f"Real-time sliding: window={window_sec}s step={step_sec}s")
    for chunk, start_sec in strategy(waveform, SAMPLE_RATE):
        chunk_len_sec = len(chunk) / SAMPLE_RATE
        end_sec = start_sec + chunk_len_sec
        _log(verbose, "asr", f"Chunk {segment_id + 1}: {start_sec:.1f}s - {end_sec:.1f}s")
        if use_backend:
            try:
                from asr_backend_client import recognize as backend_recognize
                out = backend_recognize(chunk.astype(np.float32), start_sec=start_sec, sample_rate=SAMPLE_RATE)
                if out is None:
                    segment_id += 1
                    continue
                text, words = out
                text = (text or "").strip()
            except Exception as e:
                _log(verbose, "asr", f"Backend error: {e}")
                segment_id += 1
                continue
        else:
            results = adapter.recognize([chunk], sample_rate=SAMPLE_RATE)
            res = results[0] if isinstance(results, list) and results else results
            if res is None:
                segment_id += 1
                continue
            text = (res.text or "").strip()
            words = _timestamped_to_words(
                res.text or "",
                getattr(res, "tokens", None),
                getattr(res, "timestamps", None),
                time_offset_sec=start_sec,
            )
        full_parts.append(text)
        all_words.extend(words)
        segment_results.append({
            "segment_id": segment_id + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text,
            "word_count": len(words),
        })
        asr_results_for_merger.append({
            "utterance_text": text,
            "words": words,
            "start_sec": start_sec,
            "end_time": end_sec,
            "segment_id": f"chunk-{segment_id + 1}",
            "timestamp": int(end_sec * 1000),
        })
        segment_id += 1

    if merge_with and asr_results_for_merger:
        _log(verbose, "merge", f"Running merger: {merge_with}")
        from merger_bridge import run_merger
        full_text = run_merger(asr_results_for_merger, merger=merge_with).strip()
    else:
        full_text = " ".join(full_parts).strip()

    _log(verbose, "done", f"Transcribed {len(all_words)} words in {len(segment_results)} chunks")

    meta: dict[str, Any] = {
        "window_sec": window_sec,
        "step_sec": step_sec,
        "num_segments": len(segment_results),
        "duration_sec": round(duration_sec, 2),
    }
    if use_backend:
        meta["asr_backend"] = os.environ.get("ASR_BACKEND_URL", "")
    if merge_with:
        meta["merge_with"] = merge_with
    if suppress_silence:
        meta["suppress_silence"] = True
        meta["suppress_gain"] = suppress_gain
        meta["suppress_smooth_sec"] = suppress_smooth_sec

    return {
        "full_text": full_text,
        "words": all_words,
        "segments": segment_results,
        "asr_results": asr_results_for_merger,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time transcription sim: sliding window (append 0.48s new audio each step).",
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV or m4a.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Parakeet ONNX model directory.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Window length in seconds (default 30).")
    parser.add_argument("--step-sec", type=float, default=0.48, help="Step (new audio appended each chunk) in seconds (default 0.48).")
    parser.add_argument("--merge-with", type=str, default=None, metavar="NAME", help="Run merger (e.g. python, node). Default: concat only.")
    parser.add_argument("--vtt", action="store_true", help="Export WebVTT to same base path.")
    parser.add_argument("--srt", action="store_true", help="Export SubRip to same base path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log each chunk and merge.")
    parser.add_argument("--suppress-silence", action="store_true", help="Apply VAD-based silence suppression.")
    parser.add_argument("--suppress-gain", type=float, default=0.0, metavar="G", help="Gain in non-speech when --suppress-silence (0=mute).")
    parser.add_argument("--suppress-smooth-sec", type=float, default=0.05, help="Smoothing at speech boundaries (seconds).")
    parser.add_argument("--vad", choices=["energy", "ten_style"], default="energy", help="VAD mode for silence suppression.")
    args = parser.parse_args()

    _default_model = Path("N:/models/onnx/nemo/parakeet-tdt-0.6b-v2-onnx")
    if args.model_dir is None:
        args.model_dir = Path(os.environ.get("MODEL_PATH", _default_model))

    audio_path = args.audio
    if audio_path.suffix.lower() == ".m4a":
        from audio import m4a_to_wav
        audio_path = m4a_to_wav(audio_path)

    out_path = args.output or audio_path.with_suffix(".realtime_transcript.json")
    out_path = Path(out_path)

    t0 = time.perf_counter()
    data = transcribe_realtime_sliding(
        audio_path,
        args.model_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        merge_with=args.merge_with,
        verbose=args.verbose,
        suppress_silence=args.suppress_silence,
        suppress_gain=args.suppress_gain,
        suppress_smooth_sec=args.suppress_smooth_sec,
        vad_mode=args.vad,
    )
    t1 = time.perf_counter()
    elapsed_sec = t1 - t0
    duration_sec = data["meta"].get("duration_sec") or 0.0
    rtfx = round(duration_sec / elapsed_sec, 2) if elapsed_sec > 0 else 0.0
    data["meta"]["elapsed_sec"] = round(elapsed_sec, 2)
    data["meta"]["rtfx"] = rtfx

    from json_float import dumps_transcript
    out_path.write_text(dumps_transcript(data), encoding="utf-8")
    if args.verbose:
        print(f"[write] {out_path}")
    if args.vtt or args.srt:
        from subtitle_export import export_subtitles
        formats = ["vtt"] if args.vtt else []
        if args.srt:
            formats.append("srt")
        out_base = out_path.with_suffix("") if out_path.suffix else out_path
        written = export_subtitles(data, out_base, formats=formats, mode="sentences", sbd_lang="en")
        for p in written:
            if args.verbose:
                print(f"[write] {p}")

    export_parts = []
    if args.vtt:
        export_parts.append("vtt")
    if args.srt:
        export_parts.append("srt")
    summary = (
        f"Wrote {out_path} | "
        f"window={data['meta']['window_sec']}s step={data['meta']['step_sec']}s chunks={data['meta']['num_segments']} words={len(data['words'])} | "
        f"duration={data['meta']['duration_sec']}s elapsed={data['meta']['elapsed_sec']}s RTFx={data['meta']['rtfx']}"
    )
    if export_parts:
        summary += " | exported " + ", ".join(export_parts)
    print(summary)


if __name__ == "__main__":
    main()
