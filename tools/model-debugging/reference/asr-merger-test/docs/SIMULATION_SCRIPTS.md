# Simulation scripts: raw outputs and precomputed VAD/RMS

Scripts in this repo that produce **raw** or **precomputed** data for simulations and agents. Run them from the **repo root** so imports resolve. Default outputs go to `simulation_inputs/` unless overridden.

**Frame resolution:** Chunk and frame-slice boundaries use the ASR-aligned grid (20 ms for Parakeet; `timestamp_align.ASR_FRAME_SEC`). Fixed-window chunk boundaries are multiples of 20 ms; VAD-chunk split boundaries and waveform slice indices are aligned to the same grid so outputs align with the ASR model.

---

## 1. Precomputed VAD and RMS (no ASR)

**Script:** `scripts/precompute_vad_rms.py`

Precomputes VAD (frame-level boolean array + segment timings) and per-frame RMS so agents can load them instead of recalculating. Uses the same 20 ms frame resolution as the ASR pipeline (energy VAD in `long_file.vad`).

**Output:** One `.npz` file containing:

- `vad_frames`: bool array, one value per frame (indexed by frame index).
- `vad_segments`: (N, 2) float array; each row is (start_sec, end_sec) for a speech segment.
- `rms`: float32 array, per-frame RMS (same frame boundaries as VAD).
- `frame_sec`, `sample_rate`, `duration_sec`, `vad_mode`, `energy_threshold`, `ten_threshold`.

**Load in Python:**

```python
import numpy as np
d = np.load("simulation_inputs/myfile_vad_rms.npz")
vad_frames = d["vad_frames"]       # bool, shape (n_frames,)
vad_segments = d["vad_segments"]  # (n_segments, 2) start_sec, end_sec
rms = d["rms"]                    # float32, shape (n_frames,)
frame_sec = float(d["frame_sec"])
# Frame index i corresponds to time i * frame_sec .. (i+1) * frame_sec
```

**Run to render:**

```powershell
python scripts/precompute_vad_rms.py --audio test-data/out.m4a -v
```

Optional: `--vad-mode energy|ten_style`, `--energy-threshold`, `--ten-threshold`, `--output`, `--output-dir`.

---

## 2. Fixed-window raw transcription (1s, 5s, 8s)

**Script:** `scripts/transcribe_raw_fixed_8s.py`

Transcribes with **fixed non-overlapping** windows aligned to the ASR frame grid (20 ms). Raw ASR only (no boundary trim, no SBD). Output filename includes chunk size: `_fixed1s.json`, `_fixed5s.json`, `_fixed8s.json`.

**Output:** JSON with `segments`, `asr_results`, `full_text`, `words`, `meta` (e.g. `segmenter: "fixed_8s"`, `chunk_sec`). Use `segments` and `asr_results` for simulation input.

**Run to render 1s, 5s, and 8s:**

```powershell
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 1 -v
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 5 -v
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 8 -v
```

Outputs: `simulation_inputs/<basename>_fixed1s.json`, `_fixed5s.json`, `_fixed8s.json`.

Optional: `--model-dir`, `--output`, `--output-dir`, `--chunk-sec` (default 8).

---

## 3. VAD-chunked raw transcription

**Script:** `scripts/transcribe_raw_vad_chunks.py`

Splits audio on VAD speech segments and sub-splits so no chunk is longer than `--max-chunk-sec` (default 5s). Chunk boundaries and waveform slice indices are aligned to the 20 ms ASR grid. Raw ASR only.

**Output:** JSON with same shape as fixed-window; `meta.segmenter` is `"vad_chunks"`, `meta.max_chunk_sec`, `meta.vad_mode`.

**Run to render:**

```powershell
python scripts/transcribe_raw_vad_chunks.py --audio test-data/out.m4a -v
```

Optional: `--max-chunk-sec`, `--vad-mode energy|ten_style`, `--model-dir`, `--output`, `--output-dir`.

Output: `simulation_inputs/<basename>_vad_chunks.json`.

---

## 4. Long-form reference (VAD + SBD, optional full detail)

**Scripts:** `scripts/long_file_transcribe.py`, `scripts/long_file_transcribe_full.py`

Full pipeline: VAD, sentence-boundary cursor, optional trim for display. Used for reference transcripts and `simulation_ref_full.json` / `.vtt`. Not “raw-only”; they store raw in `segments`/`asr_results` and cleaned in `full_text`/`words`/`sentences`. See `docs/LONG_FILE_TRANSCRIPTION.md` and `docs/FULL_JSON_OUTPUT.md`.

**Run (typical):**

```powershell
python scripts/long_file_transcribe.py --audio test-data/out.m4a -v
python scripts/long_file_transcribe_full.py --audio test-data/out.m4a -v
```

---

## Summary: run all to render raw values

From repo root, for a single input (e.g. `test-data/out.m4a`):

```powershell
# Precomputed VAD + RMS (no ASR)
python scripts/precompute_vad_rms.py --audio test-data/out.m4a -v

# Fixed 1s, 5s, 8s raw transcription
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 1 -v
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 5 -v
python scripts/transcribe_raw_fixed_8s.py --audio test-data/out.m4a --chunk-sec 8 -v

# VAD-chunked raw transcription (max 5s per chunk)
python scripts/transcribe_raw_vad_chunks.py --audio test-data/out.m4a -v
```

Resulting files in `simulation_inputs/` (with basename from the audio file):

- `<basename>_vad_rms.npz` – VAD frames, segments, RMS (load with `np.load`).
- `<basename>_fixed1s.json`, `<basename>_fixed5s.json`, `<basename>_fixed8s.json` – raw ASR by fixed window.
- `<basename>_vad_chunks.json` – raw ASR by VAD-based chunks.

Set `MODEL_PATH` or `--model-dir` if the ONNX model is not at the default path. For m4a input, scripts convert to WAV automatically when ffmpeg is available.

---

## 5. Real ASR for a time window (with cache)

Agents can request **real ASR** for a specific time window when needed, while still using precomputed raw JSON for the rest. Results are **cached by (audio, start_sec, end_sec)** so the same window returns the stored result instead of running ASR again.

**Backend:** When `ASR_BACKEND_URL` is set (e.g. `localhost:9999`), recognition uses the shared backend. Otherwise local onnx-asr is used (slower if many windows).

**Cache:** Stored under `simulation_inputs/asr_cache/<audio_basename>/` as JSON files keyed by window, e.g. `0.80_8.00.json` for the window 0.8 s to 8.0 s. Cache key uses start and end rounded to 2 decimal places so requests for the same logical window hit the same entry.

**Python API:**

```python
from asr_window_client import recognize_window

result = recognize_window("test-data/out.wav", 0.8, 8.0)
# result: utterance_text, words, start_sec, end_time, segment_id, timestamp; from_cache: True if hit
```

**CLI:**

```powershell
python scripts/recognize_window_cli.py --audio test-data/out.wav --start 0.8 --end 8.0
```

Optional: `--cache-dir`, `--no-cache`, `--no-backend`. Output is one JSON object (asr_results-style) to stdout.

To run the backend (one model in VRAM for multiple consumers): `python scripts/asr_backend.py [--port 9999]` then set `ASR_BACKEND_URL=localhost:9999` when calling `recognize_window` or the CLI.
