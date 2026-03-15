# Offline Long-Form JSON as Chunk Source

Using a **fully detailed offline transcription** (long-form script or any ASR that outputs rich JSON) as the single source of truth for testing avoids running real-time ASR during parameter search. Simulators and agents consume this JSON and point to the **required timestamps** to derive chunk streams. Real ASR can be used later or when needed.

## Why offline JSON first

- **Speed:** One long-form run produces token/word/utterance-level data; all merger and strategy experiments replay from that file. No per-trial ASR.
- **Parallel search:** Many workers can read the same JSON and evaluate different mergers, window parameters, or custom chunking methods in parallel.
- **Reproducibility:** Same input transcript for every trial; only merger logic and chunk boundaries vary.
- **Real ASR when needed:** For validation or live behavior, point tools at the ASR backend; the same simulators and metrics still apply.

## Required structure for simulators

Simulators need to **point to timestamps** in the JSON to build or slice chunks. The following are the contract for "fully detailed" long-form JSON:

### Minimum (chunk replay)

- **`words`** – list of objects with:
  - `start_time`, `end_time` (seconds)
  - `text` (string)
  - optional: `confidence`
- **`full_text`** – full transcript (reference).
- **Reference sentences** – either:
  - **`segments`** – list of `{ "text", "start_sec", "end_sec", ... }` (sentence/segment level), or
  - any structure your reference pipeline uses; simulators can derive sentences from segments or from `full_text` + SBD.

### Optional (richer replay and techniques)

- **`asr_results`** – pre-built chunk list. Each item: `utterance_text`, `words`, `start_sec`, `end_time`, `segment_id`, `timestamp`. If present, merger sim uses it directly; otherwise chunks are derived from `words` (and optionally `segments`).
- **Token-level** – if your long-form output has token-level timings, agents or custom code can slice by token for finer-grained chunking or confidence-based strategies.
- **Utterance-level** – utterance boundaries and labels can drive chunking (e.g. one chunk per utterance, or merge by pause).

### Timestamps simulators use

- **Word-level:** `words[].start_time`, `words[].end_time` – to assign words to segments, or to slice chunks by time windows (sliding/fixed window over the timeline).
- **Segment-level:** `segments[].start_sec`, `segments[].end_sec` – to treat each segment as one chunk, or to align reference sentences for metrics.
- **Chunk-level (when using pre-built asr_results):** `asr_results[].start_sec`, `asr_results[].end_time`, `asr_results[].words` – used as-is for merger replay.

So: **alternative long-form transcription scripts** only need to produce at least `words` (with timestamps) and `full_text`; simulators can derive chunks from that. Providing `segments` or `asr_results` gives more control and avoids re-slicing when your chunk definition matches. For the complete schema (all fields for simulations), see [FULL_JSON_OUTPUT.md](FULL_JSON_OUTPUT.md).

## Deriving chunks from detailed JSON

Two main ways to get a chunk stream without running real-time ASR:

1. **Pre-built chunks** – JSON already has `asr_results` or segment-level chunks; simulators use them as-is.
2. **Time-window slicing** – From `words` (and optional `full_text`), build chunks by sliding or fixed windows:
   - For each window `[t, t + window_sec]` (e.g. step `step_sec`), collect all words whose interval overlaps the window.
   - Emit one merger-style chunk per window (utterance_text = concatenation of words in window, words list, start_sec, end_time, segment_id, timestamp).

The repo provides a helper for (2): **`chunk_from_transcript.asr_results_from_words_by_window(data, step_sec, window_sec)`**. So any long-form JSON that has `words` with timestamps can be used to simulate different step/window settings for strategy_sweep or Optuna without re-running ASR.

## Agents and custom techniques

Agents are **not limited** to the merger or window strategies defined in this repo. They can:

- Define their own **chunking methods** (e.g. by utterance, by silence, by confidence, by token boundaries).
- Use the same offline JSON and **required timestamps** (`words[].start_time/end_time`, segments, etc.) to produce their own `asr_results`-style list and run mergers.
- Introduce new parameters (trigger periods, dynamic window length, word/token confidence thresholds) and plug them into the same metrics pipeline (WER, sentence recall, etc.).

So long as the JSON has word-level (and optionally token/utterance) timestamps, agents can create their own versions of techniques and parameters and obtain "chunks of transcriptions" by simulating real audio transcription from the offline file.

## Raw simulation input scripts and folder

Two scripts produce **raw ASR-only** JSON (no boundary trim, no SBD) for use as simulation input:

- **Fixed 8s chunks:** `scripts/transcribe_raw_fixed_8s.py` – non-overlapping 8-second windows; output `simulation_inputs/<basename>_fixed8s.json`.
- **VAD chunks:** `scripts/transcribe_raw_vad_chunks.py` – VAD speech segments split so no chunk exceeds a few seconds (default 5s); output `simulation_inputs/<basename>_vad_chunks.json`.

Outputs are written to the **`simulation_inputs/`** folder by default. See that folder's README and [FULL_JSON_OUTPUT.md](FULL_JSON_OUTPUT.md) for schema and raw vs cleaned fields.

**Precomputed VAD/RMS:** `scripts/precompute_vad_rms.py` writes a single `.npz` with VAD frame array, segment timings, and per-frame RMS so agents can load them without recalculating. See [SIMULATION_SCRIPTS.md](SIMULATION_SCRIPTS.md) for usage and loading.

## When to use the real ASR backend

- **Validation** – Confirm that optimized parameters still behave well when chunks come from live ASR instead of sliced JSON.
- **Live or on-demand runs** – When the goal is real-time behavior or new audio not in the offline JSON.
- **Backend:** Run `scripts/asr_backend.py`; set `ASR_BACKEND_URL` so scripts and agents use it. Offline JSON remains the default for fast parallel parameter search; the backend is optional and on demand.

## Summary

| Input | Use case |
|-------|----------|
| Offline long-form JSON (words + timestamps, optional segments/tokens/asr_results) | Fast replay, parallel parameter search, strategy_sweep, Optuna. Simulators point to required timestamps; agents can define custom chunking. |
| Real ASR backend | When you need live transcription or validation; agents can call the same backend. |
