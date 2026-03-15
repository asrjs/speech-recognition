# Python HF Whisper pipeline observations

This repo now includes a Python-side inspection script so we can compare `transformers.js` Whisper behavior with the original Hugging Face `transformers` pipeline on the same audio.

Script:

- [python-hf-whisper-inspect.py](N:\github\ysdede\transformers-v4-parakeet-demo\scripts\python-hf-whisper-inspect.py)

Recommended command on this machine:

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' `
  .\scripts\python-hf-whisper-inspect.py `
  --audio N:\JFK.wav `
  --model openai/whisper-base `
  --output .\docs\results\hf-whisper-base-jfk-python.json
```

Direct Node baseline commands used in this repo:

```powershell
node .\scripts\node-whisper-inspect.mjs `
  --audio N:\JFK.wav `
  --mode segments `
  --model Xenova/whisper-base `
  > .\docs\results\whisper-base-jfk-node-segments.json

node .\scripts\node-whisper-inspect.mjs `
  --audio N:\JFK.wav `
  --mode words `
  --model Xenova/whisper-base `
  > .\docs\results\whisper-base-jfk-node-words.json
```

Comparison helper:

```powershell
node .\scripts\compare-whisper-python-node.mjs `
  --python .\docs\results\hf-whisper-base-jfk-python.json `
  --python-case segments_chunked `
  --node .\docs\results\whisper-base-jfk-node-segments.json
```

What the script checks:

- plain ASR output: `{ text }`
- segment timestamps: `return_timestamps=True`
- word timestamps: `return_timestamps='word'`
- explicit windowing: `chunk_length_s=30`, `stride_length_s=5`

Why this matters for Parakeet TDT:

- Hugging Face pipeline compatibility is mainly about the public task contract, not NeMo hypothesis richness.
- Whisper segment `chunks` are timestamp-token spans produced by the decoder state machine, not sentence-aware segments.
- Long-form Whisper can still produce awkward chunk boundaries or overlap artifacts, especially when explicit chunking is used.
- That means a Parakeet TDT pipeline does not need to invent semantic sentence chunks to be compatible. Matching the HF shape and broad behavior is enough.

Interpretation guidance:

- `return_timestamps=True` should be treated as "timestamped decoder chunks", not "true sentence segmentation".
- `return_timestamps='word'` should be treated as a flattened list of word chunks.
- Rich NeMo outputs like token timings, confidences, debug traces, and raw decoder metadata still belong on direct model calls, not the default pipeline return.

The generated JSON result file is intended to be the source of truth for concrete behavior checks against `transformers.js`.

Observed on `N:\JFK.wav`:

| Source | Case | Observation |
|---|---|---|
| Python `transformers` 4.48.0 and 4.51.3 | `plain` | Fails on 146.326s audio unless timestamps are enabled or explicit chunking is used |
| Python `transformers` 4.48.0 and 4.51.3 | `segments_auto` | Returns 17 chunks, but the final timestamps are not monotonic and include `null` |
| Python `transformers` 4.48.0 and 4.51.3 | `words_auto` | Returns 206 word chunks; the final word timestamp reaches `153.02s`, beyond the audio duration |
| Python `transformers` 4.48.0 and 4.51.3 | `segments_chunked` | Returns only 3 coarse chunks for `30s/5s` windowing |
| Python `transformers` 4.48.0 and 4.51.3 | `words_chunked` | Returns 197 word chunks; the last word has a `null` end timestamp |
| Current local `transformers.js` Whisper | `segments` with `30s/5s` | Returns 23 smaller chunks on the same audio |
| Current local `transformers.js` Whisper | `words` with `30s/5s` | Returns 208 word chunks on the same audio |

Practical conclusion:

- Whisper itself is not a clean semantic reference for sentence-like chunks.
- Current Python HF Whisper and current `transformers.js` Whisper do not match exactly on long-form chunk granularity.
- For Parakeet TDT, target the HF public pipeline contract first: `{ text }` and `{ text, chunks }`.
- Keep NeMo-rich outputs on direct model calls. Full NeMo hypothesis parity is optional, not required for HF-style pipeline compatibility.
