# Return granularity in Conformer TDT (transformers.js)

The Nemo Conformer TDT implementation now follows the shared `automatic-speech-recognition` pipeline contract more closely:

- **pipeline mode** returns `{ text }` by default and `{ text, chunks }` when timestamps are requested
- **direct model calls** keep the richer JS-first outputs such as `words`, `tokens`, grouped `confidence`, grouped `debug`, and `metrics`

---

## 1. Pipeline mode

Pipeline mode is for task-level compatibility:

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
```

Supported timestamp behaviors:

| Option | Output |
|--------|--------|
| `return_timestamps: false` | `{ text }` |
| `return_timestamps: true` | `{ text, chunks }` with sentence-like timestamped chunks |
| `return_timestamps: 'word'` | `{ text, chunks }` with word-level timestamped chunks |

Examples:

```js
await pipe(audio);
// { text }

await pipe(audio, { return_timestamps: true });
// { text, chunks: [{ text, timestamp: [start, end] }, ...] }

await pipe(audio, { return_timestamps: 'word' });
// { text, chunks: [{ text, timestamp: [start, end] }, ...] }
```

Notes:

- Pipeline mode does **not** expose `returnWords`, `returnTokens`, or the debug flags.
- For long audio, Nemo internally uses sentence-window restart logic, but the public pipeline shape remains the same.
- `return_timestamps: true` is now intended to return finalized sentence-like chunks, not Whisper-style arbitrary long-form chunk artifacts.
- `return_timestamps: 'word'` returns a flat list of timestamped words.

---

## 2. Direct model call

Direct calls expose the full Nemo decode options:

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
const inputs = await pipe.processor(audio);
const out = await pipe.model.transcribe(inputs, {
  tokenizer: pipe.tokenizer,
  returnTimestamps: true,
  returnWords: true,
  returnTokens: true,
  returnMetrics: true,
  timeOffset: 0,
});
```

Main options:

| Option | Effect |
|--------|--------|
| `returnTimestamps` | Enables utterance timestamps/confidence output |
| `returnWords` | Adds `words` |
| `returnTokens` | Adds `tokens` |
| `returnMetrics` | Adds timing metrics |
| `returnFrameConfidences` | Debug frame confidences |
| `returnFrameIndices` | Debug frame indices |
| `returnLogProbs` | Debug log probs |
| `returnTdtSteps` | Debug TDT step trace |
| `timeOffset` | Adds a global timestamp offset in seconds |

Typical direct outputs:

- `{ text }`
- `{ text, isFinal, utteranceTimestamp, confidence }`
- same plus `words`
- same plus `tokens`
- same plus `metrics` and `debug`

Direct output shape:

- `text`
- `isFinal`
- `utteranceTimestamp`
- `words: [{ text, startTime, endTime, confidence? }]`
- `tokens: [{ id, token, rawToken, isWordStart, startTime, endTime, confidence? }]`
- `confidence: { utterance?, wordAverage?, averageLogProb?, frames?, frameAverage? }`
- `metrics: { preprocessMs, encodeMs, decodeMs, tokenizeMs, totalMs, rtf, rtfX }`
- `debug: { frameIndices?, logProbs?, tdtSteps? }`

For transition, the direct model call still accepts legacy snake_case option names such as `return_timestamps`, but the returned object is camelCase.

---

## 3. Demo app behavior

The demo app in this repo now mirrors that split:

- **Direct Nemo call: on**
  - `Return timestamps` is a boolean
  - `Words`, `Tokens`, `Metrics`, and debug flags are available
- **Direct Nemo call: off**
  - the app uses pipeline mode
  - `Pipeline timestamps` has three options:
    - `off`
    - `sentences`
    - `words`

The demo now also makes the distinction between these two controls explicit:

- **Load mode**
  - decides how the transcriber is constructed
  - `pipeline (auto)` uses the regular `pipeline(...)` loader
  - `explicit (local export)` builds `processor + tokenizer + model + AutomaticSpeechRecognitionPipeline` manually
- **Inference mode**
  - decides whether the app calls the HF-style pipeline API or direct `model.transcribe()`

The transcribe view also generates copyable JS snippets from the active UI toggles:

- **Copy options**
  - copies the exact current inference options object
- **Copy JS**
  - copies a full load + infer example matching the selected load mode and inference mode

For pipeline mode, those snippets now reflect the new public behavior:

- `return_timestamps: true` -> sentence-like chunks with timestamps
- `return_timestamps: 'word'` -> word chunks with timestamps

For long audio in pipeline mode, the app now documents the actual merge strategy:

- transcribe a window
- finalize stable sentences and keep their timestamps
- drop the last immature sentence
- restart the next window from the end time of the latest finalized sentence

Relevant code:

- [App.jsx](N:\github\ysdede\transformers-v4-parakeet-demo\src\App.jsx)

---

## 4. Summary

Use **pipeline mode** when you want compatibility with the standard ASR task API.  
Use **direct `model.transcribe()`** when you want the full Nemo-native outputs.

The important compatibility target is the **public contract**, not Whisper's exact chunk boundaries. For Parakeet TDT in this repo, pipeline mode now returns a cleaner sentence-oriented chunk list while remaining compatible with the standard `{ text }` / `{ text, chunks }` ASR interface.
