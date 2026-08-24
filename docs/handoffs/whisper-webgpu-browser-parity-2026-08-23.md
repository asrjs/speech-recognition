# Whisper WebGPU Browser Parity Handover

Date: 2026-08-24
Branch: `feat/whisper-cleanup-beam-temperature`
Workspace: `N:\github\asrjs\speech-recognition`
Browser harness: `N:\github\asrjs\webgpu-agent-test`
Previous checkpoint: `ec24cef` Whisper beam/alignment implementation commit

## Objective

Keep the WebGPU path compatible with PyTorch Whisper large-v3 / WhisperX:
fast FP16 greedy, stable beam as the correctness oracle, language auto-detect,
and word timestamps without downloading GPU KV.

## Browser Matrix (custom 4-graph, `fp16io-fp16-webgpu`)

Model: `ysdede/whisper-large-v3-turbo-onnx-4graph`
Local encoder `fp16_iofp32_fp16out` + decoder `fp16`. Do not substitute
`onnx-community/whisper-large-v3-turbo`.

| Case | Audio | Mode | Total | RTFx | KV | Downloads | Language |
| ---- | ----- | ---- | ----: | ---: | -- | --------: | -------- |
| EN greedy GPU-KV | 29.9s JFK | greedy, `maxNewTokens=50` | `983.075ms` | **30.4192x** | `gpu-buffer` | `0` | `en` |
| EN stable beam | 29.9s JFK | `numBeams=2`, CPU-KV | `10465.995ms` | 2.8573x | `cpu` | `0` | `en` |
| TR auto greedy GPU-KV | 18.6s TDK | `language=auto` | `1230.63ms` | 15.1474x | `gpu-buffer` | `0` | **`tr`** |
| EN greedy timestamps | 10.0s JFK | `noTimestamps=0` | `608.13ms` | 16.4509x | `gpu-buffer` | `0` | `en` |

English greedy and stable beam produced **identical 50-token sequences** and
the same transcript prefix. Beam remains the correctness oracle; GPU-KV stays
greedy-only.

Turkish `language=auto` selected `<|tr|>` (`languageDetectionMs=199.61`) and
transcribed the TDK fixture in Turkish. Expected text is the TDK sentence about
epidemics; output matched with minor morphological/punctuation differences
(`duraklamaktaydı` vs `duraklamakta idi`).

Timestamped greedy emitted timestamp tokens and, after the interpolation
fallback, **17 words** on the 10s JFK clip. First word: `{ text: "In", startTime: 0, endTime: 0.556 }`.

## Code Change In This Slice

Splitgraph word timestamps used decoder-align when present and otherwise
returned an empty list, even when `<|0.00|>` / `<|t|>` tokens were already in
the sequence. `coalesceWhisperWordTimestamps()` now falls back to timestamp-token
interpolation, matching Whisper's own fallback. Attention/DTW alignment is
still preferred when it yields words.

Harness additions:

- Turkish fixture `/audio/tr-tdk-18s.wav`
- `?language=auto` and `?noTimestamps=0`
- posted `detectedLanguage` and `words`
- `scripts/run-webgpu-matrix.mjs` (kills Chrome per case so the profile is not left locked)

```powershell
cd N:\github\asrjs\webgpu-agent-test
npm run dev   # already required
node scripts/run-webgpu-matrix.mjs
node scripts/run-webgpu-matrix.mjs en-greedy-timestamps
```

## Next Implementation Order

1. Compare DTW/attention word timestamps against faster-whisper / WhisperX on
   English and Turkish fixtures. Interpolation is a fallback, not the quality
   target.
2. Keep `experimentalBatchedBeam` opt-in until more beam sizes, EOS, timestamp,
   and Turkish beam coverage exist.
3. Keep GPU-KV restricted to greedy argmax.
4. Run `tests/smoke/whisperx-runner.mjs` on real EN/TR files after word times
   match a reference.

`condition_on_previous_text`, hotwords, and numeral suppression remain
deprioritized.

## Beam-size and policy matrix revalidation - 2026-08-24

Using the same custom FP16 splitgraph artifact and the existing headless Chrome
harness:

| Case | Stable | Batched | Exact tokens | Calls | Result |
| ---- | ------: | ------: | ------------ | ----: | ------ |
| EN 30s, beam 5 | 245 | 49 | yes | 5x fewer | `check`, capped at 50 tokens |
| EN 10s, timestamped beam 2 | 40 | 20 | yes | 2x fewer | 17 words and timestamps match |
| TR 18s, auto beam 2 | 158 | 79 | yes | 2x fewer | both detect `tr` |

The English beam-5 totals were `24764.285ms` stable versus `18703.1ms`
batched. The timestamped totals were `4484.46ms` versus `3877.41ms`; the
Turkish totals were `16097.825ms` versus `13772.135ms`. All beam runs used CPU
KV and reported zero GPU tensor downloads. The browser status is `check` only
because the harness deliberately caps generation; no inference or page errors
were observed.

## Runner and enhanced-pipeline polish - 2026-08-24

- The WhisperX-compatible runner now carries raw decoder-init no-speech logits
  and selected beam traces into temperature fallback gates instead of relying
  on per-step vocab-logit snapshots. Its sampler scans the 51,865-token vocab
  without `Math.max(...values)`, and `--model-dir` is mapped correctly.
- `EnhancedWhisperExecutor` slices both typed-array compatibility inputs and
  normal planar/interleaved `AudioBufferLike` inputs by frame, then merges
  Whisper-native segments, words, warnings, and language without the generic
  post-processing shape mismatch.
- Regression coverage is in `tests/whisperx-runner-cli.test.ts`,
  `tests/whisper-enhanced-executor.test.ts`, and
  `tests/whisper-long-audio.test.ts`. Full validation is now 115 files / 720
  passing tests, with 4 skipped; real runner EN/TR fixtures remain pending.
