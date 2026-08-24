# Whisper WebGPU Current Handover

Date: 2026-08-24
Branch: `feat/whisper-cleanup-beam-temperature`
Workspace: `N:\github\asrjs\speech-recognition`
Browser harness: `N:\github\asrjs\webgpu-agent-test`

## Current Checkpoint

The last committed checkpoints are:

- `ec24cef feat(whisper): finish beam quality and alignment pipeline`
- `378c8ea docs(whisper): record independent healthy WebGPU benchmark`
- `8b15be5 fix(whisper): source no-speech quality from decoder init`

The implementation continuation is committed in `ec24cef`. It includes
selected-beam quality traces, splitgraph word alignment, bounded beam
candidate selection, runner fallback hardening, and native VAD merge fixes.
Keep the browser and real-fixture limitations below explicit when continuing.

The implementation target is the custom splitgraph model repository:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
```

Do not substitute `onnx-community/whisper-large-v3-turbo` for the WebGPU
speed-path validation. The merged `onnx-community/*` models are secondary
compatibility paths.

## Active Browser Model

The remote preset names the encoder artifact:

```text
fp16_iofp32/encoder_model.onnx
```

The local browser harness uses the optimized fp16-output copy:

```text
encoder: fp16_iofp32_fp16out
decoder: fp16
preset: fp16io-fp16-webgpu
```

These are genuine FP16 artifacts. ONNX inspection found 487 FP16 encoder
initializers, 101 decoder-init initializers, and 88 decoder-step initializers.
Decoder logits/KV interfaces are FP16; mel input remains FP32.

## Healthy WebGPU Evidence

Independent headless Chrome validation after the workstation restart:

| Fixture | Total | RTFx | Encoder | Decoder steps | Step p50/p95 | KV | Downloads |
| ------- | ----: | ----: | ------: | ------------: | ------------: | -- | --------: |
| 29.9043s JFK | `1175.81ms` | `25.6993x` | `183.49ms` | `657.165ms / 49` | `13.395 / 15.430ms` | `gpu-buffer` | `0` |
| 10.0043s JFK | `731.205ms` | `13.856x` | `183.235ms` | `282.275ms / 18` | `15.220 / 18.305ms` | `gpu-buffer` | `0` |

Adapter: NVIDIA Blackwell. WebGPU features included `shader-f16`,
`timestamp-query`, and `subgroups`. The transcript was coherent and stable.
Expected ORT warnings concern CPU-assigned shape operations; no inference or
page errors affected these runs.

Raw result files:

- [30-second browser result](N:/github/asrjs/webgpu-agent-test/_results/fp16io-fp16-webgpu-2026-08-23T18-39-46-022Z.json)
- [10-second browser result](N:/github/asrjs/webgpu-agent-test/_results/fp16io-fp16-webgpu-2026-08-23T18-40-12-558Z.json)

The earlier `~8x` result was degraded GPU state. Historical `25-28x` results
are now corroborated by the independent `25.6993x` run. Always warm the model,
reuse the correct local variant, and measure longer audio before optimizing.

## Resume Validation - 2026-08-24

Beam expansion now computes entropy and only the bounded `beamSize + 1`
candidate set; it no longer allocates a full-vocabulary log-softmax array for
each active beam. Float32 rounding is retained for ranking compatibility.

The new deterministic contract test is:

```powershell
npm run benchmark:whisper-beam
```

It covers stable-vs-batched parity at beam sizes 2, 3, and 5. The real Chrome
matrix also passed the following cases on the custom FP16 splitgraph artifact:

| Case | Stable calls | Batched calls | Token/text parity | Extra parity |
| ---- | -----------: | ------------: | ----------------- | ------------ |
| EN 30s, beam 5 | 245 | 49 | exact | CPU KV, 0 downloads |
| EN 10s, timestamped beam 2 | 40 | 20 | exact | 17 word timestamps exact |
| TR 18s, auto beam 2 | 158 | 79 | exact | detected `tr`, CPU KV |

The capped browser runs report `check` because the harness intentionally uses a
token cap; this is not an inference error. Stable beam remains the correctness
oracle, batched beam remains opt-in, and GPU-KV remains greedy-only.

## Implemented In The Last Checkpoint

`8b15be5` fixes Whisper no-speech metric provenance:

- `onDecoderInitLogits` exposes a copied raw first-position decoder vector
  before timestamp/suppression processing.
- The callback is threaded through core greedy/beam decode, splitgraph,
  GPU-KV greedy, and merged-decoder paths.
- `no_speech_token_id` is parsed from generation config.
- The executor resolves the token from generation config, tokenizer, then
  compatibility fallback `50362`.
- The enhanced executor passes raw init logits and token ID through
  `QualityGateContext` into temperature fallback.
- Direct generic gate callers remain backward compatible.

Focused tests cover raw-vs-processed logits, dynamic token IDs, gate-context
forwarding, beam-init callback behavior, and generation-config parsing.

## Verification

Run from `N:\github\asrjs\speech-recognition`:

```powershell
npm test -- --run
npm run typecheck
npm run lint
npm run build
```

Last result: 115 test files passed, 1 skipped; 720 tests passed, 4 skipped;
typecheck passed; build passed; lint had 0 errors and 6 existing warnings.

## Next Implementation Order

1. ~~Implement selected-beam quality metrics without retaining full-vocabulary
   logits for every beam.~~ Done. See
   [selected-beam quality handover](N:/github/asrjs/speech-recognition/docs/handoffs/whisper-webgpu-selected-beam-quality-2026-08-23.md).
2. ~~Add fixture tests proving compression/logprob rejection and temperature
   recovery, including a beam case once selected metrics exist.~~ Done.
3. ~~Revalidate browser parity on the custom splitgraph model for English and a
   Turkish fixture. Keep stable CPU-KV beam as the correctness oracle.~~ Done.
   See [browser parity handover](N:/github/asrjs/speech-recognition/docs/handoffs/whisper-webgpu-browser-parity-2026-08-23.md).
   Warmed English greedy GPU-KV reached `30.4192x` on 29.9s JFK with zero GPU
   downloads; stable `numBeams=2` matched greedy tokens exactly. Turkish
   `language=auto` detected `tr`. Word timestamps now fall back to timestamp-token
   interpolation when decoder-align is empty (`17` words on 10s JFK).
4. ~~Wire splitgraph `decoder_align` + DTW word timestamps (lazy session load,
   CPU encoder states, text-token rows only, crop padded 30s frames to audio
   duration, per-timestamp-span DTW, spread duplicate jumps).~~ Browser 10s JFK
   greedy GPU-KV: `In` 0.00–0.16, last word ~9.98–10.00, `long` no longer
   zero-duration. Turbo emitted only a 0s/10s timestamp pair, so `history`
   still over-spans; compare against faster-whisper / WhisperX (wav2vec2) next.
5. Keep `experimentalBatchedBeam` opt-in until broader beam-size, EOS,
   timestamp, and Turkish beam coverage is complete.
6. Keep GPU-KV restricted to greedy argmax. Do not enable GPU-KV beam without
   correct KV cloning/reordering and output parity.

## Explicitly Deprioritized

`condition_on_previous_text`, hotwords, and numeral suppression remain out of
the next critical path. Prior testing did not show a useful gain, and context
carryover can amplify errors. Revisit only after a reproducible fixture shows
clear benefit.

Related documents:

- [Browser parity handover](N:/github/asrjs/speech-recognition/docs/handoffs/whisper-webgpu-browser-parity-2026-08-23.md)
- [Selected-beam quality handover](N:/github/asrjs/speech-recognition/docs/handoffs/whisper-webgpu-selected-beam-quality-2026-08-23.md)
- [Completion plan](N:/github/asrjs/speech-recognition/docs/plans/whisper-webgpu-completion-plan.md)
- [Validation handoff](N:/github/asrjs/speech-recognition/docs/handoffs/whisper-webgpu-completion-validation-handoff.md)
- [Optimization logbook](N:/github/asrjs/speech-recognition/docs/Whisper-Optimizations.md)
- [Agent task list](N:/github/asrjs/speech-recognition/docs/AGENT_TASKS.md)
