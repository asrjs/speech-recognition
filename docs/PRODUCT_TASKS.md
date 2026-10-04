## C49 — Moonshine streaming: the toolchain silently randomizes its encoder

Status: BLOCKED. The export that looked clean is invalid.

`moonshine-ai/moonshine-streaming-tiny` stores its encoder as
`model.encoder.embedder.conv1.weight`, where the non-streaming checkpoint stores
`model.encoder.conv1.weight` directly. The installed transformers maps none of
the streaming tensors, loads the model anyway, and randomly initializes the
entire encoder. It reports this only as a warning.

The load listed `model.encoder.conv1.weight`, `conv2`, `conv3`, `groupnorm`,
`layer_norm` and every layer's input and post-attention layernorms as newly
initialized. Greedy decode then emitted `"I, I, I, ..."` — plausible output from
random weights.

That makes the graphs exported earlier in this session invalid: they encode the
randomly initialized encoder, not the checkpoint. The receipt now says
`EXPORT_INVALID_RANDOMLY_INITIALIZED_ENCODER`, and the artifacts are retained
only as a record of the attempt.

This is the same failure class that has already cost this session three times: a
load or a decode that succeeds, produces the right shapes, and only warns. The
lesson generalizes — a warning during model load is not a diagnostic, it is
evidence that something was not loaded.

Prerequisites, in order: a transformers version expecting
`model.encoder.embedder.*` or a state-dict remap into the expected layout; then
re-export; then a reference transcript; then a gate. Until the first is
available nothing about this checkpoint can be gated.

Receipt: `moonshine/streaming-export-2026-10-04.json`.

## C48 — Correlation matching is only valid for 2-D weight matrices

Status: a methodological rule, extracted from the redux substitution work.

Mapping one model's anonymous ONNX constants to another's modules by correlation
works for weight matrices and is unsound for biases. A bias vector has no
correlation structure, so a match above any threshold is chance rather than
evidence. In the redux map the weakest substituted conv tensors were all
`conv.norm.bias` at correlation 0.567 to 0.585 with margin near 0.5, while the
weight matrices sat between 0.6 and 0.86 with margins up to 2.5.

The guard is now in `substitute_weights.py`: anonymous 1-D initializers keep
their teacher values instead of taking a correlation-derived assignment.

This is recorded as **hygiene, not as a diagnosis**. The substitution ladder had
already shown that cosine collapses at the gentlest weight class, and removing
the guessed bias matches did not restore adjacency, it moved cosine from 0.0339
to 0.0215. Biases of magnitude 0.1 to 1.0 cannot produce a 3.4x output-std
inflation, so the large effect comes from the 2-D weight matrices.

Receipt: `model-candidates/redux-substitution-2026-10-04.json`.

## C47 — Pair verified on unseen audio; both seed artifacts retired

Status: JFK oracle match plus a clean second fixture. Streaming re-scoped.

The BOS self-attention block was cargo cult. With cross-attention alive, an
**empty** self-attention cache reproduces the JFK oracle exactly, 24 tokens in
25 steps, which is how PyTorch's `DynamicCache` starts. Every earlier seed
variant "failed" only because it was compensating for dead cross-attention, so
both the 32 MB seed cache and the 13,824-byte BOS block are gone. The runtime
now starts from an empty cache, which also removes the dynamic-shape
specialization question: the graph accepts length 0.

Multi-fixture passes with quality. On `librivox.org.wav`, 1073 frames, audio
never seen during export or tuning:

> Preface of A Year with the Birds, this is a LibriVox recording. All LibriVox
> recordings are in the public domain.

Long-form needs a repetition guard, and the model window is not the limit. On
`librivox-blankgaps-synthetic.wav`, 2329 frames and about 56 s, plain greedy
decoding opens correctly and then loops: _"of a year with the birds of a year
with the birds of a year with..."_, 7 distinct tokens out of 120. A soft
repetition penalty of 1.1 changes nothing. Banning the last 40 emitted tokens
breaks the loop outright: 46 steps, 45 distinct tokens of 45, and

> of a year with the birds. ... This is a LibriVox recording. All LibriVox
> recordings are in the public domain. For more information ...

So the loop is a selection artifact, not a model window. An earlier reading of
this fixture blamed greedy decoding and then blamed the window; both were wrong,
and the record is corrected here because the soft penalty alone does not
exonerate greedy selection. The guard improves selection but does not recover
punctuation or casing, so a sampling or beam policy is still unmeasured.

Streaming is re-scoped, not deferred. It needs the `moonshine-streaming`
checkpoint, which ships safetensors only, so it is a separate port with its own
export and chunked decode loop rather than a measurement of this pair.

## C46 — Self-exported Moonshine pair matches the oracle

Status: `PASS_ORACLE_MATCH` on the JFK fixture with our own fp32 graphs.

C45 named the fix and this closes it. Passing real encoder states into the
decoder wrapper makes the graph compute cross-attention internally. The
sensitivity probe that proved the fix:

| probe                                           | before |     after |
| ----------------------------------------------- | -----: | --------: |
| max abs logit difference vs zeroed `cross_past` |    0.0 | **27.01** |
| first-step argmax                               |  19314 |  **1126** |

1126 is the oracle's first token. JFK now decodes to
`And so my fellow Americans ask not what your country can do for you, ask what
you can do for your country.` in 24 tokens, 25 steps and 446 ms, matching both
token ids and text.

The port also gets smaller and more general. **The 32 MB seed cache is gone**:
cross-attention memory is computed in-graph, and self-attention starts from one
exported BOS-step block of 13,824 bytes. Utterance length is no longer pinned to
an exported seed.

Long-form now **executes** where it previously failed with a tensor size error:
`librivox.org.16k.wav` at 1558 frames runs in 580 ms. Its transcript is junk
(two byte-fallback tokens), and the emitted ids indicate input outside the
model's training window rather than a graph fault, so long-form quality is
explicitly **not** claimed. Streaming and partial-result behaviour remain
unmeasured.

Receipts: `moonshine/pair-oracle-match-2026-10-04.json`,
`moonshine/encoder-divergence-2026-10-04.json`.

## C45 — Exported decoder ignores cross-attention; root cause measured

Status: root cause confirmed by measurement. The exported decoder graph is
unsound and is not wired into the runtime.

The export wrapper passed `encoder_hidden_states=None`. Moonshine's attention
gates the entire cross-attention block on that argument, so every one of the six
decoder layers traced the self-attention branch and `cross_past` became a dead
input. Probing the graph settles it in one run:

| cross_past | max abs logit difference | argmax |
| ---------- | -----------------------: | -----: |
| all zeros  |                  **0.0** |  19314 |
| halved     |                  **0.0** |  19314 |

The logits are bit-identical. The decode was running a decoder with no audio
memory at all, which is why the output was locally plausible English word salad.

Five hypotheses were tried and falsified before this was found, and every one
produced the **same first token**: keeping cross-attention constant, seeding
self-attention with zeros, seeding it empty, seeding it from the shipped cache,
and recomputing it from a true empty PyTorch cache. An identical first token
across all of them is the signature of an ignored input rather than a wrong
value, and testing input sensitivity is cheaper than another seed variant.

The fix is named and small: pass encoder states into the wrapper, which makes
the graph compute cross-attention key and value internally. That also retires
the separate cross-attention graph and the self-attention seed, since the
decoder can then start from an empty self-attention cache. It has not been
implemented or measured here.

The runtime stays on the previous configuration, which still produces the JFK
transcript exactly. The export tool keeps both graphs because the encoder is
verified at 4.447e-04 and remains the fix for the original drift; the decoder
graph is recorded as unsound rather than deleted, since deleting it would lose
the encoder path that ships with the same receipt.

Receipt: `moonshine/encoder-divergence-2026-10-04.json`.

## C44 — Encoder drift fixed by re-export; decoder now exports too

Status: the encoder is faithful and verified. The self-exported pair does not
yet decode to the oracle, so the runtime stays on the previous configuration.

The `onnxscript` prerequisite resolved the export blocker from C43. With
`onnxscript` 0.7.2 installed in the reference venv, `torch.onnx.export(dynamo=True)`
traces the encoder that the legacy tracer rejected:

| encoder input                    | shipped graph |    our export |
| -------------------------------- | ------------: | ------------: |
| max abs error                    |         1.609 | **4.447e-04** |
| mean abs error                   |         0.032 | **4.229e-06** |
| cosine                           |             — |  0.9999998808 |
| `allclose(atol=1e-4, rtol=1e-3)` |         false |      **true** |

That is roughly a 3600x improvement and it settles the C42 question: the shipped
browser graphs carry weights from a different checkpoint revision, not a frontend
difference. Two advisor hypotheses were tested and rejected on measurement. The
weights are fp32, not fp16, and fp16-rounding the checkpoint does not reproduce
them, so this is not a precision artefact. The drift is structured, with a 0.50
sign balance and a -0.46 correlation against the weights themselves.

The decoder exports too, with past state as two stacked tensors
(`self_past`, `cross_past`, 12 slots, 8 heads, 36 head dimensions) rather than
24 named ones. The dynamo exporter traces a `*args` wrapper as a single argument,
and per-tensor names contain dots that are not valid Python identifiers. Logits
need `proj_out`, which lives at model level, not on the decoder.

**The pair is not yet usable.** Its decode does not match the oracle, so the
runtime stays on the previous configuration, which still produces the JFK
transcript exactly. Two leads are recorded, both unverified: cross-attention
memory is a fixed projection, so feeding `cross_present` back on every step may
be wrong, and the tokenizer vocab lookup returns empty text.

One correction to an earlier receipt. The cross-attention parity result reported
`PASS` at 6.68e-06, but the comparison reshaped both sides identically and so
could not detect a layout error. It proves the graph reproduces `k_proj` and
`v_proj`; it does not prove the cache layout. The layout bug is real and was
found later: reinterpreting `[batch, frames, heads * headDim]` as
`[batch, heads, frames, headDim]` preserves the element count without permuting
the data. The symptom is a repeating token loop rather than an error.

Receipts: `moonshine/encoder-divergence-2026-10-04.json`,
`moonshine/cross-kv-parity-v2-2026-10-04.json`.

## C43 — Moonshine encoder root cause found; re-export is a bounded blocker

Status: root cause identified and reproducible. The fix needs one prerequisite
this session does not have.

**The shipped encoder graph is not this checkpoint's weights.** Comparing graph
initializers against `MoonshineForConditionalGeneration` by name and shape:
**30 of 30 comparable tensors mismatch, 0 match exactly.** The values are close
rather than random, which rules out a corrupt export:

| tensor         | max abs diff | torch std | onnx std |
| -------------- | -----------: | --------: | -------: |
| `conv1.weight` |       0.0461 |    0.1927 |   0.1905 |
| `conv2.weight` |       0.0303 |    0.1059 |   0.1052 |
| `conv3.weight` |       0.0390 |    0.1296 |   0.1286 |

Same architecture, same tensor names, a different checkpoint revision. That
explains the encoder drift measured in C42 without needing any frontend theory,
and it invalidates the normalization hypothesis for good.

**The encoder cannot be re-exported with the available toolchain.**
`torch.onnx.export(..., dynamo=False)` fails with `AssertionError: All tensors
must have the same rank`. The third conv yields a rank-4 tensor and the legacy
tracer rejects the mixed-rank stack. Flattening the rotary input does not help,
because the rank-4 tensor originates upstream in the conv stack; four variants
were tried and all reproduce the failure. The dynamo exporter that this model
does trace with needs `onnxscript`, which is not installed.

So the missing prerequisite is exactly one of:

- `onnxscript` installed in the reference venv, which unlocks the dynamo
  exporter, or
- a graph rewrite that removes the rank-4 conv output.

Until one lands, long-form windowing, streaming and per-fixture seeds stay
blocked, and the port keeps the checkpoint seed cache it already passes with.
This is a bounded blocker rather than an open question: the failure is
reproducible and the fix is named.

Receipt: `moonshine/encoder-divergence-2026-10-04.json`.

## C42 — Moonshine cross-attention graph exported; encoder drift found

Status: the new graph is numerically exact. The port stays on the seed-cache
path because the encoder graph is not faithful.

**A 3.98 MB graph replaces a 32 MB seed cache, and it is exact.**
`export_moonshine_cross_kv.py` exports the decoder's cross-attention key and
value projections as a standalone graph with a dynamic frame axis. Verified
against PyTorch hooks on all twelve tensors: **max abs error 6.68e-06**
(`cross-kv-parity-v2-2026-10-04.json`). Two layout facts came out of the
verification: the projection emits `[batch, frames, heads * headDim]` while the
decoder cache wants `[batch, heads, frames, headDim]`, and
`onnxruntime-node` returns run results **keyed by output name, not positionally**.
That second one cost the most time: reading results positionally yields
`undefined` silently, which fed a decoder with zeroed history and produced 256
tokens of noise instead of an error.

**The encoder graph drifts from PyTorch, and that blocks the switch.**
`compare_encoder_parity.py` measures `last_hidden_state` on the JFK fixture:

| encoder input           | max abs error | mean abs error |
| ----------------------- | ------------: | -------------: |
| raw PCM                 |         1.609 |          0.032 |
| feature-extractor input |         4.878 |          0.067 |

Gate is 1e-3. Input normalization is ruled out: both paths drift by a similar
amount, and the graph takes only `input_values`, so no attention mask is
involved. The drift is inside the graph and is not yet localised to a layer.

The consequence is structural, and it is the reason the port still seeds
cross-attention from the checkpoint: **the encoder output is computed but never
reaches the decode.** Cross-attention key and value are a fixed projection, so
the transcript comes out right regardless of what the encoder produces. That is
why the earlier JFK and browser passes were clean while this was invisible, and
why utterance length is pinned to the exported seed.

Self-attention cannot substitute for the seed. Zeroed history yields 256 tokens
with no EOS, and the blank pass's own `present` output is length 2 while the feed
must be length 1, because the graph appends the token being decoded.

So the runtime stays on the verified seed path. Cross-attention projection,
long-form windowing and streaming remain blocked on fixing the encoder export,
and the next step is per-layer comparison inside the graph.

Receipts: `moonshine/cross-kv-parity-v2-2026-10-04.json`,
`moonshine/encoder-divergence-2026-10-04.json`.
Full suite: 1,477 passed / 33 skipped.

## C41 — Moonshine registered and measured in the browser

Status: PASS with transcript parity on both backends. WASM is the better placement.

`UsefulSensors/moonshine-tiny` is now a built-in preset (`moonshine`) with a
local-folder adapter, a descriptor, and a browser case. Measured on the JFK
fixture against the recorded oracle, with the same PCM digest on both sides:

| backend                   | result                |      load | transcribe |      RTFx |
| ------------------------- | --------------------- | --------: | ---------: | --------: |
| WebGPU (NVIDIA Blackwell) | `baselineMatch: true` | 2932.0 ms |  4176.7 ms |      2.63 |
| WASM                      | `baselineMatch: true` | 2000.7 ms |   739.2 ms | **14.88** |

**WASM wins by about 5.6x on this fixture.** The encoder is a small conv
subsampler plus Conformer and gains nothing from the GPU, while the
autoregressive decoder is launch-bound. This is a placement measurement for one
model and one fixture, not a general rule.

Registered wiring: `src/models/moonshine/session.ts` and its tokenizer,
`src/presets/moonshine-factory.ts`, `src/presets/moonshine-local-adapter.ts`, the
descriptor, and the browser `MODELS` entry. The family reuses the shared CTC
collapse only for reporting; decoding is autoregressive, so no CTC collapse is
applied to the emitted tokens.

Named limits: local artifact source only, because no validated Hub export exists;
streaming and partial-result behaviour are unmeasured; long-form windowing is
unmeasured; one fixture per backend.

Receipt: `tools/data/results/moonshine/moonshine-browser-parity-2026-10-04.json`.
Full suite: 1,477 passed / 33 skipped.

## C40 — Moonshine tiny ported; omnilingual divergence resolved

Status: PASS for the Node gate. Browser registration for Moonshine is next.

**Omnilingual browser divergence resolved.** The earlier record compared
different fixtures: the Node passes ran on the four shipped language clips, the
browser failure ran on `jfk-short`. Two probes settled it. First, a PCM digest
showed the browser audio path was also wrong: the default `browser-target-rate`
strategy round-trips through AudioContext and produced `584bb2fe...` while Node
produced `ebd52851...` for the same file. The probe now reports `audio.pcmSha256`
and decodes with `native-rate`, so the digests match. Second, with identical PCM
the Node decode of the same int8 graph still returned garbage on `jfk-short`, so
the graph was never the cause. On its own fixture
`tools/data/fixtures/audio/omnilingual/en.wav` the WebGPU run reports
`baselineMatch: true`. That model does not transcribe `jfk-short`; that fixture is
outside its demonstrated domain and must not be read as a quality signal.

**Moonshine tiny ported.** `UsefulSensors/moonshine-tiny` runs through
`PASS_ORACLE_MATCH` on both token ids and text, 457 encoder frames, 25 decode
steps, 360 ms on CPU. Contract verified from the checkpoint: `feature_size` 1 and
`do_normalize` false, so raw 16 kHz samples with no mel frontend. The PyTorch
encoder does not trace under the available exporter, so
`tools/moonshine-onnx-export/export_moonshine_onnx.py` validates the browser fp32
pair and records the decoder contract instead of emitting a new graph.

Two decoder facts were measured, not assumed. The graph returns self-attention
history only, 13 outputs for 6 layers, so cross-attention key and value are
computed once from the encoder states and stay constant for the utterance. The
`EncoderDecoderCache` layout also differs from the graph's input names, so the seed
is mapped by name.

Receipts: `tools/data/results/moonshine/`, `tools/model-debugging/ports/moonshine-tiny.port.json`,
and `tools/data/results/omnilingual-ctc/browser-webgpu-divergence-resolved-2026-10-04.json`.
Full suite: 1,477 passed / 33 skipped.

## C39 — Browser harness measures transcript parity and timings for all three candidates

Status: TWO of three candidates pass real browser transcript parity on WebGPU;
the third is a measured failure, not an unverified gap.

New harness work in the established `webgpu-agent-test` loop: `ctc-asr.html`,
`src/ctc-asr-webgpu.js` and `scripts/run-ctc-asr-webgpu.mjs`, driven by URL
parameters (`model`, `backend`, `audio`, `oracle`, `repeat`, `warmup`, `clipSec`),
posting to `/__test_result__` and read back through `/__test_results__` with a
preset-keyed diff. The page imports the library through relative source paths so
the gate exercises live source, and serves local graphs through `/@fs/` under
the existing `server.fs.allow` entry. Building it surfaced three real harness
defects: the POST endpoint is singular, `decodeAudioSourceToMonoPcm` takes a Blob
rather than an ArrayBuffer, and the capture plugin expects `baselineMatch` and
`tokens`.

Measured on Chrome headless WebGPU with an NVIDIA Blackwell adapter:

| model         | graph                 | result                       |      load | transcribe |  RTFx |
| ------------- | --------------------- | ---------------------------- | --------: | ---------: | ----: |
| `wav2vec2-tr` | our own fp32 export   | **pass**, baselineMatch true | 5839.2 ms |   135.5 ms | 81.17 |
| `ctaguchi`    | our own fp32 export   | **pass**, baselineMatch true | 3330.7 ms |   581.2 ms | 18.93 |
| `omnilingual` | released dynamic-int8 | **fail**, transcript garbage | 4384.7 ms | 11574.4 ms |  0.95 |

For the two passing models the browser transcript is byte-identical to the Node
baseline. The Turkish model's output is its correct Turkish transliteration of
English audio. The ctaguchi run first reported fail only because the oracle
omitted that model's own "askk" quirk; the browser output matched Node throughout.

The one failure is informative: both graphs that survive the browser are **our
own fp32 exports**, while the third-party dynamic-int8 graph, carrying 106
`DynamicQuantizeLinear` and 146 `MatMulInteger`, degenerates in the browser while
decoding correctly in Node. That is consistent with the execution-provider kernel
hypothesis, and it argues that fp32 self-export, not a repacked int8 graph, is
what carries a candidate to WebGPU.

Named limits: token ids are reported empty because `returnTokenIds` does not
populate them for these families, so browser receipts carry no per-frame token
evidence; the omnilingual browser run used the JFK fixture rather than its own
four-language set; and each configuration was run once, with no repeat-stability
or long-form browser evidence.

Receipt: `tools/data/results/candidates-browser-harness-2026-10-04.json`, with
the int8 divergence detailed in
`tools/data/results/omnilingual-ctc/browser-webgpu-divergence-2026-10-04.json`.

## C38 — All three candidate models now load and warm up in the browser

Status: PASS (warm-up) for ctaguchi 110M, omnilingual CTC 300M and Turkish
wav2vec2 XLS-R, on a real Chrome NVIDIA Blackwell adapter, offline through the
pinned ORT sidecars. Transcript parity in the browser is still unverified.

Following the established workflow, each candidate now has a descriptor, preset
factory, local-folder adapter and a companion-UI browser gate:

| model                                       | graph                 | browser gate |
| ------------------------------------------- | --------------------- | ------------ |
| `ctaguchi/parakeet-hf-ctc-110m-en`          | our own fp32 export   | PASS         |
| `OpenVoiceOS/omnilingual-asr-ctc-300m-onnx` | released dynamic-int8 | PASS         |
| `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`   | our own fp32 export   | PASS         |

Three real defects surfaced while wiring the browser lane, each caught by
running rather than by types:

1. **Preset key mismatch.** The Turkish descriptor used `preset: 'wav2vec2'`,
   which has no local-folder adapter, so the browser reported "does not yet have
   a browser local-folder loader". It now has its own `wav2vec2-tr` preset key,
   which resolves the adapter and the preset factory while still using the
   wav2vec2 executor and tokenizer.
2. **Wrong artifact field names.** The wav2vec2 executor reads
   `artifacts.modelUrl` / `modelDataUrl` / `modelDataFilename`; the adapter
   supplied `encoderUrl` / `encoderDataUrl` / `encoderFilename`, so ORT received
   an undefined path and rejected it with "must be 'path' or 'buffer'".
3. **ORT Web rejects blob URLs.** In-memory local models arrive as `blob:`
   URLs, which ORT Web will not load. `createOrtSession` now fetches blob-backed
   graphs to bytes first, and the shared `OrtModuleLike` signature was widened to
   `string | Uint8Array` so the in-memory path typechecks.

The browser gates are **warm-up only**: the demo verifies its own bundled
sample, so these receipts prove the family loads, creates sessions and runs a
transcription in Chrome, not that browser output matches an oracle. Node-side
oracle gates are the ones that compare tokens and text.

Receipts: `tools/data/results/omnilingual-ctc/` and
`tools/data/results/wav2vec2-tr/`. Full suite: 1,477 passed / 33 skipped.

## C37 — Turkish wav2vec2 XLS-R: first self-exported candidate with Turkish coverage

Status: PASS through the harness on two clips; browser gate and model-accuracy
parity not covered.

`mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` (cc-by-4.0, language `tr`, 530k
downloads) is the candidate the filter was looking for: non-NVIDIA, **Turkish**,
and transformers-loadable, so we can export ONNX ourselves instead of adopting a
third-party repack. That is exactly the Parakeet v2/v3 and Whisper discipline.

Export: new `tools/wav2vec2-onnx-export/export_wav2vec2_ctc_onnx.py` produced a
1,262,344,930-byte opset-17 graph with a dynamic sample axis from the official
PyTorch checkpoint, input `input_values [batch, samples]` with the convolutional
frontend inside. Contract matches the library's existing wav2vec2 and omnilingual
CTC paths exactly, so the port is wiring plus a 38-token character vocabulary
where `[PAD]` is the blank and `|` is the word separator.

Both harness gates are `PASS_ORACLE_MATCH` on token ids and text, with the graph,
vocabulary, fixtures and baseline digest-bound. Named limits: the baseline is
self-consistency of our own export plus the decode path, since no Turkish
reference transcript exists for the clips; the clips are English speech
transcribed into Turkish orthography, which shows the graph performs real work
but is not a Turkish accuracy claim; descriptor registration compiles and
suite-tests, but the **browser gate has not been run** for this family, and no
local-folder adapter or preset wiring is registered yet.

Receipts: `tools/data/results/wav2vec2-tr/`. Manifest:
`tools/model-debugging/ports/mpoyraz-wav2vec2-turkish.port.json`.

## C36 — Omnilingual CTC 300M: second model ported, export path assessed

Status: PORT VERIFIED through the harness; self-export path BLOCKED upstream.

`OpenVoiceOS/omnilingual-asr-ctc-300m-onnx` (base `facebook/omniASR-CTC-300M`,
apache-2.0 and ungated, non-NVIDIA) is ported and passes the shared harness on
four languages with exact token-id and text match against a recorded baseline:

| clip | result                                                                          |
| ---- | ------------------------------------------------------------------------------- |
| `de` | "alles hat ein ende nur die wurst hat zwei"                                     |
| `en` | "ask not what your country can do for you ask what you can do for your country" |
| `es` | "no preguntes que puede hacer tu pais por ti..."                                |
| `fr` | "ne vous demandez pas ce que votre pay peut faire..."                           |

The port is small because the contract is: raw 16 kHz waveform in
`x [batch, num_samples]`, convolutional frontend inside the graph, blank id 0,
10288-token vocabulary of real spaces, no mel parity work. It reuses
`normalizeWav2Vec2Waveform` and the shared `CtcDecoder`. All four gates are
`PASS_ORACLE_MATCH` with the graph, tokens, fixtures and baseline digest-bound.

The baseline is **self-consistency** against a Python decode of the same released
graph, not upstream-oracle parity: the model card spot check uses Portuguese and
Swahili clips that are not obtainable, and the English clip is JFK but the model
omits its opening clause.

**Surgical ONNX export is blocked for this candidate, on measured evidence.**
The released graph is dynamic int8: 106 `DynamicQuantizeLinear` plus 146
`MatMulInteger` over 325,983,100 stored parameters (325.6M int8 plus 0.35M fp32,
measured, not inferred). Activation scales are computed from live inputs, so the
quantization cannot be stripped statically without a re-export, and stripping it
would destroy the int8 behavior just validated. Upstream offers only
`omniASR-CTC-300M.pt` (1.3 GB) plus a sentencepiece model, with no
`config.json` and no transformers class; its README points at
`pip install omnilingual-asr`, a custom Meta package. An fp32 self-export would
therefore require installing that package, which is an authority question, so
the released graph is this candidate's ceiling.

Named limits for C36: four-clip `PASS_ORACLE_MATCH` through the Node harness
against a Python self-consistency baseline of the same int8 graph; **browser
gate not yet run for this family**; preset and local-folder adapter wiring is not
registered yet (descriptor and family factory are, both compile and suite-test);
fp32 self-export blocked on the `omnilingual-asr` install plus the 1.3 GB
checkpoint; WebGPU unverified because the graph carries dynamic-quant ops that
the CPU path executed. Session lifetime is per-session with a disposal test, after
review caught a shared-cache bug where a second session transcribing through a
disposed model.

This is the same rule that produced the Parakeet v2/v3 and Whisper work: prefer
our own export when the candidate exposes exportable source weights. It applies
to candidates with NeMo `.nemo` or transformers checkpoints, as the ctaguchi
110M port demonstrates, where we exported our own ONNX rather than adopting a
repack. The Omnilingual line is the counter-case: third-party ONNX only.

## C32 — Parakeet CTC 110M candidate: export verified, runtime port pending

Status: PASS with named limits — superseded mid-port; the library side now
exists (frontend, decoder family, preset, local adapter) and passes the port
harness. Limits that remain: 137 of 88,080 feature coordinates outside
`atol=1e-5, rtol=1e-4`; browser evidence is load + warm-up, not oracle parity;
WebGPU inference for this family unverified; local-folder only, no Hub source.

`ctaguchi/parakeet-hf-ctc-110m-en` is the only verified non-NVIDIA-originated
Parakeet-line student: `ParakeetForCTC` / `parakeet_ctc`, 109,287,937
parameters, cc-by-4.0, safetensors plus a sentencepiece tokenizer. It loads under
the installed transformers 4.57.6 despite being saved by 5.4.0.

Export: new `tools/parakeet-ctc-onnx-export/export_parakeet_ctc_onnx.py`
produces a 437,740,665-byte opset-17 graph with dynamic frame axes, using the
legacy TorchScript exporter because torch 2.13 defaults to the dynamo exporter
that requires `onnxscript`, and installing that was not authorized.

Verified on real audio: JFK mel features `(1, 1101, 80)` produce logits
`(1, 138, 1025)` in ONNX Runtime, the 138 frames matching the encoder's 8x
subsampling and the 1025 vocabulary. That is shape evidence, not a transcript.

The feature contract differs from the NeMo 128-bin processor this library shares
elsewhere: **80 mel bins with 0.97 preemphasis** (n_fft 512, hop 160, win 400).
So the remaining work is real: a ParakeetForCTC feature extractor, a CTC decoding
family for this vocabulary/tokenizer, then a preset, local adapter and port
manifest so it gets the same digest-bound oracle checks as Orukeet.

**Feature contract now resolved.** Feeding the official `ParakeetProcessor.__call__`
features into the locally exported ONNX graph and greedily decoding with the
model's tokenizer reproduces the oracle transcript exactly. Feeding the
no-preemphasis path to the same graph decodes to `.` alone, which proves the
earlier 7.40 measurement used a contract the model does not consume. The correct
reference is therefore the preemphasis path (1101x80), and against it the current
frontend measures max abs `18.02` / mean `9.25` over 88,080 coordinates.

**BROWSER: LOAD AND WARM-UP, NOT AN ORACLE COMPARISON.** The companion UI loads
`ctaguchi/parakeet-hf-ctc-110m-en` from a local folder in Chrome on a real NVIDIA
Blackwell adapter (`PASS_COMPANION_LOCAL_WARMUP`, 5.1 s, zero page errors, offline
through the pinned ORT sidecars). The receipt's console shows our adapter
selecting `model.onnx` and `tokenizer.json`, sessions created, and the app's own
verification transcription reporting success, so the family graph did run. It
does **not** compare a browser transcript with the oracle: the demo's warm-up
uses its own bundled sample, so browser oracle parity is still unverified.
Reaching load at all needed a preset factory plus a `SpeechModelFactory` whose
`createSession` owns the model lifecycle, matching the other CTC families.

Post-refactor functional evidence: `PASS_FAMILY_ENTRY_ORACLE_MATCH` through the
real `loadParakeetCtcFromOptions` entry (138 frames, 34 tokens, oracle text),
because the `createSession` rewrite had to be proven to still infer rather than
merely compile.

**REGISTERED AND VERIFIED THROUGH THE SHARED HARNESS.** `ctaguchi/parakeet-hf-ctc-110m-en`
is now a built-in descriptor (preset `parakeet-ctc`, local-folder only, no Hub
source because the repository ships no ONNX) with a local-folder adapter, and it
runs through `tools/model-debugging/scripts/port_manifest.py`:
`PASS_ORACLE_MATCH` on both token ids and text, with the graph, tokenizer,
original safetensors and oracle all bound by declared SHA-256 before execution.
Runner: `tools/model-debugging/scripts/transcribe_parakeet_ctc_local.mjs`.
Manifest: `tools/model-debugging/ports/ctaguchi-parakeet-ctc-110m-en.port.json`.

**THE PORTED FAMILY NOW TRANSCRIBES.** Running the new `src/models/parakeet-ctc`
family on JFK through the exported graph yields 34 tokens and the text "And so,
my fellow Americans, askk not what your country can do for you. Ask what you can
do for your country." — the oracle, apart from one leading space from the
sentencepiece word marker. Three wiring defects surfaced and were fixed: the
attention mask must be rank 2 `[batch, frames]`; the frontend is mel-major while
the graph is `[batch, frames, mels]`, so without a transpose the graph emits
zero tokens; and the pad id 1024 lives in `tokenizer.json` `added_tokens`, not
`model.vocab`, so a vocab-only tokenizer mis-resolves the blank.

**ROOT CAUSE FIXED.** `ParakeetFeatureExtractor.__call__` normalizes mel features
per band after the log (mean over valid frames, unbiased variance, then
`(x - mean) / (std + 1e-5)`, masked). The private `_torch_extract_fbank_features`
method I had been comparing against does not. Enabling that normalization drops
feature error from max abs `18.02` / mean `9.25` to **max abs `2.04e-4` / mean
`9.76e-7`** over 88,080 coordinates, leaving **137 failing coordinates (0.16%)**
at `atol=1e-5, rtol=1e-4`, concentrated in 12 mel bands. That residual is a
per-band reduction precision difference, the same class as the Orukeet
pure-silence divergence, not a spectral or filterbank error.

Frame counts settled too: the official centered STFT yields 1101 frames, but
`features_lengths` and the official mask cover 1100, so the existing clamp is
correct; enabling `includeCenteredTailFrame` regressed parity to max `3.28` and
was reverted.

**Stage-wise localization (superseded by the fix above).** With preemphasis disabled our frontend matches the
official extractor to max abs `4.08e-4` / mean `1.92e-6` across 88,080
coordinates, so the STFT, window, power spectra, librosa bank, log guard and
centered padding are all correct. Enabling preemphasis is the sole break
(`18.02`), and applying the same preemphasis formula in JS before the
preprocessor reproduces that same error, so our arithmetic is not at fault: the
extra transform lives inside `ParakeetFeatureExtractor.__call__`, which
`ParakeetProcessor` only delegates to. Next: diff the tensor `__call__` hands to
`_torch_extract_fbank_features` against our own, then compare per-frame power
spectra before mel.

**Frontend parity BLOCKED, localized.** `src/models/parakeet-ctc/frontend.ts`
encodes the official contract (80 bins, n_fft 512, hop 160, win 400, 0.97
preemphasis, hann-symmetric, centered STFT, HTK mel with slaney norm, power
spectra, `log(mel + 2^-24)`, `[frames, mels]`). Measured against the official
numpy extractor on JFK it does not yet agree: max abs `16.63`, mean abs `2.25`
over 88,000 coordinates, with no frame shift improving it (shift 0/±1 all
similar), so this is arithmetic scale, not alignment.

**Progress: the mel scale was wrong and is now fixed.** The frontend passed
`melScale: 'htk'`, but `librosa.filters.mel` defaults to the **Slaney** mel
scale (f_sp 200/3, 1000 Hz breakpoint, log(6.4)/27). Switching to `'slaney'` with
`slaneyNorm: true` cut max abs error from `16.63` to `7.40` and mean from `2.25`
to `1.70`. Parity still FAILS (87,671 of 88,000 coordinates outside 1e-4), and
the per-band ratio is now a smooth monotone function of band index
(`0.002, 0.008, 0.027, 0.091, 0.202, 0.429, 1.708`) instead of scattering,
which points at band-edge or slaney-area-normalization construction rather than
the mel scale.

Root cause history: the filterbank shape, not a scale factor. The
per-mel-band linear ratio official/ours varies by orders of magnitude (mel 0
`0.001`, mel 5 `0.007`, mel 10 `0.206`, mel 20 `14.16`, mel 30 `0.639`, mel 40
`3.729`), which rules out a single missing normalization factor. The shared
`src/models/lasr-ctc/mel.ts` builds the NeMo/Kaldi-style HTK filterbank, while
ParakeetForCTC needs librosa's `filters.mel(..., norm='slaney')` on the default
HTK scale.

Verified as already correct, so they are not causes: window placement (our
`createPaddedWindow` centers the 400-sample window with `leftPad = 56`, matching
`torch.stft`), window flavor (`hann` divides by `winLength - 1`, i.e.
`periodic=False`, matching `torch.hann_window(400, periodic=False)`), power
spectra (`re^2 + im^2`), and the log guard (`log(mel + 2^-24)`, both sides at
the same `-16.6355` floor).

The filterbank is shared code used by the LASR and GigaAM CTC frontends. A
family-local reconstruction of librosa's bank was attempted and reverted: it
produced obviously wrong magnitudes and is not committed. What landed instead
are two additive, backward-compatible options in `src/models/lasr-ctc/mel.ts` —
`melFilterbank` (family-supplied matrix) and `includeCenteredTailFrame` — which
are inert until a family opts in. The tail-frame option was verified to move the
family frontend from 1100 to the official 1101 frames. Full suite: 1,470
passed / 33 skipped. 2. **Frame count clamp.** The official centered STFT yields 1101 frames for the
JFK fixture; our `validFrameCount` clamps to 1100 and discards the final
centered frame. Same shared-code caveat; lower priority than the filterbank.

Receipt: `tools/data/results/parakeet-ctc/parakeet-ctc-frontend-rootcause-2026-10-04.json`.

The oracle transcript for this model is
`And so, my fellow Americans, askk not what your country can do for you. Ask
what you can do for your country.` ("askk" is the model's own output). No preset,
adapter or manifest is registered, so nothing ships on unverified features.

Receipt: `tools/data/results/parakeet-ctc/parakeet-ctc-110m-en-export-2026-10-04.json`.
Feature measurements: `parakeet-ctc-frontend-parity-2026-10-04.json`.

Note on `primeline/parakeet-primeline`: verified it has **no config.json at all**,
only a `2_95_WER.nemo` checkpoint and a README. Its distillation status and
architecture stay unverified, so it is not yet a defensible alternative.

## C35 — parakeet-redux is not smaller; quantization path collapses

Status: CORRECTION — an earlier claim in this session was wrong.

`moondream/parakeet-redux` was reported as "148,923,399 parameters, about 4.2x
smaller than the 627M line". That number is the Hugging Face safetensors
_file-size_ estimate. Ternary weights pack five elements per byte, so the file is
small while the parameter count is unchanged.

Verified from the repository's own `ternary.json`: `n_quantized_params`
603,979,776 plus `n_dense_params` 23,290,911 equals **627,270,687** parameters,
24 more than the teacher `nvidia/parakeet-tdt-0.6b-v3` (627,270,663). The
architecture is `parakeet_tdt` with a 24-layer, 1024-hidden, 128-mel encoder,
identical to Parakeet TDT v3.

It therefore fails scope twice: it is NVIDIA-derived (`source.teacher` names the
NVIDIA model) and it is the same size class, not a student. Exporting it to ONNX
requires dequantizing ternary weights back to float32, which yields a full-size
graph with no runtime saving, because ONNX Runtime has no ternary matmul
operator. The authorized quantization path buys download and disk size only.

Receipt: `tools/data/results/model-candidates/parakeet-redux-size-correction-2026-10-04.json`.

## C34 — Parakeet candidate survey is now conclusive

Status: SURVEY COMPLETE — every candidate resolved; one port shipped, no smaller
student exists in scope.

Reading the first 2 MB of `primeline/parakeet-primeline` (a plain tar, not
gzipped) settles the last open candidate without a 2.5 GB download:
`model_weights.ckpt` is 2,508,649,755 bytes, about 627M float32 parameters, and
`model_config.yaml` declares `ConformerEncoder` with `n_layers: 24`,
`d_model: 1024`, `feat_in: 128` plus an `RNNTDecoder`. That is the same
architecture and scale as Parakeet v3 and Orukeet, so the earlier
"genuine distillation" label is disproved and it is reclassified
`DEFER_FULL_SIZE_NOT_A_DISTILLATION`. The `2_95_WER` name indicates a full-size
retrain, not a size reduction.

Consequence: **no non-NVIDIA Parakeet-line model smaller than 627M exists that we
may port.** The only sub-300M record is `moondream/parakeet-redux`, a 1.58-bit
ternary repack of TDT v3, which the standing integer-quantization rule excludes.
Whether quantization counts as a permitted path for it is a rule decision for
the user, not a technical blocker; if permitted it is a separate evidence-gated
port, not a distillation.

Receipt: `tools/data/results/model-candidates/parakeet-distil-verification-v3-2026-10-04.json`.

## C31 — Parakeet distillation and new-model survey

Status: SUPERSEDED BY C32 — read C32 for the live verdict. Kept for the survey
method and the exclusion reasons. Do not resume the parakeet-redux target from
this record.

Correction carried into C32: `moondream/parakeet-redux` is a 1.58-bit ternary
TDT v3 repack, so it is excluded by the standing integer-quantization rule
rather than blocked on a download, and the strongest portable candidate is
`ctaguchi/parakeet-hf-ctc-110m-en`, whose weights were subsequently fetched and
exported under C32. This record predates both facts and its "no weights
downloaded" line describes only this survey pass.

Scope is Parakeet-line distillations and new models; NVIDIA-originated and
Nemotron models are excluded. Surveyed 62 `parakeet` records and verified the
strongest candidates against the Hugging Face API.

| candidate                                |      params | license   |    ONNX | verdict                                                 |
| ---------------------------------------- | ----------: | --------- | ------: | ------------------------------------------------------- |
| `moondream/parakeet-redux`               | 148,923,399 | cc-by-4.0 |    none | strongest; about 4.2x smaller than the 627M line        |
| `moondream/parakeet-ultra`               | 627,270,663 | cc-by-4.0 |    none | deferred: same size class, not a student                |
| `istupakov/parakeet-tdt-0.6b-v3-onnx`    |         n/a | cc-by-4.0 | 5 files | third-party export of stock v3, not a new model         |
| `primeline/parakeet-primeline`           |  unverified | cc-by-4.0 |    none | deferred: architecture and size unverified              |
| `handy-computer/parakeet-primeline-gguf` |         n/a | cc-by-4.0 |    none | deferred: GGUF is not an executable runtime format here |

`parakeet-redux` is the only verified sub-300M model on the Parakeet line and
the only candidate that is a real size win. It publishes no ONNX graphs, so
porting it means downloading weights and running our own export; both are
authority-gated. Its quantization character must also be re-checked against the
standing rule before any promotion, rather than assumed from its size.

Excluded by scope and recorded with reasons: `nvidia/parakeet-ctc-0.6b` (a
usable ONNX export exists via `onnx-community`, worth revisiting), the 114.6M
`nvidia/parakeet-tdt_ctc-110m` student, its fine-tunes, and the Nemotron
families.

**Corrections after identity and architecture verification.** The unqualified
ids `parakeet-redux` and `parakeet-tdt-0.6b-v3-distill` return HTTP 401: they
are not public repositories and were search artifacts of my own include list, so
they were never candidates. `moondream/parakeet-redux` is **not** a distilled
student at all: upstream documents it as a 1.58-bit ternary TDT v3, an
integer-quantized repack, which the standing rule excludes outright rather than
merely gating on a download.

That leaves two real keepers and no already-portable model:

- **`ctaguchi/parakeet-hf-ctc-110m-en` — keep, strongest portable.** Verified
  `ParakeetForCTC` / `parakeet_ctc`, 109,305,345 parameters, cc-by-4.0,
  sentencepiece tokenizer, safetensors weights, non-NVIDIA repository. Its
  safetensors-plus-config shape gives the shortest export path. Porting needs a
  new transformers-based ONNX export plus a ParakeetForCTC emission path; this
  repository has CTC runtimes (gigaam-ctc, wav2vec2, lasr-ctc) but no
  ParakeetForCTC family and no parakeet-ctc exporter. Verify tokenizer and
  emission layout before writing the family.
- **`primeline/parakeet-primeline` — keep as a future export task.** A genuine
  non-NVIDIA English distillation, but it publishes `.nemo` weights only, so it
  needs our existing `tools/nemo-tdt-onnx-export` path.

Receipts: `hf-parakeet-distil-2026-10-04.json` (62 records),
`parakeet-distil-verification-2026-10-04.json` and its corrected
`parakeet-distil-verification-v2-2026-10-04.json`.

## C30 — Orukeet preset contract and in-repo demo

Status: PASS for the preset contract and demo wiring; the demo page itself is
browser-UI work verified by build and API-shape checks, not by an automated
browser run yet.

Registering Orukeet in the catalog was not sufficient on its own. The preset
factory resolves configuration from
`src/presets/parakeet/manifest.ts`, so without a manifest
`supports('oruk/orukeet')` stayed false and the config merge was empty. The
native CLI hid this because it takes an explicit runtime config. Added the
audited Orukeet manifest (vocabulary 8193, blank token 8192, mel 128, the 25
supported languages, sentencepiece tokenizer, pinned HF repo) plus a regression
that asserts both the manifest and the factory-resolved request, so this gap
cannot reopen behind an explicit config.

Two real defects were caught by driving the page: the script called
`model.transcribe(pcm, 16000, options)`, which passes the sample rate as the
options object (the PCM entry point is `transcribeMonoPcm`), and it imported a
package specifier the demo cannot resolve, unlike the working canary page which
imports `../../../src/runtime/load.ts`. Both are fixed; the page now uses the
relative public-API import and `transcribeMonoPcm`.

**Resolved and verified.** Serving the ORT runtime through Vite's `/@fs/` route
to the repository's installed `onnxruntime-web/dist/` (already permitted by
`server.fs.allow`) makes the page work with no copy, symlink or new dependency.
Driving the page with real Chrome, the Orukeet local folder and the JFK fixture
gives `PASS_ORUKEET_SMOKE`: 7.6 s load, NVIDIA Blackwell adapter, encoder/decoder
fp32 selection, two identical transcripts that **exactly match the official
oracle**, clean disposal and zero page errors. Receipt:
`tools/data/results/orukeet/orukeet-demo-jfk-browser-2026-10-03.json`.

**Hub source removed.** Verified against the Hugging Face API at revision
`b59c13a7`: the repo has no root-level `encoder-model*.onnx`,
`decoder_joint-model*.onnx` or `vocab.txt`; graphs live under
`onnx/combined-v0.1.0-int8/` and `onnx/sherpa-v0.1.0-int8/` with a
`tokenizer.json`. Advertising a Hub source would 404 on the application's
default load, so the manifest entry no longer declares one, the catalog entry
sets `supportsHubSource: false`, and a regression asserts it. Orukeet is
local-folder only until a matching validated export source exists; the audited
fp32 local preset and config are unchanged.

Demo: `examples/demo/public/orukeet-smoke.html` with
`examples/demo/src/orukeet-smoke.js`, following the existing canary smoke page.
It loads `oruk/orukeet` from a local folder through the public browser API,
transcribes a chosen 16 kHz mono WAV, repeats for determinism, and awaits
disposal. The script uses only real exported API (`text`, `tokens`, `metrics`,
`warnings`); an invented `disposeSpeechModelLocalEntries` call was removed when
the export list proved it does not exist.

Full suite 1,469 passed / 33 skipped. Automated browser verification of this page
is the next step; the companion-UI gate already covers Orukeet browser loading
and warm-up.

## C29 — Port manifests and Orukeet as a built-in model

Status: PASS — one manifest drives four real runs (Node/Bun long and short,
companion UI) against the official oracles. Not feature-parity, fp16 or
performance acceptance.

Orukeet had **no built-in descriptor at all**, so it could not be selected in the
companion demo, had no local-folder loader, and could not be exercised by any
UI-driven gate. Registering `oruk/orukeet` in the Parakeet catalog (25
languages, 627,057,286 parameters, revision
`b59c13a733fce5cf193230d0fa317ee7f145a108`) reuses the `parakeet` preset key, so
the existing local adapter, demo selection and browser gates apply with no new
family code. The companion UI now loads Orukeet from a local folder and verifies
warm-up on a real NVIDIA Blackwell adapter (15.8 s, no page errors).

Added the reusable port harness: `tools/model-debugging/ports/*.port.json` plus
`tools/model-debugging/scripts/port_manifest.py`. A manifest declares original
weights, local artifacts, fixtures, oracles and the Node/Bun/browser cases;
the runner binds every file to its recorded SHA-256 **before** any execution,
then compares token ids and text per case. It is fail-closed by construction: on
first use it rejected a runtime-config digest that had been copied from an older
receipt, and a browser case writes its own receipt instead of racing the
manifest receipt.

Executed through the manifest, the three Node/Bun cases pass with exact oracle
match and the browser case passes its warm-up gate:

Two fail-closed gaps found in review were closed before this was called done: a
native run with a missing or empty `compare` list reported a pass without
comparing anything, and artifact/oracle digests were recorded but not enforced.
Every consumed file is now bound by declared SHA-256 before execution, including
the 2.4 GB external-data sidecar, and `tests/test_port_manifest.py` covers
missing/empty/unknown comparison fields, a missing fixture, mismatched output, a
changed sidecar, changed graphs and a changed oracle (10 regressions, 0.5 s
against a synthetic bundle with the real file names).

| case                                | backend                       | result                                                                                                           |
| ----------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `node-native-jfk`                   | native-cpu                    | 38/38 tokens, exact text, RTFx 10.80                                                                             |
| `node-native-librivox-repeat8`      | native-cpu                    | 722/722 tokens, exact text, RTFx 7.22                                                                            |
| `bun-native-librivox-repeat8`       | native-cpu                    | 722/722 tokens, exact text, RTFx 7.37                                                                            |
| `companion-ui-warmup-webgpu-hybrid` | WebGPU encoder + WASM decoder | `PASS_WARMUP`: load + warm-up on Chrome 154.0.8037.93, no oracle comparison (the UI uses its own warm-up sample) |

Receipts: `tools/data/results/orukeet/port-manifest-*-2026-10-03.json` plus the
paired `.browser.json` for the UI case. Full suite unchanged at 1,468 passed /
33 skipped. The deferred production-feature coordinate gate is untouched.

# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C28 — Automated browser gates: fixed, parameterized, and honest limits

Status: MIXED — Parakeet v2/v3 companion-UI browser gates PASS with real
inference; the Nemotron 3.5 browser path is PARTIAL/BLOCKED. No production
acceptance is claimed for any browser lane.

The companion UI smoke (`tests/smoke/browser-demo-smoke.py`) never passed with a
real model. The root cause was not the model: the app resolved the ORT runtime
sidecar from `cdn.jsdelivr.net`, which the no-download route aborted, so every
run died with "no available backend found". The smoke now fulfills those exact
URLs from the installed, version-pinned `onnxruntime-web` package, records URL,
path and sha256 for each, and fails closed unless a model load consumed both.

Parameterized the gate on `--model-id`, `--runtime-backend` and real-file
artifact-layout detection, then ran it on a real Blackwell adapter:

| model                             | layout        | backend       | result                 |
| --------------------------------- | ------------- | ------------- | ---------------------- |
| `parakeet-tdt-0.6b-v3`            | nemo-tdt      | webgpu-hybrid | PASS, warm-up verified |
| `parakeet-tdt-0.6b-v2`            | nemo-tdt      | webgpu-hybrid | PASS, warm-up verified |
| `parakeet-tdt-0.6b-v3`            | nemo-tdt      | wasm          | PASS, warm-up verified |
| `nemotron-3.5-asr-streaming-0.6b` | nemotron-rnnt | webgpu-hybrid | FAIL, see below        |

Three findings for the newest port, each recorded with receipts:

1. **No browser local-folder loader existed** for Nemotron 3.5; the registry
   raised "does not yet have a browser local-folder loader". Added
   `src/presets/nemotron/local-adapter.ts` plus the shared, model-neutral
   `src/runtime/local-entry-files.ts` helpers, with two unit regressions. The
   adapter also forwards `encoderFilename`/`decoderFilename`/`jointFilename`,
   without which ORT cannot resolve `.data` beside a blob URL.
2. **Matching-contract local-folder browser inference is UNTESTED.** The adapter
   is unit-tested only. Its one exercised browser path used
   `nemotron-3.5-asr-streaming-0.6b-onnx`, whose encoder is a _different export
   contract_ (`input_features`, `prompt_ids`, `cache_mask`, per-layer KV and
   conv caches, separate first/continuation encoders) than the executor's
   six-input `audio_signal`/conformer-cache contract. The two naming generations
   are **not interchangeable filenames**; supporting the two-encoder export is a
   separate contract adaptation, not quantization-suffix detection. The
   matching-contract folder run stopped at smoke layout detection and has not
   been re-run.
3. **Direct-URL harness inference FAILS.** The established
   `webgpu-agent-test` runner builds the executor from artifact URLs directly
   and never touches the new adapter, so this is separate evidence. fp32 fails
   at session creation with `WebAssembly.Memory(): Property 'initial': value
74903 is above the upper bound 65536` in both hybrid and all-WebGPU
   compositions — a WASM32 memory ceiling. fp16 clears it (session 11.7 s, 119
   frames, RTFx 3.04) but emits **zero tokens**, so the transcript is empty and
   baselineMatch is false.

So: Parakeet v2/v3 companion-UI browser gates pass; Nemotron 3.5 browser
decoding is **not** verified on either path. Next: run the local-folder adapter
against a matching-contract export, and localize the zero-token decode in the
`nemotron-rnnt-webgpu.js` runner before touching the family. The inherited dirty
`src/models/nemotron-rnnt/*` and `src/presets/descriptors.ts` work stayed
untouched, so this adapter is deliberately unlanded on top of it.

## C27 — Long-form port fix: encoder layout beyond 82 seconds

Status: PASS — real public-API inference, exact oracle parity, regression tests.
The production-feature gate is untouched and still deferred.

Resuming the recorded P02 gap (`long_library_native_node_bun` and
`accumulated_tokens_text` were UNTESTED) exposed a genuine library defect. The
first 149.7 s run failed with `Got invalid dimensions for input: encoder_outputs
... index: 1 Got: 1872 Expected: 1024`. The NeMo TDT executor inferred the
encoder-output layout by comparing axis magnitudes, so any clip with more
encoder frames than the 1024-feature dimension flipped it and fed a transposed
tensor to the decoder — about 82 seconds of audio at the 10 ms / 8x stride. Every
previously recorded library run used an 11 s fixture, which hid it.

The executor now resolves the layout from the encoder graph's own
`encoded_lengths` output, crops a padded time axis, and fails closed when the
length matches neither axis. Two regression tests cover both cases and fail
against the previous implementation (2 failed / 14 passed) while the full file
passes after it (16 passed).

With the fix, Node 26.2.0 and Bun 1.3.14 both reproduce the original-engine
reference **exactly** on the 149.7 s fixture: 722/722 identical token ids,
byte-identical 1899-character transcripts, 1872 encoder frames, 806 decode
iterations matching the official 806 joint calls, zero warnings. This closes the
P02 long-decoding gap for the native CPU fp32 path. WASM/WebGPU long-form
decoding, streaming-controller behavior at this length, and multi-fixture
soak remain UNTESTED.

Receipt: `tools/data/results/orukeet/long-repeat8-end-to-end-parity-2026-10-03.json`.
Reusable lesson recorded as step 8 of the
[porting parity playbook](../tools/model-debugging/playbooks/model-porting-parity.md#recommended-flow):
verify a port on a clip longer than the encoder feature dimension, not only a
short fixture.

Follow-up: the first fix tested a padded time axis before the
`dims[1] === encoded_lengths` equality case, so a time-major output whose frames
stay below the feature count could be read as padded feature-major. The final
order is equality-first, then the two padded cases, then fail closed. Layout
regressions now cover frames-above-features, time-major-below-features,
padded-crop and ambiguity; removing the padded branches fails the padded test
(1 failed / 17 passed) and the file passes 18.

Final verified state on the corrected ordering: JFK 38/38 tokens and exact text;
the 149.7 s fixture 722/722 tokens, exact text, 1872 frames and 806 decode
iterations on both Node and Bun. Receipt
`tools/data/results/orukeet/long-short-form-parity-final-2026-10-03.json`
supersedes the first-fix receipt, which is preserved.

## C26 — Frontend conformance gate policy for future ports

Status: EVIDENCE RECORDED from existing receipts, decision left open. No new
inference, no weight downloads, no official-capture refresh, no runtime or
tolerance change.

The `atol=1e-5, rtol=1e-4` feature-coordinate gate on Orukeet was never met by
any NeMo port shipped from this repository. Recorded official-versus-library
frontend comparisons: `parakeet-realtime-eou-120m-v1` max `4.0801039` /
mean `9.019221e-4` with exact token and text parity, `canary-180m-flash`
asrjs `2.1461394` / `8.918627e-4`, meljs `0.7285282` / `9.190078e-4`, and
Orukeet `3.4236908e-4` / `1.226810e-6` with oracle-matching 38-token output.

Orukeet's frontend is the closest to its official oracle of any port here:
6268x, 2128x and 11917x smaller maximum absolute error than those three, and
727x, 749x and 735x smaller mean absolute error. `3.42e-4` is not a port
defect on this evidence, and Orukeet is not harder to port than the models
already shipping.

This is **evidence, not a reclassification**. The coordinate gate remains
Orukeet's production-feature gate and stays FAIL / DEFERRED at unchanged
tolerances; nothing here relaxes or removes it, and no numerical work is called
done. Contracts differ between the rows (the Parakeet v1 and Canary receipts use
`validLengthMode: centered` with `normalization: none`, meljs compared 140,800
values, and the Parakeet v1 maximum sits entirely on boundary frame 1100 of
1101), so the ratios compare magnitude classes, not interchangeable gates.

Whether the strict per-coordinate gate keeps binding for Orukeet is the user's
call. Either it stays binding and `full_asr` stays unproven, or the historical
functional standard the shipped ports used is adopted. The
[focus goal](GOAL_MODEL_PORT_HARNESS.md#two-standards-what-shipped-and-what-this-lane-gates-on)
records both options; this repository does not pick (b) on its own.

Pure silence remains a separate recorded divergence: an official masked fp32
mean reduction on a constant row, not spectrum fidelity.

Receipt: `tools/data/results/orukeet/frontend-conformance-family-comparison-2026-10-03.json`
(aggregation of six existing receipts with their SHA-256). No official stock
v2/v3 feature capture is stored locally, so a same-contract stock v3
comparison remains unavailable without original weights.

## C25 — Existing Qwen harness reuse and bounded numerical disposition

Status: PASS for the focused reuse exercise; Orukeet production-feature
conformance remains **DEFERRED / FAIL**, not model acceptance. No runtime,
family math, tolerance, exporter or shared-control implementation changed.

The existing Qwen Node/WASM CLI ran local official-export artifacts on JFK
with dynamic encoder/fp16 decoder and produced 25 tokens, WER/CER zero and
normalized exact match against the saved native reference. Cold elapsed time
was 111899.6057 ms. It exited successfully through its awaited disposal path.
This did not rebuild the library, refresh official features or prove browser,
performance or resource-reclamation acceptance.

The unchanged shared attribution control also ran the real Qwen dynamic
encoder. Reference features came from the current library processor, not a
fresh official-engine capture. A deliberate same-shape frame-major/mel-major
misinterpretation was localized to `input_features`; replay reproduced the
candidate output with zero numerical error. The receipt binds the waveform,
prepared feed and seven unchanged source/artifact hashes. Three actual
encoder calls, one six-line callback; zero new inference loops, exporters,
weight downloads, shared-attribution changes or permanent adapter scripts.
The one-input diagnostic proves real cross-family reuse, not a production bug
or multi-input discrimination.

Receipts under `tools/data/results/qwen/`:
`goal-harness-existing-port-jfk-2026-10-03.json` and
`goal-harness-encoder-layout-attribution-jfk-2026-10-03.json`.
The existing [Qwen guide](../tools/model-debugging/reference/qwen3-asr-0.6b/README.md#shared-encoder-boundary-control)
now exposes the tested control recipe and its limits.

The [focus goal](GOAL_MODEL_PORT_HARNESS.md#bounded-orukeet-numerical-disposition)
records the bounded Orukeet disposition: official-complex/native-real-FFT
controls pass downstream feature gates, portable full-512 FFT already fails
end-to-end, and pure silence has a separate measured reduction mismatch.
No rejected FFT experiment was repeated or numerical gate relaxed. Browser
real-hybrid scores retain their existing fixture-specific PASS. Complete
Orukeet feature and full ASR acceptance remain unproven.

## C24 — Python reference environment inventory

Status: AUDITED — read-only environment discovery, real import/CPU smoke and
dependency checks. Not model parity, CUDA acceptance or environment migration.
User requested project-owned environments per runtime/reference library and
inspection of venv, Conda and uv installations.

Eleven interpreters were inspected. Six reference stacks passed actual imports
and Torch CPU arithmetic. Only the isolated project Qwen environment passed
`pip check` among those six. The project NeMo environment is an overlay:
NeMo 3.0.0 is local, while Torch/ONNX/ORT/NumPy come from Conda `nemo`, whose
own NeMo version is 2.4.0. Both import ASR successfully but have dependency
conflicts; they are not interchangeable oracle identities. The sibling
GigaAM environment also inherits from system Python and is not project-owned.

Conda is installed but absent from `PATH`; explicit invocation locates `nemo`,
`medasr_env`, `asrds_py311`, `vgtts` and base. uv supplies a managed Python
3.13.15 interpreter, but no uv tool environment or project uv manifest/lockfile
was found. Its visible NumPy comes from the user site, not an isolated ASR stack.

The [tool map](../tools/README.md#python-reference-environments), local skill
and [workstation inventory](WORKSPACE_CONTEXT.md#python-reference-environment-inventory)
now expose interpreter selection, package provenance, isolation and actual
dependency-check outcomes. The recommended unit is a version-pinned reference
dependency stack, shared across compatible models, with separate environments
for incompatible reference versions. No package was installed or upgraded;
no environment or inherited Nemotron change was edited.

Receipt: `tools/data/results/python-runtime-environments-2026-10-03.json`.
Imports/CPU arithmetic passed for project NeMo, project Qwen, Conda NeMo,
Conda MedASR, Conda onnx-asr and sibling sherpa/FunASR. Dependency-check failures
are preserved, not suppressed. Provisioning isolated replacements remains
separate from this completed inventory.

## C23 — Project-local skill and tool discovery

Status: PASS — documentation discovery and CLI-help smoke; not model inference
or numerical acceptance. User requested easier discovery after agents missed
existing model-porting skills, reference engines and local artifacts.

The tracked startup now exposes the project-local Model Debugging Skill and
the [task-to-tool map](../tools/README.md#task-to-tool-map) before long task
records. The map routes ports, feature contracts, original captures, stage
comparison, attribution, artifact audits and browser checks to existing tools.
No registry, second runner, inference loop or global skill install was added.
Core runtime and the twelve inherited Nemotron changes remain untouched.

Toolkit startup links and the family-playbook index now use relative paths.
The script index exposes the existing offline Qwen library CLI and correctly
groups frontend options with the frontend tool, not the artifact packer.
It warns about the legacy frontend's sibling default, missing help handler and
different reference schema. The Qwen guide's PowerShell capture command is
repaired; its offline runner section names actual supported arguments.
The workspace map supplies exact local NeMo, onnx-asr and sherpa-helper paths.
Local environments and paths are discovery leads, not fresh-clone guarantees.

Two fresh-agent, read-only exercises started at `docs/AGENT_ENTRYPOINT.md`.
Both located the local skill, family workflow, original reference helpers and
existing runners. Qwen discovery also located local ONNX artifacts and the
already implemented `qwen-asr` family. NeMo discovery found the project-local
reference interpreter instead of treating a default-Python import failure as
missing tooling. Their CLI/schema findings were checked against source.
The Qwen guide was rechecked after its command and runner-route corrections.

Validation: 122 local Markdown file/heading links resolve across the ten edited
discovery/goal guides. `node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --help`
exits successfully with the documented flags. The stage comparator's `--help`
also exits successfully. The discovered reference interpreter imports NeMo
3.0.0 and Torch 2.6.0+cu118. Formatting and whitespace checks pass.
No real model inference was run for this documentation-only change.

The focus goal record retracts the unsupported claim that Qwen needs a new
family/download merely to run local ONNX artifacts. Its historical original
PyTorch snapshot path is absent, which is a separate prerequisite for fresh
official-engine capture. Existing ports do not automatically demonstrate the
focus goal's cheaper-port/cross-family-attribution criterion. Orukeet feature
parity remains FAIL; all original tolerances and model ordering stay unchanged.

## C22 — Real-provider disposal drain and post-C21 native refresh

Status: PASS — actual onnxruntime-node inference through the public example
CLI and inspector; not feature-parity, browser, fp16 or measured-memory
acceptance.

C18-C21 shipped with simulated/deferred-call evidence and explicitly left real
provider shutdown open. This slice records actual native inference at the
current HEAD and closes the "accepted work during disposal" question with the
real executor rather than an injected one.

Refreshed public-consumer output on a freshly source-bound build (rebuilt
`dist`, build snapshot hash `1d74ddfc3b463007`, 1403 files, identical across
all four runs). All four public example CLI runs exit 0 with awaited cleanup
PASS, actual `native-cpu`/`onnxruntime-node`/`cpu` identity, `fp32` precision
and pinned node ORT `1.25.0-dev.20260228-6e72d31970`: Node 26.2.0 and Bun
1.3.14 each match the official oracle on JFK (38 tokens) and on the
same-captured-PCM speech/silence control (69 tokens, waveform comparison PASS).
Receipts under `tools/data/results/orukeet/`:
`public-example-node-jfk-post-c21-2026-10-03.json`,
`public-example-bun-jfk-post-c21-2026-10-03.json`,
`public-example-node-speech-silence-post-c21-2026-10-03.json`,
`public-example-bun-speech-silence-post-c21-2026-10-03.json`. All remain
**PARTIAL_PASS_OUTPUT_ONLY**.

Lifecycle: real mid-decode abort plus two oracle repeats pass again
(`native-cancel-reuse-post-c21-2026-10-03.json`; abort observed at decode
1/508, handle identity unchanged). New `--dispose-drain` control
(`native-disposal-drain-post-c21-2026-10-03.json`) requests disposal while a
call is in flight, on a directly created family session first and then on the
loaded public handle. Both phases settle with oracle-matching output before
disposal resolves (dispose 3341.73ms / 3373.48ms), the next call on each is
rejected, and `createSession()` after disposal is rejected. The control fails
closed if disposal settles before the accepted call, if the accepted output
diverges, if disposal itself fails, or if a post-disposal call is accepted.
Seven new simulated regressions cover those behaviors and the exclusive
loaded-session requirement (`tests/nemo-inspector-disposal-drain.test.ts`);
they are not real provider acceptance.

Full regression: 1,462 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/post-c21-disposal-drain-full-suite-2026-10-03.json`.
TypeScript (library and examples) and scoped ESLint pass; the only lint output
is pre-existing `max-lines` warnings. TLS verification/system CA were enabled
only for that test child.

No runtime, decoder internals, family math, tolerance or public API changed: the
new control lives in the existing inspector adapter.

The disposal record now carries the observed settle order
(`["acceptedCall","dispose"]` on both real-provider phases in
`native-disposal-drain-settle-order-post-c21-2026-10-03.json`) instead of an
asserted boolean; the derived flag is computed from that observation.

Follow-up on the open browser gate: the head call 20 normalized-score failure
is now attributed to the real-encoder feed, not the WASM head. Perturbing only
the recorded encoder delta (`1.4566e-6`) on official reference inputs
reproduces the failure exactly (`5.8079e-4`, same 3 mismatches, same first
channel); predictor-state deltas contribute nothing, and the head response is
linear at about 399x gain. Details and the two read-only probes are in
`PARAKEET_VARIANT_TASKS.md`. The gate remains FAIL at unchanged tolerances.

Still open, with the unchanged failing gates cited by receipt: production
feature parity remains FAIL at the unchanged `atol=1e-5, rtol=1e-4`
(JFK 313/140,928, `filterbank-feature-jfk-2026-10-03.json`; speech/silence
74/519,808 on the default path versus 73 under the diagnostic width-8
normalizer, `production-raw-width8-speech-silence-2026-10-03.json`; pure
silence 12,800/12,928, `filterbank-feature-pure-silence-2026-10-03.json`).
Also open: the browser real-hybrid normalized-score failure at head call 20,
source-audio resampling acceptance, held-out quality, fp16 promotion, a hung
provider still delaying disposal, and measured memory/VRAM recovery after
teardown.

## C21 — Direct public TDT session shutdown

Status: PASS — public runtime/model/session lifecycle with injected executor;
not real ORT/WebGPU teardown, raw executor or cross-family certification.

High-level model handles already used an operation lease, but sessions returned
by direct model.createSession() could dispose their executor during transcription.
Two new deferred-call regressions reproduced premature executor disposal for
session and model owners. The TDT session now consumes the existing shared
runtime operation lease: accepted transcription (including canonical mapping)
settles before executor teardown, new calls reject once session disposal enters,
and errors release their lease. No decoder internals or shared helper changed.
Dependency direction remains shared runtime ownership primitive -> family session.

Seven new lifecycle cases include the public root runtime.loadModel().createSession()
path: session/model/runtime shutdown with successful or failing inference, plus
multiple active calls and an executor cleanup failure. Runtime disposal first
settles inflight loads and reaches model/session cleanup asynchronously; tests
observe actual session-dispose entry, not a fixed microtask count or an instant
closure guarantee for all runtime-owned models. A faulty timing-assumption test
extension was reverted before rebuilding that synchronization. Direct session
disposal closes its own lease synchronously. Repeated cleanup retains the same
failure and disposes the executor once; callers retain their inference error.

Eight session tests and seventy-five related executor/initialization/high-level/
streaming-owner tests pass with TLS verification enabled. TypeScript, scoped
ESLint and formatting pass. Full suite: 1,455 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-tdt-public-session-disposal-full-suite-2026-10-03.json`.
TLS verification/system CA are test-child-only settings.

Real provider shutdown and resource recovery still need an actual inference
check. Raw OrtNemoTdtExecutor callers and other family sessions are not covered
by this change; a never-settling inference still delays disposal. Orukeet parity
remains FAIL and no model promotion, tolerance change or speed claim follows.

## C20 — Preprocessor active-run disposal ownership

Status: PASS — deterministic deferred-run/session-creation simulations;
not real provider teardown, whole executor or long-session certification.

Two regressions reproduced disposal completing while a preprocessor run was
still pending, for both successful and failing inference. Active calls are now
registered synchronously and drained through their tensor cleanup before the
processor releases its session. New calls reject once disposal begins, and a
post-acquisition check prevents a warm-session run from starting after disposal.
Accepted runs already executing retain their result/error; disposal waits, it
does not interrupt the provider. A never-settling provider run still prevents
disposal completion; no unsafe forced release or timeout was added.

Six new tests cover success/failure, cleanup-before-release, multiple active
runs, pending creation success/failure, warm acquisition and repeated disposal.
Thirty-one processor and fifty-six processor/native-bridge/RNNT/lifecycle tests
pass; TypeScript, scoped ESLint, formatting and independent review pass.
Full regression: 1,448 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-preprocessor-run-disposal-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child.

Ownership remains local to the NeMo processor; no model math, shared runtime,
decoder or public API changed. Broader executor encoder/decoder disposal during
active transcription remains open and must be checked through public sessions
before claiming whole-model shutdown safety. Orukeet feature parity remains FAIL.

## C19 — Preprocessor cleanup failure isolation

Status: PASS — injected tensor-disposal failures and session reuse;
not real native/browser memory-leak or concurrent-run certification.

Following C18, a throwing output disposer still skipped later outputs and both
inputs; a throwing input disposer skipped the next input. Cleanup could also
mask an inference or validation error. Eight new regression cases reproduced
the failures before the production fix; one cleanup control already passed.
The processor now attempts each unique transient tensor once despite earlier
disposal failures. Single errors preserve identity; multiple errors are reported
in an AggregateError, with the original operation failure first and as cause
when present (including a thrown undefined). Aliased outputs are not disposed
twice. No shared helper, decoder, feature mathematics or public API changed.

Twenty-five processor tests and fifty processor/native-bridge/RNNT/lifecycle
tests pass. TypeScript, scoped ESLint, formatting and independent review pass.
Full regression: 1,442 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-preprocessor-disposal-cleanup-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child. These are
mocked fault-injection regressions, not actual ORT allocation/disposal failures
or measured GPU/CPU memory recovery. Concurrent process/dispose remains open.

## C18 — Preprocessor partial input allocation cleanup

Status: PASS — injected ORT input-allocation failure/reuse ownership;
not actual out-of-memory, browser/native VRAM or full lifecycle certification.

NeMo TDT's ONNX processor allocated both inputs before its cleanup scope.
If the second Tensor constructor threw, the first input was never disposed;
the executor's session teardown did not own those per-call tensors. A regression
reproduced the missing dispose call. Both allocations now occur inside the
existing try/finally; optional references release whichever inputs succeeded.
Original error identity is preserved, inference does not start on allocation
failure, later calls reuse the session, and repeated disposal releases it once.
No model mathematics, public API, backend selection or generic runtime changed.

Forty-one processor/native-bridge/RNNT/lifecycle tests (including 16 processor
tests) pass; TypeScript, scoped ESLint, formatting and diff checks pass.
Full regression: 1,433 PASS / 33 SKIP / 0 FAIL. Full regression
receipt: `tools/data/results/harness/nemo-preprocessor-allocation-cleanup-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child. Real native
allocation-failure injection, cleanup errors during tensor disposal and
concurrent processor-run/disposal behavior remain separate acceptance work.

## C17 — Repeatable real-browser local lifecycle smoke

Status: PASS — bounded stock-v3 hybrid cancellation/reuse/disposal smoke;
not Orukeet parity, GPU-kernel interruption or long-session soak acceptance.

`npm run test:browser-local-lifecycle -- --consumer-root <browser-demo-root> --model-root <stock-v3-model-root> --audio <fixture> --expected-phrase "<full phrase>" --encoder-precision fp16 --output <new-receipt.json>`
uses the existing family-owned browser page and public library inference API,
with cooperative decode cancellation enabled. Paths and expected text are
required, never guessed or downloaded. Requires existing Python/Playwright,
Chrome, Node, the sibling's installed Vite toolchain and local model files.
It builds the harness in memory from the current source aliases; no dist
rebuild, sibling edit, dependency installation or public model promotion.
Use the direct Python entry point if selecting a particular interpreter.

The runner now independently requires two matching nonempty results, actual
WebGPU encoder/WASM decoder metrics, inference/disposal evidence and no
execution/cleanup errors. Requested cancellation must reject with
`PipelineAbortedError` at a partially completed decode, without completion.
Full expected-transcript normalization stays in the existing shared browser
smoke comparator, not a second Python decoder or ASR quality metric.
`npm run test:browser-local-lifecycle:unit` runs ten fast preflight/receipt tests;
they do not launch model inference. Invalid PASS labels and partial lifecycle
evidence are rejected. The real browser command is opt-in, not part of Vitest.

Fresh Chrome 154 / ORT Web 1.29.0 run through the npm command passed:
two identical Harvard transcripts, decode aborted at 5 / 55 units, successful
reuse, awaited disposal, no blocked requests/page errors, verified engine/build
bytes, unchanged inputs, preview process stopped and port closed.
Receipt: `tools/data/results/harness/local-lifecycle-npm-2026-10-03.json`.
Existing ORT node-placement warnings remain; session provider metrics are not
proof that every graph node ran on GPU. No production runtime code changed.

Post shared-cleanup refresh (2026-10-03): the same npm opt-in with fresh receipt
`tools/data/results/harness/local-lifecycle-shared-closure-2026-10-03.json`
passes real Chrome154/ORT Web1.29.0 stock-v3 fp16-encoder/fp32-head inference.
Decode cancellation rejects at5/55 units with PipelineAbortedError, without a
complete event. Both successful results are exactly
"The boy was there when the sun rose." with WebGPU encoder/WASM decoder metrics;
awaited disposal, unchanged protected inputs, served build/engine byte hashes,
zero blocked requests/page errors and stopped preview all pass. Shared closure
probe explicitly reports ECONNREFUSED. Single-run times2223.65ms and339.79ms
are cold/reused observations, not a matched optimization benchmark or soak.
This closes real lifecycle revalidation of the helper reuse, not app UI model
loading, Orukeet numerical parity or every-node GPU placement.

## C15 — Preserve requested remote artifact revisions

Status: PASS — network fetch revision contract; existing cache authentication
and real remote/browser downloads are not certified by these simulated tests.

Removed the shared asset handle's silent HF revision-404 fallback to `main`.
Byte reads, streamed reads and blob locators now reject the requested missing
artifact without fetching or caching a different revision. Provider controls
preserve successful branch/commit reads and the omitted-revision `main` default.
The original failure was reproduced on all three paths before the fix.
54 focused IO/external-data/runtime/builtin tests passed; full suite: 1,414
passed, 33 skipped, zero failed. Receipt:
`tools/data/results/harness/hf-revision-strict-full-suite-2026-10-03.json`.
TypeScript and scoped ESLint passed. Migration and old-cache limitations are
documented in `architecture.md`; no cache entries or sibling files were deleted.

Legacy built-in cache isolation (2026-10-03): a prepopulated wrong-main entry
under a pinned key reproduced the old-cache hole before implementation. The
built-in HF helper now uses a versioned JSON-tuple key for non-main revisions;
main/default keys and all caller-owned provider keys are untouched. A network
read populates the new key and the second read reuses it; tests prove the old
entry remains and no delete occurs. All 35 targeted HF/IO/IndexedDB tests and
source typecheck pass. Status: PARTIAL PASS pending independent review and wider
regression. This uses injected cache/network controls, not an actual remote
download or persistent browser migration; first materialized pinned reads may
require downloading again. No handle-level override of custom cache keys was
retained. Namespace isolation does not authenticate arbitrary cache contents.

Wider regression checkpoint: fresh full suite with TLS verification enabled in
the test child process passes 1,431 tests /33 SKIP /0 FAIL. Receipt:
`tools/data/results/harness/hf-legacy-cache-isolation-full-suite-tls-2026-10-03.json`.
The preceding run inherited an environment TLS-disable flag and is not used as
TLS-enabled evidence; no persistent environment setting was changed. Scoped
ESLint passes. New TS hunks were formatted; unrelated baseline test formatting
was restored and eight HF tests re-passed. Independent final review and commit
remain next; actual remote/browser cache migration is not certified here.

Independent final review: PASS, no correctness findings. Reviewer reran the
32 HF/handle tests and source/examples typechecks; recorded full-suite counts
were verified. Baseline formatting warnings outside new test hunks remain
unchanged. Built-in key isolation is accepted; real browser persistence remains
an explicit next integration rung, not implied by injected cache tests.

Real Chrome persistence rung (2026-10-03): PASS using
`python tests/smoke/hf-cache-browser-smoke.py --output <new-receipt.json>`.
Chrome154 fresh context seeds legacy wrong pinned bytes and compatible main
bytes, intercepts one tiny pinned response, then reloads the same origin with
all remote requests blocked. Pinned text persists in real IndexedDB, main text
is reused, and both old entries remain unchanged. No blocked request/page error;
owned preview closure confirms ECONNREFUSED. Receipt:
`tools/data/results/harness/hf-cache-real-chrome-bound-2026-10-03.json`.
Source, tool and in-memory bundle hashes are bound; no model download or sibling
mutation. This exercises internal production helpers, not published imports,
real HF/TLS, model inference, cross-browser or browser-restart persistence.
Existing-output preflight rejects before build/browser; optimized Python is
rejected because it would disable assertions.

## C16 — Caller cancellation during a coalesced model load

Status: PASS — cooperative pipeline wait/cache cancellation contract;
prompt cancellation and real browser integration are not certified here.

Two deterministic tests reproduced aborted joined/cached callers receiving
handles before the fix. `src/runtime/load.ts` now checks the caller signal
before mutation/cache access, after mutation/disposal waits and after a joined load.
A cancelled joiner rejects without cancelling the live owner or disposing its
shared handle. Pre-aborted forced reload does not evict an existing handle.
The creator still owns the underlying load signal: its cancellation rejects
live joiners too; explicit retry creates a fresh usable handle. A third test
records that ownership/cleanup contract, without silent retries or transfers.
40 focused lifecycle/API/builtin tests passed; TypeScript passed. ESLint has
zero errors and two max-lines warnings. These are simulated generic runtime
tests, not real model/browser acceptance. Boolean-only `AbortSignalLike` is
checked cooperatively at await boundaries, not immediately during a pending
promise. Generation/forced-reload/disposal regressions stay in the same suite.
Final full regression: 1,418 passed / 33 skipped / zero failed; independent
read-only review passed after adding the post-forceReload-disposal guard.
The test environment emitted an existing TLS-disabled warning; no global TLS
settings were changed, and these results do not certify secure remote loading.

Post-commit package verification (2026-10-03): `npm run build`,
`node tests/smoke/node-import-smoke.mjs`,
`node tests/smoke/browser-root-import-graph.mjs` and
`node tests/smoke/packed-native-absent-smoke.mjs` passed on the current working
tree (including preserved paused Nemotron edits; not a clean-commit artifact).
The browser static closure covers 298 modules / 27 entry points. The isolated
offline packed consumer imports without native ORT, passes strict public
declaration checks and rejects an explicitly unavailable native backend before
asset/model creation, without silently substituting available WASM.
The retained fixture is `C:\Users\steam\AppData\Local\Temp\asrjs-packed-consumer-kZkUuL`.
These are packaging/import and unavailable-backend controls, not actual browser
inference, ASR parity, native execution or a packed-consumer cancellation test.

## C14 — Companion in-memory browser preview

Status: PARTIAL PASS — actual browser-demo UI boot/settings; model inference,
local file-picker/load, service-worker behavior and other companion UI remain open.

`node tests/smoke/companion-build.mjs <consumer-root> --serve` now serves the
existing in-memory build on an ephemeral 127.0.0.1 port. It writes no sibling
build/config/cache files. Only the explicit build outputs and bounded top-level
public files enter its immutable route map; directories and symlinks are not
recursively traversed, individual public files above 10 MiB fail. COOP/COEP,
exact Content-Length, WASM MIME and no-store headers support browser checks.
This is not a general Vite preview replacement or a file/model directory server.
Default build-only behavior is unchanged; three existing CLI negatives pass.
Syntax, focused diff checks and actual browser-demo source-alias build pass.

Actual Chrome154.0.8037.93 control (2026-10-03): fresh context, service workers
blocked and external requests rejected; title/Load Model/Idle state render,
Parakeet v3 selection keeps WebGPU encoder + WASM decoder, and Local folder
selection exposes the local-model panel. No page errors or ONNX weight fetches.
Four HTTP missing/traversal/source/malformed-path negatives return 400/404;
the main JS served bytes match build SHA256
`314ec796f0bdb0bb159778ec933a7ce42d83ee2aa0848c30c1e9a3d7cae4be53`.
Config SHA256 remains `05b797e775018c124b38ca310802453e5bb72854bcc7aebf68d525ffbd0dbbcb`.
The app attempts font/badge and Hugging Face metadata requests at startup/model
selection; this bounded control blocks them rather than claiming offline support.
Evidence is terminal output, not an automated persisted browser receipt.

Automation refresh (2026-10-03): the existing
`python tests/smoke/browser-demo-smoke.py --consumer-root <browser-demo-root> --output <new-receipt.json>`
already implements this scenario and verifies served HTML/application bytes
against its in-memory build receipt. Fresh Chrome154 run passed boot, v3 hybrid
selection, WASM selection, local-source panel and missing-folder recovery with
no page errors or weight requests. Font/badge/HF metadata requests remain
blocked and recorded; this is not a claim that the app is fully offline.
Receipt: `tools/data/results/harness/browser-demo-ui-closure-2026-10-03.json`.

Fixed a cleanup evidence defect: socket errors of any kind no longer count as
closed-port proof. UI and local inference smoke runners share a bounded Node
probe that accepts only explicit ECONNREFUSED; reachable ports, timeouts,
unknown errors and malformed/contradictory evidence cannot pass cleanup.
Two new regressions failed before implementation;8 UI-runner and10 local
lifecycle tests now pass, with Python syntax checks. The real UI receipt records
the stopped preview and `{closed:true, code:'ECONNREFUSED'}`. No sibling source
or published runtime changed; local inference was not rerun for this tool-only
helper reuse. Next cover real app-local artifact loading/cancellation and
transcription separately from the existing public-library lifecycle harness.
UI-selected hybrid settings do not prove graph placement or ASR correctness.

## C01 — Opt-in native ORT installation

Status: PARTIAL PASS — optional-peer metadata, resolution-isolated smoke and
actual packed-archive consumption. The offline package-manager comparison has
been attempted but fails on missing cached dependencies; completed installation
footprints remain UNVERIFIED.

Purpose: browser-only consumers should not need to install a native engine
that the runtime does not automatically register.

Baseline evidence as of 2026-10-01: `package.json` previously required
`onnxruntime-node@1.25.0-dev.20260228-6e72d31970`. Its local installed tree has
237 files / 223,239,822 bytes (212.9 MiB), including multiple platform assets.
This is an installed-tree measurement, not a published archive-size claim.
The architecture already keeps native CPU composition explicit and browser
imports free of eager native bindings.

Implemented: the unchanged pinned native version is an optional peer and a
development dependency. Offline lockfile-only refresh ran with install scripts
disabled; no package versions changed. Native loading remains lazy and explicit.
`npm run test:native-optional-peer` verifies metadata, built root/browser imports
with exact native package resolution blocked, unavailable probe, typed rejection
without fallback, and an installed CPU binding positive control. This process-local
simulation is not proof of a real packed-consumer installation or model inference.

Packed consumer control (2026-10-01): `npm run test:packed-native-absent`
creates a real local archive with scripts disabled, extracts it outside the
checkout and links only existing browser dependencies/type helpers. With native
ORT physically absent, actual package root/browser imports, explicit native
unavailable rejection and strict NodeNext public declarations passed (no
`skipLibCheck`). The public `runtime.loadModel()` control explicitly requests
native CPU with an available WASM backend also registered: it rejects with
`backend-unavailable` before model creation or asset resolution, without fallback.
No model inference or browser GPU execution is certified.
Fixtures are retained in the printed temporary directory for inspection.

Next bounded step: compare actual package-manager install footprints and repeat native Node/Bun local
example/resource inference controls before marking this task complete.
Do not silently install an engine on a missing-binding failure. Do not change
the public API or fork model executors by host.

Acceptance:

- A packed-package consumer without native ORT imports root/browser surfaces.
- Explicit native probe/load reports a clear missing-engine error.
- A native-present consumer runs the same Node/Bun local example and cleanup.
- Browser root import closure and package tarball smoke checks pass.
- Native resource lifecycle tests pass; affected docs explain opt-in setup.
- Measure comparable install/package footprints; do not equate lazy imports
  with lower install size. Package-manager or artifact checks unavailable in
  this environment remain explicitly UNTESTED.

Relevant existing checks: `tests/smoke/browser-root-import-graph.mjs`,
`tests/smoke/package-tarball-smoke.mjs`, `tests/package-tarball-contract.test.ts`,
`tests/native-cpu-backend.test.ts`, `tests/native-resource-smoke.test.ts`, and
`examples/nemo-tdt-native-local.mjs`.

### C01 resource refresh (2026-10-02)

The existing real native resource harness was rerun explicitly with Node
v26.2.0 and Bun 1.3.14 against the same small ONNX Identity fixture. Both
successful acquisition (two sessions) and partial acquisition failure (one
session) pass exact acquired/release-attempt/released counts. Repeated dispose
does not add releases. These are resource-only controls, not model inference,
numerical parity, or install-footprint acceptance. Small evidence pointer:
`tools/data/results/harness/native-resource-refresh-handoff-2026-10-02.json`.

Next portability gap: the prerequisite Vitest tests currently spawn
`process.execPath` (Node), and the manually used Identity fixture lives under
a local ONNX virtual environment. Add an explicit opt-in Bun prerequisite path
and a portable repository-owned fixture; retain requested-missing-fixture FAIL
and optional-unrequested SKIP semantics. Existing manual Bun evidence is valid
but does not make that automatic test path implemented.

### C01 portable automatic resource controls (2026-10-02)

Implemented the bounded portability slice: own 125-byte Identity graph and
synthetic vocabulary, temporary exclusive writes and hash/overwrite controls,
Node/Bun opt-in real executor checks, and the same four prerequisite negatives
under explicitly requested Bun. No local model/Python fixture dependency or
native engine install was introduced. Strict changed-test compilation passes.
Both requested hosts pass all 10 smoke checks; opt-ins unset gives 4 PASS /
6 SKIP. Explicitly requested Bun without an executable produces 5 FAIL, never
an accidental PASS/SKIP (negative-run JSON retained locally). Fixture unit
control passes; shared gate regression gives 8 PASS / 1 intentional SKIP.
See `tests/fixtures/README.md` for build prerequisite and opt-in commands.
This closes the automatic resource-fixture gap, not C01 install-footprint or
full model/native/browser acceptance.

### C01 offline install comparison harness (2026-10-02)

`tests/smoke/install-footprint.mjs` packs one actual built archive and installs
it into two fresh temporary consumers: package alone versus the same package
plus its exact optional native peer. npm is invoked without a shell, offline,
with install scripts disabled. Receipts hash tool/metadata/archive, preserve
both consumers, bind lockfile versions and require actual native resolution
identity. A failed install produces no footprint delta; existing receipts are
refused before npm runs. Seven tests (including actual CLI negative controls)
pass; strict changed-test compilation and build pass. Related gates give
38 PASS / 6 intentional SKIP; browser import graph and 1,401-file package
tarball contracts pass. This is not native/model/browser execution acceptance.

Actual Node v26.2.0/npm 11.13.0 offline runs fail both consumers with
`ENOTCACHED` for `@types/node@26.4.0`; partial module trees and npm cleanup
warnings are retained, not measured as successful installs. The packed archive
is 1,093,345 bytes compressed / 5,152,384 bytes unpacked. These archive sizes
are not installed footprint savings. See
`tools/data/results/harness/offline-install-footprint-handoff-2026-10-02.json`.

Next obtain an authorized cache-complete environment and rerun with a new
receipt. Do not switch to network fetching or lower success checks silently.
Even a future PASS measures script-disabled logical file bytes only: native
assets generated/downloaded by the native install script and allocated disk
blocks remain a separate footprint/runtime gate. C01 stays PARTIAL.

## C02 — Local text/JSON cancellation consistency

Status: PASS — bounded Blob read checkpoint fix; not universal load cancellation.

`BlobAssetHandle.readText()` previously ignored its request signal, unlike
`readBytes()`. Pre-aborted text reads returned success; abort during text reads
returned late success, and JSON reads could report a parse error instead of
cancellation. Three focused regression cases reproduced these failures.

Text reads now check the existing model-neutral abort helper before and after
`Blob.text()`. JSON reads inherit the same checks before parsing. Successful
reads and genuine Blob read errors retain their original semantics. This does
not interrupt the platform's Blob materialization or add a new public API,
model branch, native dependency, timer or framework state.

Verification: strict changed-test compilation, five Blob cases and all 39
related IO/provider/tokenizer/ORT-abort tests pass; build and browser root
import graph pass (297 modules / 27 entries). The built public IO provider
with real Blob objects passes two successful reads and four cancelled reads
on Node v26.2.0 and Bun 1.3.14 separately. Real browser/app cancellation and
full model inference were not run for this slice. Companion source/API changes
are unnecessary; broader cancellation integration remains a separate gate.

## C03 — Segmented transcription respects explicit finalization

Status: PASS — bounded controller lifecycle fix; live microphone/model and
sibling application acceptance remain separate.

`transcribeUtterance()` previously bypassed the finalized guard used by
`pushAudio()` and `flush()`. After explicit `finalize()` it could call the model
and reset/mutate merger and latency state. Two regressions failed before the
fix: submission after completed finalization and submission queued behind
finalization. The existing guard now runs inside the serialized operation,
after stale-generation rejection and before any utterance state changes.
Reset restores use; repeated segmented utterances before explicit finalization
remain supported. A segment's `reason: 'stop'` does not itself finalize the
whole controller. No public types, model mathematics or framework glue changed.

Verification: all 11 controller tests and 39 related realtime/browser-controller
simulation tests pass. Source and TS-example compilation, strict changed-test
compilation with `skipLibCheck=false`, built browser import closure (297 modules /
27 entries), and built public realtime API checks under Node v26.2.0 and Bun
1.3.14 pass. Those public checks use deterministic callbacks, not ASR inference.
The existing abort test had a narrower-than-public callback parameter; it now
accepts `AbortSignalLike` and asserts/narrows the actual controller-owned native
signal before listening. Runtime cancellation semantics are unchanged.

Full suite: 1,317 PASS / 33 SKIP / 0 FAIL in
`tools/data/results/harness/realtime-finalized-guard-full-suite-tls-2026-10-02.json`.
This run explicitly sets `NODE_TLS_REJECT_UNAUTHORIZED=1` only in the test child
environment; the first run inherited a disabled-TLS setting and is retained
separately. No process-wide or user environment setting was changed.

Affected consumer audit: streaming-demo routes detected segments through this
controller, resets it on capture start, and normally uses stop/flush rather
than explicit transcription-controller finalization. Its ordinary flow needs
no source migration. Direct consumers that call explicit `finalize()` must call
`reset()` before submitting another utterance, consistently with push/flush.
Playground and benchmark-demo have no matching source calls in the scoped
audit. Sibling build and real browser/microphone execution are UNTESTED here:
streaming-demo's build writes to its read-only checkout. No sibling edits were
made. Continue the Orukeet FFT boundary work alongside broader resource and
application gates; this lifecycle slice does not close those workstreams.

## C04 — Disposal during browser microphone/VAD startup

Status: PASS — deterministic lifecycle coverage only; no production fix was
needed for these paths. Real browser microphone/model acceptance stays open.

Added `tests/browser-controller-dispose.test.ts` through the public browser API.
Controlled Promise gates verify disposal while microphone acquisition is
pending and while VAD initialization is pending (both late success and late
rejection). Tests require immediate inactive state, VAD abort, exactly-once
capture stop, starter stop with `flush:false`, subscription cleanup, monitor
and starter disposal, and rejection of a subsequent start. The manual case
also verifies late chunk/error callbacks cannot mutate state or emit an
utterance, plus repeated disposal does not stop/dispose resources again.
No new model-specific branch, framework binding, dependency or API is added.

Verification: all three cases pass; strict standalone changed-test compilation
with `skipLibCheck=false` passes; the browser controller, browser realtime
starter and transcription controller suites together pass 27 tests. These use
injected capture/starter/monitor resources, not a real device, VAD model, ASR
or WebGPU. Source implementation is unchanged, so no companion migration is
required; sibling/browser hardware integration remains UNTESTED.

Next product gate: exercise the same lifecycle under the existing browser
loop with real capture/model resources where available. Keep Orukeet's open
mel2/frame1027 numerical investigation separate from these lifecycle checks.

## C05 — Capture cleanup survives release failures

Status: PASS — bounded resource-owner fix with deterministic injected resources;
real microphone/browser failure integration remains UNTESTED.

Four new regressions failed before the fix: an early disconnect skipped later
releases, a rejected context close skipped owned tracks, a concurrent stop
returned before cleanup completed, and startup cleanup swallowed its own failure.
Capture now shares one cleanup promise, attempts every independent release,
preserves a single original cleanup error, and aggregates multiple errors.
Startup failures retain both initialization and cleanup errors. Repeated stop
observes the same completed or rejected cleanup rather than rerunning releases.
Borrowed streams remain untouched unless track ownership is explicitly opted in.
PCM, sample-rate, resampling and model behavior are unchanged.

Verification: seven new cases and all 41 related capture/browser/realtime tests
pass; source/example compilation, strict changed-test compilation with
`skipLibCheck=false`, and built browser import closure (297 modules / 27 entries)
pass. These checks use injected resources, not live capture or model inference.
No companion edits or new dependencies were made. Callers must still handle
`stop()` rejection; broader controller disposal after a capture cleanup failure
is a separate acceptance gate, not proven by this owner-level fix.

The late starter-event gate is now covered by C06 below. Next run current-source
real-browser capture lifecycle using a root-owned server
or separately authorized sibling cache writes. Do not validate against the
stale sibling build or claim mocked tests as microphone/WebGPU acceptance.

## C06 — Ignore starter events once disposal begins

Status: PASS — bounded event-delivery lifecycle fix; actual VAD/browser and
model inference acceptance remain separate.

Two regressions failed before the fix: retained starter events changed status
and error state while disposal waited for VAD startup or capture shutdown.
The subscription callback now returns immediately when disposed, before reading
segment PCM, publishing an utterance, calling transcription or changing state.
Tests also invoke the retained callback after cleanup has completed. The guard
does not depend on microphone-active state: ordinary speech-detect stop sets
the mic inactive before flushing its last segment, which must still be delivered.
A positive flushed-stop test and existing active-segment transcription coverage
remain passing. No decoder, framework state or public contract changes.

Verification: two RED-to-GREEN regressions and one positive stop-flush case;
all 44 related capture/browser/realtime tests pass. Source/example compilation,
strict changed-test compilation with `skipLibCheck=false`, scoped lint and built
browser import closure (297 modules / 27 entries) pass. Tests use controlled
starter/capture resources and a real transcription-controller instance, not a
VAD model, physical microphone, ASR inference or WebGPU. No sibling writes.

Next product gate: current-source real-browser capture lifecycle, then cleanup
of controller-owned resources when capture stop itself rejects (C05 only proves
the capture owner's independent releases). Keep these separate from Orukeet's
unresolved feature and recurrent WebGPU parity gates; no model promotion.

## C07 — Real Chromium capture owner lifecycle

Status: PASS — real browser/fake WAV device, capture owner only. Controller,
VAD/ASR, physical device, soak and cleanup-failure browser gates remain open.

Added a root-owned Python Playwright runner and small JS probe importing the
built public `dist/browser.js` entry. No sibling Vite/cache writes or server
are required: page HTTP requests are locally fulfilled from `dist` plus the
explicit existing `pako` ESM dependency; unexpected requests fail. The initial
attempt stopped before capture because bare `pako` lacked an import map. Adding
the verified local package entry fixed harness resolution, without core edits.
Python/Playwright/Chrome were already installed; nothing was downloaded.

After refreshing the library build, Chrome 154.0.8037.93 passed two independent
capture sessions: four contiguous 256-frame chunks each, finite nonzero PCM,
48 kHz device/context and 16 kHz output, one actual AudioWorklet module load,
shared concurrent stop, context `closed` and owned track `ended`. No extra
chunks arrived during each 150 ms post-stop observation. This uses real browser
media/audio APIs, not injected fake audio nodes. WAV content may be changed by
browser capture processing; no exact PCM or speech-quality claim is made.

Evidence: `tools/data/results/harness/real-browser-capture-owner-reviewed-2026-10-02.json`
hashes the probe, runner, capture source, fixture and all 269 served resources
(267 library modules, probe and pako). Inputs were stable before/after the run.
Python/JS syntax and help checks pass. Existing-output preflight rejects before
browser launch and preserves the receipt hash. README provides build/replay
commands and requires a fresh build; source-to-build equivalence is not
automatically certified by the runner. The receipt is exclusively created.

Next: extend this actual-browser path to the public microphone controller's
stop/restart/dispose and pending-handoff lifecycle, preserving capture-owner
controls. Independently cover controller cleanup when capture stop rejects.
Orukeet feature/recurrent parity and model promotion are still unresolved.

## C08 — Real Chromium manual microphone controller lifecycle

Status: PASS — real browser/fake WAV device, explicit manual mode only.

Extended the offline runner with a separate controller scenario, preserving the
capture-owner control. The public controller, default starter and monitor are
real library instances. Instrumentation wraps actual capture handles and actual
AudioContext/AudioWorklet resources; it does not substitute fake audio nodes.

Chrome 154.0.8037.93 passed stop/flush with exact delivery of accepted captured
PCM, restart, active disposal without an extra utterance, stable disposed state
and rejection of start after disposal. A separate controlled handoff delays the
return of an already-live capture handle: disposal accepts no PCM and closes
the context and owned track after handoff. All three sessions loaded one actual
worklet, stopped through the controller once, ended tracks and closed contexts.
This is not a microphone permission-dialog race or exact source-WAV parity.

Evidence: `tools/data/results/harness/real-browser-manual-controller-2026-10-02.json`.
The separate capture-owner regression also passed:
`tools/data/results/harness/real-browser-capture-controller-control-2026-10-02.json`.
Receipts bind source/probe/runner/audio and served-module hashes. Fresh source
build, JS/Python syntax, CLI help and diff checks pass. No downloads or sibling
writes. Speech-detect/VAD, ASR, WebGPU, physical-device and cleanup-failure
acceptance remain open; no Orukeet/model promotion follows from these checks.

Next: prove controller-owned cleanup when capture stop rejects, independently
of capture-owner cleanup and model-parity work.

## C09 — Controller disposal after release failures

Status: PASS — controlled resource-failure regressions; real-browser healthy
manual lifecycle control also passes. Browser failure injection is untested.

A rejecting capture stop previously skipped starter unsubscription, monitor
disposal and starter disposal. Since the controller was already marked disposed,
a second dispose could not complete those releases. The new single-failure
regression failed at the missing unsubscription before the implementation fix.

Dispose now attempts each owned release independently in the original order,
clears buffered manual PCM and listeners, preserves a single error by identity,
and aggregates multiple failures in release order. Two tests cover a capture
failure and simultaneous capture/unsubscribe/monitor/starter failures, including
safe repeated disposal. Normal stop/flush behavior is deliberately unchanged.

Verification: 46 related capture/browser/realtime tests pass; independent review
reran all eight disposal tests. Source and example compilation, strict changed-
test compilation, scoped lint, formatting and diff checks pass. Fresh-build
Chromium manual-controller healthy control passed all three real capture sessions:
`tools/data/results/harness/real-browser-controller-disposal-control-2026-10-02.json`.
That browser receipt does not certify injected cleanup failures, VAD or ASR.

Next: separately specify and test normal stop when capture stop rejects; return
to Orukeet's earliest feature divergence without relaxing numerical acceptance.

## C10 — Logical stop after capture cleanup failure

Status: PASS — simulated failure matrix and real-browser healthy manual control.

Previously a rejected capture stop skipped capture-info reset, speech-detect
starter stop/manual flush, stopped status and monitor finalization. Four new
manual/speech-detect × flush true/false regressions failed before the fix.
Stop now independently attempts those operations, preserving the existing
flush option and surfacing errors afterward: single errors retain identity,
multiple failures are aggregated in operation order. Buffered manual PCM is
delivered once only when requested; flush false discards it. Speech-detect
still delivers the final starter segment when flushing despite capture failure.

Six focused cases cover that matrix, capture/starter/monitor simultaneous
failure, and disposal overtaking a pending failing stop. Review caught late
manual publication in the first implementation; that source edit was reverted
before adding the race guard. Disposal now suppresses buffered publication and
late status/monitor updates while queued disposal owns remaining cleanup.
The revised implementation passed independent review (20 focused tests).
Monitor mocks are cleared after startup to ensure stop finalization is
actually observed. All 52 related capture/browser/realtime tests pass. Source
and example compilation, strict changed-test types, scoped lint, format and diff
checks pass. Current-build Chromium healthy manual lifecycle passed three real
sessions: `tools/data/results/harness/real-browser-controller-stop-reviewed-2026-10-02.json`.
This is not browser cleanup-failure injection, physical-device, VAD-model or ASR
acceptance. Sibling consumer execution remains untested; no sibling edits.

Next: bounded package/runtime acceptance and cancellation/repeated-session
checks, while Orukeet's portable feature and recurrent browser gates stay open.

## C11 — Cached pipeline reuse after cancellation

Status: PARTIAL PASS — generic controlled-model regression; real-model host
reuse remains untested.

The existing cancellation test now also proves a fresh-signal transcription
succeeds after an aborted call, using one cached model and disposing it once.
Both calls use an explicit stable cacheKey: an AbortSignal-bearing request is
not automatically serializable, so expecting automatic cache reuse would test
the wrong contract. No production implementation changed.

Verification: 28 high-level API/lifecycle tests, scoped lint, formatting and
diff checks pass. Standalone strict test compilation reports a generic
TranscriptResponse mismatch in the unchanged createProbeRuntime mock at line
60; that mock typing remains open and this is not a clean typecheck claim.
This test does not certify real ONNX session recovery or browser GPU reuse.

Typing follow-up (2026-10-03): reproduced the line-60 factory error. The mock
session returned canonical output for every generic response flavor. It now
uses actual public input/options/conditional response types and returns native,
canonical or envelope output as requested. Three public loaded-handle cases
independently assert exact single/batch response shapes and once-only disposal.
Strict standalone NodeNext compilation with skipLibCheck disabled now passes;
31 high-level API/lifecycle tests, scoped lint, format and diff checks pass.
No production API or type was weakened. Cancellation reuse remains a controlled
model result, not real ONNX recovery acceptance.

## C12 — Global window timestamp contract

Status: PARTIAL PASS — controlled two-window public API regression.

Token-detail work in progress (2026-10-03): detailed window requests now retain
tokens, offset available seconds, assign global token indices and remap word
references without mutating window-local outputs. Only matching ID, text,
raw text and positive time intervals (within 1e-6 seconds) identify duplicates;
matching is one-to-one. Untimed or unmatched tokens inside audio overlap are
retained with `pipeline.windowing.token-overlap-unresolved`, not guessed away.
Missing word references are omitted with the same warning. Tokens can therefore
contain unresolved window evidence rather than a fully reconciled final sequence.
Model-local frame/step metadata is preserved, not converted to global indices.
The direct planner also resolves `returnTokens` via the existing canonical
detail resolver, preserving explicit detail precedence.

Current source typecheck, scoped ESLint, diff check and 22 focused windowing
tests pass. Final independent review approved the corrected ambiguity behavior.
The final full suite passed 1,427 tests with 33 skips, with TLS verification
enabled only in the child process. Formatting was confined to new code; existing
unrelated formatting and paused Nemotron changes were preserved.
Next: real model/browser token-detail acceptance and bounded long-audio memory
measurement. This conservative merge is not numerical ASR parity acceptance.

Real stock-v3 token-detail control (2026-10-03): the existing local Chrome
public API harness now has opt-in `--verify-window-tokens`. The same decoded
short speech fixture is placed in two disjoint, silence-padded 20-second windows.
Independent validation requires detailed output, two actual windows, contiguous
global token/word indices, finite global seconds in both windows, valid word-token
references and repeated identical detail. This is no-overlap real-model control,
not overlap reconciliation, numerical parity, Orukeet or long-session acceptance.

The first real run correctly failed: windowed metrics discarded actual component
backend identity. A failing-first regression exposed the same generic defect;
the shared string accumulator now retains encoder/decoder/joint backend identity
and emits `mixed` when windows disagree. Source typecheck, scoped lint and 23
windowing tests pass;11 Python harness tests and JS/Python syntax checks pass.
The second real Chrome154 run exited0 with two repeated 40-second results, each
22 tokens and16 words, WebGPU encoder/WASM decoder, awaited disposal, unchanged
inputs, no page errors/blocked requests and explicit ECONNREFUSED cleanup proof.
Receipts (preserved separately):
`tools/data/results/harness/local-window-tokens-v3-2026-10-03.json` (FAIL),
`tools/data/results/harness/local-window-tokens-v3-providers-2026-10-03.json` (PASS).
Next: finish independent harness review, full regression/static checks and
scoped commit, then actual overlap and bounded-memory controls.

Review checkpoint: runtime provider telemetry is committed separately as
`2641f9e`, with 1,428 full-suite PASS /33 SKIP,23 targeted tests, typecheck and
scoped lint. The harness extension remains uncommitted pending two regressions:
reject oversized decoded fixtures before allocating the doubled window buffer;
reject in-range word-token references whose token timing belongs to the wrong
window. Existing successful real receipt is preserved, but these validation
holes must close before accepting/committing the harness extension.

Resolved checkpoint (2026-10-03): both review regressions now pass. The fixture
rejects empty or over-19-second mono PCM before allocating a fixed 40-second
repeated buffer (2,560,000 bytes); this does not bound the original media decode
allocation or certify VRAM/soak behavior. Word-token references must also match
the word's timing interval, rejecting an in-range reference into another window.
Independent read-only review approved the corrected code. Twelve Python runner
tests and two Node fixture tests pass. The corrected actual Chrome hybrid run
exits 0 with receipt
`tools/data/results/harness/local-window-tokens-v3-bounded-2026-10-03.json`:
two repeat transcriptions each retain 22 tokens, 16 words and two windows, with
identical canonical detail, WebGPU encoder/WASM decoder, no page errors or
blocked requests, unchanged input hashes, awaited disposal and ECONNREFUSED
preview closure. This remains stock-v3 no-overlap smoke, not Orukeet parity,
overlap reconciliation, timestamp accuracy or a long-session memory benchmark.

Actual shared-speech overlap diagnostic (2026-10-03): opt-in
`--verify-window-tokens --window-token-overlap` places the short speech once at
15 seconds in a bounded 34-second PCM fixture. Two 20-second windows overlap
by six seconds, sharing the complete speech; source PCM must be at most five
seconds. A failing-first Node fixture test now passes, together with all three
fixture tests and 13 Python runner tests. CLI rejects overlap without requested
token detail; the validator requires the explicit overlap policy and still
checks canonical indices, timing/reference associations and repeated detail.
Receipt `tools/data/results/harness/local-window-tokens-v3-overlap-2026-10-03.json`
exits 0: both actual Chrome hybrid results contain the single expected phrase,
eight words and 15 tokens, with unchanged inputs and successful cleanup.
Status is PARTIAL PASS: word/text duplication control passes, but token overlap
remains unresolved. Four context-shifted occurrences remain beyond the 11-token
phrase; both runs emit `pipeline.windowing.token-overlap-unresolved`. Global
token indices therefore do not imply a unique chronologically ordered token
sequence. Do not hide this evidence by fuzzy timing deduplication or call it
complete reconciliation. Next: review this diagnostic, then determine a safe
canonical token ownership contract using the captured real ambiguity before
changing generic merging; long-session memory measurements remain separate.

Review correction: two windows alone did not prove overlap was applied. A
failing-first negative removes the overlap warning from an otherwise valid
two-window result; the diagnostic now requires the captured ambiguous-overlap
warning in both repeats. All 13 Python tests pass. The existing real receipt
passes the tightened validator by offline post-processing; this is not a new
browser run with the updated runner hash. This fixture deliberately certifies
the unresolved-overlap diagnostic, not all future successfully reconciled output.
Public transcript JSDoc now explains token-array indices, known word references
and why retained ambiguous tokens must not be joined to reconstruct final text.
No token selection/deduplication behavior or model mathematics was changed.

Added word and segment-only cases through root-exported
`transcribeWithWindowing`. Four seconds of synthetic PCM produce two explicit
two-second windows without overlap: local times 0.25..1.25 become
0.25..1.25 and 2.25..3.25 in the final canonical result. The model-local
result objects retain their original times. Segment-only detail does not
invent words. No production mathematics, API or model code changed.

Verification: 16 windowing/stage tests, scoped ESLint, diff check and strict
standalone NodeNext test compilation with skipLibCheck disabled pass.
Still open: overlap-retained timestamp checks, token-detail behavior and
real supported-model/browser/companion-demo acceptance. This bounded
simulation does not complete P08 or certify timestamp quality of any model.

Overlap regression (2026-10-03): a two-window case initially failed with
`alpha shared alpha shared omega` and a timestamp rollback. `windowWords`
already included pending words, but both accumulation sites concatenated that
history again. Reused the existing interval-aware merge at those two sites;
the unchanged acceptance now retains the longer shared word at 1.6..2.0 s,
and a three-window extension retains earlier history and final tail times.
Source typecheck, strict test typecheck, scoped lint and 18 targeted tests pass.
The full suite before the third-window extension passes 1,378 / 33 SKIP,
including a repeat with inherited TLS-disable removed only in that child.
Build/browser/real-model/companion acceptance is not claimed; dist is untouched
while the live DFT diagnostic binds compiled asset hashes.

Browser control (2026-10-03): Chrome 154.0.8037.93 passed all four synthetic
public-root cases (two/three overlap windows and word/segment-only offsets).
`tests/smoke/window-timestamps-browser.py` bounds the asynchronous probe to
60 seconds and checks served JS bytes against the requested local build,
with build/tool hashes before and after execution. Wrong-build and existing
receipt negative controls fail without writing a success receipt. Evidence:
`tools/data/results/harness/window-timestamps-browser-bundled-2026-10-03.json`.
The isolated tsc output was bundled with installed esbuild for browser package
resolution; raw ESM correctly fails to resolve bare `pako`. Main dist remains
untouched. A directly owned, quiet loopback HTTP server was used after the
external shell-based helper and route.fetch interception proved unsuitable;
response hashing now observes original browser requests, not proxy requests.
This certifies synthetic windowing in a real browser only, not model accuracy,
GPU execution, token detail or sibling consumers. Those gates remain open.

## C13 — Compact public-root consumer bundles

Status: PARTIAL PASS — source consumer bundle closure, not inference acceptance.

Run `node tests/smoke/compact-browser-bundle.mjs` for a stdout-only receipt and
`node --test tests/smoke/compact-browser-bundle.test.mjs` for four regressions.
Installed esbuild (through the existing development toolchain) bundles two
public-root imports in memory: createSpeechRuntime is 22,074 bytes and
transcribeWithWindowing is 17,971 bytes, minified browser ESM. Each has a
32,768-byte regression budget; this is not a universal application size target.
Both retain no model/preset, ORT engine or native backend code and no external
imports. Metafile visited inputs are not confused with retained bytes.

The tool hashes JS/TS bytes supplied to esbuild and rechecks them after both
builds, plus root package metadata and its own source. Receipts include actual
esbuild version and bundle hashes. This is not exhaustive dependency-resolution
metadata provenance. Nothing is emitted to dist, and no dependency is installed.
Negative tests reject family/engine/native retention, Windows path variants,
external imports, exceeded budgets and missing/malformed output evidence.
Syntax, four tests and formatting pass. These tests use Node's test runner;
the existing Vitest glob does not include them.

Still open: published-package consumer bundles, startup cost and application
integration. No production optimization, model inference or WebGPU claim.

## C14 — Sibling consumer build acceptance

Status: PARTIAL PASS — actual in-memory Vite builds; browser workflows remain open.

`node tests/smoke/companion-build.mjs N:\github\asrjs\browser-demo` and the
same command for `streaming-demo` pass with their installed Vite 7.3.1 and
current public aliases into this checkout's src. Browser-demo transforms 307
modules, streaming-demo 344; transcription and VAD worker bundles are included.
Browser-demo CSS is 29,900 bytes with the correct consumer working directory,
not the earlier 5,852-byte wrong-cwd diagnostic. Streaming-demo CSS is 30,915
bytes. These are build measurements, not rendering or performance acceptance.

The normal Vite config bundler writes a temporary module into the sibling's
node_modules; its diskless runner instead fails on these configs' \_\_dirname.
The new probe compiles captured config bytes into in-memory ESM, supplies the
original absolute dirname, resolves packages from the consumer, and disables
Vite config reloading/build writes. Its process working directory is restored.
Only object configs with a vite.config.js entry are covered. Arbitrary custom
config/plugins are executable code: review them before using this probe; the
write:false option is not a security sandbox for plugins.

Stdout receipts bind config/package/tool hashes, actual Vite version and all
in-memory output byte counts/hashes, not exhaustive application source identity.
No dependency installation, sibling source edit or dist rewrite was performed.
Current checkout includes preserved paused Nemotron changes; this is not a
clean-commit release build. Browser-demo's existing config hash stays
05b797e775018c124b38ca310802453e5bb72854bcc7aebf68d525ffbd0dbbcb;
streaming-demo's stays ec887bf5b4c056e8a4181dd37b771ee422ebbde6cbb3df551087df414d4744de.
Browser-demo retains its classic coi-serviceworker script warning, and both
report stale Browserslist data; no package update was made to silence warnings.

Syntax, formatting and three prerequisite negatives pass using
`node --test tests/smoke/companion-build.test.mjs`: missing root, surplus
arguments and requested missing config fail without a PASS receipt.
These Node tests are separate from the existing Vitest glob.
Next: real browser model management/capture/hybrid workflows and other affected
companions. These builds do not complete those integration gates.

Additional consumer builds (2026-10-03): the same reviewed-config in-memory
harness passes for playground (304 modules, main JS720,714 bytes, CSS33,392),
vad-demo (344 modules, main JS936,946 bytes, CSS30,915), and benchmark-demo
(288 modules, main JS692,208 bytes, CSS4,428). Installed Vite7.3.1, same tool
SHA25649635ee0d807c15573bf8cae968739cba1ef1e4fab9c29624a56d90fb4173357.
Actual config SHA256s respectively:
`518e81c022fb9d320ad8e9683a0bb998292a919c10d3cda38e176b2efe4396d8`,
`a044208026c577751638462eb785194ba25366023d2f0d2a9bace8a3fd2d0f89`,
`9a4242d6740dcaec1f2154f0e5b86bdf87d6eb22605a02a8589537ab5838f197`.
Each command exits0 and binds config/package/tool/output identities in stdout;
no sibling output files or source edits. Stale Browserslist warnings persist
where reported, benchmark retains the classic coi-serviceworker warning.
These broaden source consumer build coverage, not browser app acceptance.

Independent actual-Chrome manual-controller check (2026-10-03): existing
`capture-lifecycle-smoke.py --scenario microphone-controller` passes against
the current untouched dist with the pinned ItsLifeJim WAV fake device.
Receipt: `tools/data/results/harness/manual-controller-current-dist-2026-10-03.json`.
Two stop/flush sessions each accept4 chunks; the active-disposal session accepts0.
All three close their AudioContexts and end owned tracks, with one stop call
and one worklet load each. The runner also checks delayed-handle disposal.
This is REAL_BROWSER_FAKE_DEVICE_MANUAL_CONTROLLER, not a sibling app, physical
microphone, ASR/VAD/WebGPU, or fresh source-to-build equivalence certification.
Dist was deliberately not rebuilt while the live DFT run binds compiled bytes.
Actual sibling model-management/capture/hybrid workflows remain open.
