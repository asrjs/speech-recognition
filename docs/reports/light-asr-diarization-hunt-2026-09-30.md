# Browser speech candidate review — 2026-09-30

## Decision and scope

Continuation clarification — 2026-10-08: the user explicitly resumes Parakeet
TDT distills **and quantized/faster derivatives**, including Redux/Ultra benefit
investigation even when parameter count is unchanged. Parakeet CTC is on hold;
new unrelated NVIDIA work remains excluded. Use the active P10 checklist in
[PARAKEET_VARIANT_TASKS.md](../PARAKEET_VARIANT_TASKS.md), routed from the agent
entrypoint. Research ranking and published scores below retain their dated scope;
this clarification is not model acceptance or a permission to adopt low-bit defaults.

This is a research/triage decision, **not** a port or a benchmark by this
library. The differentiator of `@asrjs/speech-recognition` is doing the
browser/WebGPU/WASM port ourselves. An existing ONNX, GGUF, Core ML, or Node
runtime is evidence and a comparison point, not a prerequisite for selecting
a candidate and not proof that its graphs work in our browser runtime.

The next bounded ASR feasibility slice is Orukeet on the existing Parakeet
TDT v3 implementation, followed by genuine Parakeet distillations and new
non-NVIDIA ASR candidates. The user's latest scope excludes new NVIDIA
model work beyond existing TDT v2/v3 for now: Nemotron 3 Diarization below
is retained as historical research, not an active queue item. Moonshine v2
remains a future candidate. Nemotron 3.5 Streaming ASR is paused at partial
browser parity; its own export and native/WASM evidence remain reusable. The
candidate queue must not displace public API, streaming, examples, model
management, memory, and lifecycle work when those provide more user value.

Priorities combine expected quality, language coverage (including Turkish),
browser port feasibility, runtime size/speed, license, and effort. Published
model-card scores are claims to test, not our acceptance evidence. A candidate
is promoted only after official-weight provenance and official/reference
inference, our own optimized fp32 graphs, stagewise numerical/token/text
parity, fp16 revalidation if useful, real browser WebGPU/WASM execution, and
measured load/latency/memory/quality on held-out audio. Integer quantization
is a separate evidence-gated decision, never the default because a third-party
INT8 or 4-bit file exists.

## Research ranking (active scope is stated above)

| Priority | Candidate                                                                                  | Fit and disposition                                                                                                                                                                                                                                                                                                                                                                                | First discriminating experiment                                                                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | [Orukeet](https://huggingface.co/oruk/orukeet)                                             | High reuse, 627M Parakeet TDT v3 weight variant, 25 European languages but **no Turkish**. Own browser port/preset, not a new `src/models` family. Weight license CC BY-SA 4.0; code MIT.                                                                                                                                                                                                          | Pin `.nemo`; compare exact model config, 128-mel preparation, vocab/tokenizer, graph I/O and decoder semantics with our v3 path. Capture official NeMo features/logits/tokens/text; export own fp32 graphs. |
| 2        | [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)             | Newly released, distinct from Nemotron 3.5 Streaming ASR: ~100M, streaming/offline, up to eight speakers. Strong browser diarization opportunity, not an ASR transcript model. Official license is OpenMDW-1.1; review obligations before distribution. A [community ONNX/browser experiment](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX) is a reference, not parity proof. | Capture official diarization output on labeled overlap/turn fixtures; audit graph ops, state/cache and WebGPU memory; compare speaker-turn semantics before choosing the public API.                        |
| 3        | [Moonshine v2 / Streaming](https://github.com/moonshine-ai/moonshine-v2)                   | Small English streaming models and a portable ORT/C++ reference make this attractive for low-latency browser UX. Different language weight licenses require separate review; no Turkish in the verified English offerings.                                                                                                                                                                         | Reproduce official streaming partial/final behavior, inspect graph/export path and benchmark in real browser; preserve canonical transcript semantics.                                                      |
| 4        | [Fun-ASR MLT Nano 2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)          | Keep the older genuine export-spike candidate. ~800M/31-language multilingual model, but the published list does not establish Turkish support. Larger and less direct reuse than Orukeet. Distinguish MLT Nano from the smaller-language [Fun-ASR Nano 2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512).                                                                               | Official-engine output and weight/license audit, then narrow browser graph and memory feasibility.                                                                                                          |
| 5        | [ARK-ASR 0.6B](https://huggingface.co/Edge0/ark-asr-0.6b-int8-onnx)                        | Split audio encoder/adapter/autoregressive decoder resembles a hybrid execution plan, but published INT8 package targets Python ORT/CUDA and its 19 listed languages exclude Turkish. Package is not proven small in browser memory or WebGPU-compatible.                                                                                                                                          | Compare official full-precision checkpoint and INT8 package contracts; inspect op support, KV growth, peak browser memory before any port commitment.                                                       |
| 6        | [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)   | Apache-2.0, strong multilingual quality claim, but ~2B and its 14 listed languages exclude Turkish; access/runtime terms and browser memory need checking. Existing third-party ports make it an adapt/compare target rather than the next own export.                                                                                                                                             | Audit actual artifact access and reference decoder behavior; benchmark browser budget before prioritizing.                                                                                                  |
| 7        | [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Valuable causal-streaming design, Apache-2.0 and 13 languages, but no Turkish and several billion parameters: likely outside mainstream browser memory budget. Existing [community ONNX](https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX) is only a feasibility precedent.                                                                                                | Define target device/VRAM and run actual browser load, chunk latency, KV-memory and output-parity gate.                                                                                                     |

The Orukeet developer says it preserves v3 architecture, tokenizer, and
inference operators, with Gabor kernels stored as ordinary convolutions; this
supports low engineering-risk **hypothesis**, not a weight-swap guarantee.
Its final checkpoint changed other tensors. The published FLEURS pooled
comparison and LibriSpeech test-other claims are **not clean held-out gains**:
the [data provenance](https://github.com/Oruk-AI/orukeet/blob/main/docs/data-and-licenses.md)
lists FLEURS in training as well as evaluation, while the
[model card](https://huggingface.co/oruk/orukeet) says test-other contributed
to final adaptation/checkpoint selection. Require fresh source-separated
non-training sets and document the limitation. Our v3 catalog's language
metadata also needs audit against official v3/Orukeet lists before adding a
preset; never infer Turkish support from a generic multilingual label.

## Related models and components, not immediate ASR ports

- Parakeet v3 variants: [Swedish Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv)
  and [Basque fine-tune](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque)
  show locale-specific reuse, not a Turkish model. The Basque tokenizer/graph
  contract differs and needs independent validation. A
  [community fp16 export](https://huggingface.co/shahrukhx01/parakeet-tdt-0.6b-v3-fp16-onnx)
  is an artifact comparison, not evidence of a new distilled model. The
  `Nordln/...v3-int8` identifier in the supplied notes was not verified to an
  exact authoritative repository, so it stays unranked. Search for an
  independently evaluated Turkish fine-tune rather than claiming these solve
  Turkish.
- [Desert Ant Align](https://github.com/Desert-Ant-Labs/desert-ant-core/blob/main/docs/models/align.md)
  is a forced word-alignment component, not an ASR or diarization model. Its
  current documented JavaScript path is Node-oriented and does not provide a
  browser build. **That is a possible porting opportunity, not a reason to
  exclude it.** Evaluate LiteRT-to-ORT/ONNX conversion, timestamp parity,
  memory and source/commercial license before a browser alignment spike.
- [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
  [Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
  and CAM++ remain comparison/component paths. They are not interchangeable:
  full diarization requires VAD/segmentation, speaker embeddings, clustering
  or streaming speaker assignment, plus overlap policy. The user note's
  “no new diarization model” is superseded by Nemotron 3 Diarization.
- [Phonon-1 Micro](https://huggingface.co/FermionResearch/Phonon-1-Micro)
  is an English-focused low-bit Qwen ASR derivative. Its 285 MB size and
  self-reported scores warrant watching, but do not establish a portable
  fp32/fp16 browser ONNX path or Turkish utility. Keep below the near-term
  queue; a community Parakeet 4-bit score is not our cross-backend result.
- Moonshine's small English streaming path is the strongest additional
  browser-first ASR candidate from this review; Nemotron 3 Diarization is the
  strongest additional diarization candidate. Continue dated surveys rather
  than treating this list as exhaustive.

## Next implementation handoff

1. Run a narrow Orukeet config/asset audit against
   `src/presets/parakeet/` and `src/models/nemo-tdt/`; record revision,
   hashes, license and exact mismatches. Do not download/publish weights to
   the Git repository.
2. If compatible, obtain pinned original `.nemo` weights and official NeMo
   inference for fixed plus fresh non-training audio. Capture features,
   encoder, decoder/joiner, token/duration decisions, timestamps and text as
   applicable; use explicit tolerances and find the earliest divergence.
3. Produce our own browser-oriented fp32 split ONNX export, optimize only with
   parity preserved, then fp16 if beneficial. Test native ORT, ORT Web WASM,
   real WebGPU and hybrid placement; record load time, warm latency, RTFx,
   GPU/CPU/WASM memory and accuracy separately.
4. Integrate only after those gates: thin Parakeet-branded preset, coherent
   root API, model management, streaming/long-audio behavior and at least one
   relevant sibling demo (`../streaming-demo`, `../browser-demo`, or
   `../playground`). If blocked, preserve artifacts and move to the next
   highest-value product or model slice.

No model weights were downloaded, no code was ported, and no browser parity
was established by this research update.
