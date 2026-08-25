# Candidate ASR artifact boundaries

Date: 2026-08-25
Workspace: `N:\github\asrjs\speech-recognition`

This note records what is currently actionable for the next non-Whisper
backends. It intentionally does not add a preset or runtime implementation
without a local, reproducible model artifact.

## Continuation audit (2026-08-25)

The FireRed source checkout was re-audited after the upstream repository added
its model-release links. `N:\github\ysdede\FireRedASR2S` still contains source,
runtime code, and example assets but no `pretrained_models` checkpoint bundle;
the local `N:\models` and Hugging Face cache likewise contain no FireRed ASR2
checkpoint, `cmvn.ark`, or Qwen3-ASR snapshot. The only cached Qwen model found
is an unrelated text-instruction model. No model hub download or third-party
ONNX conversion was performed.

The checked-in FireRed capture/export/verify scripts and Qwen reference capture
script pass Python 3.11 bytecode compilation and `--help` smoke checks. They
remain ready for the first approved local artifact, but this audit does not
create a correctness claim without native reference outputs.

## FireRedASR2-AED

The local reference source is `N:\github\ysdede\FireRedASR2S`. The AED model
is a model-specific Conformer encoder plus Transformer decoder with a CTC head
used for token timestamps. Its feature contract is also model-specific:

- 80-bin Kaldi fbank, 25 ms frame length, 10 ms frame shift, `snip_edges=true`;
- optional Kaldi CMVN from `cmvn.ark`;
- encoder inputs `padded_input [batch, frames, 80]` and
  `input_lengths [batch]`;
- decoder output token IDs come from the AED vocabulary; timestamp refinement
  uses the CTC branch and the encoder subsampling factor.

The checked local checkout contains the implementation and TensorRT-oriented
encoder export helper, but no FireRed ASR checkpoint files in its example
`pretrained_models` directory. The package therefore has FireRed VAD support,
not an artifact-backed FireRed ASR runtime.

The official model source is
[FireRedTeam/FireRedASR2-AED](https://huggingface.co/FireRedTeam/FireRedASR2-AED).
There are also public third-party ONNX conversions, including
[42ailab/FireRedASR2-AED-ONNX](https://huggingface.co/42ailab/FireRedASR2-AED-ONNX)
and a batch-oriented int8 conversion at
[Kn90688/FireRedASR2-AED-int8-batch-onnx](https://huggingface.co/Kn90688/FireRedASR2-AED-int8-batch-onnx).
Neither conversion has been downloaded or accepted as a correctness artifact
for this package. Their model cards are useful for understanding the required
encoder/decoder/CTC file boundary, not as a substitute for upstream parity.

### Required FireRed implementation sequence

1. Obtain one approved local checkpoint or ONNX bundle and record its SHA-256,
   source revision, vocabulary, CMVN, feature parameters, and export options.
2. Add a FireRed-specific Python reference runner that emits token IDs,
   decoded text, batch lengths, beam settings, CTC timestamps, and latency.
3. Export/validate encoder, AED decoder, and CTC timestamp graphs separately.
   Check dynamic batch with mixed lengths; padding must not alter a shorter
   utterance's output.
4. Compare PyTorch/reference, native ONNX Runtime, WASM, and WebGPU on the
   same fixture set before adding `src/models/firered-aed` and a preset.

The batch boundary is important: a public int8 conversion reports that its
encoder mask and decoder batch dimensions were changed specifically to avoid
padding pollution and mixed-length repetition. That behavior must be proven
in our own parity fixture before exposing a `batchSize` option.

## Qwen3-ASR-0.6B

The current candidate is
[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), not the older
Qwen2-Audio material in earlier notes. The official model card describes a
BF16 speech model with language identification, offline/streaming inference,
batch inference, and Turkish support. Its reference package is `qwen-asr`,
with Transformers and vLLM backends.

No Qwen3-ASR snapshot is present in the checked local model/cache directories,
so there is currently no ground-truth output to use for an ONNX export or
WebGPU claim. Qwen3-ASR is an audio-conditioned language model rather than a
Whisper 4-graph encoder/decoder; it must not be routed through
`whisper-seq2seq` merely because both produce token text.

### Qwen implementation update (2026-08-26)

The artifact-gated browser integration is now implemented under
`src/models/qwen-asr`. It uses the existing public browser ONNX conversion as
an explicit source and does not download or commit its weights. The family
implements the Qwen frontend, ByteLevel BPE, audio-token-mask trimming,
prefill/KV-step greedy decode, canonical transcript mapping, and deterministic
mock graph tests. It intentionally has no preset, no fake scaffold transcript,
no batch option for the published batch-1 graph, and no Whisper-style
timestamps.

The implementation is a runtime boundary, not an end-to-end quality claim.
There is still no local Qwen reference snapshot or browser run in this
workspace. See
[`qwen3-asr-webgpu-2026-08-26.md`](qwen3-asr-webgpu-2026-08-26.md) for the
artifact contract and verification gate.

### Canary 180M Flash browser verification (2026-08-26)

Canary is now the next fully exercised browser candidate rather than a
scaffold. The existing `src/models/nemo-aed` implementation was run against
the approved local smoke bundle at
`N:/models/onnx/nemo/canary-180m-flash-smoke` with the JavaScript feature
extractor and FP16 encoder/decoder graphs.

The fixed 11-second `jfk-short.wav` fixture produced exact text in both local
WASM and browser WebGPU. The browser run was cross-origin isolated and
reported 30.355 ms preprocessing, 413.935 ms encoding, 774.925 ms decoding,
1,219.88 ms total, and 9.0173x RTFx. It emitted 44 tokens over 45 decode
iterations with no warnings. This is a smoke/reference result, not a claim of
model-wide WER; broader language and quantization parity remain separate
gates.

The repeatable browser entry point is
`examples/demo/public/canary-smoke.html`. It is deliberately artifact-gated:
no model weights are downloaded or committed, and the Vite file allow-list
only exposes the repository and `N:/models`.

### Required Qwen verification sequence

1. Capture an approved `qwen-asr` reference environment and a fixed fixture
   manifest containing audio identity, requested language, detected language,
   text, timestamps, and batch order.
2. Inspect the official model's audio frontend, multimodal input packing,
   generation prompt, cache shapes, and stopping policy. Freeze those as a
   Qwen-specific artifact manifest before writing a runtime bridge.
3. Export the smallest independently testable graph boundary first, then run
   token/logit parity in native ORT. Add dynamic batch only after batch-1
   parity is exact and mixed-length attention masks are tested.
4. Run the implemented runtime against the approved artifact in native/WASM
   first, then WebGPU. Keep timestamps as a separate alignment contract; do
   not infer Whisper-style timestamp tokens.

## Tooling now available

The repository now contains local-only reference and conversion helpers:

- FireRed capture records checkpoint hashes, feature lengths, encoder states,
  decoder teacher-forced logits, token IDs, text, and optional CTC timestamps.
- FireRed export emits separate encoder, full-prefix AED decoder, and CTC ONNX
  graphs. The full-prefix decoder is a parity boundary, not yet the final
  cached browser decoder contract.
- FireRed ONNX verification compares encoder states/lengths/masks, decoder
  logits, and CTC logits against the captured reference.
- The Qwen capture helper uses the official local qwen-asr Transformers API
  with offline flags and preserves language, text, timestamps, and batch
  order.
- The shared Node ONNX auditor hashes local bundles, reports external-data
  candidates, and loads each graph through native CPU ORT.

These tools intentionally do not download a checkpoint or promote a
third-party conversion to a supported artifact.

## Current decision

The next Qwen change should be driven by an approved local artifact and a
reference transcript fixture. The family is executable when that source is
provided, but it must not be promoted to a preset or called quality-verified
until the reference/browser matrix passes. FireRed remains artifact-gated and
should follow the same sequence rather than receiving another empty model
class.
