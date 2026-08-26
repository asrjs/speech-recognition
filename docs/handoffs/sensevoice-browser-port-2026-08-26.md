# SenseVoiceSmall browser port handoff

Date: 2026-08-26
Workspace: `N:\github\asrjs\speech-recognition`

## Candidate decision

SenseVoiceSmall is the next bounded prototype after the verified Parakeet v3
and Canary paths. It is not a Turkish replacement for Parakeet: the released
SenseVoiceSmall checkpoint covers Mandarin, Cantonese, English, Japanese, and
Korean. Its value is a materially different browser trade-off: one encoder
pass, CTC greedy decoding, language identification, inverse text
normalization, emotion, and audio-event tags without an autoregressive decoder
loop.

Primary references:

- Upstream: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- Browser-relevant ONNX contract:
  [OpenVoiceOS/sensevoice-small-onnx](https://huggingface.co/OpenVoiceOS/sensevoice-small-onnx)
- Related ONNX runtime reference:
  [onnx-asr](https://github.com/istupakov/onnx-asr)
- Candidate ONNX collection:
  [DakeQQ/Automatic-Speech-Recognition-ASR-ONNX](https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX)

The model weights use the FunASR Model Open Source License, not Apache-2.0.
The source runtime and model artifact must therefore remain separately
attributed, and no model weight is downloaded or committed as part of this
port.

## Frozen graph contract

The `onnx-asr` conversion exposes:

| Input/output | Contract |
| --- | --- |
| `features` | float32 `[B,T,80]` Kaldi fbank |
| `features_lens` | int64 `[B]`, valid fbank frames |
| `language` | int64 `[B]`; `auto=0`, `zh=3`, `en=4`, `yue=7`, `ja=11`, `ko=12` |
| `textnorm` | int64 `[B]`; `withitn=14`, `woitn=15` |
| `logprobs` | float32 `[B, ceil(T/6)+4, 25055]` |
| `logprobs_lens` | int64 `[B]` |

The graph folds the FunASR low-frame-rate stack (`lfr_m=7`, `lfr_n=6`),
CMVN, and four prompt frames into the ONNX graph. The JavaScript processor must
only reproduce the 16 kHz Wespeaker/Kaldi fbank: 400-sample Hamming window,
160-sample hop, 512-point FFT, 80 bins, dither 0, per-frame DC removal,
per-frame preemphasis 0.97, and `snip_edges` behavior.
The graph pads each item with its last valid frame, so mixed-length batch
results must be compared against equivalent single-item runs.

## ASR.js implementation boundary

Add `src/models/sensevoice/` as a model family, rather than claiming the
existing `lasr-ctc` implementation is generic enough. The family should own:

- the SenseVoice fbank processor and CMVN/shape contract;
- prompt language and ITN option mapping;
- SentencePiece vocabulary loading and prompt-token filtering;
- ONNX Runtime Web execution for WASM and WebGPU;
- batch padding and `logprobs_lens` trimming;
- CTC collapse with token spans and canonical transcript mapping;
- native metadata for detected language, emotion, and audio event;
- direct and Hugging Face artifact sources, progress, cleanup, and limits.

The shared CTC collapse/timing utilities may be reused. Do not add a decoder
loop, Whisper timestamps, or a `batchSize` option until mixed-length parity is
proven. VAD/long-form chunking remains a runtime concern; the model graph is
short-clip oriented even though the upstream FunASR wrapper can compose VAD.

## Implemented runtime slice

The artifact-gated `src/models/sensevoice` family now includes the configured
80-bin frontend, prompt mapping, SentencePiece vocabulary loading, direct and
Hugging Face ONNX sources, WASM/WebGPU ORT execution, single-item canonical
transcription, and a true padded `transcribeBatch` graph path. The batch path
uses one graph invocation and trims each item with `logprobs_lens`. It is
exposed on the family-specific `SenseVoiceBatchSession`; the package-wide
generic batch contract still needs to be designed around mixed model families.

The contract layer and runtime registration are covered by local tests. No
real SenseVoice artifact is present in the approved local model directories,
so no quality or browser parity result is claimed yet.

The first processor draft used the existing MedASR Hann/global-preemphasis
defaults. That was rejected after comparison with the onnx-asr
`KaldiPreprocessorNumpy(name="wespeaker")` source; SenseVoice now selects the
Hamming, frame-local DC/preemphasis path explicitly while existing MedASR
callers retain their defaults.

## Verification gate

Before adding a preset, an approved local artifact must pass:

1. reference Python output on fixed English and at least one supported
   non-English fixture, including prompt IDs and metadata tokens;
2. frontend statistics and graph input parity;
3. native ORT, WASM, and WebGPU output/token parity;
4. single-item versus mixed-length batch equivalence;
5. silence, repeated inference, EOS, punctuation/ITN, and long-audio window
   tests;
6. model size, load time, peak memory, latency, RTFx, and WebGPU provider
   assignment measurements.

Until that artifact is explicitly available, SenseVoice remains a documented
prototype target, while Parakeet v3 remains the Turkish production reference
and Canary remains the currently verified new WebGPU backend.
