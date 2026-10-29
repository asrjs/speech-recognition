# HF model artifact publication inventory — 2026-10-25

This is a per-artifact publication record for local ONNX ports. An uploaded file set is not a blanket model-quality pass; execution plans remain experimental unless linked to the exact tested artifacts, backend, quantization and receipts.

## Published

### Qwen3-ASR 0.6B custom ONNX bundle

- Repo: https://huggingface.co/ysdede/qwen3-asr-0.6b-onnx
- Immutable runtime artifact revision: `2cbf53f6894e29d0d927b57931d40a6882e0ffe8`; current self-documenting repo revision: `ac0bedcc0f1614c0b3982940c14bdb169ff2365d`.
- Source: `Qwen/Qwen3-ASR-0.6B` revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, Apache-2.0. Upstream revision has no ONNX graphs.
- Uploaded runtime set (2,267,257,419 bytes): `audio-encoder-dynamic.onnx`, `decoder-prefill-fp16.onnx`, `decoder-step-fp16.onnx`, the shared `decoder-fp16.onnx.data`, and `tokenizer/tokenizer.json`.
- Every file SHA-256 was computed from the exact directory recorded in real-weight receipts and matched an independently hashed duplicate local port. The ONNX initializer inspection reports the encoder's stored initializers as float32; decoder graph names and weights identify FP16. Pinned upstream `config.json`, `generation_config.json` and `preprocessor_config.json` are also included under `provenance/source/`, with their source revision and digests; they are reference metadata and do not replace the custom graph contract. This does not approve all provider/precision combinations.
- Status: **EXPERIMENTAL / PARTIAL PASS**. Qwen v16 four-demo matrix: 16/18 exact tested flows; native streaming unsupported. Physical microphones and broad multilingual quality remain unverified.
- Card and artifact manifest: repo `README.md`, `asrjs-manifest.json`, `LICENSE`.

### Moonshine Streaming Tiny custom four-graph bundle

- Repo: https://huggingface.co/ysdede/moonshine-streaming-tiny-onnx
- Published revision: `cb347a0760c2d7d6cf9083eb770376ba108e5b31`.
- Source: `moonshine-ai/moonshine-streaming-tiny` revision `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`; streaming Tiny model is MIT licensed.
- Exact runtime set (670,530,748 bytes): four FP32 ONNX graphs, their four required external-data shards, tokenizer and five processor/configuration files. All 14 hashes and sizes match the independent parent artifact receipt. Local machine paths were removed from the public provenance copy.
- Status: **EXPERIMENTAL / PARTIAL PASS**. The v18 four-demo matrix was 13/18: padded manual/recording flows produced empty output, and WebGPU FireRed-VAD missed one word. Only fixture-specific flows count as accepted; physical microphone and broad accuracy remain unverified.
- Card and artifact manifest: repo `README.md`, `asrjs-manifest.json`, `asrjs-provenance.json`, `LICENSE`.

Evidence summary: [public core handoff](HANDOFF-MODERN-MODELS-2026-10-25.md#historical-checkpoint-modern-model-e2e-matrix). Detailed app receipts live in the private companion demos repository and are not copied into these model repos.

## Held for rights or provenance review

| Local export | Provenance and current evidence | Why held |
| --- | --- | --- |
| SenseVoice Small | `FunAudioLLM/SenseVoiceSmall` official revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`; export manifest pins the source checkpoint. Four-demo matrix 16/18. | The model card declares a custom `other` / model-license. The local original checkout has no license text. Need include/read the exact official model agreement and confirm derivative ONNX redistribution terms before upload. |
| GigaAM multilingual CTC | Local ONNX hashes and YAML/provenance are in the companion `parent-model-artifacts.json`; export is marked `experimental-official-export`. Four-demo matrix 10/18. | The source repository code has MIT licensing, but the local checkpoint provenance does not identify checkpoint license/revision clearly enough to conclude that these model weights may be republished. |
| GigaAM v3 E2E RNNT | Three graph hashes, YAML, vocabulary and export provenance are recorded locally; export is marked `experimental-official-export`. Four-demo matrix 12/18. | Same code-versus-checkpoint license gap; needs source checkpoint revision/redistribution basis. |
| X-ASR 160 ms | Encoder/decoder/joiner/token hashes exist locally; CPU reference control is explicitly not a runtime pass. Four-demo matrix 14/18, including four VAD transcript misses. | Model source revision and model-weight license are not established in the inspected export files. Do not infer them from the GigaAM adapter/runtime or sherpa-onnx implementation. |

No files from these held exports were uploaded. The model repo names can be prepared after exact weight licensing and source revisions are confirmed. Existing `ysdede` repos for Parakeet, Whisper, MedASR, Canary, Nemotron and wav2vec2 were found and should be reused rather than duplicated.

## Reproduction material

`tools/data/results/hf-publication-20261025/` contains the exact Qwen/Moonshine card, license, hash manifest and provenance material used for publication. The hash helper scripts contain local input paths and are internal evidence only; they were excluded from the Hub allowlists.



## Publication and remote verification record

The repos were confirmed absent before creation. Authenticated account was `ysdede`; no access token was printed or recorded. Commands used:

```powershell
hf repo create ysdede/qwen3-asr-0.6b-onnx --repo-type model --exist-ok
hf upload-large-folder ysdede/qwen3-asr-0.6b-onnx N:/github/asrjs/demos/tmp/real-e2e/qwen-loading-20261023/model-fp16 --repo-type model --num-workers 4 --include audio-encoder-dynamic.onnx decoder-prefill-fp16.onnx decoder-step-fp16.onnx decoder-fp16.onnx.data tokenizer/tokenizer.json
hf upload ysdede/qwen3-asr-0.6b-onnx N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/qwen3-asr --repo-type model --include README.md LICENSE asrjs-manifest.json --commit-message "Add experimental asr.js Qwen ONNX metadata"
hf download Qwen/Qwen3-ASR-0.6B --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 --include config.json generation_config.json preprocessor_config.json --local-dir N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/qwen3-asr/provenance/source
hf upload ysdede/qwen3-asr-0.6b-onnx N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/qwen3-asr --repo-type model --include provenance/source/config.json provenance/source/generation_config.json provenance/source/preprocessor_config.json provenance/source/source-manifest.json --commit-message "Add pinned upstream Qwen configuration metadata"
hf repo create ysdede/moonshine-streaming-tiny-onnx --repo-type model --exist-ok
hf upload-large-folder ysdede/moonshine-streaming-tiny-onnx N:/github/asrjs/demos/tmp/real-e2e/remaining-models-20261024/native-moonshine-streaming-tiny --repo-type model --num-workers 4 --include encoder_model.onnx encoder_model.onnx.data encoder_chunked_model.onnx encoder_chunked_model.onnx.data decoder_prefill_model.onnx decoder_prefill_model.onnx.data decoder_step_model.onnx decoder_step_model.onnx.data tokenizer.json config.json generation_config.json preprocessor_config.json processor_config.json tokenizer_config.json
hf upload ysdede/moonshine-streaming-tiny-onnx N:/github/asrjs/speech-recognition/tools/data/results/hf-publication-20261025/moonshine-streaming-tiny --repo-type model --include README.md LICENSE asrjs-manifest.json asrjs-provenance.json --commit-message "Add experimental asr.js Moonshine ONNX metadata"
```

Remote Hub API inventory (`HfApi.model_info(..., files_metadata=True)`) confirms the Qwen repo at `ac0bedcc0f1614c0b3982940c14bdb169ff2365d` has the five runtime files plus four provenance files, and Moonshine at `cb347a0760c2d7d6cf9083eb770376ba108e5b31` has the exact 14 approved files. Remote sizes match the manifests; the remote LFS SHA-256 for each large ONNX file and external-data shard matches the corresponding manifest digest. Local manifest digests were recomputed from the receipt-backed source directories before upload. Qwen runtime-only revision `2cbf53f6894e29d0d927b57931d40a6882e0ffe8` is retained as the initial immutable artifact checkpoint; use current revision `ac0bedcc0f1614c0b3982940c14bdb169ff2365d` to fetch both bundle and upstream config provenance.

The standard `hf upload-large-folder` resumable cache was generated under the source directories in `demos/tmp`; it is ignored runtime metadata and was left intact.
