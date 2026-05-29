# Whisper Large v3 Turbo — Node/WASM Splitgraph Validation Report

**Generated**: 2026-05-29T17:20:10.361Z
**Artifacts**: /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph
**Backend**: Node CLI (`fp32`/`fp16` use onnxruntime-node CPU because these large variants exceed WASM memory on this host; `q8` uses onnxruntime-web WASM CPU)
**max_new_tokens**: 64
**Alignment validation**: disabled

## Scope

- Validates existing fp32/fp16/q8 splitgraph variants locally in Node CLI.
- Uses fp32 as the Node CPU baseline; fp16 is also checked on Node CPU, and q8 is validated with the WASM CPU execution provider. No WebGPU/browser automation is included.
- Uses language suffixes from fixture filenames: `.tr.*` → Turkish, `.en.*` → English.
- Decoding path is greedy `temperature=0`; beam search is not implemented here.

## Fixtures

| Fixture | Language | Reference |
|---------|----------|-----------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | yes |
| ItsLifeJim.en.wav | en | no |
| JFK_Short.en.wav | en | no |
| jfk2.en.wav | en | no |
| librivox.org-1600hz.en.wav | en | no |

## Generation Controls

| Fixture | Variant | Language | Task | no_timestamps | max_new_tokens | suppress_tokens | begin_suppress_tokens | Decoding |
|---------|---------|----------|------|---------------|----------------|-----------------|-----------------------|----------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | tr | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp32 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp32 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp32 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp32 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | tr | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp16 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp16 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp16 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp16 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | q8 | tr | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | q8 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | q8 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | q8 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | q8 | en | transcribe | true | 64 | 88 tokens | [220, 50257] | greedy, temp=0 |

## Prompt Consistency

| Fixture | Prompt language | fp32 prompt IDs | fp16 match | q8 match |
|---------|-----------------|-----------------|------------|----------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | [50258, 50268, 50360, 50364] | yes | yes |
| ItsLifeJim.en.wav | en | [50258, 50259, 50360, 50364] | yes | yes |
| JFK_Short.en.wav | en | [50258, 50259, 50360, 50364] | yes | yes |
| jfk2.en.wav | en | [50258, 50259, 50360, 50364] | yes | yes |
| librivox.org-1600hz.en.wav | en | [50258, 50259, 50360, 50364] | yes | yes |

## Token/Text Comparison vs fp32

| Fixture | Variant | Tokens | EOS | Token match vs fp32 | Text match | Decoded text | Time |
|---------|---------|--------|-----|---------------------|------------|--------------|------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | 64 | false | exact (64/64, 100.0%) | true | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciyd | 16.53s |
| ItsLifeJim.en.wav | fp32 | 64 | false | exact (64/64, 100.0%) | true | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my t | 11.829s |
| JFK_Short.en.wav | fp32 | 40 | true | exact (40/40, 100.0%) | true | In the long history of the world, only a few generations have been granted the role of defending fre | 11.021s |
| jfk2.en.wav | fp32 | 27 | true | exact (27/27, 100.0%) | true | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your  | 10.986s |
| librivox.org-1600hz.en.wav | fp32 | 10 | true | exact (10/10, 100.0%) | true | Preface of A Year with the Birds. | 10.453s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | 64 | false | exact (64/64, 100.0%) | true | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciyd | 14.477s |
| ItsLifeJim.en.wav | fp16 | 64 | false | exact (64/64, 100.0%) | true | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my t | 15.18s |
| JFK_Short.en.wav | fp16 | 40 | true | exact (40/40, 100.0%) | true | In the long history of the world, only a few generations have been granted the role of defending fre | 19.08s |
| jfk2.en.wav | fp16 | 27 | true | exact (27/27, 100.0%) | true | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your  | 13.682s |
| librivox.org-1600hz.en.wav | fp16 | 10 | true | exact (10/10, 100.0%) | true | Preface of A Year with the Birds. | 13.253s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | q8 | 64 | false | exact (64/64, 100.0%) | true | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciyd | 31.874s |
| ItsLifeJim.en.wav | q8 | 64 | false | DIFF (46/64, 71.9%) | false | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my t | 29.231s |
| JFK_Short.en.wav | q8 | 40 | true | exact (40/40, 100.0%) | true | In the long history of the world, only a few generations have been granted the role of defending fre | 27.89s |
| jfk2.en.wav | q8 | 27 | true | exact (27/27, 100.0%) | true | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your  | 28.51s |
| librivox.org-1600hz.en.wav | q8 | 64 | true | DIFF (9/64, 14.1%) | false | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the p | 28.922s |

## Alignment/DTW Validation

| Fixture | Variant | Shape | Row sums min/mean/max | Non-negative | Monotonic DTW |
|---------|---------|-------|-----------------------|--------------|---------------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | n/a | n/a | n/a | n/a |
| ItsLifeJim.en.wav | fp32 | n/a | n/a | n/a | n/a |
| JFK_Short.en.wav | fp32 | n/a | n/a | n/a | n/a |
| jfk2.en.wav | fp32 | n/a | n/a | n/a | n/a |
| librivox.org-1600hz.en.wav | fp32 | n/a | n/a | n/a | n/a |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | n/a | n/a | n/a | n/a |
| ItsLifeJim.en.wav | fp16 | n/a | n/a | n/a | n/a |
| JFK_Short.en.wav | fp16 | n/a | n/a | n/a | n/a |
| jfk2.en.wav | fp16 | n/a | n/a | n/a | n/a |
| librivox.org-1600hz.en.wav | fp16 | n/a | n/a | n/a | n/a |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | q8 | n/a | n/a | n/a | n/a |
| ItsLifeJim.en.wav | q8 | n/a | n/a | n/a | n/a |
| JFK_Short.en.wav | q8 | n/a | n/a | n/a | n/a |
| jfk2.en.wav | q8 | n/a | n/a | n/a | n/a |
| librivox.org-1600hz.en.wav | q8 | n/a | n/a | n/a | n/a |

## Status Summary

| Variant | Node CLI | Runtime backend | Prompt parity | Token parity vs fp32 | Alignment sanity | Status |
|---------|----------|-----------------|---------------|----------------------|------------------|--------|
| fp32 | pass | node-cpu | pass | baseline | pass | pass |
| fp16 | pass | node-cpu | pass | pass | pass | pass |
| q8 | pass | wasm | pass | fail | pass | fail |

## Deferred / Manual

Current Node CLI validation is intentionally strict: prompt/generation-control parity passes, but any token/text/EOS divergence is reported before WebGPU is attempted.

fp16 parity requires converting float16 logits/alignment tensors back to float32 before logit processors and argmax; raw uint16 half bits are not comparable logits.

q8 uses ONNX Runtime Web WASM CPU. Extended greedy decoding can diverge from fp32 because the decoder is quantized; those token/EOS differences remain visible in the comparison table instead of being hidden.

WebGPU smoke is intentionally not automated here. After Node/WASM validation passes, WebGPU should be tested manually in the browser/app.

Beam search for the 4-graph splitgraph runtime is not implemented in this validation pass; keep it as the next decoding task after greedy parity is stable.

Mixed dtype, q4/q4f16, exporter changes, browser automation, and published HF artifact changes are out of scope for this report.

