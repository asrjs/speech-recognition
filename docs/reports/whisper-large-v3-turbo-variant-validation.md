# Whisper Large v3 Turbo — Node/WASM Splitgraph Validation Report

**Generated**: 2026-05-29T15:58:38.900Z
**Artifacts**: /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph
**Backend**: Node CLI (`fp32`/`fp16` use onnxruntime-node CPU because these large variants exceed WASM memory on this host; `q8` uses onnxruntime-web WASM CPU)
**max_new_tokens**: 16
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
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | tr | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp32 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp32 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp32 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp32 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | tr | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp16 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp16 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp16 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp16 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | q8 | tr | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | q8 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | q8 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | q8 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | q8 | en | transcribe | true | 16 | 88 tokens | [220, 50257] | greedy, temp=0 |

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
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | 16 | false | exact (16/16, 100.0%) | true | Bulaşıcı hastalıkların beklenmedik zamanlarda yapt | 10.799s |
| ItsLifeJim.en.wav | fp32 | 16 | false | exact (16/16, 100.0%) | true | Incredible. Not only should it have been destroyed by our phasers, it | 11.283s |
| JFK_Short.en.wav | fp32 | 16 | false | exact (16/16, 100.0%) | true | In the long history of the world, only a few generations have been granted the | 14.64s |
| jfk2.en.wav | fp32 | 16 | false | exact (16/16, 100.0%) | true | And so, my fellow Americans, ask not what your country can do for you | 10.418s |
| librivox.org-1600hz.en.wav | fp32 | 10 | true | exact (10/10, 100.0%) | true | Preface of A Year with the Birds. | 9.712s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | 16 | false | DIFF (0/16, 0.0%) | false | Weihnerne실히 volley�-calmuggageugesificant�zcz obligedfruitygenieth | 13.368s |
| ItsLifeJim.en.wav | fp16 | 16 | false | DIFF (0/16, 0.0%) | false | quilteremony mauvpeareberishBooks Dahłec��ىcaustfunding々lementullainel | 14.679s |
| JFK_Short.en.wav | fp16 | 16 | false | DIFF (0/16, 0.0%) | false | � updmidtmate kahkahaleighassiswartsiłhillobed PRIhölideelve� | 18.199s |
| jfk2.en.wav | fp16 | 16 | false | DIFF (0/16, 0.0%) | false | spanning評 reappreckςżen鬼 Burchothe kahkahacade‑Hubsentsстанов | 13.846s |
| librivox.org-1600hz.en.wav | fp16 | 16 | false | DIFF (0/16, 0.0%) | false | twentardonteokulifwichalid��in�注docknai IKEESINENNIS | 14.072s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | q8 | 16 | false | exact (16/16, 100.0%) | true | Bulaşıcı hastalıkların beklenmedik zamanlarda yapt | 29.065s |
| ItsLifeJim.en.wav | q8 | 16 | false | exact (16/16, 100.0%) | true | Incredible. Not only should it have been destroyed by our phasers, it | 26.814s |
| JFK_Short.en.wav | q8 | 16 | false | exact (16/16, 100.0%) | true | In the long history of the world, only a few generations have been granted the | 27.585s |
| jfk2.en.wav | q8 | 16 | false | exact (16/16, 100.0%) | true | And so, my fellow Americans, ask not what your country can do for you | 26.778s |
| librivox.org-1600hz.en.wav | q8 | 16 | false | DIFF (9/16, 56.3%) | false | Preface of A Year with the Birds. This is a LibriVox | 26.147s |

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
| fp16 | pass | node-cpu | pass | fail | pass | fail |
| q8 | pass | wasm | pass | fail | pass | fail |

## Deferred / Manual

Current Node CLI validation is intentionally strict: prompt/generation-control parity passes, but any token/text/EOS divergence is reported before WebGPU is attempted.

WebGPU smoke is intentionally not automated here. After Node/WASM validation passes, WebGPU should be tested manually in the browser/app.

Beam search for the 4-graph splitgraph runtime is not implemented in this validation pass; keep it as the next decoding task after greedy parity is stable.

Mixed dtype, q4/q4f16, exporter changes, browser automation, and published HF artifact changes are out of scope for this report.

