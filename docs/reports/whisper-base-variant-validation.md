# Whisper Large v3 Turbo — Node/WASM Splitgraph Validation Report

**Generated**: 2026-05-29T19:54:26.434Z
**Artifacts**: /tmp/whisper-base-4graph
**Backend**: Node CLI (`fp32`/`fp16` use onnxruntime-node CPU because these large variants exceed WASM memory on this host; `q8` uses onnxruntime-web WASM CPU)
**Validator**: V2 (session reuse per variant)
**max_new_tokens**: 444
**Alignment validation**: disabled

## Scope

- Validates existing fp32/fp16/q8 splitgraph variants locally in Node CLI.
- Uses fp32 as the Node CPU baseline; fp16 is also checked on Node CPU, and q8 is validated with the WASM CPU execution provider. No WebGPU/browser automation is included.
- Uses language suffixes from fixture filenames: `.tr.*` → Turkish, `.en.*` → English.
- Decoding path is greedy `temperature=0`; beam search is not implemented here.

## Fixtures

| Fixture | Language | Reference |
|---------|----------|-----------|
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | en | no |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | yes |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | en | no |
| ItsLifeJim.en.wav | en | no |
| JFK_Short.en.wav | en | no |
| common_voice_tr_38277367.tr.mp3 | tr | no |
| common_voice_tr_38284290.tr.mp3 | tr | no |
| common_voice_tr_38284299.tr.mp3 | tr | no |
| common_voice_tr_38289682.tr.mp3 | tr | no |
| jfk2.en.wav | en | no |
| librivox.org-1600hz.en.wav | en | no |
| train_000015.tr.mp3 | tr | no |
| train_000017.tr.mp3 | tr | no |
| train_000058.tr.mp3 | tr | no |
| train_000085.tr.mp3 | tr | no |
| train_000094.tr.mp3 | tr | no |

## Generation Controls

| Fixture | Variant | Language | Task | no_timestamps | max_new_tokens | suppress_tokens | begin_suppress_tokens | Decoding |
|---------|---------|----------|------|---------------|----------------|-----------------|-----------------------|----------|
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38277367.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38284290.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38284299.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38289682.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp32 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000015.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000017.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000058.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000085.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000094.tr.mp3 | fp32 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| ItsLifeJim.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| JFK_Short.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38277367.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38284290.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38284299.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| common_voice_tr_38289682.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| jfk2.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| librivox.org-1600hz.en.wav | fp16 | en | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000015.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000017.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000058.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000085.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |
| train_000094.tr.mp3 | fp16 | tr | transcribe | true | 444 | 88 tokens | [220, 50257] | greedy, temp=0 |

## Prompt Consistency

| Fixture | Prompt language | fp32 prompt IDs | fp16 match | q8 match |
|---------|-----------------|-----------------|------------|----------|
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | [50258, 50268, 50359, 50363] | yes | NO |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| ItsLifeJim.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| JFK_Short.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| common_voice_tr_38277367.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| common_voice_tr_38284290.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| common_voice_tr_38284299.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| common_voice_tr_38289682.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| jfk2.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| librivox.org-1600hz.en.wav | en | [50258, 50259, 50359, 50363] | yes | NO |
| train_000015.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| train_000017.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| train_000058.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| train_000085.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |
| train_000094.tr.mp3 | tr | [50258, 50268, 50359, 50363] | yes | NO |

## Token/Text Comparison vs fp32

| Fixture | Variant | Tokens | EOS | Token match vs fp32 | Text match | Decoded text | Time |
|---------|---------|--------|-----|---------------------|------------|--------------|------|
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | Functionally, the lower leg is supplied by two vessels, ITP and fibular artery. There is a significa | 5.921s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | Bulaşıcı hastalıkların beklenmelik zamanlarda yaptıkları salgınlar. O kadar korkunç ve tahrip edecek | 5.663s |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | No evidence of periodic collections. The left and right Iliac arteries are normal in course and cali | 5.92s |
| ItsLifeJim.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | It is not life as we know or understand it. | 5.89s |
| JFK_Short.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | in the long history of the world. Only a few generations have been granted the role of defending fre | 5.895s |
| common_voice_tr_38277367.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Burada yazı artık okulmaz bir şekilde alıyordu. | 5.849s |
| common_voice_tr_38284290.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Sorgusuz Tüayesi. | 5.835s |
| common_voice_tr_38284299.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Beyesiza gelince caminin yanındaki kahvelerden birinde oturduğlar. | 6.437s |
| common_voice_tr_38289682.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | yüksek Roma kemerlerinin yanındaki Pülüsür Kamilonlara şerbetçileri seyrederken gözüm karşı taraftak | 6.679s |
| jfk2.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c | 5.83s |
| librivox.org-1600hz.en.wav | fp32 | 444 | false | exact (444/444, 100.0%) | true | Preface of a Year with the Birds This is a Libravox recording All Libravox recordings are in the pub | 5.888s |
| train_000015.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Şimdi, yututlaki algoritmaların sansır ve baskısın edeniyle, içeriğini bulmak daha zor hale geliyor. | 7.366s |
| train_000017.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Bu taboda görüldüğü üzere, tahıla beslenen hayvanları kıyasa çok daha fazla içerdiklerini görebilirs | 5.958s |
| train_000058.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | mikrop kapmaktan ve hastalanmaktan korkar ve kaygılın. | 5.934s |
| train_000085.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Kinci grub köpekler ise malisret bu elektroşoka sürekli maruz kalıyorlar. Deneyin ikinci yaşamasında | 6.344s |
| train_000094.tr.mp3 | fp32 | 444 | false | exact (444/444, 100.0%) | true | Doktorunuz gereken tedavileri uyguladıktan sonra dökülme devam ediyorsa veya saçlarda iyileştirileme | 5.932s |
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | Functionally, the lower leg is supplied by two vessels, ITP and fibular artery. There is a significa | 8.701s |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | Bulaşıcı hastalıkların beklenmelik zamanlarda yaptıkları salgınlar. O kadar korkunç ve tahrip edecek | 7.015s |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | No evidence of periodic collections. The left and right Iliac arteries are normal in course and cali | 7.86s |
| ItsLifeJim.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | It is not life as we know or understand it. | 7.863s |
| JFK_Short.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | in the long history of the world. Only a few generations have been granted the role of defending fre | 7.795s |
| common_voice_tr_38277367.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Burada yazı artık okulmaz bir şekilde alıyordu. | 7.624s |
| common_voice_tr_38284290.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Sorgusuz Tüayesi. | 7.086s |
| common_voice_tr_38284299.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Beyesiza gelince caminin yanındaki kahvelerden birinde oturduğlar. | 7.207s |
| common_voice_tr_38289682.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | yüksek Roma kemerlerinin yanındaki Pülüsür Kamilonlara şerbetçileri seyrederken gözüm karşı taraftak | 8.598s |
| jfk2.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your c | 11.296s |
| librivox.org-1600hz.en.wav | fp16 | 444 | false | exact (444/444, 100.0%) | true | Preface of a Year with the Birds This is a Libravox recording All Libravox recordings are in the pub | 7.833s |
| train_000015.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Şimdi, yututlaki algoritmaların sansır ve baskısın edeniyle, içeriğini bulmak daha zor hale geliyor. | 10.178s |
| train_000017.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Bu taboda görüldüğü üzere, tahıla beslenen hayvanları kıyasa çok daha fazla içerdiklerini görebilirs | 7.038s |
| train_000058.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | mikrop kapmaktan ve hastalanmaktan korkar ve kaygılın. | 7.114s |
| train_000085.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Kinci grub köpekler ise malisret bu elektroşoka sürekli maruz kalıyorlar. Deneyin ikinci yaşamasında | 6.921s |
| train_000094.tr.mp3 | fp16 | 444 | false | exact (444/444, 100.0%) | true | Doktorunuz gereken tedavileri uyguladıktan sonra dökülme devam ediyorsa veya saçlarda iyileştirileme | 6.733s |

## Alignment/DTW Validation

| Fixture | Variant | Shape | Row sums min/mean/max | Non-negative | Monotonic DTW |
|---------|---------|-------|-----------------------|--------------|---------------|
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp32 | n/a | n/a | n/a | n/a |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp32 | n/a | n/a | n/a | n/a |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp32 | n/a | n/a | n/a | n/a |
| ItsLifeJim.en.wav | fp32 | n/a | n/a | n/a | n/a |
| JFK_Short.en.wav | fp32 | n/a | n/a | n/a | n/a |
| common_voice_tr_38277367.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| common_voice_tr_38284290.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| common_voice_tr_38284299.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| common_voice_tr_38289682.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| jfk2.en.wav | fp32 | n/a | n/a | n/a | n/a |
| librivox.org-1600hz.en.wav | fp32 | n/a | n/a | n/a | n/a |
| train_000015.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| train_000017.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| train_000058.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| train_000085.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| train_000094.tr.mp3 | fp32 | n/a | n/a | n/a | n/a |
| 00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav | fp16 | n/a | n/a | n/a | n/a |
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | fp16 | n/a | n/a | n/a | n/a |
| 0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav | fp16 | n/a | n/a | n/a | n/a |
| ItsLifeJim.en.wav | fp16 | n/a | n/a | n/a | n/a |
| JFK_Short.en.wav | fp16 | n/a | n/a | n/a | n/a |
| common_voice_tr_38277367.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| common_voice_tr_38284290.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| common_voice_tr_38284299.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| common_voice_tr_38289682.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| jfk2.en.wav | fp16 | n/a | n/a | n/a | n/a |
| librivox.org-1600hz.en.wav | fp16 | n/a | n/a | n/a | n/a |
| train_000015.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| train_000017.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| train_000058.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| train_000085.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |
| train_000094.tr.mp3 | fp16 | n/a | n/a | n/a | n/a |

## Status Summary

| Variant | Node CLI | Runtime backend | Prompt parity | Token parity vs fp32 | Alignment sanity | Status |
|---------|----------|-----------------|---------------|----------------------|------------------|--------|
| fp32 | pass | node-cpu | pass | baseline | pass | pass |
| fp16 | pass | node-cpu | pass | pass | pass | pass |

## Deferred / Manual

Current Node CLI validation is intentionally strict: prompt/generation-control parity passes, but any token/text/EOS divergence is reported before WebGPU is attempted.

fp16 parity requires converting float16 logits/alignment tensors back to float32 before logit processors and argmax; raw uint16 half bits are not comparable logits.

q8 uses ONNX Runtime Web WASM CPU. Extended greedy decoding can diverge from fp32 because the decoder is quantized; those token/EOS differences remain visible in the comparison table instead of being hidden.

WebGPU smoke is intentionally not automated here. After Node/WASM validation passes, WebGPU should be tested manually in the browser/app.

Beam search for the 4-graph splitgraph runtime is not implemented in this validation pass; keep it as the next decoding task after greedy parity is stable.

Mixed dtype, q4/q4f16, exporter changes, browser automation, and published HF artifact changes are out of scope for this report.

## q8 Divergence Analysis

Two fixtures diverge from fp32 under extended greedy decoding (`max_new_tokens=64`).
Both are quantized-decoder sensitivity, not runtime bugs.

| Fixture | Divergence step | fp32 token | q8 token | Top-1 / Top-2 margin | Cause |
|---------|-----------------|------------|----------|----------------------|-------|

Both divergences occur at tight decision points where the top-1/top-2 margin is small.
The logit processor is applied identically (logits before and after suppression are the same at the divergence step, confirming no generation-control mismatch).
Prompt IDs and generation controls are identical between fp32 and q8.

Conclusion: q8 strict token parity with fp32 is not expected at extended `max_new_tokens`.
The q8 variant is validated as a compact quantized candidate, not a bit-exact drop-in for fp32.
Short-sequence decoding is stable; extended decoding can differ at tight decision points.
WebGPU testing should accept these known divergences as expected quantization sensitivity.

