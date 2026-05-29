# Whisper Large v3 Turbo — Variant Validation Report

**Generated**: 2026-05-29 18:07:38
**Model**: openai/whisper-large-v3-turbo (4-graph ONNX)
**Artifacts**: /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph

## Environment

| Item | Value |
|------|-------|
| OS | Linux 5.15.167.4-microsoft-standard-WSL2 |
| Python | 3.12.3 |
| ONNX Runtime | 1.26.0 |
| PyTorch | 2.12.0+cu130 |
| CPU | x86_64 |
| Providers | CPUExecutionProvider |

## Fixtures

| # | Filename | Language | Duration | Sample Rate | Size | Reference |
|---|----------|----------|----------|-------------|------|-----------|
| 1 | 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | 18.6s | 16000 Hz | 583 KB | ✓ |
| 2 | ItsLifeJim.en.wav | en | 22.4s | 44100 Hz | 1930 KB |  |
| 3 | JFK_Short.en.wav | en | 17.1s | 16000 Hz | 536 KB |  |
| 4 | jfk2.en.wav | en | 11.0s | 16000 Hz | 344 KB |  |
| 5 | librivox.org-1600hz.en.wav | en | 18.7s | 16000 Hz | 585 KB |  |

## Artifact Metrics

| Variant | Total Size | Files | encoder_model | decoder_init | decoder_step | decoder_align |
|---------|-----------|-------|--------------|-------------|-------------|---------------|
| fp32 | 4541 MB | 13 | 0 MB | 909 MB | 606 MB | 593 MB |
| fp16 | 2272 MB | 12 | 1215 MB | 455 MB | 303 MB | 297 MB |
| q8 | 1410 MB | 9 | 615 MB | 228 MB | 414 MB | 150 MB |

## Variant: fp32

Load time: 5.229s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav (tr)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | tr |
| Prompt token IDs | [50258, 50268, 50360, 50364] |
| Decoded text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik geli |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 84.6% |
| Encoder time | 10.5548s |
| Decoder init time | 0.2526s |
| Step total time | 5.6045s |
| Step avg / token | 25.02ms |
| Total decode time | 16.424s |

### ItsLifeJim.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 6.1371s |
| Decoder init time | 0.139s |
| Step total time | 5.5085s |
| Step avg / token | 24.59ms |
| Total decode time | 11.797s |

### JFK_Short.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 6.0924s |
| Decoder init time | 0.1261s |
| Step total time | 5.4811s |
| Step avg / token | 24.47ms |
| Total decode time | 11.712s |

### jfk2.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 6.4695s |
| Decoder init time | 0.1276s |
| Step total time | 5.5258s |
| Step avg / token | 24.67ms |
| Total decode time | 12.135s |

### librivox.org-1600hz.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 6.1689s |
| Decoder init time | 0.1265s |
| Step total time | 5.4786s |
| Step avg / token | 24.46ms |
| Total decode time | 11.786s |

## Variant: fp16

Load time: 5.162s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav (tr)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | tr |
| Prompt token IDs | [50258, 50268, 50360, 50364] |
| Decoded text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik geli |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 84.6% |
| Encoder time | 7.5254s |
| Decoder init time | 0.1457s |
| Step total time | 6.842s |
| Step avg / token | 30.54ms |
| Total decode time | 14.568s |

### ItsLifeJim.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 7.546s |
| Decoder init time | 0.132s |
| Step total time | 6.7661s |
| Step avg / token | 30.21ms |
| Total decode time | 14.492s |

### JFK_Short.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 7.4967s |
| Decoder init time | 0.1226s |
| Step total time | 6.8268s |
| Step avg / token | 30.48ms |
| Total decode time | 14.502s |

### jfk2.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 7.5035s |
| Decoder init time | 0.1408s |
| Step total time | 6.8083s |
| Step avg / token | 30.39ms |
| Total decode time | 14.505s |

### librivox.org-1600hz.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 7.9943s |
| Decoder init time | 0.1412s |
| Step total time | 7.4175s |
| Step avg / token | 33.11ms |
| Total decode time | 15.606s |

## Variant: q8

Load time: 2.372s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav (tr)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | tr |
| Prompt token IDs | [50258, 50268, 50360, 50364] |
| Decoded text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik geli |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 84.6% |
| Encoder time | 5.4513s |
| Decoder init time | 0.0921s |
| Step total time | 5.293s |
| Step avg / token | 23.63ms |
| Total decode time | 10.849s |

### ItsLifeJim.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 5.1981s |
| Decoder init time | 0.0861s |
| Step total time | 5.4788s |
| Step avg / token | 24.46ms |
| Total decode time | 10.778s |

### JFK_Short.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 5.0335s |
| Decoder init time | 0.0815s |
| Step total time | 4.5247s |
| Step avg / token | 20.2ms |
| Total decode time | 9.653s |

### jfk2.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 4.7868s |
| Decoder init time | 0.0849s |
| Step total time | 4.4874s |
| Step avg / token | 20.03ms |
| Total decode time | 9.372s |

### librivox.org-1600hz.en.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 4.7013s |
| Decoder init time | 0.0837s |
| Step total time | 4.5169s |
| Step avg / token | 20.16ms |
| Total decode time | 9.315s |

## Performance Comparison (first fixture)

| Variant | Encoder | Init | Step Total | Step/tok | Total | Tokens |
|---------|---------|------|------------|----------|-------|--------|
| fp32 | 10.5548s | 0.2526s | 5.6045s | 25.02ms | 16.424s | 224 |
| fp16 | 7.5254s | 0.1457s | 6.842s | 30.54ms | 14.568s | 224 |
| q8 | 5.4513s | 0.0921s | 5.293s | 23.63ms | 10.849s | 224 |

## Prompt Consistency

| Fixture | Prompt language | Prompt token IDs | Consistent across variants |
|---------|-----------------|------------------|----------------------------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav | tr | [50258, 50268, 50360, 50364] | yes |
| ItsLifeJim.en.wav | en | [50258, 50259, 50360, 50364] | yes |
| JFK_Short.en.wav | en | [50258, 50259, 50360, 50364] | yes |
| jfk2.en.wav | en | [50258, 50259, 50360, 50364] | yes |
| librivox.org-1600hz.en.wav | en | [50258, 50259, 50360, 50364] | yes |

## Status Summary

| Variant | Native ORT | Smoke Decode | Accuracy vs FP32 | Status |
|---------|-----------|-------------|-----------------|--------|
| fp32 | pass | pass | reference | baseline |
| fp16 | pass | pass | compare prompt-consistent output vs fp32 | WebGPU candidate |
| q8 | pass | pass | compare prompt-consistent output vs fp32 | compact candidate |

## Conclusion

This report uses one fixed prompt token sequence per fixture across all variants before comparing outputs.
Fixture language is read from explicit `.en` / `.tr` filename suffixes when present, then legacy filename hints.
If variants disagree on a fixture, treat it as a real variant/runtime difference, not a language-prompt difference.
Turkish fixture accuracy is now prompt-valid only when the Prompt Consistency table shows `tr` with identical token IDs for fp32/fp16/q8 and the fp32 baseline agrees with the reference.

## Known Limitations

- fp32 is native/reference only (~4.5 GB) — not for browser/WebGPU.
- fp16 is export-time FP16 only. Post-export converter is broken (Cast mismatch).
- q8 text quality and timestamp sanity must be verified per-fixture before claiming equivalence.
- WebGPU validation is pending for both fp16 and q8.
- Mixed dtype and q4/q4f16 are deferred.

## Recommended Next Tasks

1. Browser/WebGPU smoke for fp16
2. Browser/WebGPU smoke for q8
3. Mixed graph-level dtype resolver
4. q4/q4f16 research
5. External benchmark dataset evaluation
