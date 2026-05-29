# Whisper Large v3 Turbo — Variant Validation Report

**Generated**: 2026-05-29 17:30:16
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
| 1 | 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav | unknown | 18.6s | 16000 Hz | 583 KB | ✓ |
| 2 | Its Life Jim.wav | en | 22.4s | 44100 Hz | 1930 KB |  |
| 3 | JFK_Short.wav | en | 17.1s | 16000 Hz | 536 KB |  |
| 4 | jfk2.wav | en | 11.0s | 16000 Hz | 344 KB |  |
| 5 | librivox.org-1600hz.wav | en | 18.7s | 16000 Hz | 585 KB |  |

## Artifact Metrics

| Variant | Total Size | Files | encoder_model | decoder_init | decoder_step | decoder_align |
|---------|-----------|-------|--------------|-------------|-------------|---------------|
| fp32 | 4541 MB | 13 | 0 MB | 909 MB | 606 MB | 593 MB |
| fp16 | 2272 MB | 12 | 1215 MB | 455 MB | 303 MB | 297 MB |
| q8 | 1410 MB | 9 | 615 MB | 228 MB | 414 MB | 150 MB |

## Variant: fp32

Load time: 5.745s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | The pandemic was so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad  |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 0.0% |
| Encoder time | 8.0592s |
| Decoder init time | 0.1891s |
| Step total time | 7.0305s |
| Step avg / token | 31.39ms |
| Total decode time | 15.293s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 7.3569s |
| Decoder init time | 0.1675s |
| Step total time | 7.5621s |
| Step avg / token | 33.76ms |
| Total decode time | 15.103s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 7.1513s |
| Decoder init time | 0.147s |
| Step total time | 7.072s |
| Step avg / token | 31.57ms |
| Total decode time | 14.387s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 7.312s |
| Decoder init time | 0.147s |
| Step total time | 8.2178s |
| Step avg / token | 36.69ms |
| Total decode time | 15.691s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 8.3381s |
| Decoder init time | 0.1948s |
| Step total time | 8.5992s |
| Step avg / token | 38.39ms |
| Total decode time | 17.152s |

## Variant: fp16

Load time: 7.357s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | The pandemic was so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad  |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 0.0% |
| Encoder time | 10.4736s |
| Decoder init time | 0.2245s |
| Step total time | 10.7086s |
| Step avg / token | 47.81ms |
| Total decode time | 21.466s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 11.5227s |
| Decoder init time | 0.2085s |
| Step total time | 10.35s |
| Step avg / token | 46.21ms |
| Total decode time | 22.14s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 10.2375s |
| Decoder init time | 0.1931s |
| Step total time | 10.3615s |
| Step avg / token | 46.26ms |
| Total decode time | 20.856s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 10.9636s |
| Decoder init time | 0.1845s |
| Step total time | 10.4522s |
| Step avg / token | 46.66ms |
| Total decode time | 21.661s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 10.3682s |
| Decoder init time | 0.2008s |
| Step total time | 10.689s |
| Step avg / token | 47.72ms |
| Total decode time | 21.317s |

## Variant: q8

Load time: 3.059s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik geli |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 84.6% |
| Encoder time | 8.8972s |
| Decoder init time | 0.1313s |
| Step total time | 9.7136s |
| Step avg / token | 43.36ms |
| Total decode time | 18.762s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 9.6645s |
| Decoder init time | 0.199s |
| Step total time | 9.1983s |
| Step avg / token | 41.06ms |
| Total decode time | 19.078s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 9.7467s |
| Decoder init time | 0.1648s |
| Step total time | 6.9299s |
| Step avg / token | 30.94ms |
| Total decode time | 16.862s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 6.6313s |
| Decoder init time | 0.1246s |
| Step total time | 7.652s |
| Step avg / token | 34.16ms |
| Total decode time | 14.424s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Prompt language | en |
| Prompt token IDs | [50258, 50259, 50360, 50364] |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 6.4224s |
| Decoder init time | 0.1208s |
| Step total time | 7.133s |
| Step avg / token | 31.84ms |
| Total decode time | 13.693s |

## Performance Comparison (first fixture)

| Variant | Encoder | Init | Step Total | Step/tok | Total | Tokens |
|---------|---------|------|------------|----------|-------|--------|
| fp32 | 8.0592s | 0.1891s | 7.0305s | 31.39ms | 15.293s | 224 |
| fp16 | 10.4736s | 0.2245s | 10.7086s | 47.81ms | 21.466s | 224 |
| q8 | 8.8972s | 0.1313s | 9.7136s | 43.36ms | 18.762s | 224 |

## Prompt Consistency

| Fixture | Prompt language | Prompt token IDs | Consistent across variants |
|---------|-----------------|------------------|----------------------------|
| 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav | en | [50258, 50259, 50360, 50364] | yes |
| Its Life Jim.wav | en | [50258, 50259, 50360, 50364] | yes |
| JFK_Short.wav | en | [50258, 50259, 50360, 50364] | yes |
| jfk2.wav | en | [50258, 50259, 50360, 50364] | yes |
| librivox.org-1600hz.wav | en | [50258, 50259, 50360, 50364] | yes |

## Status Summary

| Variant | Native ORT | Smoke Decode | Accuracy vs FP32 | Status |
|---------|-----------|-------------|-----------------|--------|
| fp32 | pass | pass | reference | baseline |
| fp16 | pass | pass | compare prompt-consistent output vs fp32 | WebGPU candidate |
| q8 | pass | pass | compare prompt-consistent output vs fp32 | compact candidate |

## Conclusion

This report uses one fixed prompt token sequence per fixture across all variants before comparing outputs.
If variants disagree on a fixture, treat it as a real variant/runtime difference, not a language-prompt difference.
Do not claim Turkish accuracy from this report unless the same Turkish prompt was used for every variant on that fixture and the fp32 baseline agrees.

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
