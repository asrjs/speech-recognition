# Whisper Large v3 Turbo — Variant Validation Report

**Generated**: 2026-05-29 16:45:43
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

Load time: 5.55s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | The pandemic was so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad  |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 0.0% |
| Encoder time | 7.8982s |
| Decoder init time | 0.2208s |
| Step total time | 6.2953s |
| Step avg / token | 28.1ms |
| Total decode time | 14.429s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 6.692s |
| Decoder init time | 0.1991s |
| Step total time | 6.2723s |
| Step avg / token | 28.0ms |
| Total decode time | 13.177s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 6.7691s |
| Decoder init time | 0.1353s |
| Step total time | 6.3725s |
| Step avg / token | 28.45ms |
| Total decode time | 13.292s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 6.7134s |
| Decoder init time | 0.1348s |
| Step total time | 6.4782s |
| Step avg / token | 28.92ms |
| Total decode time | 13.343s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 6.8825s |
| Decoder init time | 0.1492s |
| Step total time | 6.002s |
| Step avg / token | 26.79ms |
| Total decode time | 13.048s |

## Variant: fp16

Load time: 5.599s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | The pandemic was so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad and so bad  |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 0.0% |
| Encoder time | 13.2893s |
| Decoder init time | 0.1987s |
| Step total time | 7.1127s |
| Step avg / token | 31.75ms |
| Total decode time | 20.657s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 7.9501s |
| Decoder init time | 0.1534s |
| Step total time | 7.3563s |
| Step avg / token | 32.84ms |
| Total decode time | 15.505s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 7.9995s |
| Decoder init time | 0.1346s |
| Step total time | 7.3765s |
| Step avg / token | 32.93ms |
| Total decode time | 15.575s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 7.9666s |
| Decoder init time | 0.1391s |
| Step total time | 7.2961s |
| Step avg / token | 32.57ms |
| Total decode time | 15.443s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 7.9094s |
| Decoder init time | 0.143s |
| Step total time | 7.6497s |
| Step avg / token | 34.15ms |
| Total decode time | 15.754s |

## Variant: q8

Load time: 2.041s

### 019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.wav (unknown)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik geli |
| Reference text | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta, ülkelerin sosyal ve ekonomik gel |
| Word overlap | 84.6% |
| Encoder time | 5.697s |
| Decoder init time | 0.1417s |
| Step total time | 6.5138s |
| Step avg / token | 29.08ms |
| Total decode time | 12.376s |

### Its Life Jim.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Incredible. Not only should it have been destroyed by our phasers, it does not even register on my tricorder. Captain, it doesn't even look real. It is not life as we know or understand it. Yet it is  |
| Encoder time | 6.2826s |
| Decoder init time | 0.0901s |
| Step total time | 5.2186s |
| Step avg / token | 23.3ms |
| Total decode time | 11.605s |

### JFK_Short.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I welcom |
| Encoder time | 7.4598s |
| Decoder init time | 0.1451s |
| Step total time | 8.4567s |
| Step avg / token | 37.75ms |
| Total decode time | 16.08s |

### jfk2.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Tha |
| Encoder time | 4.8489s |
| Decoder init time | 0.0798s |
| Step total time | 4.4499s |
| Step avg / token | 19.87ms |
| Total decode time | 9.392s |

### librivox.org-1600hz.wav (en)

| Metric | Value |
|--------|-------|
| Tokens generated | 224 |
| EOS reached | False |
| Decoded text | Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit LibriVox.org. Read by Olivia. A Yea |
| Encoder time | 4.6422s |
| Decoder init time | 0.0781s |
| Step total time | 4.4306s |
| Step avg / token | 19.78ms |
| Total decode time | 9.164s |

## Performance Comparison (JFK_Short / first fixture)

| Variant | Encoder | Init | Step Total | Step/tok | Total | Tokens |
|---------|---------|------|------------|----------|-------|--------|
| fp32 | 7.8982s | 0.2208s | 6.2953s | 28.1ms | 14.429s | 224 |
| fp16 | 13.2893s | 0.1987s | 7.1127s | 31.75ms | 20.657s | 224 |
| q8 | 5.697s | 0.1417s | 6.5138s | 29.08ms | 12.376s | 224 |

## Status Summary

| Variant | Native ORT | Smoke Decode | Accuracy vs FP32 | Status |
|---------|-----------|-------------|-----------------|--------|
| fp32 | pass | pass | reference | baseline |
| fp16 | pass | pass | pending | WebGPU candidate |
| q8 | pass | pass | pending | compact candidate |

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
