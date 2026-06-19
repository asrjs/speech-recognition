# ASR.js Whisper Engine — Complete Handoff

**Branch**: `main`
**Date**: 2026-06-13 (Flexo → Bev migration)
**Source machine**: P520 (WSL2, RTX 5060 Ti 8GB) → **Target**: P520 (Windows native, RTX 5060 Ti)

## Quick Recall (Bev için)

1. `cd ~/github/asrjs/speech-recognition`
2. Bu dosyayı oku → sonra `docs/AGENT_TASKS.md`
3. `asrjs-dev` skill'ini yükle

## Proje Özeti

**Amaç:** Whisper large-v3-turbo'yu browser'da WebGPU ile çalıştırmak. Model, HuggingFace'te `ysdede/whisper-large-v3-turbo-onnx-4graph` adresinde, 4-graph splitgraph formatında (encoder + decoder_init + decoder_step + decoder_align). 5 quantization variant: fp32, fp16, fp16io, q8, mixed.

## Zaman Çizelgesi

### 19 June — Whisper decode parity + fp16 WebGPU beam functional pass ✅

Branch `feat/whisper-cleanup-beam-temperature` now has a correctness-first
Whisper decode pass aligned with OpenAI Whisper/faster-whisper semantics:

- `temperature=0` uses greedy or beam argmax.
- `temperature>0` uses sampling and disables beam search.
- `bestOf` is sampling-only and ignored for `temperature=0`.
- Enhanced temperature fallback now passes each retry temperature to the
  vanilla executor and preserves caller `onTokenLogits` callbacks.
- Language auto-detection uses decoder-init SOT logits and filters real Whisper
  language tokens.
- Beam survivor KV caches stay aligned when completed beams are retained during
  patience-based continuation.
- fp16 splitgraph beam caches carry typed-array storage plus per-beam tensor
  dims through the decoder-init -> decoder-step callback boundary.

Windows Chrome headless harness (`N:\github\asrjs\webgpu-agent-test`) validated:

- Greedy `fp16io-fp16-webgpu&gpuKv=1`: functional pass, zero GPU tensor
  downloads, KV location `gpu-buffer`, RTFx about `28.29`.
- Beam `fp16io-fp16-webgpu&numBeams=2&patience=1`: functional pass on stable
  CPU-KV splitgraph path, KV location `cpu`, RTFx about `2.36`.

Important: beam search is correct but not optimized. Treat the stable CPU-KV
beam path as the oracle for future batched beam work. Batched beam should not be
accepted as a speedup until token parity, EOS behavior, timestamp policy, and KV
parent reordering are proven against this path.

### 14 Haziran — Full fp16 WebGPU + mel perf ✅

Windows Chrome + WebGPU'da custom repo doğrulandı:

- Model source: `ysdede/whisper-large-v3-turbo-onnx-4graph`
- Preset: `fp16io-fp16-webgpu`
- Encoder: `fp16_iofp32`
- Decoder: `fp16`
- 29.9s JFK fixture, 50 token cap, doğru transcript + EOS
- Stage metrics: preprocess `234.63ms`, encode `1732.64ms`, decode `3837.28ms`, total `5812.04ms`, RTFx `5.1452`

Önemli: Loglarda `onnx-community/whisper-large-v3-turbo` görünürse yanlış repo
yükleniyor demektir. Demo/library preset custom 4-graph kaynağına resolve etmeli.

Whisper mel tarafı da optimize edildi: `n_fft=400` korunarak direct DFT yerine
cached Bluestein FFT kullanılıyor. 30s mel benchmark yaklaşık `9185ms` →
`204ms`.

### 30 Mayıs — Entry 023: WebGPU Pipeline İLK ÇALIŞMA ✅

~150 tool call sonunda **ilk başarılı WebGPU Whisper transkripti**:

- **fp16io encoder** (fp16 internal + fp32 I/O) + **fp32 decoder**
- JFK transkripti: "And so, my fellow Americans, ask not what your country can do for you..."
- 25.57s total (encoder 2.13s, decoder 3.32s WebGPU'da)

**Kök neden 6 policy bug (precision DEĞİL):**

1. Yanlış task token (translate → transcribe)
2. Yanlış no_timestamps token ID (50363 → 50364)
3. `suppress_tokens` eksik (~80 token)
4. `begin_suppress_tokens [220, 50257]` eksik → EOS erken
5. Encoder KV preservation eksik → step 2+ crash
6. Custom decode loop → kütüphane kullanılmalı

**Kritik değişiklik:** Tüm decode logic `src/models/whisper-seq2seq/` altında. Browser test page'leri sadece UI shell — kodu library'den sync ediyor.

### 31 Mayıs — Verification + Browser Testing

#### ✅ Node ORT Verification (Steps 1-5, HEPSİ PASS)

| Test                   | Sonuç       | Eşik       |
| ---------------------- | ----------- | ---------- |
| Step 1: Mel            | MSE=0       | ✅         |
| Step 2: Encoder cosine | 0.999987    | ≥0.999 ✅  |
| Step 2: Encoder MSE    | 4.9368e-6   | <0.01 ✅   |
| Step 4: Transcript     | IDENTICAL   | birebir ✅ |
| Step 5: Token-by-token | 27/27 match | 100% ✅    |

**Sonuç:** fp16io encoder Node ORT'de fp32 ile bit-identical. Quality tuning GEREKSİZ.

#### ❌ Browser Testing — 4 Bloker

| #   | Sorun                           | Detay                                             |
| --- | ------------------------------- | ------------------------------------------------- |
| 1   | **fp32 encoder 2.4GB**          | Browser fetch limit ~1.5-2GB → `Failed to fetch`  |
| 2   | **WASM fp16 desteksiz**         | fp16io WASM'de çöp çıktı ("a, a,")                |
| 3   | **WASM heap limit**             | ~1.5GB → encoder+decoder birlikte yüklenemiyor    |
| 4   | **Headless browser WebGPU yok** | GPU adapter yok (WSL2) → **Windows Chrome gerek** |

### 13 Haziran — IO + Cache Rework (bugün pushlanan)

Az önce pushlanan 4 commit:

```
04b3bfb feat(whisper): beam search patience parameter for early stopping
a73ca0e chore: lint fixes — const declarations, wav2vec2 builtin imports
51dcbeb feat(wav2vec2): external data support, model family registration
5f89cec feat: io handles, whisper executor enhancements, IndexedDB cache rework
```

**io/cache.ts** ve **io/handles.ts** rework — IndexedDB cache altyapısı güçlendirildi:

- `IndexedDbAssetCache` → blob storage eklendi
- `resolveAssetHandle` → HF URL builder + BlobAssetHandle
- Wav2Vec2 model ailesi registration + external data desteği

## WebGPU Testleri — Bev'in Rolü

**Bev (Windows host, RTX 5060 Ti + gerçek Chrome) WebGPU testlerini yapıyordu.** Flexo (WSL2) headless browser'da WebGPU adapter alamıyor.

Test sayfası: `/mnt/n/github/asrjs/webgpu-agent-test/index.html`

- Tüm variant'lar: fp32, fp16, fp16io, q8, mixed
- Tüm backend'ler: WebGPU, WASM
- Cross-validation mode
- Library-synced (kendi decode loop'u YOK)

**Bev'in yapacağı testler:**

```bash
cd /mnt/n/github/asrjs/webgpu-agent-test
npx serve -l 8765
# Chrome'da https://localhost:8765/ aç
# Önce fp16io + fp32 decoder dene (çalışan tek kombinasyon)
```

## Model Boyutları ve Hangi Backend'de Çalışır

| Variant    | Encoder      | Decoder      | WASM         | WebGPU   | Not                        |
| ---------- | ------------ | ------------ | ------------ | -------- | -------------------------- |
| fp32       | 2.4GB        | 761MB        | ❌ heap      | ❌ fetch | Çok büyük                  |
| fp16       | 1.2GB        | 381MB        | ❌ fp16 yok  | ✅       | Küçük ama WASM'e uymaz     |
| **fp16io** | 1.2GB        | 761MB        | ❌ fp16 yok  | ✅       | **Ana hedef**              |
| q8         | 616MB inline | 643MB inline | ❌ bad_alloc | ?        | En küçük ama decoder büyük |
| mixed      | 616MB inline | 582MB        | ?            | ?        | Test edilmedi              |

## Proje Yapısı

```
~/github/asrjs/speech-recognition/
  src/
    models/whisper-seq2seq/
      core.ts              — decode (greedy, beam, bestOf, patience)
      processor.ts         — WhisperTimestampLogitProcessor
      executor.ts          — ORT bridge, splitgraph, KV management
      enhanced-executor.ts — production pipeline
      generation-config.ts — config parsing
    models/wav2vec2/       — Wav2Vec2 executor, ORT bridge, types (yeni)
    audio/whisper-mel.ts   — mel spectrogram + padToFrames
    io/cache.ts            — IndexedDbAssetCache (blob desteği eklendi)
    io/handles.ts          — resolveAssetHandle, HF URL builder
    presets/whisper/manifest.ts — onnx-community/whisper-large-v3-turbo preset
  tests/smoke/
    verify-step1-mel.mjs              — Mel (MSE=0 ✅)
    verify-step2-encoder.mjs          — Encoder fp16io vs fp32 (cos=0.999987 ✅)
    verify-step3-5-decode.mjs         — Full decode (27/27 tokens ✅)
    whisper-large-v3-turbo-native.mjs — Native ORT smoke
    whisperx-runner.mjs               — WhisperX-compatible runner
  examples/demo/         — YENİ: HTTPS Vite demo app (certs/, audio samples)
  docs/
    SESSION_HANDOVER.md  — Bu dosya
    AGENT_TASKS.md       — Görev listesi
```

```
/mnt/n/github/asrjs/webgpu-agent-test/  (Windows N: sürücüsü)
  index.html       — Verification suite (library-synced)
  models/          — fp32, fp16, fp16_iofp32, q8, mixed ONNX modeller
  INSTRUCTIONS.md  — Test talimatları
```

## Sıradaki Görevler (Bev için)

| #   | Görev                                     | Öncelik      | Not                                                                                |
| --- | ----------------------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| 1   | IndexedDB cache ile browser model loading | 🔴 Yüksek    | `createSpeechPipeline({ cacheModels: true })` veya low-level `IndexedDbAssetCache` |
| 2   | WebGPU verification (gerçek Chrome)       | 🔴 Yüksek    | Bev'in GPU'suyla test sayfasını çalıştır                                           |
| 3   | fp16io + fp32 decoder WebGPU'da test      | 🔴 Yüksek    | Çalışan tek kombinasyon, doğrula                                                   |
| 4   | q8/mixed variant'ları dene                | 🟡 Orta      | Daha küçük modeller, WASM'e uyabilir                                               |
| 5   | int8 model generation for WASM            | 🟡 Orta      | `onnxruntime.quantization.quantize_dynamic`                                        |
| 6   | Batched encoder                           | ⬜ Ertelendi | CPU'ya faydası yok                                                                 |

## Doğrulama Komutları

```bash
cd ~/github/asrjs/speech-recognition

# Node ORT verification (hepsi WSL2'de çalışıyor)
node tests/smoke/verify-step1-mel.mjs            # Mel MSE=0
node tests/smoke/verify-step2-encoder.mjs        # Encoder cosine > 0.999
node tests/smoke/verify-step3-5-decode.mjs       # Decode 27/27 match

# Standard tests (601 test)
npm run typecheck && npm run lint && npm test
npm run build

# Browser testing (SADECE Windows Chrome'da)
cd /mnt/n/github/asrjs/webgpu-agent-test
npx serve -l 8765 --ssl
# Chrome'da aç: https://localhost:8765/
```

## Önemli Notlar

1. **WASM'de fp16 ÇALIŞMAZ** — ORT Web WASM backend fp16 ops'ları desteklemiyor. fp16io encoder WASM'de "a, a," gibi çöp çıktı üretiyor. fp16io **sadece WebGPU** için.
2. **Encoder KV preservation** — decoder_step her çağrıda encoder KV'leri düşürüyor. Library'de `executor.ts` bunu handle ediyor.
3. **IndexedDB cache** — 2.4GB model dosyalarını fetch etmek için IndexedDB şart. Library'de altyapı var (`io/cache.ts`, `io/handles.ts`), test page'de henüz yok.
4. **fp16io = fp32 kalitesinde** — Node ORT'de kanıtlandı. WebGPU'daki kalite sorunları policy bug'lardandı, precision'dan değil.
5. **ORT WebGPU bilinen bug'lar** (test edildi 2026-05-31): fp16 decoder NaN, q8 decoder KV cache quantization error, ConvInteger düzeltildi (ORT 1.26.0). fp32 decoder tek sağlıklı seçenek.

## Oturum Kayıtları

- Session `20260530_182641_988654`: Entry 023 — WebGPU pipeline working, 6 policy bug fixed
- Session `20260531_031225_84ffded3`: fp16io verification pipeline complete, browser testing blocked
- Skill: `asrjs-dev` (proje genel), `whisper-model-verification-pipeline` (verification workflow)
