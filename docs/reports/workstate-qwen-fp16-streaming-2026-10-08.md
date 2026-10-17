# Qwen fp16 ve Moonshine streaming — çalışma durumu ve devir notu

Tarih: 2026-10-08. Yazan: OMP oturumu. Amaç: bu oturumda yürütülen aktif işi
ana ajana devretmek. Bu bir hedef belgesi veya yeni kuyruk değildir. Kalıcı
hedef [GOAL_PROMPT_RECOMMENDED.md](../GOAL_PROMPT_RECOMMENDED.md), iş kayıtları
[PRODUCT_TASKS.md](../PRODUCT_TASKS.md) ve
[PARAKEET_VARIANT_TASKS.md](../PARAKEET_VARIANT_TASKS.md) içindedir. Durum
özeti ve çalışma rehberi
[model-port-status-and-guide-2026-10-08.md](model-port-status-and-guide-2026-10-08.md)
dosyasındadır.

## 0. Kısa özet — şu an ne ile uğraşıyorum

Kullanıcının **"Whisper'da yaptığımız gibi float16 weights ve surgical graph
creation kullan"** talimatı üzerine Qwen3-ASR 0.6B'yi browser'a sığdırma
yolundayım. Bu turda üç danışman bulgusu doğrulandı ve iki önemli sonuç ölçüldü:

| Konu | Ölçülen sonuç |
| --- | --- |
| fp16 dönüşümü (ORT converter) yüklenmesi | Node'da `graphOptimizationLevel: 'basic'` ve `'extended'` ile **LOAD OK**; varsayılan `'all'` bu ORT build'inde çöküyor. String option sözleşmesi geçerli. |
| fp16 decoder **çalıştırma** | **BAŞARISIZ.** `Concat` node'u buffer yeniden kullanımında çakışıyor. Hem Node 1.25-dev hem Python ORT 1.30'da, past uzunluğu 0 ve 1'de, tüm session seçeneklerinde tekrarlıyor. fp32 graph aynı feed ile sorunsuz. |
| Moonshine streaming yüklemesi | transformers 5.18.0 ile **temiz yükleme: 0 missing / 0 unexpected / 0 mismatched key**, conv1 `(640, 320, 5)`. Eski "random encoder" engeli çözüldü; diskteki graph'lar hâlâ geçersiz, yeniden export gerekiyor. |
| "320-bin mel frontend" iddiası | **YANLIŞ.** 320 = `hidden_size`. Encoder ham dalga-formu işler: kare `frame_ms=5.0` → 80 örnek → `Linear(80→320)` → causal conv'lar. Mel yok. |

Ayrıca: provenance düzeltmeleri ve Türkçe durum rehberi yazıldı (bölüm 4),
diskteki kirli/atık durum belgelendi (bölüm 5).

## 1. Bağlam — neden bu iş

- Kullanıcı talimatı (2026-10-05): Whisper WebGPU çalışmasındaki gibi fp16
  weights ve surgical split graph'lar kullan. Bu izin tolerance düşürme veya
  integer quantization değildir.
- Qwen3-ASR fp32 seti 3.75 GB; browser WASM session'ı ~6.99 GB tahsis isteğiyle
  4 GB WASM32 sınırını aşıyor (kayıt: `tools/data/results/browser/asr-browser-measurements-2026-10-04.json`).
  fp16 hedefi tam bu yüzden: toplamı yarıya indirmek.
- Üç danışman bulgusu doğrulandı: (T-13) eski runner'da optimization option'ı
  yoktu, string option sözleşmesi geçerlidir; (T-9) 320 mel değil hidden_size;
  (T-21) rapor mevcut sıralı kayıtlara dayanmalı ve in-flight işleri
  tamamlanmış gibi göstermemeli.

## 2. Qwen3-ASR fp16 — ölçülen durum

### 2.1 Dönüşüm (ORT converter yolu) — yükleme çalışıyor, çalıştırma çalışmıyor

- Kaynak: `models/onnx/qwen3-asr-0.6b/` (fp32, 3.75 GB ağırlık).
- Dönüşüm: `python tools/nemotron-onnx-export/convert_fp16.py --src-dir models/onnx/qwen3-asr-0.6b --dst-dir models/onnx/qwen3-asr-0.6b-fp16`
  (~57 s). Bayraklar: `keep_io_types=True`, `op_block_list={'Softmax'}`,
  `disable_shape_infer=True` (`convert_fp16.py:41-50`).
- Sonuç set: **`models/onnx/qwen3-asr-0.6b-fp16/` = 1.8 GB**:
  - `audio_front_model.onnx{,.data}` = 1.67 MB + 372.8 MB
  - `decoder_step_model.onnx{,.data}` = 1.60 MB + 1,192,085,504 B (1.19 GB)
  - `embed_tokens_model.onnx{,.data}` = 407 B + 311.2 MB
  - Artı kopyalanan destek dosyaları: `tokenizer.json`, `jfk_features.npy`,
    `jfk_features_mask.npy`, `jfk_cu_seqlens.bin`,
    `prompt_prefix_embeds.npy`, `prompt_suffix_embeds.npy`.
- fp16 decoder `.data` referanslı toplam = dosya boyutu = 1,192,085,504 B
  (kompakt; fp32'deki referanslı 2,384,171,008 B'nin tam yarısı).

**Optimization seviyesi probe'u** (graph: fp16 `decoder_step_model.onnx`):

| Seviye | Node `onnxruntime-node` 1.25.0-dev | Python `onnxruntime` 1.30.0 |
| --- | --- | --- |
| `disabled` | Type Error (`InsertedPrecisionFreeCast_/...` float vs float16) | Type Error (aynı) |
| `basic` | **LOAD OK** | LOAD OK |
| `extended` | **LOAD OK** | LOAD OK |
| `all` (varsayılan) | `GetIndexFromName` çöküşü: `InsertedPrecisionFreeCast_/language_model/layers.19/self_attn/q_norm/Constant_output_0` | LOAD OK |

Yani: string `graphOptimizationLevel` sözleşmesi `onnxruntime-common`
`inference-session.d.ts:69` ile uyumlu çalışıyor; eski "enum yok, kontrol yok"
çıkarımı yanlıştı. Varsayılan çöküş bu ORT build'inin LayerNorm füzyonuna özgü.

**Runner düzeltmesi (uygulandı, doğrulandı):**
`tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs` artık
`--opt-level=<disabled|basic|extended|all>` (varsayılan `all`) alır ve üç
session'a
`sessionOptions = { executionProviders: ['cpu'], graphOptimizationLevel }`
geçirir. Not: önceki iki deneme python `str.replace` ile yapılmıştı ve **sessizce
uygulanmamıştı** (eşleşme yoksa da koşulsuz `print` yazıyordu). Bu yüzden 'basic'
daha önce hiç test edilmemişti. Şimdi `edit` aracıyla yapıldı ve dosya okunarak
doğrulandı; `node --check` geçti.

**Çalıştırma hatası (güncel bloker):**

```
node tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs --model-dir=models/onnx/qwen3-asr-0.6b-fp16 --opt-level=basic
→ Non-zero status code returned while running Concat node. Name:'/Concat_2'
  Status Message: ... Shape mismatch attempting to re-use buffer.
  {28,1,8,0,128} != {28,1,8,2,128}. Validate usage of dim_value ... in shapes in the model.
```

Tekrarlama matrisi (hepsi aynı hata):

| Koşul | Sonuç |
| --- | --- |
| Node, basic, boş past (0) | FAIL |
| Node, basic, past uzunluğu 1 | FAIL |
| Node, basic + `enableMemPattern:false` / `enableCpuMemArena:false` / ikisi | FAIL |
| Node, extended | FAIL |
| Python ORT 1.30, `all`, boş past | FAIL |
| **fp32 graph, aynı feed** | **RUN OK** (`logits 1x2x151936`, `present_key 28x1x8x2x128`) |

Yapısal gözlem: fp16 graph'ta `keep_io_types` dönüşümü nedeniyle
`/Concat_1` çıktısı `graph_output_cast_1`, `/Concat_2` çıktısı
`graph_output_cast_2` olarak yeniden adlandırılmış; graph çıktıları hâlâ
`present_key`/`present_value`. fp32 graph'ta bu Concat'ler doğrudan
`present_*` üretir ve aynı feed ile çalışır. Değer bilgisi (value_info) kaydı:
fp16'da 6 kayıt var (`graph_input_cast_1/2`, `graph_output_cast_1` vb.);
fp32'de hiç value_info yok.

**Sonuç:** Dönüştürülmüş fp16 decoder bu ORT build'lerinde çalıştırılamıyor.
Bu bir session-option sorunu değil; graf/planner sorunu. Çözüm yolları bölüm
6.1'de.

### 2.2 Doğrudan fp16 export (PyTorch'tan) — yarım

- Dosya: `tools/qwen3-asr-onnx-export/export_qwen3_asr_fp16.py` (untracked, bu
  oturumun işi). Şu an yalnızca **front + embed** export ediyor; **decoder
  export'u yok**.
- Bilinen hata: `RuntimeError: Input type (float) and bias type (struct c10::Half)
  should be the same`. Wrapper `input_features`'i `.to(dtype)` yapıyor ama
  **mask** float32 kalıyor; audio tower'ın conv mask yolu fp16 bias ile float
  girdi görüyor. İlk denenecek düzeltme: `input_features_mask.to(dtype)` (veya
  tower içi conv mask hesabını dtype'a çevirmek).
- Ek not: script'in içindeki `capture_schedule(model, features)` çağrısı
  tower'ı wrapper'dan önce çalıştırır; yalnızca wrapper'a cast eklemek bunu
  değiştirmez. Çağrı da dtype tutarlı olmalı.
- Bu yol tercih edilir uzun vadede: fp16 cache/logits tensörlerini doğrudan
  üretir ve converter'ın geometri/füzyon artefaktlarını taşımaz. Whisper
  planındaki "surgical graph" anlayışı budur.

### 2.3 fp32 baseline (referans)

- `node tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs --model-dir=models/onnx/qwen3-asr-0.6b`
  → oracle metni birebir, 30 step, 4333 ms. (Runner düzenlemesinden **sonra**
  tekrar koşuldu; varsayılan `all` ile aynı davranış, regresyon yok.)
- Oracle: `tools/data/results/model-candidates/qwen3-asr-0.6b-baseline-2026-10-04.json`
  → `And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.`

## 3. Moonshine streaming — düzeltilen gerçekler

### 3.1 Mimari (kanıt: transformers 5.18 kaynak kodu)

`tools/model-debugging/reference/transformers5/Lib/site-packages/transformers/models/moonshine_streaming/modeling_moonshine_streaming.py`:

- Encoder **ham dalga-formu** alır: `input_values` → `frame_len =
  sample_rate * frame_ms / 1000` (config: `sample_rate=16000`, `frame_ms=5.0`
  → **80 örnek**) → `cmvn` → `Linear(80, 320)` → `transpose` →
  `CausalConv1d(320, 640, k=5, stride=2)` → `silu` →
  `CausalConv1d(640, 320, k=5, stride=2)` (satırlar ~288-313).
- `hidden_size = 320` (hem kök config hem `encoder_config`). `head_dim = 40`,
  6 katman, 8 head, `vocab_size = 32768`, `max_position_embeddings = 4096`.
- Yayınlanan `preprocessor_config.json` = `Wav2Vec2FeatureExtractor`,
  `sampling_rate=16000`, `do_normalize=false`; streaming export receipt'inde
  `pad_to_multiple_of: 80`. Bu **ham dalga-formu sözleşmesiyle uyumludur**;
  non-streaming Moonshine ile aynı frontend. **Mel frontend yok.**

Eski C49/streaming-export kaydındaki "320-dimensional mel features" ve
"needs a mel frontend" ifadeleri yanlıştı; düzeltildi (bölüm 4).

### 3.2 Yükleme — ÇÖZÜLDÜ (5.18 ile)

```
tools/model-debugging/reference/transformers5/Scripts/python.exe -c "...from_pretrained('.cache/moonshine-streaming-tiny', output_loading_info=True)"
→ transformers 5.18.0
→ config class: MoonshineStreamingConfig
→ missing keys: 0 | unexpected keys: 0 | mismatched keys: 0
→ enc conv1 shape: (640, 320, 5)   # checkpoint ile birebir
```

Yani eski "installteki transformers streaming sınıfına sahip değil, encoder
rastgele initialize ediliyor" engeli **artık geçerli değil**: 5.18 stack'i ile
checkpoint doğru yükleniyor. C49'daki `EXPORT_INVALID_RANDOMLY_INITIALIZED_ENCODER`
kaydı o tarihte (4.57.6) doğruydu; bu not 5.18 durumunu belgeler.

### 3.3 Diskteki graph'lar GEÇERSİZ — yeniden export gerekli

- `models/onnx/moonshine-streaming-tiny-fp32/` içindeki üç graph'ın sha256'ları,
  `tools/data/results/moonshine/streaming-export-2026-10-04.json` (20:16) ile
  **birebir aynı** ve o receipt'in status'ü
  `EXPORT_INVALID_RANDOMLY_INITIALIZED_ENCODER`. Graph'lar 20:05'te yazılmış;
  receipt 11 dakika sonra geçersiz ilan etmiş. Yani diskteki set = geçersiz
  export. **Kullanılmamalı.**
- sha256 eşleşmeleri: encoder `cea305f8…`, crossKv `7b666362…`,
  decoder `17d0430d…`. Klasördeki `export-receipt.json` "validated checkpoint"
  ifadesi taşıyor ama o yükleme doğrulaması yapılmamıştı; geçerlilik kararı
  20:16 receipt'indedir.
- Sıradaki adım: 5.18 stack'i ile **yeniden export** (aynı araç:
  `tools/moonshine-onnx-export/export_moonshine_pair.py`; runner geometriyi
  receipt'ten okur, `slots=12 heads=8 headDim=40`).
- Oracle engeli: eski receipt'te `model.generate` `ValueError: text input must
  be of type str...` veriyordu (4.57.6). 5.18 ile **tekrar denenmeli**; oracle
  olmadan hiçbir streaming gate kurulamaz.
- Streaming decode sözleşmesi henüz yok: chunked/righ-context decode ayrı iş
  (receipt: "a streaming-trained decoder fed whole-utterance encoder states is
  out of distribution").

## 4. Bu oturumda yazılan/değişen dokümanlar (uncommitted)

Yeni dosyalar:

- `docs/reports/model-port-status-and-guide-2026-10-08.md` — Türkçe durum
  raporu + çalışma rehberi (kullanıcının istediği "oku anla toparla raporla
  guide hazırla" teslimatı).
- Bu dosya: `docs/reports/workstate-qwen-fp16-streaming-2026-10-08.md`.
- `tools/qwen3-asr-onnx-export/restore_dynamic_axes.py` — converter'ın sembolik
  ekseni sabitlemesi olasılığına karşı yazıldı; bu graph'larda **gerekmedi**
  (eksenler zaten sembolik: `past_length`). Gelecekteki dönüşümler için duruyor,
  şu an kullanılmıyor.
- `tools/qwen3-asr-onnx-export/export_qwen3_asr_fp16.py` — doğrudan fp16 export
  (bölüm 2.2; front+embed var, decoder eksik).

Düzenlenen dosyalar (hepsi bu oturumun):

- `docs/AGENT_ENTRYPOINT.md` — yeni rehbere routing satırı.
- `docs/PRODUCT_TASKS.md` — üstte "2026-10-08 model provenance and status
  correction" bölümü; C32'de `ctaguchi` satırı düzeltildi.
- `docs/PARAKEET_VARIANT_TASKS.md` — "Provenance correction — 2026-10-08"
  bölümü; P10 başlığı ve içeriği "bounded survey" olarak yeniden yazıldı.
- `tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs` — `--opt-level`.
- `tools/qwen3-asr-onnx-export/capture_cu_seqlens.py` — `capture_schedule()`
  fonksiyonu eklendi (yeniden kullanılabilir).

Provenance düzeltmesinin kaynağı:
`https://huggingface.co/ctaguchi/parakeet-hf-ctc-110m-en/raw/main/README.md`
→ "Converted from `nvidia/parakeet-tdt_ctc-110m`". Yani repo sahipliği model
kökeni değildir; mevcut küçük CTC portu korunur ama "bağımsız non-NVIDIA
student" etiketi geri çekildi.

## 5. Diskteki kirli durum (in-flight) ve atıklar

### 5.1 Git durumu

- **Modified (18):** 10 protected Nemotron dosyası + `capture_cu_seqlens.py` +
  `transcribe_qwen3_asr_local.mjs` + 3 doküman (bu oturum) +
  `docs/GOAL_PROMPT_RECOMMENDED.md`, `tools/parakeet-ternary/ternary.py`,
  `tools/parakeet-ternary/test_ternary.py` (**bu oturuma ait değil; kullanıcı/
  önceki oturum işi — dokunulmadı**).
- **Untracked (seçme):** yukarıdaki yeni dosyalar + `docs/reports/light-asr-diarization-hunt-2026-09-30.md`,
  `tools/model-debugging/reference/transformers5/` (venv),
  `tools/model-debugging/reference/hf-parakeet-port/probe_ort_dft.py` (+test).
- Kural: yalnızca açık yol ile stage et; commit öncesi
  `git diff --cached --name-only` yazдır. `git add -A` yok. Protected 10 dosya
  asla stage edilmez.

### 5.2 Atıklar ve boyutlar

- `models/onnx/qwen3-asr-0.6b/onnx__MatMul_*`: **412 dosya, toplam 5.5 GB**.
  Üç graph'ın hiçbiri bu dosyaları referans etmiyor (doğrulandı: tüm
  initializer'ların `location` alanı yalnızca kendi `.onnx.data` dosyasını
  gösteriyor). Ölü debris; silinmesi güvenli (silme sonrası fp32 gate tekrar
  koşulmalı). **Henüz silinmedi.**
- `models/onnx/qwen3-asr-0.6b/decoder_step_model.onnx.data` = **7.15 GB** ama
  graph'ın referansladığı toplam veri **2.38 GB** (offset+length toplamı;
  dosya sonuna uzanan bölge dışında ~4.77 GB referanssız bayt var). Bu, geçmiş
  bir çift-save artefaktı. Kompaksiyon hash'leri değiştirir; mevcut receipt'ler
  o hash'lere bağlı olduğundan **şimdilik dokunulmadı**; karar ana ajanda.
- `.cache/levelprobe.mjs`, `.cache/pastprobe{,2,3}.mjs` — bu turun geçici
  probe'ları (.cache, gitignored).
- `dev server` (webgpu-agent-test :8765) bu turda kullanılmadı.

## 6. Sıradaki adımlar (öncelik sırasıyla)

### 6.1 Qwen fp16 — decoder'ı çalıştırılabilir yap (öncelik 1)

İki yol var; ucuzdan pahalıya:

1. **Graf yapısını karşılaştır.** fp32 vs fp16 `decoder_step_model.onnx`
   içinde `/Concat_1`, `/Concat_2` ve çevresindeki `Cast` zincirini, ayrıca
   bu düğümlerin çıktılarına giden value_info'ları karşılaştır. Hipotez:
   converter `keep_io_types` ile çıktı Cast'i eklerken Concat çıktısını
   yeniden adlandırdı ve ORT planner'ı bu çıktı için yanlış buffer planlıyor.
   Deney: Cast'i koruyup Concat'in ara çıktısını ayrı isimle geri koymak
   (graph surgery), veya `disable_shape_infer=False` ile yeniden dönüştürmek.
2. **Doğrudan fp16 decoder export'u** (bölüm 2.2): `export_qwen3_asr_fp16.py`
   içine decoder export'u ekle. Referans: `tools/qwen3-asr-onnx-export/export_qwen3_asr_decoder.py`
   (legacy tracer, **boş olmayan past örneği** ile trace — boş past mask'i tek
   pozisyona özelleştirip ikinci adımda 2.603 sapma üretmişti; uzunluk-4 örnek
   bunu düzeltmişti; sidecar konsolidasyonu `all_tensors_to_one_file=True`).
   Wrapper'da mask'i de dtype'a çevir (bölüm 2.2). Front için de aynı dosya
   kullanılır.
3. fp16 çalışınca: **Node gate** (oracle metni) → **parity metrikleri**
   (fp16 vs fp32: front `audio_embeds`, decoder ilk-adım `logits`; max/mean
   abs, allclose(1e-4,1e-3), cosine — açık kayıt) → **browser WASM ve WebGPU**
   ölçümü. Bellek hesabı: fp16 toplam 1.88 GB; fp32'de WASM ~6.99 GB istiyordu,
   fp16'da ~3.5 GB beklenir — 4 GB WASM32 sınırına **yakın**, ölçülmeden
   "sığar" denemez. Browser harness: `N:/github/asrjs/webgpu-agent-test`
   (`npm run dev` :8765), `node scripts/run-ctc-asr-webgpu.mjs --model=qwen3asr ...`.

### 6.2 Moonshine streaming — yeniden export + oracle (öncelik 2)

1. `tools/model-debugging/reference/transformers5/Scripts/python.exe` ile
   streaming export'u yeniden koş (`export_moonshine_pair.py`, çıktı yeni
   klasöre; eski geçersiz set korunur ama kullanılmaz).
2. 5.18 ile reference transcript üret (eski `generate` hatası tekrar denenir).
3. Chunked/right-context streaming decode sözleşmesini kur; partial/final
   bitiş davranışını oracle ile karşılaştır.
4. Geometri receipt'ten okunur (`slots=12 heads=8 headDim=40`); runner artık
   sabit değerlere bağlı değil.

### 6.3 Doküman ve temizlik (öncelik 3)

- Türkçe rehbere (T-21 eksikleri): `docs/reports/hf-multilingual-asr-candidate-survey-2026-08-30.md`
  kaynaklı ek lead'ler (Granite Speech 4.1 2B, GLM-ASR-Nano-2512, VibeVoice-ASR,
  Voxtral Mini 3B ONNX, Canary 1B v2 ONNX, Qwen 1.7B) tek paragraf halinde
  eklenmeli; bunlar adapt/defer kararlarıdır, port taahhüdü değildir.
- 412 shard'ın silinmesi + fp32 gate'in tekrar koşulması (kanıt: gate PASS).
- `.cache` probe scriptlerinin silinmesi.
- Commit: yalnızca sahiplenilen dosyalar (bölüm 4 listesi), explicit path ile.
  Protected Nemotron dosyaları ve kullanıcı dosyaları hariç.

## 7. Bu turda öğrenilen tuzaklar

- **Python `str.replace` sessizce eşleşmeyebilir.** Koşulsuz `print` bunu
  gizler. Dosya düzenlemeleri `edit` aracıyla yapılmalı ve sonrasında dosya
  okunarak doğrulanmalı. (Runner düzenlemesi bu yüzden iki kez uygulanmamıştı.)
- **`console`/`ls` çıktıları kırpılır.** Tam metin için `artifact://<id>`
  okunmalı (örn. hata mesajları).
- **ORT float16 converter çıktısı runtime'da bozuk olabilir.** Yükleme
  başarısı çalıştırma kanıtı değildir; graph'ı gerçek feed ile koş.
- **`disabled` optimization seviyesi dönüştürülmüş graph'ta type hatası verir;**
  `basic`/`extended` çalışır. Varsayılan `all` bu Node build'inde çökebilir.
- Boş past feed'i (`[28,1,8,0,128]`) fp32 graph'ta sorunsuz; fp16 dönüşümünde
  Concat buffer planı bozuluyor.

## 8. Doğrulama sınırı (bu turda YAPILMAYANLAR)

- fp16 decoder hiçbir runtime'da çalıştırılamadı; **fp16 parity ölçümü yok**.
- Browser ölçümü bu turda yapılmadı (fp16 Node'da çalışmadığı için).
- Streaming yeniden export edilmedi; diskteki set hâlâ geçersiz.
- Tam test suite bu turda koşulmadı; yalnızca fp32 Node gate (PASS) ve
  `node --check` koşuldu. Son bilinen suite durumu: 1,479 passed / 33 skipped
  (önceki tur; bu turun değişiklikleri test kapsamı dışı).
- 412 shard ve 7.15 GB `.data` dosyası silinmedi/kompakte edilmedi.
- Türkçe rehberdeki 320-mel ifadesi ve fp16 satırı bu turda düzeltildi
  (rehber §5 satır 82 ve 84, §7.C satır 154).

## 9. Ana ajan için ilk üç komut

```powershell
# 1. fp32 baseline hâlâ yeşil mi (hızlı):
node tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs --model-dir=models/onnx/qwen3-asr-0.6b

# 2. fp16 hatasını yeniden üret (beklenen: Concat_2 buffer hatası):
node tools/model-debugging/scripts/transcribe_qwen3_asr_local.mjs --model-dir=models/onnx/qwen3-asr-0.6b-fp16 --opt-level=basic

# 3. Streaming load kontrolü (beklenen: 0/0/0, conv1 (640,320,5)):
tools/model-debugging/reference/transformers5/Scripts/python.exe -c "from transformers import AutoModelForSpeechSeq2Seq; m,i=AutoModelForSpeechSeq2Seq.from_pretrained('.cache/moonshine-streaming-tiny', dtype='float32', output_loading_info=True); print(len(i['missing_keys']), len(i['unexpected_keys']), len(i['mismatched_keys']))"
```
