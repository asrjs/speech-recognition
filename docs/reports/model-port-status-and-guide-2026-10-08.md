# ASR model portları: durum raporu ve çalışma rehberi

Tarih: 2026-10-08.

**Hedef, model isimlerini çoğaltmak değil; doğrulanmış modelleri kompakt bir browser ve native speech runtime içinde sunmak.** Model sırası: Orukeet → gerçek küçük Parakeet student/distill adayları → sıralanmış non-NVIDIA adayları. FireRed ve Qwen ASR 0.6B özgün ilgi alanları olarak listede kalır. Nemotron çalışması duraklatılmıştır.

Bu belge, kullanıcının durum incelemesi ve rehber isteğini karşılar. Yeni bir handoff, goal veya rakip task kuyruğu değildir. Kalıcı hedef [GOAL_PROMPT_RECOMMENDED.md](../GOAL_PROMPT_RECOMMENDED.md), işler [PRODUCT_TASKS.md](../PRODUCT_TASKS.md) ve [PARAKEET_VARIANT_TASKS.md](../PARAKEET_VARIANT_TASKS.md) içinde kalır.

## 1. Yapmamız gereken iş

1. Mevcut Orukeet portunu ve açık acceptance kapılarını koru.
2. Parakeet adaylarında gerçek parameter sayısını ve model kökenini doğrula.
3. Gerçek küçük student ile quantization, format conversion ve full-size fine-tune ayrımını yap.
4. Non-NVIDIA adaylarını kalite, Türkçe/çok dilli değer, lisans, browser belleği ve export yolu ile değerlendir.
5. Seçilen modeli özgün ağırlık → original-engine oracle → kendi fp32 ONNX exportumuz → sayısal karşılaştırma → public library → gerçek host/provider zincirinden geçir.
6. fp16 ve surgical split graphs kullanırken aynı oracle zincirini koru.
7. Her porttan mevcut harness'e tekrar kullanılabilir bir iyileştirme bırak.

Ürün kapsamı ayrıca API, asset loading, cancellation, reset/disposal, uzun ses, realtime, örnekler ve paket boyutudur. Yalnızca converter ve benchmark üretmek teslimat değildir. Kaynak: [kalıcı hedef §3–10](../GOAL_PROMPT_RECOMMENDED.md), [charter](../PROJECT_CHARTER.md).

## 2. Önceki akışta düzeltilmesi gereken iddialar

| Önceki iddia | Kanıta dayalı düzeltme |
| --- | --- |
| Orukeet tamamen bitti. | Native uzun-ses token/text kontrolü ve belirli browser kontrolleri geçti. Production-feature kapısı hâlâ FAIL. Evrensel model acceptance yok. |
| Redux yaklaşık 149M parameter; küçük distill. | Ternary dosya büyüklüğü parameter sayısı sanılmış. Gerçek toplam 627,270,687. Küçük student değil. |
| Ultra ve primeline küçük Parakeet alternatifleri. | Ultra 627,270,663; primeline yaklaşık 627M. İncelenen adaylar full-size sınıfında. |
| `ctaguchi/...110m-en` bağımsız non-NVIDIA student. | Upstream model kartı açıkça `nvidia/parakeet-tdt_ctc-110m` format conversion diyor. Repo sahibi model kökeni değildir. Küçük CTC yolunun mevcut kodunu koru; bunu yeni bağımsız distillation diye sunma. |
| NeMo kurulu değil; yeniden kurmak gerekiyor. | Mevcut NeMo environment kayıtları var. Interpreter ve sürüm kimliğini doğrula; eski oracle'ı başka stack ile sessizce değiştirme. |
| Qwen 0.6B hiçbir browser backend'inde kullanılamaz. | Son `-hf` fp32 export/host yolu bellek sınırına çarptı. Ayrı, daha eski dynamic encoder + native fp16 decoder yolu için gerçek browser output PASS kayıtları var. |
| `graphOptimizationLevel` için JS enum zorunlu; enum yoksa kontrol yok. | Kurulu `onnxruntime-common` contract'ı string kabul ediyor. Enum export'unun olmaması kontrolün olmadığı anlamına gelmez. Son Node runner'da `--basic-opt` session options'a bağlı değil. |
| İkinci Qwen fixture reddedildi; iş tamamlandı. | Bu bir FAIL/genelleme açığıdır. Windowing önerisi çalışan ve gated bir çözüm değildir. |
| fp16 dosyaları yarı boyuta indi; port tamamlandı. | Dosya boyutu load/parity/peak-memory başarısı değildir. Son fp16 script decoder export etmiyor; son deneme dtype hatasıyla durdu. |
| Moonshine'da `basic` sonucu, fusion'ın neden olamayacağını kesin kanıtlıyor. | Kaydedilen toggle bu configuration'da çıktıyı düzeltmedi. Gerçek options, ORT bundle ve optimized partition kimliği olmadan bütün kernel/fusion nedenlerini dışlama. |

Small-model köken düzeltmesinin birincil kaynağı: [ctaguchi model kartı](https://huggingface.co/ctaguchi/parakeet-hf-ctc-110m-en/raw/main/README.md), bu incelemede okundu. `Converted from nvidia/parakeet-tdt_ctc-110m` ifadesi, C32/P10'un önceki köken etiketini geçersiz kılar. Yeni NVIDIA işleri dışlama kuralını bu format conversion ile aşma. Mevcut portun korunması, yeni model çalışmasının kapsam onayı değildir.

## 3. Parakeet hattı

| Aday | Doğrulanmış ayrım | Karar |
| --- | --- | --- |
| `oruk/orukeet` | Yaklaşık 627M Parakeet v3 fine-tune; küçük student değil. | Mevcut portu kullan. Açık kapıları ayrı göster. |
| `moondream/parakeet-redux` | 603,979,776 ternary + 23,290,911 dense = 627,270,687 parameter. | Küçük-distill hedefini karşılamaz. Öğretmen grafiğine substitution denemesi başarısız; tekrar başlatma. Integer/ternary kararını fp16 izniyle karıştırma. |
| `moondream/parakeet-ultra` | 627,270,663 parameter. | Küçük model kazancı yok; mevcut gerekçeyle defer. |
| `primeline/parakeet-primeline` | Tar/config incelemesinde yaklaşık 627M, 24 layer / 1024 hidden. | Full-size fine-tune; küçük distill diye seçme. |
| `ctaguchi/parakeet-hf-ctc-110m-en` | Gerçek küçük CTC yolunda 109,287,937 parameter kaydı var. Upstream, NVIDIA 110M modelinin Transformers conversion'ı olduğunu söylüyor. | Kod/export/preset var; output kontrolleri kayıtlı. Bağımsız non-NVIDIA student değil. Yeni NVIDIA işleri kuralı altında otomatik next target yapma. |

C32'deki küçük CTC portunun feature karşılaştırması: max abs `2.04e-4`, mean abs `9.76e-7`; `atol=1e-5, rtol=1e-4` ile 137/88,080 coordinate FAIL. Browser kaydı load + warm-up kontrolüdür; browser oracle acceptance değildir.

Kaynaklar: [C32/C34/C35](../PRODUCT_TASKS.md), [redux size correction](../../tools/data/results/model-candidates/parakeet-redux-size-correction-2026-10-04.json), [v3 survey](../../tools/data/results/model-candidates/parakeet-distil-verification-v3-2026-10-04.json). Survey'in bazı alanları eski/çelişkili kaldı. Örneğin redux için 149M iddiası correction içinde sürüyor; küçük CTC için “non-NVIDIA-originated” etiketi upstream ile çelişiyor. Dosyanın `CONCLUSIVE` status'u bütün içeriği doğru yapmaz.

**Sonuç:** İncelenen redux/ultra/primeline kayıtlarında gerçek küçük student yok. Bu, dünyada hiçbir küçük Parakeet checkpoint olmadığı iddiası değildir. Bu inceleme yeni bir genel model taraması yapmadı. Gerçek student bulma işini eski üç yanlış adayla kapatma.

## 4. Yeni ve özgün ASR adayları

Aşağıdaki sıra, [2026-09-30 araştırma kaydındaki](light-asr-diarization-hunt-2026-09-30.md) non-NVIDIA sırasını korur. FireRed/Qwen özgün ilgi alanlarıdır; o rapor bunlar için birbirine göre yeni bir sıra belirlemiyor. Yeni ranking uydurma. Araştırma listesi, tamamlanmış port listesi değildir.

| Grup / kayıt sırası | Aday ve kaynak kimliği | Değer ve ilk doğrulama |
| --- | --- | --- |
| Özgün ilgi | FireRed ASR; [FireRedASR2S ailesi](https://huggingface.co/collections/FireRedTeam/fireredasr2s) | AED ile LLM checkpoint'ini ayrı seç. FireRedVAD başarısını ASR acceptance sayma. Özgün ağırlık, lisans, Kaldi-fbank ve decoder oracle'ını doğrula. |
| Özgün ilgi | `Qwen/Qwen3-ASR-0.6B`; ayrıca `Qwen/Qwen3-ASR-0.6B-hf` | İki artifact/runtime yolunu ayır. Mevcut fp16/split-graph yolu yeniden keşfedilecek bir eksik değil. Aynı audio ve artifact identity ile karşılaştır. |
| Dated non-NVIDIA sıra 1 | Moonshine v2 / `moonshine-ai/moonshine-streaming-tiny` | Küçük English streaming adayı. Offline Tiny ile aynı architecture değil. Gerçek streaming partial/final ve cache davranışı gerekir. |
| Dated non-NVIDIA sıra 2 | `FunAudioLLM/Fun-ASR-MLT-Nano-2512` | Yaklaşık 800M, 31 dil araştırma kaydı; Türkçe doğrulanmamış. Nano ile MLT Nano'yu karıştırma. Official engine sonrası dar export feasibility. |
| Dated non-NVIDIA sıra 3 | ARK-ASR 0.6B; lead `Edge0/ark-asr-0.6b-int8-onnx` | 19 dil kaydı Türkçe içermiyor. Bu INT8 lead, özgün full-precision ağırlık kimliği veya WebGPU parity değildir. Önce original checkpoint'i bağla. |
| Dated non-NVIDIA sıra 4 | `CohereLabs/cohere-transcribe-03-2026` | Yaklaşık 2B, Apache-2.0, 14 dil kaydı; Türkçe yok. Access, original decoder ve browser peak memory kontrolü. |
| Dated non-NVIDIA sıra 5 | `mistralai/Voxtral-Mini-4B-Realtime-2602` | Yaklaşık 4B causal streaming; 13 dil kaydı, Türkçe yok. Hedef cihaz, chunk latency ve KV büyümesi ölçülmeden browser verdict verme. |
| İzleme | `FermionResearch/Phonon-1-Micro` | English odaklı low-bit Qwen türevi. 285 MB dosya, fp32/fp16 browser yolu veya parameter küçülmesi kanıtı değil. Integer kararını ayrıca ele al. |
| Mevcut aile + yeni varyant araştırması | [GigaAM](https://github.com/salute-developers/GigaAM) | CTC/RNN-T kodu ve eski gerçek backend kayıtları var. Yeni varyantın exact ID/revision/dil listesini bağla; aileyi sıfırdan port etme. |

Nemotron 3 Diarization, dated tabloda görünse de aktif model sırasından çıkarılmıştır. Alignment/diarization araştırmalarını ASR transcript portu ile bir tutma. Bu listedeki dil, boyut ve lisans bilgileri dated research kapsamındadır; her yeni ciddi portta güncel original kaynağı tekrar doğrula.

Türkçe için ayrıca mevcut `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` çalışması var. Bu bir Parakeet distill değildir. Mevcut CTC/çok dilli çalışmaları koru; kullanıcı aday kuyruğuna yeni model eklemek için bunları gerekçe yapma.

## 5. Çalışan şeyler ve açık kapılar

| Yol | Kayıtlı başarı | Açık kalan sınır |
| --- | --- | --- |
| Orukeet native fp32 | 149.7125 s fixture: Node 26.2.0 ve Bun 1.3.14, 722/722 token, aynı metin, 806 decode iteration. | Tek fixture/config. Browser uzun-ses, geniş kalite ve bütün feature kapıları buradan çıkmaz. |
| Orukeet browser hybrid | JFK'de doğru ORT WebGPU bundle ile real encoder + WASM head score kontrolü PASS. | Production features 313/140,928 coordinate FAIL; max `3.423691e-4`. `full_asr` kaydı `NOT_PROVEN_FEATURE_GATE_UNACCEPTED`. |
| Qwen mevcut `src/models/qwen-asr` | Dynamic encoder ve native fp16 decoder exportları mevcut. 2026-08 browser matrix JFK output PASS kaydediyor. 2026-10-03 library WASM kontrolü normalized WER/CER 0. | Dated output controls; yeni build'in tüm stage/quality/host acceptance'ı değil. |
| Qwen yeni `src/models/qwen3-asr` | `-hf` weights ve kendi fp32 front/decoder/export kayıtları var. JFK oracle eşleşmesi kayıtlı. | İkinci fixture schedule axis nedeniyle reddedildi. Input bazlı schedule/PCM feature yolu tam genelleşmiş değil. Browser fp32 yolu 4 GB WASM32 sınırını aştı. |
| Qwen fp16 (converter yolu) | Converter ile 1.8 GB fp16 set üretildi (`models/onnx/qwen3-asr-0.6b-fp16/`); Node'da `graphOptimizationLevel: 'basic'/'extended'` ile session açılıyor. | Decoder çalıştırmada `/Concat_2` buffer-reuse hatası veriyor (Node ve Python'da, past uzunluğu 0 ve 1). fp16 parity ölçülmedi. Doğrudan export scripti (`export_qwen3_asr_fp16.py`) front+embed üretiyor; decoder export'u eksik. Ayrıntı: [çalışma durumu](workstate-qwen-fp16-streaming-2026-10-08.md). |
| Moonshine offline Tiny | Session stacked caches kullanıyor; seed-cache zorunluluğu kaldırılmış. JFK token/text kontrolü ve unseen audio kayıtları var. Browser WASM: 24 token, 25 step, RTFx 9.0. | Aynı fp32 set WebGPU'da bozuk token/text üretiyor. Stage karşılaştırması yok; genel kernel kök nedeni kanıtlanmadı. |
| Moonshine streaming | Ayrı checkpoint ve ayrı architecture. Transformers 5.18 ile yükleme doğrulandı: 0 missing / 0 unexpected / 0 mismatched key, conv1 `(640, 320, 5)`. Encoder ham dalga-formu işler: `frame_ms=5.0` → 80 örnek → `Linear(80→320)` → causal conv'lar; 320 = `hidden_size`, mel yok. | Diskteki graph'lar hâlâ geçersiz (random-initialized encoder; 4.57.6 dönemi). 5.18 ile yeniden export, official streaming oracle ve chunked decode sözleşmesi gerekir. Ayrıntı: [çalışma durumu](workstate-qwen-fp16-streaming-2026-10-08.md). |
| FireRed ASR | Architecture dokümanı ve VAD ile ilgili çalışma var. | `src/models/firered-llm.ts` boş topology class içeriyor; gerçek ASR backend kanıtı değil. |

Kanıt yolları:

- [Orukeet long repeat8](../../tools/data/results/orukeet/long-repeat8-end-to-end-parity-2026-10-03.json).
- [Orukeet corrected bundle](../../tools/data/results/orukeet/browser-hybrid-webgpu-bundle-jfk-2026-10-03.json).
- [Eski Qwen browser matrix](experimental-model-browser-matrix-2026-08-28.md).
- [Mevcut Qwen library output kontrolü](../../tools/data/results/qwen/goal-harness-existing-port-jfk-2026-10-03.json).
- [Yeni Qwen export/ikinci fixture kayıtları](../../tools/data/results/model-candidates/qwen3-asr-0.6b-2026-10-04.json).
- [Moonshine/Qwen son browser kaydı](../../tools/data/results/browser/asr-browser-measurements-2026-10-04.json).
- [C49/C50 kaynak uzlaştırması](../PRODUCT_TASKS.md).

Browser receipt'in üst `status` alanı eski kernel failure adını taşıyor; Moonshine satırı ise numeric divergence diyor. Qwen satırının genel “model browser-shippable değil” sonucu da fazla geniş. Ham failure'ı koru; verdict'i executed artifact/host kapsamıyla oku. Eski JSON'u sessizce PASS yapma.

## 6. fp16 ve surgical graph çalışma rehberi

**Kullanıcının yönü geçerli: fp16 weights ve kendi surgical graph'larımızı kullan.** Bu izin, parity toleransını düşürme veya integer quantization izni değildir.

1. Aynı özgün checkpoint revision/config/tokenizer/hash zincirini sabitle.
2. Mevcut fp32 baseline'ı ayrı dizinde koru.
3. Encoder, projector/embedding, decoder prefill, decoder step ve state sınırlarını aile contract'ına göre seç.
4. Dynamic frame/window/mask eksenlerini gerçek inference contract'ıyla eşleştir.
5. Weight dtype, graph I/O dtype, cache dtype ve sensitive arithmetic dtype'ını ayrı kaydet.
6. Gerektiğinde normalization/Softmax gibi sensitive işlemleri fp32 tutmayı kontrollü A/B deneyi olarak ölç.
7. Graph/data external ranges ve hash kimliğini doğrula.
8. Official engine ile feature, encoder, state/cache ve logits karşılaştırmalarını kaydet: max/mean abs, cosine, explicit atol/rtol, allclose ve first divergence.
9. Kayıtlı-state replay ile independently accumulated decode'u ayrı çalıştır.
10. Full token ids ve text'i aynı oracle ile karşılaştır.
11. Kısa, farklı uzunlukta, silence/speech-silence ve uzun fixture kapılarını çalıştır.
12. Public library yolunda native Node/Bun ve gerçek browser WASM/WebGPU/hybrid kombinasyonlarını ayrı ölç.
13. Load, warm latency, RTFx, RSS/VRAM/WASM peak memory, CPU↔GPU transfer ve cleanup sonuçlarını kaydet.
14. Yalnızca doğruluk ve ölçülen ürün kazancı birlikte korununca optimize adayı promote et.

Weight dosyasının yarıya inmesi peak memory'nin yarıya ineceğini garanti etmez. Aynı embedding/LM weight'in birden çok graph'ta bulunması, ORT staging, external-data buffers ve KV cache toplamı ayrıca ölçülür. Bütün decoder'ı fp16'ya çevirmek otomatik accuracy veya WebGPU uyumluluğu sağlamaz.

Whisper'dan graph boundary, prefill/step ayrımı ve uygun GPU-KV placement derslerini al. Whisper decode/timestamp semantics'ini diğer ailelere kopyalama. Kaynak: [Whisper planı](../plans/whisper-webgpu-completion-plan.md), [export araçları](../../tools/whisper-onnx-export/README.md).

Qwen'de daha önceki `ArgMax` surgery output handling'i azalttı ama total latency'yi artırdı; candidate promote edilmedi. “Surgical” değişiklik tek başına speedup değildir. Kaynak: [ölçülen ArgMax deneyi](qwen3-asr-webgpu-argmax-surgery-2026-08-29.json).

## 7. İlk uygulanabilir teknik dilimler

Bunlar yeni queue değildir. İlgili C/P task'larının dependency sırasını netleştirir.

### A. Qwen tekrarını kaldıracak artifact/runtime karşılaştırması

- Sahipler: `src/models/qwen-asr`, `src/models/qwen3-asr`, mevcut family exporters ve local adapters.
- Önce `N:/models/onnx/qwen3-asr-0.6b-official` ile yeni `models/onnx/qwen3-asr-0.6b` setinin architecture, source revision, graph I/O, cache, tokenizer ve precision farklarını tabloya bağla.
- Var olan dynamic encoder + fp16 prefill/step yolunu baseline olarak yeniden kullan. Yeni exporter yazmayı varsayılan çözüm yapma.
- Yeni `-hf` yolunun ikinci fixture başarısızlığını açık tut. Variable-length export veya model-safe windowing seçimini yalnızca official output controls ile kabul et.
- Windowing native streaming değildir. Window sınırı, kısa tail, merge ve repeated decode davranışını doğrula.
- Son fp16 hata incelemesinde önce `capture_schedule(model, features.float())` çağrısını oku. Bu çağrı fp16 tower'ı export wrapper'ından önce çalıştırır; yalnızca wrapper'a cast eklemek önceki çağrıyı değiştirmez. Bu bir source-grounded inceleme yönüdür, doğrulanmış fix değildir.
- Exit: ikinci fixture full token/text/state gates, fp16 stage metrics, public input preparation ve gerçek browser receipt. Bu kapılar kapanmadan “Qwen tamamlandı” deme.

### B. Moonshine WebGPU ilk divergence

- Sahipler: mevcut Moonshine session ve family export/harness adapter.
- Seed-cache migration'ını tekrar yapma.
- İlk kontrol: effective PCM, graph/data hashes, seçilen ORT module ve gerçek provider/precision kimliği.
- Aynı input ile önce encoder outputs karşılaştır.
- Encoder geçerse cross-KV ve ilk decoder logits/state karşılaştır.
- İlk farklı stage'e göre yalnızca bir surgical graph/precision deneyi seç.
- Hard `repetitionBan` farklı decode policy'dir; aynı cümlenin tekrar edilmesini engellemek oracle parity değildir. Bunu reference decode ile karıştırma.
- Exit: sayısal stage comparison ve independently accumulated browser token/text kontrolü. Ardından matched performance/peak-memory A/B.

### C. Moonshine streaming doğru original model

- Sahipler: project-local Transformers5 reference stack ve streaming family tooling.
- Var olan `moonshine_streaming` implementation'ını incele; eski offline encoder'a key rename yapma.
- Missing/unexpected/mismatched/random-initialized weights olmadan load'u doğrula.
- Encoder sözleşmesini doğrulandığı gibi kur: ham dalga-formu, `frame_ms=5.0` ile 80 örnek/kare, in-graph `Linear(80→320)`, `pad_to_multiple_of: 80`. Mel formülü türetme; 320 `hidden_size`'dır.
- Official partial/final output, state continuity ve reset oracle'ı yakala.
- Geçersiz eski streaming graph'ını production asset seçimi dışında tut; failure kaydını silme.
- Exit: kendi fp32 export'u, feature/state/token parity ve gerçek incremental library/browser kontrolü.

### D. Aday sırasına dönüş

- Redux/ultra/primeline exportlarını “distill işi” diye yeniden başlatma.
- Küçük Parakeet adayının model kökeni ve teacher/training kanıtı olmadan format conversion'ı bağımsız student sayma.
- Mevcut küçük CTC portunu silme veya yeniden yazma. Yeni NVIDIA model işi kapsamına otomatik ekleme.
- Gerçek küçük student yoksa incelenen aday kümesinin gerekçeli defer kaydını koru; bütün dünyayı kapsayan yokluk iddiası kurma.
- FireRed ve Qwen özgün ilgilerini, ardından dated non-NVIDIA adaylarını mevcut queue'da tut. Kapsamı sessizce değiştirme.
- Her aday için exact upstream ID/revision, license, parameter count, Turkish/language evidence, local assets, original engine, export route ve ilk discriminating experiment kaydet.
- Yeni büyük weight download veya sibling edit gereken noktada yetki al. Mevcut yerel weights ve araçları önce kontrol et.

## 8. Araçlar ve güvenli başlangıç

Başlangıç sırası: [AGENT_ENTRYPOINT.md](../AGENT_ENTRYPOINT.md) → [safety](../AGENT_SAFETY_RULES.md) → [tool map](../../tools/README.md#task-to-tool-map) → [model debugging skill](../../tools/model-debugging/SKILL.md) → ilgili family guide → mevcut C/P task ve receipt.

| İş | Mevcut araç |
| --- | --- |
| SHA-bound port orchestration | `tools/model-debugging/scripts/port_manifest.py`; C51 mandatory digest preflight. |
| Stage comparison | `tools/model-debugging/scripts/node-compare-stage-captures.mjs`, `tools/tensor_parity.py`. |
| First-input attribution | `tools/stage_attribution.py`; aile adapter callback'i ile kullan. |
| ONNX/external-data audit | `tools/onnx_artifacts.py`, `tools/repack_onnx_external_data.py`, mevcut artifact-audit CLI. |
| Qwen mevcut inference yolu | `tests/smoke/qwen3-asr-node-wasm-benchmark.mjs`; bu incelemede `--help` çalıştı. |
| Qwen original capture/export | `tools/model-debugging/reference/qwen3-asr-0.6b/README.md`. |
| Orukeet public native consumer | `examples/nemo-tdt-native-local.mjs`; mevcut long-form receipt komutlarını kullan. |
| Gerçek browser | İlgili mevcut sibling runner; CLI, artifact routing ve library import identity'yi önce doğrula. |

Kurulu ORT session type contract'ı: `node_modules/onnxruntime-common/dist/cjs/inference-session.d.ts`, `graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'layout' | 'all'`. Contract bunu native Node/WASM için belirtir; WebGPU'da fusion kapatıldığını varsayma. Çalışan export'ta `GraphOptimizationLevel` enum'unun `undefined` olması bu string contract'ını iptal etmez.

Protected Nemotron dosyalarına dokunma. Tarihî handover 12 dosya adı korur; bu incelemede 10'u modified, ayrıca `capture_cu_seqlens.py` modified idi. Çalışma ağacı temiz değildir. Bütün inherited untracked/ignored artifacts'i koru. Stage yalnızca explicit owned paths; commit öncesi cached name list'i göster. Bu inceleme commit/push/download/install yapmaz.

## 9. Bu raporun doğrulama sınırı

Repo goals, task kayıtları, ilgili source ve JSON receipts okundu. Orukeet, iki Qwen yolu, Moonshine migration/streaming ve Parakeet candidate kökenleri ayrı uzlaştırıldı. Küçük CTC upstream model kartı doğrudan okundu. Canlı directory incelemesi eski Qwen dynamic/fp16 dosyalarını doğruladı. Bu raporun ilk yazımında yeni fp16 hedef klasörü boştu; sonrasında converter ile 1.8 GB'lık set üretildi ve ölçüldü (bkz. [çalışma durumu](workstate-qwen-fp16-streaming-2026-10-08.md)).

`node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --help` çalıştı. Native ORT introspection sürümleri ve enum yokluğunu gösterdi. Bunlar CLI/API keşif kontrolleridir; model inference acceptance değildir.

Canlı reference import kontrolleri de geçti:

- `tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe`: NeMo `3.0.0`, Torch `2.6.0+cu118`, `ASRModel` import PASS. Mevcut overlay environment'ın dependency-consistency sınırları devam eder; yeni model parity veya CUDA acceptance değildir.
- `tools/model-debugging/reference/transformers5/Scripts/python.exe`: Transformers `5.18.0`, Torch `2.14.1+cpu`, `MoonshineStreamingForConditionalGeneration` import PASS. Bu, architecture kodunun varlığını kanıtlar; doğru checkpoint load veya feature contract'ını kanıtlamaz.
- Rehberdeki 23 yerel dosya bağlantısı çözüldü. Üç startup/task dosyasının rehbere routing bağlantısı doğrulandı.

Bu rapor yeni model inference, numerical parity, full regression suite veya browser benchmark çalıştırmadı. Eski PASS'lar kendi tarih/fixture/artifact kapsamıyla raporlandı. Son konuşmadaki `1,479 passed / 33 skipped` sayısı bu incelemenin fresh test sonucu değildir. Goal açık kalır; raporun tamamlanması model roadmap'inin tamamlanması değildir.
