# ASRJS kütüphane ve demo çalışmasının bağımsız denetimi

18 Ekim 2026 — karar: **REQUEST CHANGES / çalışma kısmen tamamlanmış.**

Önceki ajanın “her şey tamamlandı, tüm kapılar geçti” sonucu mevcut kodla desteklenmiyor. Gerçek ve korunmaya değer geliştirmeler var; ancak doğrulama araçları yanlış başarı üretebiliyor, genel API bağlantıları hâlâ kırık ve demo taşınması bazı regresyonlar getirmiş. Toplu geri alma veya yeni bir yeniden yazım yerine, aşağıdaki dar düzeltmeler ve bağımsız kabul kapıları öneriliyor.

## Kapsam ve yöntem

- Kullanıcının eklediği konuşmadaki T1–T7, W3–W4, demo modernizasyonu, monorepo, Git geçmişi ve LFS iddiaları incelendi.
- Kütüphane: `26cdde26..6e53c832a6e0837a117e187e383eaef51c1f6890`; aralık 109 dosya, 32.404 ekleme ve 8.710 silme içeriyor. Bunun önemli bölümü raporlar, arşivler ve kanıt dosyaları.
- Demos: `57dae695096558f84b5a5923362e3a0896eb8add`; dört aktif uygulama, ortak Vite yapılandırması, UI paketi ve referans geçmişi.
- Üç bağımsız salt okunur inceleme model/runtime, streaming/VAD ve demo alanlarına ayrıldı. Bulgular tam fonksiyonlar ve çağıran kod üzerinden kontrol edildi; temel sorunlar ayrıca küçük yeniden üretimlerle doğrulandı.
- Test/build çalıştırmaları yapıldı. Bu denetim tüm modellerin yeni gerçek GPU çıkarım koşusu, mikrofon uçtan uca testi veya sayısal model paritesi sertifikası değildir. Mevcut tarayıcı makbuzları ve bunları üreten kod incelendi.
- İnceleme öncesinde ve testler sonrasında iki çalışma ağacı temizdi. Ürün kaynakları, model ağırlıkları ve eski kanıtlar değiştirilmedi; bu rapor eklendi. Commit/push yapılmadı.

## Yeniden çalıştırılan kontroller

| Kontrol | Sonuç | Anlamı |
|---|---|---|
| Kütüphane `npm run typecheck` | PASS | `src` ve örnekler; ana tsconfig testleri kapsamıyor |
| Kütüphane `npm run build` | PASS | Derleme başarılı |
| Kütüphane `npm test -- --reporter=dot` | **1748 passed, 5 failed, 36 skipped** | 208 test dosyası geçti, 2 başarısız, 7 atlandı |
| NeMo TDT lifecycle hedefli koşusu | **5 passed, 4 failed** | Birleşik ORT cleanup regresyonu bağımsız tekrarlandı |
| Root import smoke | PASS, **94 modül** | Mevcut testin dar kuralı geçiyor; ilk T1 hedefi geçmiyor |
| Tüm browser-safe entry smoke | PASS, 342 modül / 32 giriş | Node statik import kontrolü |
| Package tarball smoke | PASS, **1597 yayımlanacak dosya** | Export hedefleri ve declaration dosyaları mevcut |
| Dört demo `npm run build` | PASS | Showcase, playground, realtime, benchmark; build uyarıları mevcut |
| `npm run test --workspace=apps/realtime` | **5 passed, 5 failed** | `document is not defined`; jsdom ayarı taşınmamış |
| Kardeş kütüphane olmadan `ASRJS_SOURCE=npm` build denemesi | **FAIL / ENOENT** | NPM modu uygulanmamış; alias hâlâ kardeş checkout'a gidiyor |
| Küçük yeniden üretimler | **Altı kusur doğrulandı** | Gevşek oracle, matris yanlış PASS/mode, Qwen kaynak bağlantısı, VAD değerleri, release hatası yutulması, Moonshine cleanup |

Beş kütüphane hatasının **dördü kod regresyonu**. Beşinci, `tests/gigaam-ctc-frontend-diagnose.test.ts:102` içindeki `tools/data/results/gigaam/jfk-short.js-features.npy` yazımında `EPERM`; bunu model veya ürün kodu hatası olarak sınıflandırmıyorum. İlk tarball denemesi de varsayılan npm cache yazma iznine takıldı; yalnız bu süreç için cache izinli audit dizinine alınarak test başarıyla tekrarlandı. Genel makine ayarı değiştirilmedi.

Yerel çalışma kanıtları:

- [Tam kütüphane test çıktısı](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/library-tests.log)
- [Kütüphane typecheck](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/library-typecheck.log), [build](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/library-build.log)
- [Dört demo build çıktısı](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/demos-build.log)
- [Realtime test çıktısı](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/realtime-tests.log)
- [Kardeş checkout olmadan build](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/isolated-build.log)
- [Tekrarlanabilir tanı scripti](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/reproduce-audit.mjs), [JSON sonuçları](C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/reproductions.json). Script gerçek kaynak fonksiyonlarını ve yeniden derlenmiş `dist` API'sini kullanıyor; sentetik ORT/harness vakaları gerçek GPU/model paritesi iddiası taşımıyor.

## Öncelikli bulgular

### F01 — P1 — WebGPU doğrulama aracı hatalı sonuçları başarılı sayıyor

[bridge.py:162](../../tools/browser-harness/bridge.py#L162) matrisin her iki koşusunda `runOffline()` çağırıyor; `mode: 'streaming'` dikkate alınmıyor. [171. satır](../../tools/browser-harness/bridge.py#L171) metin/oracle veya GPU execution kontrolü yapmadan PASS atıyor. [cli.mjs:336](../../tools/browser-harness/cli.mjs#L336) matris spec'ine `expectJfk` aktarmıyor; [369. satırdan sonraki return](../../tools/browser-harness/cli.mjs#L369) matris FAIL/PARTIAL_PASS olsa bile CLI'yi başarısız sonlandırmıyor.

Ayrıca [cli.mjs:29](../../tools/browser-harness/cli.mjs#L29) tam transkript yerine üç anahtar ifadeye bakıyor. Gerçek fonksiyon **`fellow americans ask country`** metnini ve sonuna yanlış kelimeler eklenmiş hâlini kabul etti. Oracle istenmediğinde dahi [449. satır](../../tools/browser-harness/cli.mjs#L449) `matchExpected: true` yazabiliyor.

Yeniden üretimde `completely wrong`, sıfır GPU dispatch ve yeniden kullanılmayan warm yükleme ile matris PASS verdi; streaming istenirken iki offline ve sıfır streaming çağrısı oluştu. Mevcut [webgpu-matrix-verify.json](../../tools/data/results/canonical/webgpu-matrix-verify.json) Türkçe Wav2Vec2'nin JFK için hatalı metnini PASS sayıyor. Parakeet ve Moonshine için dispatch sıfır kayıtlı; bu tek başına GPU kullanılmadığını kanıtlamaz, ölçümün bunu doğrulayamadığını gösterir.

**Düzeltme/kabul:** execution, text/token parity ve provider kanıtını ayrı alanlar yap; bilinmeyeni `null/UNVERIFIED` tut; fixture'a bağlı gerçek oracle kullan; yanlış/eksik metin, sıfır/kanıtsız GPU, başarısız model ve streaming mode için negatif testler ekle. Matris başarısızlığında exit code sıfır olmasın. Eski makbuzları silmeden geçerlilik kapsamını düzelt.

### F02 — P1 — Qwen3-ASR genel yükleme yolu çıkarım yapamıyor

[qwen3-asr-local-adapter.ts:100](../../src/presets/qwen3-asr-local-adapter.ts#L100) düz `frontUrl/decoderUrl/embedUrl` alanları oluşturuyor; [preset:27](../../src/presets/qwen3-asr-factory.ts#L27) bunları `qwen-asr` ailesine aynen veriyor. [model.ts:54](../../src/models/qwen-asr/model.ts#L54) ise executor oluşturmak için `options.source` bekliyor.

Gerçek root `loadSpeechModel({preset:'qwen3-asr', backend:'wasm', options:{frontUrl,...}})` çağrısı yüklenmiş handle döndürdü; geçerli biçimli PCM ile ilk transkripsiyon **`ExperimentalArtifactMissingError`** verdi. Sahte URL'lere erişim dahi yapılmadı: executor hiç kurulmadı. Önceki Qwen tarayıcı kanıtı bu hatayı yakalamıyor; [verify-browser-moonshine-qwen.mjs:270](../../tools/verify-browser-moonshine-qwen.mjs#L270) önceden hesaplanmış JFK feature'ları ve doğrudan ORT session'ları kullanıyor.

**Düzeltme/kabul:** adapter/preset/family aynı artifact sözleşmesini kullanmalı; sidecar ve gerçek PCM feature yolu doğrulanmalı. En az üç fixture'ı public local-entry API'den yükleyip transkribe et; yalnız model resolve/create testi yeterli değil.

### F03 — P1 — Birleşik ORT cleanup hatalarını gizliyor

[session.ts:212](../../src/inference/ort/session.ts#L212) `session.release()` hatalarının tamamını yutuyor. [nemo-tdt/ort.ts:399](../../src/models/nemo-tdt/ort.ts#L399) artık bu helper'ı kullandığı için executor'ın `AggregateError` sözleşmesi bozulmuş. Dört lifecycle testi bu yüzden başarısız. Sentetik `release()` hatası da çağırana başarı olarak döndü.

**Düzeltme/kabul:** ortak release hatayı iletmeli; gerçekten best-effort isteyen çağıran açıkça yakalamalı. Mevcut dört testi veya beklenen hataları gevşetmeden düzelt; ardından diğer beş ORT göçünün cleanup davranışını kontrol et.

### F04 — P1 — Neural VAD runtime ile public tipler uyuşmuyor

[streaming-config.ts:68](../../src/runtime/streaming-config.ts#L68) public union'da `ten-vad-only` ve `rough-and-ten-vad` sunuyor. [streaming-detector.ts:433](../../src/runtime/streaming-detector.ts#L433) yalnız `hybrid`, `vad-only`, `rough-only` tanıyor. Hazır bir VAD ile eski iki public değerin **effectiveMode=rough-only** olduğunu yeniden ürettim. Yeni değerler public config tipiyle uyuşmuyor; testler tsc kapsamı dışında ve bir test `@ts-expect-error` kullanıyor.

**Düzeltme/kabul:** tek kanonik enum/type ve gerekiyorsa açık eski-değer dönüşümü. Her dışa açık değeri public API üzerinden test et; TypeScript tüketici örneği de derlensin.

### F05 — P1 — Worker streaming örnekleme hızını kaybediyor

[browser-transcription-worker-thread.ts:226](../../src/runtime/browser-transcription-worker-thread.ts#L226) `PUSH_STREAM_AUDIO` içindeki `sampleRate` alanını kullanmadan çıplak Float32Array iletiyor. [istemci:443](../../src/runtime/browser-transcription-worker.ts#L443) 48 kHz gibi bir değer kabul edip gönderiyor; audio normalizer çıplak diziyi 16 kHz varsayıyor. Örneğin 48.000 sample yanlışlıkla üç saniye olarak yorumlanabilir.

**Düzeltme/kabul:** örnekleme hızını taşıyan AudioInput oluştur veya desteklenen hıza açıkça resample et. Aynı dalganın 16/48 kHz girdileri eşdeğer süre/çıktı üretmeli; yalnız transfer-list testi yeterli değil.

### F06 — P1 — Nemotron implementasyonu modeller arası state taşıyan streaming değil

[streaming-transcriber.ts:208](../../src/models/nemotron-rnnt/streaming-transcriber.ts#L208) her committed parça için, [286. satır](../../src/models/nemotron-rnnt/streaming-transcriber.ts#L286) preview için tekrar `session.transcribe(pcm)` çağırıyor. Executor her çağrıda [encoder cache'lerini](../../src/models/nemotron-rnnt/executor.ts#L565) ve [predictor h/c durumunu](../../src/models/nemotron-rnnt/executor.ts#L711) yeniden sıfırlıyor. Yeni sınıf transcript/timestamp biriktiriyor; akustik ve RNNT durumunu push'lar arasında korumuyor.

**Düzeltme/kabul:** mevcut davranışı chunked transcription olarak sınıflandır; stateful kabulü için encoder/cache/predictor/joiner state'in korunmasını ve parça sınırlarını aşan sözcükleri resmi referansa karşı ölç. Bu denetimde gerçek ses üzerindeki hata miktarı ölçülmedi; “stateful” tamamlanma iddiası kodla çelişiyor.

### F07 — P2 — Mikrofon partial yolu VAD öncesi onset sesini atlıyor

[browser-controller.ts:289](../../src/runtime/browser-controller.ts#L289) detector speech-start ürettikten sonra yalnız o anki chunk'ı stream'e yollar. Detector segment başlangıcını [ring buffer'daki önceki onset'e](../../src/runtime/streaming-detector.ts#L636) geri alabiliyor; stream aynı geri alınan sesi almıyor. VAD onayı öncesinde başlayan sözcük partial girdisinde eksik kalır.

**Düzeltme/kabul:** başlangıçta buffer'dan gereken onset PCM'ini bir kez besle; geciken VAD onayı ve max-duration devamında sample coverage testleriyle kayıp/çift beslemeyi denetle.

### F08 — P2 — Moonshine kısmi yükleme başarısızlığında session sızdırıyor

[streaming-loader.ts:100](../../src/models/moonshine/streaming-loader.ts#L100) dört session'ı `Promise.all` ile açıyor, ancak `created` listesine yalnız hepsi başarıyla döndükten sonra ekliyor. Bir graph reddederken diğerleri açılırsa catch onları göremiyor. Gerçek loader'a enjekte edilen sentetik ORT ile **3 session oluşturuldu, 0 session release edildi** sonucu alındı.

**Düzeltme/kabul:** her başarılı allocation'ı takip et; kısmi yükleme/fallback ve geç tamamlanan promise senaryolarında tüm kaynakların kapandığını doğrula.

### F09 — P1 — Moonshine WASM çalışırken WebGPU raporlayabiliyor

[model.ts:168](../../src/models/moonshine/model.ts#L168) üç graph'lı offline yolda **encoder, cross-KV ve decoder'ın tamamını** `executionProviders:['wasm']` ile açıyor. Dolayısıyla önceki anlatımdaki “encoder/cross-KV WebGPU, yalnız decoder WASM” tanımı güncel kodla uyuşmuyor. [model.ts:333](../../src/models/moonshine/model.ts#L333) buna rağmen canonical sonuca istenen backend'i yazıyor. Streaming yolunda da `options.encoderBackend` yoksa loader WASM seçiyor; açık WebGPU isteğinin fallback'inde [streaming-loader.ts:143](../../src/models/moonshine/streaming-loader.ts#L143) kaydedilen `resolvedEncoderBackend` wrapper tarafından kullanılmıyor. Aynı wrapper [model.ts:322](../../src/models/moonshine/model.ts#L322) gerçek final token sayısını `tokenIds:[]` ile kaybediyor.

**Düzeltme/kabul:** istenen ve çözülen provider'ları ayrı raporla; streaming final metadata/tokenlarını koru. Zorlanmış fallback testinde hem handle hem transcript metadata kontrol edilmeli.

### F10 — P1 — Demos tek başına kurulup build edilemiyor

[vite.shared.js:27](../../../demos/vite.shared.js#L27) ve [alias listesi:60](../../../demos/vite.shared.js#L60) her zaman kardeş `speech-recognition/src` checkout'una gidiyor. Manifestlerde yayımlanmış kütüphane bağımlılığı ve önerilen `ASRJS_SOURCE=npm` seçimi yok. Kardeş yolu olmayan kontrollü build'de `ASRJS_SOURCE=npm` ayarlı olmasına rağmen `src/index.ts ENOENT` alındı.

**Düzeltme/kabul:** sürümü sabitlenmiş paket/tarball tüketimi varsayılan olsun; source geliştirme modu açıkça seçilsin. Temiz geçici klasörde yalnız demos + paket/tarball ile dört build ve browser smoke geçmeli. Henüz yayımlanmamış sürüm için yerel pack/tarball kullanılabilir; bu, publish izni değildir.

### F11 — P1 — Realtime production build dev-server WASM yolu gömüyor

[apps/realtime/vite.config.js:9](../../../demos/apps/realtime/vite.config.js#L9) mutlak checkout yolu içeren `/@fs/.../node_modules/onnxruntime-web/dist/` üretip build'de de `define` ile gömüyor. [capture hook:77](../../../demos/apps/realtime/src/shared/streaming/useStreamingCaptureController.js#L77) FireRed VAD etkinleştirildiğinde bunu worker'a override olarak aktarıyor. Statik deploy'da Vite'ın `/@fs/` route'u yok.

**Düzeltme/kabul:** production'da paketlenen asset URL'lerini kullan; build çıktısını düz statik HTTP sunucusunda açıp FireRed VAD model/WASM yüklemesini doğrula. Dev server'da çalışması bu kabulü sağlamaz.

### F12 — P2 — SRT/VTT export gerçek zamanlamayı kullanamıyor

[TranscriptPanel.jsx:23](../../../demos/apps/realtime/src/shared/browser-console/components/TranscriptPanel.jsx#L23) `item.words` ve `item.audioDuration` bekliyor. [transcriptionRecord.js:18](../../../demos/apps/realtime/src/app/transcriptionRecord.js#L18) ve [52. satır](../../../demos/apps/realtime/src/app/transcriptionRecord.js#L52) bunları kaydetmiyor; yalnız `duration` ve `wordCount` saklanıyor. Bu yoldan üretilen kayıtların export'u her segment için sabit **2,5 saniye** kullanıyor; son kelime hizalama paneli de words alamıyor.

**Düzeltme/kabul:** kayıt şeması canonical words ve capture zaman ofsetini taşımalı; iki farklı uzunlukta, arada sessizlik bulunan segmentin export zamanlarını test et. Yalnız alan adını değiştirmek sessizlik/segment ofseti sözleşmesini çözmez.

### F13 — P2 — Realtime test ortamı taşınırken kaybolmuş

[apps/realtime/vite.config.js:14](../../../demos/apps/realtime/vite.config.js#L14) yeni merge'de eski `test: {environment:'jsdom', globals:true}` bloğunu korumuyor. Hook testleri Node ortamında `document.createElement` çağrısında çöküyor. On testin beşinin başarısızlığı yeniden üretildi.

**Düzeltme/kabul:** ilgili workspace'in DOM test ortamını geri getir; `npm run test --workspace=apps/realtime` komutunun varsayılan hâliyle geçmesini sağla. Testleri skip etme veya yalnız build'i yeterli sayma.

### F14 — P2 — Playground seçilen tuning ayarlarını uygulamıyor

[App.jsx:843](../../../demos/apps/playground/src/App.jsx#L843) CPU thread/profiling options içeren loadRequest kaydediyor; [862. satırdaki gerçek yükleyici çağrısı](../../../demos/apps/playground/src/App.jsx#L862) options'ı göndermiyor. UI'daki CPU Threads kontrolü universal mode'da etkisiz; request paneli gerçekleşen çağrıyı yanlış temsil ediyor.

**Düzeltme/kabul:** gösterilen request ile gerçek loader girdisi aynı nesne/sözleşmeden türesin; seçili thread/profiling değerlerinin loader'a ulaştığını ölç.

### F15 — P2 — Üretilen universal örnek desteklenmeyen streaming çağrısı yapıyor

[App.jsx:311](../../../demos/apps/playground/src/App.jsx#L311) her model için `createStreamingTranscriber()` çağırıyor. Builtin handle bu capability'nin bulunmadığı modellerde `NotImplementedSpeechFeatureError` üretecek guard içeriyor. Uygulamanın kendi stepper'ı capability kontrolü yaparken kopyalanabilir örnek yapmıyor.

**Düzeltme/kabul:** snippet modelin gerçek capability'sine göre oluşmalı; streaming desteklemeyen bir aile ve destekleyen aile için üretilen örnekler çalıştırılmalı.

## Tamamlanma iddialarının düzeltilmiş durumu

| İddia / alan | Denetim sonucu |
|---|---|
| Kök paketten ağır executor/ORT statik importlarının çıkarılması | Gerçek iyileştirme. Mevcut dar smoke geçiyor. Ancak T1'in `<45 modül, 0 model dosyası` kabulü yerine **94 modül, 12 model dosyası ve 10 preset dosyası** var. Test yalnız `executor.js/ort.js` adlarını yasaklıyor. Tam T1 kabulü değil. |
| Eksik model/preset subpath export'ları | Paket smoke geçti; korunmalı. Bu bütün modellerin çıkarım yaptığı anlamına gelmez. |
| Fake webgl/webnn varsayılan kaydının kaldırılması | Kodda mevcut, korunmalı. Moonshine fallback metadata sorunu ayrıca açık. |
| Qwen/Moonshine gerçek browser kanıtları | Makbuzlar mevcut; yeniden GPU koşusu yapılmadı. Özel runner doğrudan ORT kullanıyor; Qwen hazır JFK feature'ları tüketiyor. Public API + üç fixture Gate-B'yi karşılamıyor. |
| W4 tüm yerel model matrisi | Makbuz yalnız üç model içeriyor; Moonshine kaydı `moonshine-tiny-fp32`, streaming tiny değil. Mode/oracle kontrolleri eksik. Whisper, Orukeet, Qwen, omnilingual vb. tüm kapsamın tamamlandığı çıkarılamaz. |
| C92/C93 son sözcük ve soft-split onarımı | Continuation overlap ve kısa parça kabul değişikliği var. `finalizeSegment('max-duration', nowFrame)` hâlâ sert sınırda çalışıyor; düşük enerji noktasına bölme uygulanmış değil. Yeni test sabit genlikli PCM segment kabulünü ölçüyor; gerçek “country” transkriptini veya sample coverage'ı doğrulamıyor. |
| Zero-copy worker | Transferable var; istemci `pcm.slice()` ve Nemotron ek kopya yapıyor. Doğru tanım kopyalanan buffer'ın transfer edilmesi. |
| Ortak UI paketi tüm uygulamalara entegre | `packages/ui` mevcut; aktif `apps/*/src` içinde **@asrjs/demo-ui import'u yok**. Paket oluşturulmuş, tüketici göçü tamamlanmamış. |
| Benchmark RTFx hesabı | İncelenen normal 16 kHz yolunda matematik hatası bulunmadı. Long-boost progress toplamı ek tekrarları saymıyor; ikincil UI kusuru. |
| 12 kaynağın tüm Git geçmişi korunmuş | 183 erişilebilir commit ve **11 subtree import kaydı** var. `transformers-v4-parakeet-demo-cr`, tek parent'lı `992d4cf` dosya ekleme commit'iyle gelmiş; bu kaynak için subtree geçmiş koruması kanıtı yok. |
| Push ve private repo | GitHub connector üzerinden iki HEAD commit'inin uzakta bulunduğu ve `asrjs/demos` visibility=private olduğu doğrulandı. |
| LFS yalnız geçmişte gerekli | Altı benzersiz LFS nesnesi listeleniyor. Aktif uygulamaların public klasörlerinde de LFS pointer dosyaları var; incelenen aktif yollar bunları kullanmıyor, core kendi asset'lerini kullanıyor. “Yalnız eski commit'lerde” ifadesi doğru değil. Uzak LFS nesneleri uçtan uca yeniden indirilip hash doğrulanmadı. |
| 10,70 GB alan açıldı | Güncel fp32 decoder sidecar **2.384.171.008 byte**; `onnx__MatMul_*` sayısı sıfır. Bunlar temizlikle tutarlı. Önceki disk ölçümü olmadan net kazanım bağımsız doğrulanamaz. |

Ek inceleme kuyruğu: unified `env.ts:53` Node için yerel kurulu ORT yerine CDN döndürüyor; eski NeMo/Qwen loader'ları yerel paket yolunu çözüyordu. Ağsız Node/WASM koşusuyla ayrıca ayrıştırılmalı. Bu denetimde bu olası regresyon için gerçek ORT yükleme testi yapılmadı.

## Toparlama sırası ve kabul kapıları

1. **Önce kanıt sistemini düzelt — F01.** PASS yalnız ölçülen koşulları ifade etsin. Hatalı oracle/provider/mode ve FAIL exit code testleri olmadan yeni “tamamlandı” makbuzu üretme. Eski başarısız ve çelişkili sonuçları koru.
2. **Kütüphane regresyonlarını küçük commit'lerle düzelt — F02–F09.** Önce mevcut dört cleanup testi; sonra Qwen public loading, VAD enum, sample-rate ve Moonshine kaynak sahipliği. Nemotron stateful çalışmasını ayrı, sayısal kabulü olan iş olarak tut; basit transcript birleştirmeyi stateful kabul etme.
3. **Monorepoyu bağımsız çalıştır — F10–F13.** Açık source/package modu, test workspace'i, statik prod WASM asset'leri ve canonical subtitle kayıt sözleşmesi. Sibling repo bulunmayan temiz dizin kapısı zorunlu olsun.
4. **Demo vaatlerini gerçek davranışla eşleştir — F14–F15 ve UI tüketimi.** Dört rolü koru: showcase, playground, realtime, benchmark. Ortak tema/word pills/subtitle/metric bileşenlerini bir uygulama üzerinden gerçek kullanıma geçir; ardından diğerlerine taşı. Sadece `packages/ui` klasörü oluşturulmasını konsolidasyon sayma.
5. **Son kabul.** Typecheck + birim/lifecycle + package smoke; desteklenen her yeni model yolu için kısa, >20 saniye ve hedef dil/gürültü fixture'ı; browser/WebGPU ve WASM fallback'in ayrı provider kanıtı; statik deploy ve mikrofon smoke. Aynı commit, fixture hash'i, model hash'i, ORT/browser sürümü ve resolved provider makbuzda yer alsın.

Korumaya değer temel: tek demos repo fikri, dört uygulama rolü, 11 kaynağın geçmiş taşıması, subpath shims, ağır root bağımlılıklarının azaltılması ve ortak ORT katmanı yönü. **Bu temel korunabilir; mevcut tamamlanma raporu kabul edilemez.**
