# Bağımsız denetim sonrası onarım kaydı

18 Ekim 2026. İlk denetim ve eski makbuzlar korunmuştur. Bu kayıt toplu PASS
sertifikası değildir. Kütüphane tabanı `6e53c832a6e0837a117e187e383eaef51c1f6890`,
demos tabanı `57dae695096558f84b5a5923362e3a0896eb8add`; düzeltmeler çalışma
ağaçlarında. Commit/push/publish yapılmadı; model ağırlıkları ve canonical
etiketler değiştirilmedi.

Alt ajanlar ayrı checkout'larda çalıştı. Ana ajan aktarılan diff'leri ve gerçek
çağıranları inceledi, ana checkout'ta testleri tekrarladı. Bu incelemede ek olarak
Moonshine double-release/geç yükleme, subtitle son zamanı, mikrofon final callback
ve hata görünürlüğü düzeltildi. Regresyon testleri eklendi; eski assertion'lar
gevşetilmedi. Özgün LF satır sonları korundu; toplu biçimlendirme yapılmadı.

## Onarımlar

| Bulgu | Düzeltme ve doğrulama |
|---|---|
| F01 | CLI/HTML/matrix ortak runner/verifier. Fixture SHA'ya bağlı tam normalize metin, gerçek mode, fresh/warm handle kimliği, provider gözlemi. FAIL=1, UNVERIFIED=2; eksik oracle PASS değil; makbuz üzerine yazılmaz. 47 hedefli test ve gerçek Chrome negatif koşuları. Warm kapalıysa warm süreleri null; summaryPhase cold olarak kaydedilir. |
| F02 | Qwen official source sözleşmesi, prefill/step, fp16/fp32 sidecar ve nested tokenizer çözümlemesi. Public local-entry API ile gerçek browser File girdilerinden 11 s, 26.45 s ve 18.64 s PCM işlendi. Ana ajan ayrıca JFK'yi tekrarladı. Türkçe çıktı hatalı kelimeler içeriyor; execution kalite/parite PASS'i değildir. |
| F03 | ORT release hataları iletilir; best-effort çağıranları açıkça yakalar/raporlar. Önceden kırılan dört NeMo lifecycle testi aynı beklentilerle geçti; 29 odaklı test. |
| F04 | Tek VAD enum'u, legacy değerlerin hybrid/vad-only dönüşümü. 44 config/detector testi ve public TypeScript tüketici örneği. |
| F05 | Worker PCM gerçek sample rate taşır. 16/48 kHz aynı dalgada süre/frekans regresyon testi. |
| F06 | Nemotron son-label predictor, nonblank state commit, push'lar arası frontend/encoder/predictor state, yeni 65-frame bloklar ve atomic failure/reset. 43 test geçti, 2 atlandı. Gerçek export/reference sayısal paritesi açık; runtime unverified uyarısı var. Stateless fallback açıkça chunked transcription. |
| F07 | Onset buffer sample cursor ile bir kez replay edilir; son PCM finalize/reset öncesi beslenir; final text callback'e ulaşır, queued error görünür. Son controller/worker grubu 22/22; physical mic kabulü ayrı. |
| F08 | Moonshine partial allocation allSettled ile kapanır; geç yükleme/public streamer dispose yarışları ve double-release kapatıldı. Negatif lifecycle/resource testleri. |
| F09 | Offline üç graph gerçek WASM provider'ını raporlar; streaming resolved provider/final token count korunur; asset yoksa capability=false. Gerçek offline WASM PASS, aynı yolda WebGPU isteği provider FAIL. Dört graph streaming WASM/WebGPU exact JFK + warm reuse PASS; GPU koşusunda 338435 dispatch. |
| F10 | Varsayılan sabitlenmiş yerel npm tarball; source modu opt-in, npm package alias'ı. Kardeş library olmadan temiz npm ci ve dört build. |
| F11 | ORT runtime public/ort içine paketlenir; base URL kullanılır, /@fs/ üretilmez. Asset testleri ve nested statik deployment kontrolü. |
| F12 | Canonical words, gerçek audio duration/capture offset; ortak SRT/VTT yuvarlaması ve eksik-word-end fallback. Sessizlikli iki segment, offset ve milisaniye taşıması testleri. |
| F13 | Realtime jsdom geri getirildi; varsayılan package-mode testleri 15/15. |
| F14 | Playground gösterilen loadRequest options nesnesini loader'a iletir; nested identity testi. |
| F15 | Snippet ve gerçek stepper supportsStreaming kontrol eder; yoksa açık offline çıkarım. İlk raporda stepper'ın zaten doğru guard kullandığı cümlesi hatalıydı; o da düzeltildi. |

Dört uygulama ortak tema/UI paketini gerçekten tüketiyor; word pills/subtitle
ortaklaştırıldı, benchmark SVG chart adapter'ları bağlandı. Long-boost progress
ek tekrarları sayıyor. İleriye dönük düşük-enerji soft split hard limit'i korur;
geçmiş PCM'e sonradan kesim yapmaz. Worker yolu buffer kopyasını transfer eder.

Node WASM kurulu ORT paketini çözüyor: gerçek Identity graph ağ kapalıyken 7→7.
Bu ASR paritesi değildir. GigaAM diagnostic testi kendine ait geçici klasöre
yazar; eski kanıtları değiştirmez.

Root mapping implementasyonları runtime'a taşındı; eski subpath'ler re-export
eder. Root statik model/native ORT implementasyonu çekmiyor. Ancak **78 modül,
istenen <45 hedefini karşılamıyor** (başlangıç 94). Smoke varsayılan exit 2,
strict budget modunda exit 1. 32 browser entry/346 modüllük sınır kontrolünün
geçmesi bütçe kabulü değildir. Public API kırılarak sayı düşürülmedi.

## Ana ajan tarafından doğrulanan kanıtlar

Log/makbuz klasörü: `tools/data/results/browser-harness/`.

| Kontrol | Sonuç / dosya |
|---|---|
| Tam test paketi | **1830 passed, 36 skipped, 0 failed; 216 dosya geçti, 7 atlandı** — `audit-full-tests-final-reviewed.log` |
| Typecheck, consumer examples, build | PASS — `audit-typecheck-final.log`, `audit-build-final.log` |
| Paket export/declaration | PASS, 1613 dosya — `audit-package-smoke.log` |
| Browser import | Sınır PASS, budget UNMET/exit 2 — `audit-browser-imports-final.log` |
| Qwen bağımsız tekrar | EXECUTED, cleanup EXECUTED, exact JFK — `audit-qwen-public-parent-jfk.json` |
| Qwen üç fixture | EXECUTED; üç audio SHA ana ajan tarafından eşleştirildi — `audit-qwen-public-3-fixtures-worker.json` |
| Moonshine bağımsız streaming | WASM exact JFK fresh/warm PASS — `audit-moonshine-streaming-parent-final.json` |
| Moonshine GPU streaming | Exact JFK fresh/warm PASS, pozitif dispatch — `audit-moonshine-streaming-cache-webgpu-streaming-20261018.json` |
| Moonshine provenance | 7 source ve staged 10 dosya SHA ana ajan tarafından yeniden hesaplandı — `audit-moonshine-provenance-parent.json` |
| Demo testleri | Realtime 15, config 3, shared/playground/benchmark 9 test geçti — `audit-realtime-package-tests.log`, `audit-demos-config-tests.log`, `audit-shared-ui-tests.log` |
| Whitespace | Her iki checkout'ta git diff --check PASS |

Moonshine'in adı streaming olan mevcut üç graph klasörü dört graph içermez.
Hash'leri eşleşen cache/tokenizer kullanıldı. Geçici kopyalar testten sonra
`N:/github/asrjs/audit-isolated/moonshine-streaming-stage-20261018` içine taşındı;
kaynak cache/eski makbuzlar değişmedi. Yeni provenance makbuzu taşınmayı kaydeder.

Demo tarball bütünlüğü:
`sha512-zZK3OgWAfd6fibYiHnhZ4+R9qyZASfDhzlfnHeRcV/yNSV6EEHfRh3BTu0JNQAJDYr+mRk/G+EBP/WoyqgTiFA==`.
Bu registry yayını değildir; demos/vendor/README.md ve lockfile kaydı mevcut.

Son ekran incelemesinde benchmark metadata yüklemesinin `Map` API'sini nesne gibi
kullandığı bulundu. Ayrıca sıralı örnek yükleme eski positional argümanları ve
response nesnesinde map kullanıyordu; random seçim totalRows vermediğinden tek
satıra düşüyordu. Demo gerçek paket sözleşmesine uyarlandı; HTTP sınırı dışındaki
gerçek library API ile config/split, seed, offset, toplam satır ve boş yanıt
testleri geçti. Benchmark toplam 9 odaklı test ve production build PASS
(`audit-benchmark-dataset-final-tests.log`, `audit-benchmark-dataset-final-build.log`).
Playground'da kolon genişliğini aşan sabit
alt grid'ler container genişliğine göre auto-fit olacak şekilde düzeltildi;
production build geçti (`audit-playground-final-build.log`). Build'ler >500 kB
chunk uyarısı veriyor; bu performans kabulü sayılmadı.

## Açık kabul kapıları

- Nemotron gerçek graph/export ve resmi referans ile sayısal streaming paritesi.
- Root import <45 modül hedefi.
- Tüm model/backend/fixture matrisi ve kalite kabulü. Qwen'in üç girdiyi işlemesi
  Türkçe doğruluk veya GPU paritesi sertifikası değildir.
- Fiziksel mikrofon ve kullanıcı görsel kabulü; sentetik capture yerine geçmez.
- Statik deployment altında FireRed model init/inference ve aralıklı tarayıcı
  navigation/asset reset sorunları; yalnız HTTP 200 bu kapıyı geçirmez.

Tarihsel düzeltmeler README'de: 11 subtree import + bir dosya snapshot'ı kanıtlı;
12 kaynağın tüm Git geçmişi taşınmış sayılamaz. Eski W4/v1 PASS makbuzları yeni
verifier kabul kanıtı olarak kullanılmamalıdır. GPU dispatch, tüm graph'ların
GPU'da çalıştığı veya tensor paritesi anlamına gelmez.

## Temiz demo kurulumu ve ekran kontrolü

İzole son checkout:
`N:/github/asrjs/audit-standalone-20261018/demos-final`.
Kardeş speech-recognition checkout'u yok, node_modules ayrı npm ci ile kuruldu.
Son vendor tarball/lockfile hash'i ana ajan tarafından doğrulandı. Dört uygulama
package-mode production build'i ve realtime npm-alias nested-base build'i geçti.
Kalıcı build/test logları checkout'un `audit-output/` klasöründe;
`final-validation-summary.json` komut exit code'larını ve kanıt yollarını içerir.
Config 3/3, realtime 15/15, UI + benchmark 14/14 test geçti. Ana ajan özetle
birlikte gerçek log sonlarını ve build çıktılarını okudu. Güncel 28 uygulama
dosyası, lockfile ve vendor tarball hash karşılaştırmasını kendi tekrar hesapladı;
tümü eşleşti.

Ana ajan dört uygulamanın ekran görüntülerini inceledi. Playground düzeltmesinden
sonra sayı alanı/düğmeler üst üste binmiyor. Benchmark gerçek metadata isteğinden
sonra Ready ve Preview Samples (6) gösterdi; `benchmark-samples-ready.png` bunu
kaydeder. Bu kontrol ASR model inference veya kullanıcı görsel kabulü değildir.

Realtime nested `/nested/demo/` için JS, service worker ve ORT WASM dosyaları
HTTP 200, WASM MIME ve COOP/COEP ile servis edildi. Ayrı sayfa koşularında render
ve crossOriginIsolated görüldü; ancak sonraki birleşik tekrarlar aralıklı büyük
JS asset reset'lerine takıldı. Ana ajanın iki bağımsız Chrome tekrarı navigation'dan
dönmedi; yalnız bu testin süreçleri sonlandırıldı ve FAIL makbuzları
`C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/parent-demo-smoke*-timeout.json`
altında korundu. Bu nedenle browser matrisi için koşulsuz toplu PASS verilmiyor.

Son değişen/yeni kod ve paket dosyalarının SHA-256 manifesti:
`C:/Users/steam/.codex/visualizations/2026/10/18/01a15056-58b5-7d92-a2a8-d77764fbd286/audit/final-reviewed-source-manifest.json`.
60 library ve 40 demo dosyasını kapsar; raporlar/çalışma sonuçları bu hash listesine
dahil değildir. Çalışma ağaçlarında commit öncesi yeni dosyalar da vardır;
yalnız tracked diff'in aktarılması yeterli değildir.

### FireRed gerçek production worker denemesi: FAIL / kabul açık

Son makbuz:
`N:/github/asrjs/audit-standalone-20261018/demos-final/audit-output/firered-production-smoke-20261018T203438.672244Z.json`.
EOU model seçimi, speech-detect, FireRed ve vad-only ile fake microphone üzerinden
16 kHz/11 s JFK verildi. UI HTTP 200, crossOriginIsolated=true ve ses akışı var.
Worker INIT gönderdi; model URL isteği 200 döndü; `/nested/demo/ort/` yolu iletildi.
Ancak **INIT ACK, PROCESS sonucu veya VAD olasılığı alınamadı**. Başlangıç PCM'i
hazırlık beklerken mevcut ring-buffer süresini aştı ve capture açık RangeError ile
durdu. Ana ajan bu startup sınırının baseline kodunda da mevcut olduğunu diff ile
doğruladı; onarım sırasında eklenen onset buffer ile karıştırılmamalı.

Bu veri, F11'in asset URL düzeltmesini destekler; gerçek VAD init/inference kabulünü
geçirmez. HTTP başarısı model readiness olarak yorumlanmadı, tampon sınırı rastgele
büyütülmedi. İlk generic/manual probe ve MIME düzeltmesinden önceki başarısız
denemeler korundu. Test süreci ve kendi statik sunucusu kapandı.
