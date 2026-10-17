# Paralel ajan görevi — Qwen doğrudan fp16 export

Tarih: 2026-10-08. Bu dosya kullanıcı tarafından diğer ajana iletilecek bir
iş sınırlandırmasıdır; mesaj gönderildiği veya işin başlatıldığı anlamına gelmez.
Yeni model kuyruğu değildir: ana ajan Orukeet / Parakeet TDT türevleri ve
2026 adayları üzerinde kalır. Parakeet CTC beklemede, Nemotron duraklatılmıştır.

## Başlangıç ve kanıt sınırı

Önce `docs/AGENT_ENTRYPOINT.md` ve yönlendirdiği güvenlik/mimari belgelerini oku.
Ardından `tools/README.md`, `tools/model-debugging/SKILL.md`, Qwen playbook'u
ve Qwen reference README'sini oku. Kaynak çalışma notu:
[workstate-qwen-fp16-streaming-2026-10-08.md](workstate-qwen-fp16-streaming-2026-10-08.md).
Bu nottaki ölçümler devralınmış kanıttır; bu görevi hazırlayan ana ajan bu turda
inference matrisini yeniden çalıştırmadı.

Önemli ayrım: `src/models/qwen-asr` ve eski official dynamic/native-fp16 portu
zaten vardır. Görev, daha yeni `models/onnx/qwen3-asr-0.6b` / Transformers-HF
export hattının fp16 adayını düzeltmektir. Eski ve yeni graph isimlerini,
tokenizer/prompt/feature/KV sözleşmelerini karıştırma; eski çalışan portu değiştirme.

## Çakışmasız çalışma sözleşmesi

- Yalnız ayrı, yazılabilir bir worktree/checkout'ta kod değiştir. Önce gerçek
  repo kökünü, HEAD'i, worktree listesini, dosya yazılabilirliğini ve kendi temiz
  baseline'ını doğrula. İzolasyon kurulamazsa ortak checkout'ta edit yapma;
  salt okunur inceleme ve önerilen patch ile dön.
- Ortak checkout kirli ve bazı Qwen dosyaları untracked. HEAD'den yeni worktree
  oluşturmak bunları taşımaz. Kendi önceki oturumuna ait aşağıdaki Qwen
  değişikliklerini içerikleri/hash'leri kayıtlı, dar bir patch olarak izole
  baseline'a taşı; başkalarının değişikliklerini topluca kopyalama veya commit etme.
- Aşağıdaki izinli yolların dışında edit gerekirse önce bildir. Başka ajanlar
  da çalışıyor: onların değişikliklerini geri alma veya üstüne yazma.
- Ana checkout'un `.cache` ağırlıkları, graph'ları ve referans ortamları yalnız
  okunur girdi olabilir; mutlak yollar kullan, aynı klasöre çıktı yazma.
- Tek seferde bir ağır export/inference işi çalıştır. GPU/browser/server :8765
  kullanımı ve yüksek RAM isteyen işler için ana ajanla zaman penceresi anlaş;
  kendi portunu/profilini kullan, başka ajanların süreçlerini kapatma.
- Yeni büyük indirme, dependency install/upgrade, sibling edit, push/upload,
  model promotion veya eski artifact/shard silme bu görev kapsamında değildir.
  Cache eksikse durumu ve ihtiyaç duyulan izinleri raporla.

## Q1 — Öncelikli teslimat: tutarlı doğrudan fp16 graph seti

Sahiplik (yalnız izole checkout'ta):

- `tools/qwen3-asr-onnx-export/export_qwen3_asr_fp16.py`
- `tools/qwen3-asr-onnx-export/test_qwen3_asr_fp16_export.py` (yeni, küçük testler)
- Kendi checkout'unda yeni `models/onnx/qwen3-asr-0.6b-fp16-direct-2026-10-08/`
  çıktı klasörü ve `tools/data/results/qwen/fp16-direct-2026-10-08/` receipt'leri.
- Sonuç raporu: `docs/reports/qwen-fp16-direct-result-2026-10-08.md`.

`export_qwen3_asr_decoder.py`, `capture_cu_seqlens.py` ve mevcut
`transcribe_qwen3_asr_local.mjs` bu teslimatta salt okunur referans/tüketicidir.
Mevcut runner fp32 floating feed kullanır; exporter sınırını buna uyumlu tut.
Bu üç dosyadan birini değiştirmek gerekirse önce exact neden ve dar diff önerisini
bildir; başlangıçta edit sahipliği yalnız exporter ve yeni test dosyasındadır.
Yeni üretim decoder loop'u yazma.

Adımlar:

1. [ ] Yerel özgün checkpoint, revision/hash, tokenizer, processor ve gerçek
   Python/Torch/Transformers/ORT sürümlerini bağla. fp32 oracle ve graph baseline'ını
   aynı etkin PCM/prompt ile doğrula. Mevcut fp32 kontrolün oracle kapsamını belirt.
2. [ ] Exporter'ın gerçek tower/mask ve `capture_schedule` API'sini incele.
   Schedule öncesindeki feature dtype'ını, wrapper feature/mask dtype'larını
   kaynak kod ve küçük negatif testle düzelt; mask tipini tahmin ederek değiştirme.
3. [ ] Aynı özgün checkpoint/builder'dan front + embed + decoder export et.
   Var olan `DecoderStep` sınırını kullan. İlk adayda mevcut runner'ın fp32
   floating I/O sözleşmesini koru; iç weights/compute fp16 olsun. I/O tipi gerçekten
   değişmek zorundaysa gizleme: sözleşme/dtype tablosu ve tüketici etkisini raporla.
4. [ ] Nonempty past ile tracing ve dinamik past/token eksenlerini doğrula.
   Boş past 0, past 1 ve çok-adımlı büyüme kontrollerini çalıştır. Load-only PASS
   yetmez; logits ile key/value present tensörlerini gerçek feed ile üret.
5. [ ] Node ve mevcut Python ORT ile kontrol et. Optimization seviyesini receipt'e
   yaz; `basic` başarısını varsayılan `all` başarısı olarak etiketleme. Dönüştürülmüş
   eski fp16 setini silme veya overwrite etme; Concat hatasını negatif kontrol
   olarak koru. Doğrudan export başarısını ölçmeden garanti etme.
6. [ ] Önce aynı fp16 PyTorch compute ile ONNX export sayısal uygunluğunu ölç;
   ayrıca özellikler, audio embeddings, decoder logits, KV cache, tokenlar ve metni
   özgün fp32 referansla karşılaştır. max/mean abs, cosine, açık atol/rtol ve ilk
   sapmayı kaydet; mevcut kabul eşiklerini gevşetme. Tek JFK metin eşleşmesi sayısal
   parity veya bütün model kabulü değildir; ikinci farklı uzunlukta fixture ekle.
7. [ ] Graph ve tüm external-data/support dosyalarını SHA-256 ile bağla; I/O
   geometri/dtype ve kullanılan source/script hash'lerini kaydet. Kendi küçük
   testlerinde eksik sidecar, yanlış dtype ve hatalı cache büyümesi kontrolleri olsun.
8. [ ] Yukarıdakiler geçtikten sonra, üzerinde anlaşılmış zaman penceresinde
   mevcut gerçek browser harness'ıyla WASM/WebGPU/hybrid bileşen yerleşimini
   ayrı ölç. Gerçek provider/bundle ve artifact kimliği, cold load, warm tekrarlar,
   bellek, token/text kontrolü ve awaited disposal kaydedilsin. Tek başına 1.88 GB
   disk boyutu browser'a sığma kanıtı değildir. Harness/shared runtime değişimi
   gerekiyorsa burada durup dar öneriyle ana ajana dön.

Exporter'ın mevcut docstring/receipt gerekçelerini de düzelt: Node optimization
option'ı olmadığı, I/O'nun zaten fp16 olduğu ve fp16'nın kesin browser'a sığdığı
iddiaları gerçek parser/wrapper ve devralınmış ölçümlerle uyuşmuyor.

## Q2 — Bağımsız küçük teslimat: fp32 sidecar audit

Q1 ağır iş başlatılamıyorsa önce yapılabilir. Salt okunur mevcut graph/sidecar
incelemesiyle location, offset, length, interval union, en büyük referenced end,
dosya boyutu, overlap/gap ve hash tablosu üret. Yalnız length toplamını canlı
veri boyutu sayma; overlap, alignment ve bütün graph referanslarını kontrol et.
Sonuç `tools/data/results/qwen/fp16-direct-2026-10-08/sidecar-audit.json` olsun.

7.15 GB dosya / 2.38 GB referenced-data raporunu hipotez olarak doğrula.
412 shard'ı veya eski `.data` dosyasını silme. Kompakt export deneyi yapılırsa
yalnız yeni klasöre yaz, eski receipt'leri koru; yeni hash ve aynı-feed sayısal
kontrollerle ayrı aday olarak raporla. Diskten küçülme tek başına browser bellek
sorununu çözdüğü anlamına gelmez. Bu, Q1 teslimatını erteleyen bir temizlik projesi değil.

## Moonshine — yalnız sonraki koordinasyon için

5.18 yüklemesi ve raw-waveform / 80-sample frame / 320 hidden_size düzeltmesi
değerli. Eski streaming graph'ları geçersiz kontrol olarak korunacak. Bu görev
Moonshine yeniden export/streaming implementation sahipliği vermez. Q1/Q2
tesliminden sonra ayrı sahiplik anlaşmasıyla ele alınabilir; whole-utterance HF
inference gerçek incremental partial/final streaming oracle'ı yerine geçmez.

## Teslim ve entegrasyon

Her alt adım PASS / FAIL / BLOCKED / UNTESTED olarak raporlansın. Exact komutlar,
ortamlar, kaynak/artifact hash'leri, sayısal metrikler, kalan engeller, own diff
ve varsa scoped commit ID'leri verilsin. Ağır binaries/venv/weights commit edilmez.
Ana ajan `AGENT_ENTRYPOINT`, goal, ortak task listeleri ve aday listesinin sahibi;
bunları değiştirme. `src/`, paket/lock/config, TDT/Whisper/Nemotron araçları ve
sibling örnekler korumalıdır. `git add -A`, ortak checkout reset/clean ve global
ayar değişikliği yok. Ana checkout'a merge/cherry-pick uygulama; ana ajan inceleyip
test ederek entegre eder. Tam parity/browser kapıları geçmediyse teslim PARTIAL'dır.
