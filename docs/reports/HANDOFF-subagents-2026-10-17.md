# Alt ajan devir kılavuzu — 2026-10-17

Bu dosya, 2026-10-17 toparlamasından sonra işe başlayacak alt ajanlar içindir.
`docs/AGENT_ENTRYPOINT.md` okuma sırasının yerini almaz; onun üstüne güncel
durum, büyük riskler ve görev kartlarını ekler. Çelişki olursa canlı Git,
kaynak kod ve receipt'ler esastır.

## 1. Anlık durum (2026-10-17)

- `main` = `origin/main`. Tam vitest paketi: 206 dosya, 1703 test PASS,
  34 skip. `npm run typecheck` PASS. Tracked working tree temiz.
- Ortam: Windows 10, PowerShell, Node v26.2.0, `onnxruntime-node` 1.30.0.
  Python referans venv'i: `tools/model-debugging/reference/transformers5/`
  (git-ignored, yalnız bu makinede; onnxruntime 1.30.0, torch 2.14.1+cpu,
  transformers 5.18.0).
- Branch'ler:
  - `wip/nemotron-rnnt-2026-09-01` (origin'de): Ağustos sonundan kalan,
    **doğrulanmamış** Nemotron değişiklikleri. Nemotron duraklatıldı; dokunma.
  - `parallel/qwen-fp16-direct-2026-10-08`: main'e fast-forward edildi.
    Rebase öncesi yedek tag'i: `backup/qwen-fp16-pre-rebase-2026-10-17`.
- Linked worktree: `U:/parallel-agents/qwen-fp16-wt`. Qwen fp16 ONNX
  artefaktları **yalnız burada** duruyor (`models/onnx/qwen3-asr-0.6b-fp16-direct-2026-10-08/`,
  git-ignored). Receipt'ler bu yolları gösterir. Worktree'yi silme.
- 1.921 izlenmeyen dosya, tamamı `tools/data/results/` altında eski receipt'ler.
  Ne commit et ne sil; T7 kartına bak.

## 2. Büyük sorunlar ve dersler

1. **Commit edilmemiş iş birikiyordu.** Görev kayıtları 9 gün, Nemotron işi
   6 hafta commit dışında kaldı. 9 Ekim'deki "1705 passed" sonucu, commit
   edilmemiş bir test dosyası sayesinde yeşildi. Commit'li test, `7ec8c8b0`'dan
   beri commit'li preset ile çelişiyordu (`8a9c02e1` ile düzeltildi).
   Kural: her dilimin sonunda sahip olduğun değişiklikleri commit et ve tam
   paketi **temiz tracked tree** üzerinde çalıştır.
2. **Main 7 hafta push edilmemişti** (377 commit). Push yalnız kullanıcı
   izniyle yapılır. Dilim sonunda "push bekliyor" diye açıkça raporla.
3. **fp16 kapısı yanlış referansa bağlıydı (Qwen).** ORT CPU, fp16 Gemm/Add/
   LayerNorm zincirlerini fp32'de çalıştırıyor ve 134 precision Cast ekliyor.
   Bu yüzden "ONNX vs native PyTorch fp16" karşılaştırması doğruluğu değil,
   yuvarlama yolunu ölçer. Kanıt: `docs/reports/qwen-fp16-direct-result-2026-10-08.md`
   QWEN-012/013. Kullanıcı, **Qwen için** göreli kapıyı onayladı (bkz. §4).
   Başka bir modele uygulamadan önce kullanıcıya sor.
4. **Yeni graph setlerinin hiçbiri gerçek browser WebGPU'da doğrulanmadı.**
   Bu kapsamda Moonshine streaming (fp32/fp16/chunked) ve Qwen fp16 var.
   WebGPU birincil backend; ürün açısından en büyük açık bu.
5. **Orukeet strict feature numerical parity çözülmedi.** Kayıtlar:
   `docs/reports/HANDOFF-chrome-auto-runner-2026-10-08.md` sonu ve
   `docs/reports/orukeet-*-2026-10-08.md`. Reddedilmiş FFT deneylerini tekrarlama.
6. **Redux packed kernel bu CPU'da çalışmıyor** (Skylake-W, VNNI yok). Hız
   iddiası Photon/x86 içindir; ORT/WebGPU'ya aktarılamaz.
7. **Paralel ajan çakışması.** Ortak dosyaları iki ajan aynı anda değiştirdi;
   sahiplik belirsiz kaldı. Kural: lane başına bir worktree; ortak dosyaya
   dokunacaksan ilk iş olarak lane iletişim dosyasında duyur.

## 3. Ortak çalışma protokolü

Başlamadan önce:
- `docs/AGENT_ENTRYPOINT.md` okuma sırasını uygula. `docs/AGENT_SAFETY_RULES.md`
  dosyasını baştan sona oku.
- `git status --short --untracked-files=no` boş olmalı. Değilse dur ve bildir.
- Kartın gerektirdiği artefaktların var olduğunu ve hash'lerinin receipt ile
  eşleştiğini doğrula.

Çalışırken:
- Lane başına ayrı worktree aç:
  `git worktree add U:/parallel-agents/<lane> -b parallel/<lane>-<tarih> main`.
  Gerekirse `node_modules` için N: checkout'una junction kur
  (Qwen worktree'sinde örneği var).
- Her deney için `tools/data/results/<aile>/...` altına JSON receipt yaz.
  Receipt içeriği: tam komut, girdi/artefakt SHA-256, runtime sürümleri,
  provider, comparator ve eşikler, ölçülen değerler, `status`. Status için
  `*_NOT_NUMERICAL_ACCEPTANCE` gibi dürüst etiketler kullan.
- Comparator: `tools/tensor_parity.py::compare_tensors`. Eşik değiştirme.
- Durum etiketleri: PASS / FAIL / PARTIAL / BLOCKED / UNTESTED. Kısmi iş PASS değildir.
- Aynı yerde iki başarısız düzenleme olursa dur, gerçek API'yi yeniden incele.

Bitirirken:
- `npx vitest run` ve `npm run typecheck` çalıştır (src/test değiştiyse). Python
  aracı değiştiyse ilgili `unittest` dosyasını çalıştır.
- Yalnız sahip olduğun dosyaları commit et. Mesaj: `test(<aile>): ...`,
  `feat(...)`, `docs(...)`.
- Görev dosyasını güncelle (`docs/PARAKEET_VARIANT_TASKS.md` P10 veya
  `docs/PRODUCT_TASKS.md`). Gerekirse lane raporuna kısa bir not ekle.
- Ana ajana rapor ver: ne yapıldı, receipt yolları, commit hash'leri, kapı
  durumları, push bekleyen branch'ler, açık kararlar.

Kullanıcı onayı olmadan yapılmayacaklar: push/publish/HF upload, global kurulum,
yeni büyük indirme, sibling repo'da düzenleme, kabul kapısı değişikliği, model
sırasını değiştirme. Nemotron ve Parakeet CTC'ye başlamak da bu kapsamda.

## 4. Kapı tanımları

- fp32 graph'lar: fp32 oracle'a karşı `compare_tensors(atol=1e-4, rtol=1e-3)`
  allclose, ardından token/metin eşitliği.
- fp16, yalnız Qwen için onaylı göreli kapı: fp32 oracle'a karşı ONNX max ve mean
  abs hatası ≤ native PyTorch fp16 hatası. Ayrıca greedy token/metin oracle ile
  aynı olmalı. Oracle: orijinal ağırlık (bf16 → fp32), fp32 compute.
- Browser: aynı oracle receipt'lerine karşı token/metin eşitliği, ardından
  numerical karşılaştırma. Provider'ın gerçekten WebGPU olduğu (fallback
  olmadığı) receipt'te kanıtlanmalı.

## 5. Görev kartları (öncelik sırası)

Kartlar bağımsızdır. T1 ve T2 GPU kullandığı için aynı anda çalıştırılmamalı.

### T1 — Qwen fp16 WebGPU browser doğrulaması
- Durum: native CPU kapıları PASS (QWEN-013). Browser UNTESTED.
- Girdiler: `U:/parallel-agents/qwen-fp16-wt/models/onnx/qwen3-asr-0.6b-fp16-direct-2026-10-08/dynamic-front/`
  (front, decoder, embed, support dosyaları). Oracle receipt'leri:
  `tools/data/results/qwen/fp16-direct-2026-10-08/decoder-precision-path-2026-10-17.json`,
  `front-precision-path-2026-10-17.json`.
- Adımlar:
  1. `tools/README.md#task-to-tool-map` üzerinden otomatik Chrome runner'ı ve
     `../webgpu-agent-test`'i bul. Mevcut Qwen browser yolunu incele
     (`src/models/` altındaki qwen ailesi, `docs/reports/model-port-status-and-guide-2026-10-08.md`).
  2. Bu dizindeki graph'ları WebGPU session'da aç. fp32 floating I/O
     sözleşmesini koru. Önce yalnız front'u çalıştırıp `features-1100.npy`
     ile `audio_embeds` çıktısını oracle'a karşı ölç.
  3. Ardından decoder döngüsü: JFK'de 30 adım, 29 token ve metin eşitliği.
- Kabul: provider WebGPU olarak kanıtlanmış, token/metin eşit, göreli kapı
  sonucu receipt'te. Bellek ve süre ölçümü ayrı alan olarak kaydedilir; tek
  çalıştırma performans kanıtı sayılmaz.
- Yasak: CPU/WASM fallback'i PASS saymak, graph'ı yeniden export etmek (önce sor).

### T2 — Moonshine streaming WebGPU browser doğrulaması
- Durum: oracle, fp32/fp16/chunked export, `StreamingTranscriber` ve public
  loader commit'li. Node-ORT PASS. Browser yok.
- Kaynak: `docs/PARAKEET_VARIANT_TASKS.md` P10 Moonshine maddesi,
  `src/models/moonshine/streaming-session.ts`,
  `tools/model-debugging/reference/moonshine-streaming/README.md`.
- Adımlar: fp32 set → fp16 set → chunked set sırasıyla WebGPU'da JFK oracle'ının
  28 token'ını üret. Streaming için reset/reuse kontrolünü tekrarla.
- Kabul: her set için ayrı receipt, token eşitliği ve provider kanıtı.
  Encoder mask zorunlu (mask'siz yol bozulur; bu bilinen bir sözleşme).

### T3 — Parakeet Ultra eşleştirilmiş hız/bellek karşılaştırması
- Durum: Kestrel oracle PASS (`ultra-official-oracle-jfk-2026-10-08.json`).
- Hedef: aynı host ve fixture'da, ısınmış ölçümlerle Ultra'yı mevcut
  Orukeet/v3 yolu ile karşılaştırmak. Dense-oracle CPU süresi bu karşılaştırma
  değildir.
- Önce: Ultra için bizim export ve runtime yolumuz var mı, kontrol et. Yoksa
  bu kart bir planlama raporuna dönüşür; kod yazmadan önce sor.

### T4 — Redux: sınırlı deney seçimi (karar raporu)
- Hedef: "packed compute deneyi" ile "etiketli dense baseline" arasında, ölçülebilir
  maliyetleriyle (decode/repack, bellek, kernel desteği) bir karar notu yazmak.
- Çıktı yalnız rapordur. Teacher decoder'ı Redux encoder ile karıştırma.
  Korelasyon tabanlı ağırlık eşlemesini tekrarlama.

### T5 — Diğer TDT student ve precision varyantlarının audit'i
- Hedef: gerçek TDT distill ve student'ları exact ID, revision ve lisans ile
  listelemek. Yalnız metadata ve kart okunur; indirme yok.
- Dosya boyutundan "küçük student" iddiası çıkarılmaz.
  `docs/reports/light-asr-diarization-hunt-2026-09-30.md` ile birleştir.

### T6 — Sıradaki non-NVIDIA aday: Fun-ASR MLT Nano oracle'ı
- Sıra: Moonshine streaming (T2) → Fun-ASR MLT Nano → ARK-ASR → Cohere
  Transcribe → Voxtral Realtime.
- İlk adım yalnız official inference koduyla oracle yakalamaktır (P02 deseni).
  Gerekli indirme veya kurulum için önce onay al.

### T7 — İzlenmeyen receipt triage'ı (hijyen)
- Hedef: `tools/data/results` altındaki 1.921 izlenmeyen dosyayı üç gruba ayırıp
  bir öneri listesi çıkarmak: commit edilecek receipt'ler, ignore edilecek
  build çıktıları (ör. `windowing-isolated-build-*`) ve büyük veya geçici dosyalar.
- Çıktı yalnız rapor ve önerilen `.gitignore` satırlarıdır. Silme yok;
  uygulama kullanıcı onayıyla yapılır.

## 6. Kapalı ya da duraklatılmış işler

- Nemotron: duraklatıldı. `wip/nemotron-rnnt-2026-09-01` doğrulanmadı.
- Parakeet CTC: ON HOLD.
- efederici int4: yalnız graph audit; hedef format değil.
- Qwen encoder lowering FAIL'i: kök neden ölçüldü, göreli kapı ile kapandı.
  Katman ikilemesini tekrarlama.
