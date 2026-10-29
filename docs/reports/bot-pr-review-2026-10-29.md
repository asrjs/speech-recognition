# Bot PR incelemesi ve Jules geçmişi — 2026-10-29

Kapsam: açık PR #240 ve #241, `asrjs/speech-recognition` deposunun tüm PR geçmişi
(229 PR, 2026-03-15 → 2026-10-29). `asrjs/demos` deposunda açık bot PR'ı yok.
Tüm PR'lar `ysdede` kimliğiyle açılmış; bot PR'ları dal adı, başlık öneki
(`Bolt:`, 🧹, 🧪, ⚡, 🔒) ve "PR created automatically by Jules" gövdesiyle ayrıldı.

## Açık PR kararları

### #240 "Bolt: reduce avoidable typed-array copies in hot paths" — kapat

CI yeşil, ama davranış değiştiriyor ve kazanç ölçülmemiş.

- Tek parçalı girişte yeni dizi yerine girişin kendisini döndürüyor. Çağıranlar
  bu buffer'ların sahibi değil: manuel mikrofon modunda `chunk` aynı anda monitör
  ve dedektöre de gidiyor; `UrlAssetHandle.openStream()` içinde cache'e yazılan
  `bytes`, tüketiciye zaten verilmiş olan parçayla aynı nesne olur
  (`MemoryAssetCache` referansı saklar; tüketici buffer'ı worker'a transfer
  ederse cache'teki veri ayrılır).
- Tek parça yalnız çok kısa girişlerde oluşur (model dosyaları onlarca parça,
  mikrofon kayıtları yüzlerce parça). Kazanç gerçek iş yükünde yok denecek kadar
  küçük; PR benchmark sunmuyor.
- "Risk near zero" iddiası sahiplik analizi içermiyor.
- Bu konu 7. kez açılıyor (27 Ağustos'tan beri).

Elle uygulanacak bir şey yok.

### #241 "Bolt: enforce strict types for production pipeline transcripts" — kapat, asıl hatayı elle düzelt

PR, `ProductionTranscript.words/segments` için `word/start/end/probability`
şeklini public tipe sabitliyor. Gerçek Whisper çıktısı bu şekilde değil:

- Üretim pipeline'ının kullandığı enhanced executor `WhisperNativeTranscript`
  döndürür; kelime ve segment alanları `text`, `startTime`, `endTime`,
  `confidence`.
- Pipeline `w.word`, `w.start`, `w.end` okuyor. Gerçek birleştirici çıktısıyla
  denendi: iki ya da daha fazla kelimede `deduplicateWords` içinde
  `TypeError: Cannot read properties of undefined (reading 'toLowerCase')`
  ile çöküyor. Testler yanlış şekilli bir mock kullandığı için hata görünmüyor.
- PR ayrıca segmentlere `probability ?? 0.9` ile uydurma güven değeri ekliyor
  ve yerel segment alanlarını (`index`, `confidence`) atıyor.

Bu konu 7. kez açılıyor (7 Eylül'den beri). Bot her seferinde tipi "düzeltiyor",
ama hiçbiri çalışma zamanı şeklini kontrol etmemiş.

Elle yapılması gereken düzeltme:

1. `whisper-production-pipeline.ts` içinde `WordLike/SegmentLike` tiplerini
   `WhisperNativeWord/WhisperNativeSegment` ile değiştir; `text/startTime/endTime`
   alanlarını post-processing şekline dönüştür.
2. Eksik güveni uydurma; `confidence` yoksa alanı boş bırak ve post-processing
   bunu kabul etsin.
3. `ProductionTranscript.words/segments` için gerçek şekli yansıtan public tip.
4. Test mock'unu `mergeWhisperChunkTranscripts` çıktısıyla değiştir ve çökme
   senaryosu için regresyon testi ekle.

Alternatif: pipeline'ın depo içinde hiç kullanıcısı yok
(`createWhisperProductionPipeline` yalnız testlerde çağrılıyor). Kök API'den
kaldırıp `./pipeline` alt yoluna taşımak da makul.

## Geçmiş: tekrar eden konular

| Konu                                        |  PR |   Birleşen | Aralık          |
| ------------------------------------------- | --: | ---------: | --------------- |
| Test kapsamı ekleme (🧪, "Add tests for …") |  58 | 0 doğrudan | Mar 15 → Eki 22 |
| Buffer pooling, LUT, DSP tahsisi            |  20 |          0 | Mar 15 → Eki 21 |
| `padToFrames` / Whisper mel kopyalama       |  19 |          0 | Nis 17 → Eki 23 |
| Token/transkript eşleme döngüleri           |  19 |          0 | Mar 15 → Haz 25 |
| Vitest path alias ekleme                    |  13 |          0 | Haz 17 → Eki 20 |
| O(1) katalog arama                          |  11 |          0 | Nis 22 → Haz 3  |
| `??` → `!` sıcak döngülerde                 |  11 |          0 | Mar 23 → Nis 14 |
| `ProductionTranscript` tipleri              |   7 |          0 | Eyl 7 → Eki 29  |
| Tek parça hızlı yolu                        |   7 |          0 | Ağu 27 → Eki 28 |
| Dispose'ta async sarmalayıcı kaldırma       |   4 |          0 | Haz 14 → Eki 25 |
| TEN-VAD worker sızıntısı                    |   4 |          0 | Eyl 24 → Eki 21 |
| `console.warn` → logger                     |   4 |          0 | Mar 15 → Ağu 31 |

Toplam: yaklaşık 205 bot PR'ı, hiçbiri olduğu gibi birleştirilmemiş. Bazı
düzeltmeler 22 Ekim'de "Consolidate reviewed bot PR fixes" PR'ında elle
toplandı. Son 30 günde 24 bot PR'ı açıldı.

Gözlemler:

- **Bot reddedilen konuları hatırlamıyor.** Aynı fikir 7–20 kez yeniden açılıyor.
  Kapatılan PR'ları kontrol etmesi istenmemiş.
- **Bot proje kurallarını görmüyor.** Kök `AGENTS.md` `.gitignore`'da;
  GitHub'daki ağaçta yok. Jules bulutta klonladığı için mimari, güvenlik ve
  "ölçülmemiş optimizasyon yok" kurallarına erişemiyor.
- **Performans PR'ları ölçümsüz.** Depoda `npm run benchmark:hot-paths` var,
  hiçbir Bolt PR'ı kullanmamış. Hedeflenen kod (katalog araması 20 kayıt,
  `padToFrames`, tek parçalı birleştirme) inference süresinin ihmal edilebilir
  kısmı; darboğaz ORT oturumları.
- **Bazı fikirler aslında doğruydu.** `console.warn` → logger (bulgu 12,
  REVIEW-2026-10-28) ve TEN-VAD worker sızıntısı gerçek konular. Sorun fikir
  değil, kapsam ve kanıt eksikliği.
- **Vitest alias döngüsü boşa.** Alias'lar yerinde; eksik olan yalnız
  `controller` ve `worker`, onları hiçbir test import etmiyor.
- **Test PR'ları değer üretmiyor.** Çoğu, zaten kapsanan saf fonksiyonlara
  yüzeysel test ekliyor. Gerçek boşluklar (ör. üretim pipeline'ının gerçek
  executor şekliyle testi) atlanıyor.

## Jules için öneriler

1. **Kuralları görünür yap.** Kurallar artık takip edilen
   [`.github/BOT_CONTRIBUTION_RULES.md`](../../.github/BOT_CONTRIBUTION_RULES.md)
   dosyasında. Jules görev prompt'u ilk iş olarak bu dosyayı okumasını istemeli.
2. **Zamanlanmış görev sıklığını düşür.** Günlük Bolt görevi yerine haftalık
   tek görev; açık bot PR'ı varken yenisini açmasın.
3. **Konu yerine backlog ver.** Bota serbest "performans bul" yerine
   `REVIEW-2026-10-28.md` aksiyon listesinden tek bir madde ata.

### Önerilen kural dosyası taslağı

```markdown
# Rules for automated contributors (Jules, Bolt and similar)

Read this file completely before choosing a task.

## Before opening a PR

1. Search closed PRs for the same files and idea:
   `gh pr list --state closed --search "<file or keyword>"`.
   If a matching PR was closed without merging, do not reopen the idea.
2. Open at most one PR at a time. If an automated PR is open, stop.
3. One concern per PR, under ~150 changed lines, with a targeted test.

## Do not propose (previously rejected)

- Replacing `??` with `!` or other micro-edits in numeric loops.
- Map/Set caches for the model catalog or preset lookups.
- Pooling, LUTs or copy avoidance in mel/padding/concatenation code.
- Returning an input array instead of a copy ("single-chunk fast path").
  Callers do not own those buffers.
- Removing `async` from dispose/cleanup callbacks (PR 238 proved a regression).
- Adding Vitest path aliases unless a test currently fails without one.
- Retyping `ProductionTranscript` without first checking the real executor
  output (`WhisperNativeTranscript` uses `text/startTime/endTime/confidence`).
- Tests that only re-assert trivial pure functions already covered.

## Performance changes

Include before/after numbers from `npm run benchmark:hot-paths` or an
end-to-end transcription timing on a real fixture. Without a measured
improvement of at least 5% on a realistic workload, do not open the PR.

## Behavior changes

State which callers own the data you change and why aliasing, ordering or
error propagation stays the same. "Risk near zero" is not an argument.

## Good targets

Pick one item from `docs/reports/REVIEW-2026-10-28.md`, "Öncelikli aksiyon
listesi", that is not already done, and reference it in the PR title.
```

### Önerilen Jules görev prompt'u

```text
You are contributing to @asrjs/speech-recognition, a TypeScript speech
recognition runtime for browsers and Node.js.

First read .github/BOT_CONTRIBUTION_RULES.md, docs/architecture.md and
.github/pull-request-rules.md completely. Follow them strictly.

Task: pick exactly one open item from the action list in
docs/reports/REVIEW-2026-10-28.md (or the item named below), implement it
with targeted tests, and run npm run typecheck, npm run lint, npm test and
npm run build. Before starting, search closed PRs for the same idea and stop
if it was rejected.

Do not open a PR if: another automated PR is open, the change matches the
"Do not propose" list, or a performance change has no measured improvement.
If nothing qualifies, open no PR and report why.

Item: <fill in, e.g. "route library console.warn calls through
SpeechRuntimeHooks.logger">
```
