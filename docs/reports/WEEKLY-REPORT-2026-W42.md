# Haftalık çalışma raporu — 12–18 Ekim 2026 + 19 Ekim devamı

Bu belge speech-recognition kütüphanesi ile demos deposundaki denetim, onarım, merkezileştirme ve gerçek model testlerini birlikte kaydeder. Tarihler Europe/Istanbul saatine göredir. 19 Ekim gece testleri önceki haftanın devamı olarak ayrıca işaretlenmiştir. Çalışma bir sürüm/parite sertifikası değildir; bilinen açık hatalarla kullanıcı tarafından checkpoint ve merge istenmiştir.

## Hedef ve yaklaşım

Önce başka ajanın PASS iddiaları denetlendi. Eksik oracle, yanlış provider/mode, gerçekte yürütülmeyen model ve atlanan testler başarı sayılmadı. Küçük değişiklikler, negatif regresyonlar, ayrı checkout kullanan yazıcı alt ajanlar ve ana ajanın bağımsız diff/test kontrolleri esas alındı. Daha sonra farklı tarihlerde farklı kütüphaneler için hazırlanmış demolar, @asrjs/speech-recognition tüketen dört merkezi uygulamaya dönüştürüldü.

## 17–18 Ekim: kütüphane denetimi ve onarımlar

Başlangıç: speech-recognition 6e53c832a6e0837a117e187e383eaef51c1f6890. Ayrıntılı bulgular companion repository docs/reports/independent-audit-2026-10-18.md; onarımlar audit-repair-progress-2026-10-18.md içinde.

- Browser harness CLI, HTML ve matrix aynı runner/verifier sözleşmesine bağlandı. Audio SHA, tam normalize metin, gerçek provider/mode, fresh/warm handle kimliği kontrol edildi. Eksik oracle UNVERIFIED, hata FAIL; eski makbuzlar korunuyor. Warm çalıştırılmadığında süre uydurulmuyor.
- Qwen local-entry yüklemesi official graph sözleşmesi, prefill/step, fp16/fp32 sidecar ve nested tokenizer çözümlemesine uyarlandı. Üç gerçek ses girdisi işlendi. JFK tekrar edildi; Türkçe çıktıdaki yanlışlar kalite/parite kabulünden ayrı tutuldu.
- ORT release hataları görünür oldu. Node ortamı kurulu runtime yolunu çözüyor; ağ kapalı Identity 7→7 kontrolü ASR testi sayılmadı.
- VAD gate enum/config dönüşümü, worker PCM sample-rate aktarımı, onset replay, finalize öncesi son PCM ve final callback/error aktarımı düzeltildi.
- Moonshine partial allocation, geç yükleme/dispose yarışları ve double-release için negatif testler eklendi. Üç graph offline WASM ile dört graph streaming ayrıldı; yanlış provider iddiası başarısız sayıldı. Streaming WASM ve WebGPU JFK fresh/warm kanıtları kaydedildi.
- Nemotron predictor/streaming state ve atomic reset düzeltmeleri test edildi. Gerçek export/reference sayısal paritesi hâlâ açık; uygulama uyarıları kaldırılmadı.
- Transcript mapping bağımlılıkları ayrıldı. Root import 94'ten 78 modüle indi fakat <45 hedefi karşılanmadı; smoke bunu başarı olarak gizlemiyor.

Kaydedilmiş onarım doğrulaması: 1830 test geçti, 36 atlandı, 0 başarısız; typecheck, tüketici örnekleri, build ve package smoke geçti. Bunlar model ailesi başına sayısal parite değildir. Bu checkpoint sırasında yeniden çalıştırılan testlerin sonucu logbook'ta ayrıca kaydedilir.

## 18–19 Ekim: merkezi demo suite

Başlangıç: demos 57dae695096558f84b5a5923362e3a0896eb8add.

Dört aktif uygulama: Showcase (dosya transkripsiyonu), Playground (API/çıktı inceleme, kayıt, benchmark), Realtime (mikrofon, segment/kelime zamanları), Benchmark (dataset/local audio ve ölçüm dışa aktarımı).

- packages/ui ortak tema, kalıcı görünüm tercihi, shell/navigation, word pills ve SRT/VTT üretimi sağlıyor.
- packages/demo-runtime ortak katalog, load request/cache kimliği ve runtime başlangıcını sağlıyor.
- Portal + dört alt yol tek suite build'iyle üretiliyor. Statik sunucu nested base, MIME, isolation header ve traversal kontrolleri içeriyor.
- Varsayılan tüketim, yerel denetlenmiş 0.2.0 tarball üzerinden package-mode. Kaynak checkout'a bağımlılık opt-in; temiz kurulum ve kardeş repo olmadan build doğrulandı. Tarball registry yayını değildir.
- Benchmark dataset metadata Map sözleşmesi, response şekli, seed/offset/population seçimi ve progress hesabı düzeltildi.
- Playground kod örneği gerçek loadRequest ile hizalandı; desteklenmeyen streaming açıkça offline olarak gösteriliyor.
- Subtitle duration/offset ve milisaniye yuvarlama, responsive grid ve ortak tema kontrolleri eklendi.

İlk suite kabulü: 55 test, dört uygulama ve portal build'i, tarayıcı/navigation/responsive kontrolleri. Bu aşama inference kabulü değildi.

## 19 Ekim: gerçek model E2E

Yerel Parakeet TDT v3, fp16 WebGPU encoder / fp32 WASM decoder; JFK WAV SHA256: 59DFB9A4ACB36FE2A2AFFC14BACBEE2920FF435CB13CC314A08C13F66BA7860E. Gerçek Chrome ve NVIDIA Blackwell. Beklenen JFK cümlesi dört uygulamada elde edildi.

| Uygulama | Doğrulanan | Sınır |
|---|---|---|
| Showcase | Yerel model warmup ve dosya transkripsiyonu | EOU120 aynı oracle'ı geçemiyor |
| Playground | Easy Mode yükleme, dosya, WAV beslenen MediaRecorder, 3 benchmark | Fiziksel mikrofon ve diğer load modları doğrulanmadı |
| Realtime | Dosya, kelime zamanları, 12.03 s worker mikrofon kaydı | Tam 12.000 s encoder layout hatası açık |
| Benchmark | Yerel model/audio, gerçek çıkarım, JSON ve pozitif finite ölçümler | Reference metni yok; self-agreement doğruluk değildir |

Production ORT 404 yolu düzeltildi; dört app'in kendi ort/ dizini kullanılıyor. Realtime worker CDN bağımlılığı kaldırıldı ve CDN bloklu gerçek koşu geçti. Benchmark seçilen quant/thread/processor seçeneklerini artık gerçek public loader'a iletiyor. Realtime canonical detail=words ile 22 zamanlı kelime görüldü. Metrik paneli canonical camelCase ve native snake_case alanlarını kabul ediyor.

Son demo regresyon sonucu: 14 Node + 45 Vitest = 59 başarılı test. Full suite build ve sonraki hedefli Benchmark/Realtime build'leri başarılı. Büyük bundle uyarıları performans kabulü değildir.

## Açık hatalar ve kabul kapıları

1. Tam 12.000 s Realtime mikrofon: [1,1024,151] encoder çıktısı ile encoded_lengths=150 reddediliyor. 12.030 s başarılı kontrol, hatayı çözmez. Önce export layout sözleşmesi doğrulanmalı; generic heuristic gevşetilmemeli.
2. EOU120 warmup fp16/int8 ve fp32/fp32 metinleri mevcut oracle'ı karşılamıyor. Beklenen metin/tolerans değiştirilmedi.
3. Fiziksel mikrofon, otomatik FireRed/TEN VAD, Hub download ve geniş model/backend matrisi doğrulanmadı.
4. Nemotron sayısal streaming paritesi ve root import <45 bütçesi açık.
5. Kullanıcı görsel kabulü, genel Türkçe doğruluk ve tüm modellerin performans/kalite kabulü açık.

## Kanıtlar ve yayın sınırı

Kütüphane: docs/reports/audit-repair-progress-2026-10-18.md ve tools/data/results/browser-harness/. Demolar: docs/suite-review-2026-10-19.md, docs/real-model-e2e-2026-10-19.md ve yerel tmp/real-e2e/ makbuzları. Alt ajan raporları tek başına PASS kaynağı yapılmadı.

Bu çalışma yalnız Git commit/push/merge checkpoint'idir. npm/Hugging Face yayını veya model ağırlığı değişikliği yok. Tamamlanmayan kapılar merge ile kapanmış sayılmaz. Devam sırası HANDOFF-2026-10-19.md; komut/teslim kaydı LOGBOOK-2026-W42.md içindedir.
