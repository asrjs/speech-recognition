# CI lint ve tam 12 saniye — 22 Ekim 2026

## Sonuç ve kapsam

CI lint hataları düzeltildi: 0 hata, 13 mevcut uyarı. Yerel tam test koşusu 1832 geçti, 36 atlandı; typecheck ve build geçti. Bu sonuç GitHub CI çalıştırıldı veya genel CI yeşil demek değildir. Root import kontrolü 78 modül ile istenen <45 hedefini karşılamıyor ve exit 2 veriyor; kapı değiştirilmedi.

Realtime tam 12.000 saniye hatası önce eski paketle yeniden üretildi, sonra kaynak ve yeni paket ile gerçek model/worker üzerinde doğrulandı. Paket modunda 12.030 saniyelik kontrol de geçti. Fiziksel mikrofon, VAD otomatik segmentasyonu ve kapsamlı model doğruluğu/parity doğrulanmadı.

## Küçük değişiklikler

1. Beş test dosyasında kullanılmayan import/argüman/değişkenler kaldırıldı; bir değişken const yapıldı. Assert veya lint kuralı gevşetilmedi. Bu adımın hedefli 52 testi geçti.
2. ONNX encoder fp32/fp16 çıkışı ve decoder girişi incelendi: [batch,1024,time]. v3 preset yapılandırmasına yalnız `encoderOutputLayout: 'bdt'` eklendi. Genel eksen tahmini, encoded_lengths doğrulaması, v2 ve EOU presetleri değiştirilmedi.
3. Gerçek hatanın [1,1024,151] / valid=150 biçimi için regresyon eklendi; 150 decoder adımının boyutu ve stride ile okunan değerleri doğrulanıyor. Üç v3 alias için manifest testi eklendi. Düzeltmeden önce iki test başarısızdı; düzeltmeden sonra ilgili 32 test geçti. Mevcut belirsiz eksen ve geçersiz uzunluk negatif testleri korunuyor.
4. Dört demo yeni, ayrı isimli yerel arşive sabitlendi. Önceki arşiv geri dönüş için korundu. 14 Node +45 Vitest testi ve dört uygulamanın paket modundaki production build'i geçti.

## Gerçek model kanıtı

Model: yerel parakeet-tdt-0.6b-v3, encoder fp16 / decoder fp32. Girdi: tests/fixtures/jfk2.en.wav; SHA256 59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e.

Chrome getUserMedia, WAV fake-device ile beslendi. Worker'a gönderilen PCM deterministik olarak 192000 örneğe (16000 Hz, tam 12.000 s) kesildi. Yaklaşık kayıt duvar saati süresi kanıt sayılmadı. Bu kontrollü capture sınırı testidir; fiziksel mikrofon testi değildir. Çıktı JFK ifade kontrolü ve kuyruk/worker tamamlanmasıyla doğrulandı; WER/oracle parity ölçümü değildir.

| Deneme | Model yükleme | Dosya | Kontrollü mikrofon |
|---|---|---|---|
| Eski paket, 192000 örnek | PASS | PASS | FAIL: layout/encoded_lengths |
| Kaynak düzeltme, 192000 örnek | PASS | PASS | PASS |
| Yeni paket, 192000 örnek | PASS | PASS | PASS |
| Yeni paket kontrol, 192480 örnek (12.030 s) | PASS | PASS | PASS |

Yeni paket: asrjs-speech-recognition-0.2.0-exact12-20261022.tgz; library base af8d147a6f09d5584c703faf84db24a65c0a4b45 + bu yerel manifest değişikliği. SHA512: ctSSYorN0oG4dj5fOb38ubXjznSEzVrdE73sUfhkYmqoFnD/1xSvXbNW4x5x/MfXcYBt+5swxkxGv6j5KstqJg==.

Library kanıtları: docs/reports/evidence/exact12-2026-10-22 (lint, tam test, önce başarısız regresyon, root import ve gerçek ONNX kontrat/hash). Browser önce/sonra JSON kayıtları ve demo test/build logları companion demos/docs/reports/evidence/exact12-2026-10-22 altında saklandı.

## Yeniden çalıştırma

Library: `npm run lint`, `npm run typecheck`, `npm test -- --reporter=dot`, `npm run build`, `node tests/smoke/browser-root-import-graph.mjs` (son komut halen exit 2).

Demos: `npm test`, `npm run build:suite`, `node scripts/serve-suite.mjs --port=4190`.

Python Playwright ve Chrome ile demos dizininden:

```powershell
python scripts/e2e/realtime-exact-duration.py --model-dir N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx --browser-executable "C:/Program Files/Google/Chrome/Application/chrome.exe" --samples 192000 --phase exact12
python scripts/e2e/realtime-exact-duration.py --model-dir N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx --browser-executable "C:/Program Files/Google/Chrome/Application/chrome.exe" --samples 192480 --phase control
```

## Bot PR incelemesi ve devam

76 açık library PR ve 0 demo PR envanteri alındı. Ayrıntılar bot-pr-review-2026-10-22.md ve JSON eşinde. Tüm difflerin satır satır doğrulandığı iddia edilmiyor; incelenmeyenler açıkça belirtiliyor. Kritik #227/#234 doğrulama kapısı ve #226 cache alias bulguları ana ajan tarafından diff ve mevcut kaynakla ayrıca denetlendi. Alt ajan incelemesindeki cache mutasyon zamanlaması düzeltildi: yeni regresyon stream tamamlandıktan sonraki mutasyondur. Hiçbir PR merge edilmedi veya GitHub yorumu gönderilmedi.

Bağımsız son kod incelemesi v3 değişikliğini dar ve testleri uygun buldu; eski paket README hash/provenance kaydı bulundu ve düzeltildi.

Sonraki işler: EOU120 warmup/oracle hatası, fiziksel mikrofon ve VAD, Hub/model-backend matrisi, root import <45. Hiçbirine PASS verilmedi. Bu tur değişiklikleri commit/push edilmedi.

Taşınabilir scripts/e2e/realtime-exact-duration.py son hâliyle ayrıca 192000 örnekte gerçek paket/model üzerinde geçti. Python -O ile assertion devre dışı bırakılması exit 1 ile reddediliyor.
