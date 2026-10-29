# WebGPU int8 ve çevrimdışı cache ölçümü — 2026-10-29

REVIEW-2026-10-28 bulgu 1 ve 5 için gerçek Chrome ölçümü.

Ortam: Windows 10, Chrome (Playwright `channel=chrome`, headed, `--enable-unsafe-webgpu`),
NVIDIA Blackwell adaptörü, onnxruntime-web 1.29.0, `dist/` main `d2e14861` sonrası build
(#242 dahil). Ses: `jfk-short.wav` (11 s). Ağırlıklar: `N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx`
(yerel, yeni indirme yok). Betikler ve ham JSON: oturum scratchpad'i (`measure.py`, `offline.py`).

## 1. int8 encoder WebGPU'da

Decoder her koşuda int8/WASM (`webgpu-hybrid`), değişen yalnız encoder.

| Encoder        | Yükleme (soğuk) | Çıkarım soğuk | Çıkarım sıcak (3 koşu) | GPU dispatch | Metin |
| -------------- | --------------- | ------------- | ---------------------- | ------------ | ----- |
| int8 / WebGPU  | 7,5 s           | 6,65 s        | 5,17 / 5,00 / 7,61 s   | 1966         | doğru |
| fp16 / WebGPU  | 31,3 s          | 3,18 s        | 0,73 / 0,67 / 0,64 s   | 1281         | doğru |
| int8 / WASM    | 15,8 s          | 7,86 s        | 7,54 / 6,66 / 5,25 s   | 0            | doğru |
| fp32 / WebGPU  | —               | —             | —                      | —            | `Failed to fetch` (2,4 GB `.onnx.data`, harness `fetch().blob()` sınırı) |

ORT her iki WebGPU oturumunda `VerifyEachNodeIsAssignedToAnEp` uyarısı veriyor; düğüm
listesi minimal build'de görünmüyor. Kanıt zamanlamadan: int8 WebGPU, tek iş parçacıklı
WASM int8 ile aynı bantta ve fp16'dan ~7× yavaş. Sonuç: int8 graf WebGPU EP'de *çalışıyor*
(doğru çıktı, oturum WASM'a düşmüyor) ama nicemlenmiş matmul'lar düğüm düzeyinde CPU'ya
gidiyor; GPU hızlanması yok.

Karar: `probe.ts` değiştirilmedi. `int8`'i `supportedPrecisions`'tan çıkarmak
`execution-context.ts:32` yüzünden açık `precision: 'int8'` + WebGPU isteklerini hataya
çevirir; shader-f16'sız adaptörde de geriye yalnız fp32 (bu harness'ta yüklenemeyen 2,4 GB)
kalır. Planlayıcı varsayılanda zaten fp16 encoder seçiyor (bölüm 2'deki istekler). Doğru
yer "desteklenir" değil "hızlandırılır" ayrımı: ör. probe notu veya planlayıcı tercih
ağırlığı. Bu tek adaptör ölçümüyle davranış değişikliği yapılmadı.

## 2. Cache'li v3 modeli ağ kapalıyken

Yöntem: kalıcı Chrome profili. Faz 1'de `huggingface.co` istekleri yerel dosyalara 302 ile
yönlendirildi ve `loadSpeechModel({ modelId: 'parakeet-tdt-0.6b-v3', backend: 'webgpu-hybrid' })`
IndexedDB'yi doldurdu (1199 MB). Faz 2'de 127.0.0.1 dışındaki her istek
`ERR_INTERNET_DISCONNECTED` ile düşürüldü. (`context.set_offline` localhost'u da
kestiği için kullanılmadı.)

| Faz      | Sonuç  | Yükleme | Çıkarım | HF istekleri |
| -------- | ------ | ------- | ------- | ------------ |
| Çevrimiçi | geçti | 29,3 s  | 2,10 s  | encoder fp16, decoder int8, vocab, tree listing |
| Ağsız    | geçti  | 6,8 s   | 0,68 s  | 4 başarısız deneme, hepsi yutuldu |

Revizyon `f88260fa…` her iki fazda pinli. Ağsız fazda doğru transkript üretildi;
bulgu 1 düzeltmesi gerçek ağırlıkla doğrulandı.

Kalan maliyet (bulgu 1 öneri a, açık): ağsız her yüklemede tree listing, metadata
listing ve iki `.onnx.data` sondası ağa gidiyor ve iki konsol uyarısı basılıyor.
Manifest artefakt listesi biliniyorsa bu dört istek gereksiz.

Not: kullanılan yerel ağırlıklar `istupakov/parakeet-tdt-0.6b-v3-onnx` kopyası; dosya adları
aynı, baytlar `ysdede` revizyonuyla karşılaştırılmadı. Test cache/çevrimdışı yolunu
doğruluyor, model baytlarının kimliğini değil.
