# Logbook — 2026 W42 ve 19 Ekim teslimi

- 17–18 Ekim: devralınan library/harness çalışması bağımsız denetlendi; 15 bulgu ve onarımlar companion repo raporlarında kaydedildi.
- 18 Ekim: library audit sonucu 1830 passed / 36 skipped / 0 failed; package/type/build kontrolleri. Ayrıntılar audit-repair-progress-2026-10-18.md.
- 18–19 Ekim: dört merkezi demo, ortak tema/runtime, suite portal ve statik deployment. Suite aşaması 55 test.
- 19 Ekim: gerçek Parakeet v3 browser testleri; ORT asset, benchmark option, mic worker path ve canonical words/metrics düzeltmeleri. Son demo testi 59/59; build başarılı.
- 19 Ekim: exact 12 s microphone FAIL, 12.03 s control PASS; EOU120 fp32 ve mixed precision warmup FAIL. Fiziksel mic/VAD/Hub UNVERIFIED. Başarısız makbuzlar korundu.
- 19 Ekim kullanıcı talebi: bu durumla commit, push, merge; haftalık rapor ve handoff yaz; sonra dur.
- Teslim hazırlığı: demos taban 57dae695096558f84b5a5923362e3a0896eb8add; library taban 6e53c832a6e0837a117e187e383eaef51c1f6890. Her iki origin/main aynı tabanlarda doğrulandı. Sandbox dışındaki mevcut keyring/SSH yetkisi çalışıyor; yeni kimlik bilgisi oluşturulmadı.

## Son teslim sonucu

Commit/PR/merge sonuçları işlem tamamlandığında aşağıya eklenir. Bu satır henüz merge kanıtı değildir.

Checkpoint yeniden doğrulaması: npm run typecheck exit 0; npm test -- --reporter=dot: 1830 passed / 36 skipped, 216 passed files / 7 skipped. Loglar tools/data/results/browser-harness/checkpoint-*-20261019.log. Kod commitleri: bba2a8d5 (harness), d99b6cb1 (runtime).
## Doğrulanmış teslim — 19 Ekim 2026

- Library PR https://github.com/asrjs/speech-recognition/pull/225 MERGED; merge SHA 35b624a75d0cc14f660e5756e19930bc0a0fb2de. Kod commitleri bba2a8d5, d99b6cb1; evidence/docs 050cd0e4.
- Demos PR https://github.com/asrjs/demos/pull/1 MERGED; merge SHA 59516bc689f4ab6051b5551e0478a9a41012df40. Kod 30e8e92; docs 4c5904e.
- Her iki PR normal merge ile birleştirildi; admin/bypass/force push kullanılmadı. Yerel main dalları origin/main'e fast-forward edildi. Bu kayıt ayrıca bir docs commit'iyle push edilir.
- Checkpoint'te library typecheck ve 1830 passed / 36 skipped tekrarlandı. Demos 14 Node +45 Vitest ve tam suite build tekrar geçti.
- GitHub library CI run 39168304794 Node22 ve Node24 lint aşamasında FAIL: 10 error, 13 warning. Kullanılmayan test değişken/import/argümanları ve prefer-const. CI unit test aşamasına ulaşmadı. Yerel test sonucu ayrı kanıttır. Kullanıcının mevcut hâliyle checkpoint talebiyle merge edildi; CI yeşil denmedi.
- Lint dosyaları: tests/moonshine-streaming-session.test.ts, moonshine-streaming-transcriber.test.ts, streaming-detector-neural-vad.test.ts, windowed-controller-demo.test.ts, worker-and-stateful-streaming.test.ts. Sonraki oturumda ilk bakım işi bunları düzeltmek ve lint + CI'ı tekrar koşmak. Root import bütçesi 78/<45 nedeniyle sonraki CI gate'i de ayrıca ele alınmalı; kriter gevşetilmemeli.
- CI hata logu library tools/data/results/browser-harness/checkpoint-ci-failure-20261019.log içinde korunur.
- GitHub demos push yanıtı default branch için 517 Dependabot uyarısı bildirdi (17 critical,277 high,188 moderate,35 low); kapsam/ulaşılabilirlik bu oturumda denetlenmedi. Legacy reference bağımlılıkları dâhil ayrıntılar ayrı inceleme konusu; bu sayılar uygulamada doğrulanmış exploit olarak yorumlanmaz.
- npm/Hugging Face publish yapılmadı. EOU120, exact12s, fiziksel mic/VAD ve sayısal parity açık durumları değişmedi. Çalışma burada bırakılıyor.
