# SenseVoiceSmall reference tooling

`capture_sensevoice_reference.py` runs the official local FunASR
`SenseVoiceSmall` implementation in offline mode. It records audio hashes,
requested language, ITN mode, native result metadata, batch order, model file
inventory, and runtime versions.

```powershell
$PYTHON = 'C:\path\to\funasr\python.exe'
& $PYTHON tools/model-debugging/reference/sensevoice/capture_sensevoice_reference.py `
  --model-dir N:\models\SenseVoiceSmall `
  --audio tools\data\fixtures\audio\jfk-short.wav `
  --output tools\data\results\sensevoice\sensevoice-small-reference.json `
  --language en `
  --use-itn `
  --batch-size 1
```

The script does not download a checkpoint. The local snapshot must contain
the model code required by FunASR, and missing files fail before a reference
JSON is written.
