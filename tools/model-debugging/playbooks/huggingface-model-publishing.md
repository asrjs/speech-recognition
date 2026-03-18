# Hugging Face Model Publishing

Use this playbook when publishing a converted model folder from
`N:\models\onnx\nemo\...` to a Hugging Face model repo with the `hf` CLI.

## Goals

- keep the model repo card aligned with our ONNX runtime packaging
- preserve large model artifacts through LFS
- publish the exact local folder we validated

## Folder Checklist

Before upload, make sure the model folder contains:

- converted runtime artifacts such as `encoder-model.onnx`, `decoder-model.onnx`,
  `decoder_joint-model.onnx`, quantized variants, tokenizer, and config
- a repo-facing `README.md` adapted to our ONNX package, not the upstream
  training/inference README
- a real `.gitattributes` file for LFS

Recommended `.gitattributes` baseline:

```gitattributes
*.onnx filter=lfs diff=lfs merge=lfs -text
*.onnx.data filter=lfs diff=lfs merge=lfs -text
*.data filter=lfs diff=lfs merge=lfs -text
```

Do not keep a plain `gitattributes` filename. Hugging Face expects
`.gitattributes`.

## README Pattern

Use an ONNX-package card like the one in:

- [N:\models\onnx\nemo\canary-180m-flash-smoke\README.md](N:\models\onnx\nemo\canary-180m-flash-smoke\README.md)

Recommended sections:

- frontmatter with license, languages, tags, `base_model`, and widgets
- short statement that this is a converted ONNX package, not the original repo
- included artifacts
- model summary
- frontend / preprocessing notes
- quantization notes
- `@asrjs/speech-recognition` usage
- upstream model and license
- references

## Upload Commands

Check auth first:

```powershell
hf auth whoami
```

Upload the whole validated folder:

```powershell
hf upload-large-folder ysdede/<repo-name> N:\models\onnx\nemo\<folder-name> --repo-type model
```

Example:

```powershell
hf upload-large-folder ysdede/parakeet-realtime-eou-120m-v1-onnx N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx --repo-type model
```

## After Upload

Verify:

- the README renders correctly
- images are visible
- large ONNX files are tracked through LFS
- filenames match what the preset/runtime expects

If needed, refresh the local model-porting docs with the repo id and any
special packaging notes.
