# WSL2 Project Handoff

This note records the local state recovered on 2026-05-28 before moving regular maintenance to WSL2.

## Current Git State

- Repository: `N:\github\asrjs\speech-recognition`
- Branch: `main`
- Upstream: `origin/main`
- The dirty changes were not random edits. They continue the Hugging Face download resilience work from the Parakeet/keet model-loading investigation.

## Why These Changes Exist

The earlier browser loading failures were caused by a few related artifact-loading problems:

- Parakeet TDT v2/v3 previously defaulted to Hugging Face branches such as `feat/fp16-canonical-v2` and `feat/fp16-canonical-v3`.
- The current repos now resolve from `main`, but existing browser IndexedDB caches may still be keyed by the old revisions.
- ONNX external-data sidecars such as `encoder-model.fp16.onnx.data` can be huge, absent, or backend-dependent. Blindly probing them creates noisy browser `404`s and sometimes forces unnecessary memory work.
- Browser IndexedDB can contain stale Blob records whose backing data is gone; reading those blobs may throw `NotFoundError`.

The uncommitted patch therefore does four things:

1. Adds `cacheKeyFallbacks` to asset requests so a new canonical cache key can read and migrate older revision keys.
2. Moves Parakeet TDT v2/v3 built-in defaults to Hugging Face `main` while keeping the old feature-branch revisions as cache-key fallbacks.
3. Uses Hugging Face repo listings to skip optional `.onnx.data` sidecar probes when the files are known absent.
4. Treats stale IndexedDB Blob `NotFoundError`s as cache misses and evicts the broken key.

## Files Changed

Runtime/cache:

- `src/types/io.ts`: adds `AssetRequest.cacheKeyFallbacks`.
- `src/io/handles.ts`: reads primary and fallback cache keys, migrates fallback hits, and keeps network fallback behavior.
- `src/io/cache.ts`: evicts stale IndexedDB Blob records when `arrayBuffer()` throws `NotFoundError`.

Model loading:

- `src/models/nemo-tdt/types.ts`: adds `cacheKeyFallbackRevisions` for Hugging Face sources.
- `src/models/nemo-tdt/executor.ts`: passes fallback cache keys and avoids known-absent optional sidecars.
- `src/models/nemo-rnnt/executor.ts`: same sidecar and cache-key behavior for RNNT.

Presets/discovery:

- `src/presets/parakeet/catalog.ts`: changes TDT v2/v3 defaults to `main` and records old revisions as fallbacks.
- `src/presets/parakeet/manifest.ts`: applies catalog revision/fallback defaults to resolved manifests.
- `src/presets/descriptors.ts`: reports Parakeet TDT default revision as `main`.

Docs/tests:

- `docs/HUGGINGFACE_DOWNLOAD_RESILIENCE.md`: documents cache-key migration, stale Blob eviction, and quiet sidecar skipping.
- `tests/io-handles.test.ts`: covers fallback cache-key reads and migration.
- `tests/indexeddb-cache.test.ts`: covers stale Blob eviction.
- `tests/nemo-tdt-executor.test.ts` and `tests/nemo-rnnt-executor.test.ts`: cover skipping absent sidecars based on repo listings.
- `tests/preset-descriptors.test.ts`: covers the new Parakeet default revision.

## WSL2 Workflow

The current WSL default distro is `Ubuntu` on WSL2. From PowerShell, the Windows checkout is visible inside WSL at:

```powershell
wsl pwd
```

When run from this repo, that resolves to:

```text
/mnt/n/github/asrjs/speech-recognition
```

For best performance and fewer Windows line-ending/file-watcher surprises, prefer a native Linux clone under the WSL filesystem for active development, for example:

```bash
mkdir -p ~/github/asrjs
cd ~/github/asrjs
git clone git@github.com:asrjs/speech-recognition.git
cd speech-recognition
npm install
npm run typecheck
npm test
```

If you keep using the Windows checkout from WSL at `/mnt/n/...`, treat it as convenient for inspection but less ideal for heavy Node installs, test watchers, and dev servers.

## Related Local Projects

Use the workspace map in `docs/WORKSPACE_CONTEXT.md` for routing:

- `speech-recognition`: source of truth for runtime/model/preset loading behavior.
- `streaming-demo`: main microphone streaming, waveform, VAD, and Hugging Face/local model loading surface.
- `browser-demo`: upload/sample-file and model-loading UX checks.
- `benchmark-demo`: performance and throughput checks.
- `vad-demo` and `firered-vad-web`: isolated VAD and FireRed frame-size/timeline checks.
- `parakeet.js`, `keet`, `onnx-asr`, and `transformers-v4-parakeet-demo`: reference repos for artifact layout and historical loader behavior.

## Verification To Run Before Pushing

Use the normal library gate:

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

If a change specifically touches browser model loading, also smoke it in `streaming-demo` or `browser-demo` after installing them against the local library checkout.
