# HuggingFace Download Resilience

This note captures the browser-side hardening added for remote Hugging Face artifact downloads.

## Context

`browser-demo` and `streaming-demo` rely on `@asrjs/speech-recognition` for model asset fetches.
When remote model loading fails, the fix should live in the shared IO/runtime layer rather than app-specific code.

## Failure Patterns We Hardened

1. Stale or broken IndexedDB cache entries causing load failure instead of network fallback.
2. Branch/revision-specific Hugging Face URLs returning `404` while `main` still exists.
3. External `.data` ONNX sidecars being resolved only via listing checks.
4. Large external-data files (multi-GB) being materialized into browser memory while creating URL locators.

## Implemented Resilience

### Cache read/write fail-open

Files:

- `src/io/handles.ts`
- `src/io/cache.ts`

Behavior:

- Cache read errors no longer fail model load.
- Broken cache keys are best-effort evicted when read fails.
- Cache write errors are logged but do not fail runtime loading.

### Revision fallback for Hugging Face assets

File:

- `src/io/handles.ts`

Behavior:

- For `provider: 'huggingface'`, if fetch for non-`main` revision returns `404`, the loader retries with `main`.
- This reduces hard failures from stale revision pins.

### Optional ONNX external-data resolution without hard listing dependency

Files:

- `src/models/nemo-tdt/executor.ts`
- `src/models/nemo-rnnt/executor.ts`

Behavior:

- Encoder/decoder `.data` files are tried directly as optional downloads.
- `404` is treated as "file absent" and ignored.
- Other errors still surface.

### Controlled URL locator materialization for HTTP assets

File:

- `src/io/handles.ts`

Behavior:

- `UrlAssetHandle.getLocator('url')` returns the original HTTP URL by default.
- Model-family loaders can opt into `preferBlobUrl` when they need the shared runtime to own the download and emit progress before handing a URL to ONNX Runtime.
- Blob URL materialization streams chunks into a `Blob` and avoids the previous contiguous `Uint8Array` allocation that caused `Array buffer allocation failed` crashes for large `.onnx.data` files.
- NeMo Hugging Face loaders use this mode so `streaming-demo` sees model download progress instead of appearing stuck inside `session-create:start`.

## Validation

Relevant tests:

- `tests/io-handles.test.ts`
- `tests/indexeddb-cache.test.ts`
- `tests/firered-indexeddb-cache.test.ts`

## Practical Impact

These changes keep the platform model-agnostic and robust: demos no longer need local one-off patches for common HF download/cache edge cases.
