# HuggingFace Download Resilience

This note captures the browser-side hardening added for remote Hugging Face artifact downloads.

## Context

`browser-demo` and `streaming-demo` rely on `@asrjs/speech-recognition` for model asset fetches.
When remote model loading fails, the fix should live in the shared IO/runtime layer rather than app-specific code.

## Failure Patterns We Hardened

1. Stale or broken IndexedDB cache entries causing load failure instead of network fallback.
2. Cached Blob records whose browser backing store disappeared and throw `NotFoundError` when read.
3. Branch/revision-specific Hugging Face URLs returning `404` while `main` still exists.
4. Revision cache-key changes forcing duplicate downloads after model repos move artifacts to `main`.
5. External `.data` ONNX sidecars creating noisy browser `404` errors when absent.
6. Large external-data files (multi-GB) being materialized into browser memory while creating URL locators.

## Implemented Resilience

### Cache read/write fail-open

Files:

- `src/io/handles.ts`
- `src/io/cache.ts`

Behavior:

- Cache read errors no longer fail model load.
- Broken cache keys are best-effort evicted when read fails.
- Cached IndexedDB blobs that throw `NotFoundError` during `arrayBuffer()` are treated as stale entries, evicted, and reported as cache misses.
- Cache write errors are logged but do not fail runtime loading.

### Cache-key migration

Files:

- `src/io/handles.ts`
- `src/types/io.ts`
- `src/presets/parakeet/catalog.ts`
- `src/presets/parakeet/manifest.ts`

Behavior:

- Asset requests can provide `cacheKeyFallbacks` for older keys that may contain the same artifact.
- A fallback-key hit is migrated back to the primary `cacheKey` so future reads use the canonical key.
- Built-in Parakeet TDT v2/v3 descriptors now default to Hugging Face `main`, with their previous `feat/fp16-canonical-v2` and `feat/fp16-canonical-v3` cache keys retained as fallback revisions.

### Revision fallback for Hugging Face assets

File:

- `src/io/handles.ts`

Behavior:

- For `provider: 'huggingface'`, if fetch for non-`main` revision returns `404`, the loader retries with `main`.
- This reduces hard failures from stale revision pins.

### Optional ONNX external-data resolution without noisy absence probes

Files:

- `src/models/nemo-tdt/executor.ts`
- `src/models/nemo-rnnt/executor.ts`

Behavior:

- Encoder/decoder `.data` files are checked against the Hugging Face repo listing when that listing is available.
- Known-absent sidecars are skipped without issuing a browser `GET`, avoiding expected-but-noisy console `404` errors.
- If the listing API is unavailable or empty, the loader falls back to the previous direct optional download probe.
- Direct probe `404` responses are still treated as "file absent"; other errors still surface.

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
