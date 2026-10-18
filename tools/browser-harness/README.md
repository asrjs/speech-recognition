# Browser verification

Build the library before running this harness (`npm run build`). The browser imports
the built public package. The CLI and page share the same runner and verifier.

```sh
node tools/browser-harness/cli.mjs --local-dir models/onnx/moonshine-streaming-tiny-fp32 --backend wasm --mode offline --expect-jfk
```

The v2 receipt separates execution, normalized full-text comparison, provider
evidence, and fresh-handle/warm-handle reuse. `cold` means a newly loaded handle;
it does not mean that HTTP, driver, or shader caches were cleared. Reuse requires
the same handle instance ID, not a rounded zero load duration.

- Exit **0 / PASS**: every requested run passed its checks.
- Exit **1 / FAIL**: execution, comparison, reuse, or provider checks failed.
- Exit **2 / UNVERIFIED**: required evidence is unavailable (including no oracle).

`--mode both` checks offline and streaming, in both fresh and reused runs.
Unsupported streaming fails; offline prefix simulation is not native streaming.
`--no-warm-rerun` skips the reused run explicitly. A matrix (`--local-dirs a,b`)
applies the same mode, oracle and reuse policy to every directory.

`--expect-jfk` binds the complete expected text to the repository's `jfk-short.wav`
SHA-256. For other audio use `--fixture name.wav --oracle oracle.json`, where the
JSON contains `fixture`, `sha256` (hash of the WAV file), and `text`. Punctuation,
case and whitespace are normalized; missing, added, or changed words fail.
This comparison is not a token, tensor, or numerical parity certificate.

Positive instrumented GPU compute dispatches establish that some inference
compute used WebGPU. They do not establish that every graph ran on GPU. Missing
instrumentation or zero observed dispatches remain UNVERIFIED, and an explicitly
reported WASM fallback fails a WebGPU request. Adapter discovery alone proves
neither execution nor parity.

Every CLI verification run saves a receipt under `tools/data/results/browser-harness/`
by default, including failures. `--out` chooses a path; existing files are never
overwritten. Historical v1 receipts remain intact but their old PASS labels are
not acceptance evidence: that runner used a loose phrase matcher, skipped matrix
streaming/oracle checks, and did not fail the process on a failed matrix.

`--timeout-ms` bounds execution; invalid modes, backends, nonpositive chunk sizes,
unknown options, and missing values fail before launching the browser.
# Large local Qwen artifacts

Chromium's `fetch(...).blob()` path may fail for the large decoder sidecar.
Use real browser File objects through the dedicated public-API smoke:

```sh
python tools/browser-harness/qwen-file-input-smoke.py --model-dir N:/models/onnx/qwen3-asr-0.6b-official --case jfk2.en.wav=/tests/fixtures/jfk2.en.wav --out tools/data/results/browser-harness/qwen-files.json
```

Repeat `--case label=/tests/fixtures/file.wav` for additional audio. This tool
records execution and cleanup, with artifact and fixture hashes. It does not
declare transcript or numerical parity; `EXECUTED` is not a quality PASS.
Receipts are created exclusively and any execution/cleanup failure exits 1.
