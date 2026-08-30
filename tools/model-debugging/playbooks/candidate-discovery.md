# Multilingual ASR Candidate Discovery

Use this playbook before starting a new model port. Popularity is a triage
signal; it is never evidence of accuracy or browser compatibility.

## Capture a dated snapshot

Run the discovery script from the repository root:

```text
node tools/scripts/survey-hf-asr-candidates.mjs --limit 50 --search asr --output tools/data/results/model-candidates/hf-asr-search-asr-YYYY-MM-DD.json
```

Use `--include owner/model,...` for high-value models that do not appear in
the first ranked page. Keep the raw JSON under `tools/data/results`; do not
overwrite an older snapshot. The script records the exact API query, counters,
tags/language signals, repository file names, and ONNX/external-data/WASM/GGUF
indicators.

## Review each candidate

For every model that looks multilingual or strategically useful:

1. Open the official model card and record revision, license, languages,
   processor, and recommended inference engine.
2. Inspect the exact repository files. A `.onnx` filename is only an artifact
   signal; check graph inputs/outputs, external-data paths, shapes, dtypes, and
   tokenizer/config files with the ONNX audit tooling.
3. Search for existing ONNX Community, Transformers.js, sherpa-onnx, wasm,
   or HF Space implementations. If a working browser path exists, choose
   **adapt/benchmark**, not a duplicate exporter.
4. Check this library and sibling repositories (`onnx-asr`, `parakeet.js`,
   `transformers-v4-parakeet-demo`, and the browser harness) for existing
   family support or reusable graph/export code.
5. Select at most one bounded objective. Follow the full chain:
   original weights → official inference → reference captures → optimized ONNX
   → native ORT → WASM/WebGPU → library executor → sibling browser integration.

Record one of `PROMOTE`, `ADAPT`, `DEFER`, or an explicit failure category
(`EXPORT_BLOCKED`, `ORT_WEB_UNSUPPORTED_OP`, `MODEL_TOO_LARGE`,
`ARCHITECTURE_NOT_BROWSER_SUITABLE`, etc.). Preserve the reason and links in a
dated report even when the answer is “do not port”.

## Current 2026-08-30 example

The first snapshot selected Nemotron 3.5 streaming 0.6B for adaptation because
it is multilingual and cache-aware RNNT with independent ONNX/WebGPU exports;
Fun-ASR MLT Nano remains the next genuinely new export spike. Granite, Voxtral,
and VibeVoice already have browser-oriented ONNX paths, while GLM-ASR is large
and has no browser artifact. Details and counters are in
`docs/reports/hf-multilingual-asr-candidate-survey-2026-08-30.md`.

## Reusable guardrails

- Keep discovery, quality labels, reference transcripts, and throughput JSON
  separate.
- Never infer WER, RTFx, or parity from downloads, likes, a demo, or tags.
- Recheck popularity and artifact availability before committing to a port;
  HF repositories and browser implementations evolve quickly.
