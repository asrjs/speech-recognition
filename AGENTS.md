# AGENTS.md

## Environment

Windows 10, powershell

## Project identity

`asr.js` is a single-package TypeScript speech runtime library.

The repository may draw on lessons learned from:
- `ysdede/parakeet.js`
- `ysdede/transformers.js`

It must not copy the architecture, public API philosophy, or folder structure of `transformers.js`.

## Architectural stance

This repo is intentionally:
- speech-first
- single-package
- ESM-first
- runtime-oriented rather than model-zoo-oriented

The library publishes one package and one public API surface from `src/index.ts`.
Internal separation is preserved through `src/` folders, not through workspace packages.
The main source-of-truth design doc is `docs/architecture.md`. If code and older research notes diverge, align the notes to the current `src/` layout rather than reintroducing removed folders.

## Internal boundaries

Use folders to preserve clean boundaries:
- `src/types`: Shared contracts, interfaces, and metric definitions.
- `src/runtime`: Session orchestration, backend bridging, caching, and chunking.
- `src/processors`: Feature extractors (e.g., mel spectrograms, waveform convolutions).
- `src/inference`: Shared architecture descriptors, backend probes, and streaming helpers.
- `src/tokenizers`: Text decoding (e.g., BPE, tiktoken).
- `src/models`: Model-family implementations and wiring.
- `src/presets`: Branded configurations and weight locators (e.g., Parakeet, MedASR).

Implementation families stay technically classified under `src/models`.
Branded families stay in `src/presets`.
Shared encoder, decoder-head, and decoding-strategy descriptors live in `src/inference/graph.ts`.
Current implementation families include NeMo TDT, HF CTC, Whisper seq2seq, and FireRed speech-LLM work.

## Rules

- keep decoder internals model-specific unless proven shared
- do not over-generalize early
- do not move model-specific logic into `src/runtime`
- preserve a stable canonical transcript contract
- backend differences must not change output semantics
- prefer direct internal modules over artificial package boundaries
- avoid generic util blobs
- keep the public API coherent and root-exported
- use the term "Feature Extractors" or "Processors" instead of "Frontend" for audio preparation.
- avoid reviving removed folder layers such as `src/encoders`, `src/decoders`, or `src/decoding` unless there is substantial shared code to justify them.

## Output expectations

When making structural changes:
- explain folder responsibilities clearly
- document dependency direction
- note important assumptions
