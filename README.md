# asr.js

`asr.js` is a single-package, speech-first TypeScript runtime library.

It keeps strong internal separation for processors, runtime orchestration, inference, model integrations, and presets, but publishes one library package with one root public API.

## Structure

```text
src/
  inference/
  models/
  presets/
  processors/
  runtime/
  tokenizers/
  types/
docs/
examples/
tests/
```

`src/inference/graph.ts` is the shared descriptor layer for encoder families, decoder-head families, and decoding strategies. Real execution code stays in `src/models/*` until it is proven reusable across multiple families.

## Usage

```ts
import { createSpeechRuntime, createWasmBackend } from 'asr.js';
```

## Commands

```bash
npm install
npm run build
npm run typecheck
npm test
```

## Status

- canonical transcript contracts: implemented
- runtime and backend registration: implemented
- processor scaffolding: implemented
- inference scaffolding: implemented
- tokenizer abstractions and stubs: implemented
- architecture-based model scaffolds: implemented
- branded preset layers: implemented
- current implementation families: NeMo TDT, HF CTC, Whisper seq2seq, FireRed speech-LLM
- production inference executors: not implemented yet
