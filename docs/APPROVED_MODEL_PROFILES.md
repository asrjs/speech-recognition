# Approved model profiles

The shared loader owns execution planning. Applications select a model and may
supply advanced overrides; they do not select graph filenames independently.

The first profile is Parakeet TDT 0.6B v3: FP16 encoder on WebGPU, INT8
combined decoder/joint on WASM, and the JavaScript nemo128 processor. Its
manifest binds graph, config and vocabulary SHA-256 identities to an immutable
Hugging Face artifact revision. The manifest is also published as
[asrjs-manifest.json](https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/blob/main/asrjs-manifest.json).

`listApprovedSpeechModelManifests()` exposes the registry.
`resolveSpeechModelExecutionPlan(modelId, input)` intersects requested component
backends and precision with supplied runtime capabilities. The high-level loader
probes the registered backends when choosing an automatic manifest-backed load.
`buildSpeechModelLoadOptions()` can consume the same capability input.

An `approved` plan requires the exact registered artifact identity, backend and
precision profile. Merely selecting a model name, finding similarly named local
files, or listing precision variants in a repo does not establish approval.
Local loads without verified identities remain `unverified`. The approved
recipe is not a claim that every downloaded byte was independently hashed by
the runtime: callers supplying artifact identity are responsible for its truth.
Unsupported explicit requests fail rather than changing the user's selection.
Supported profiles without a matching receipt remain `unverified`.

Approval is scoped to the linked evidence. The initial Parakeet receipt covers
the English JFK fixture and controlled prerecorded Chrome microphone flows.
It does not certify physical microphone hardware, every language/audio length,
or numerical parity at every inference stage. New real-model integration
receipts are recorded separately from manifest selection unit tests.

## When a device cannot run the approved profile

The planner falls back instead of failing and reports why. Without WebGPU
`shader-f16`, Parakeet v3 uses the FP32 encoder, a larger download that needs
more GPU memory. Without WebGPU, the encoder runs on WASM. Each departure is
listed in the plan's `warnings` and emitted as a `model-load:fallback` progress
event, so `createSpeechController()` exposes it in `loadWarnings`. Explicit
backend or precision choices are not reported as fallbacks, and the resulting
plan is `unverified`.

## Export for a model repository

Build the SDK, then export from its registry instead of maintaining another
backend matrix in the HF repository:

```sh
npm run build
node tools/scripts/export-approved-model-manifest.mjs parakeet-tdt-0.6b-v3 asrjs-manifest.json
hf upload ysdede/parakeet-tdt-0.6b-v3-onnx asrjs-manifest.json asrjs-manifest.json
```

The export command refuses to overwrite an existing output. Review hashes,
artifact revision and evidence scope before publishing an updated manifest.
Publishing an experimental export does not automatically register it as an
approved execution profile.
