# Documentation

## Build an application

- [Quick start](../README.md): install, load and transcribe.
- [Public API guide](PUBLIC_API_GUIDE.md): model discovery, loading, transcripts, assets and lifecycle.
- [Streaming and deployment](INTEGRATION_GUIDE.md): native streaming, windowed inference, microphone/VAD, offline assets and Node/Bun.
- [Framework bindings](FRAMEWORKS.md): React hooks, Svelte stores, Vue composables and shared controller ownership.
- [Browser model cache](BROWSER_CACHE.md): shared storage, artifact identity and the limits of sharing across sites.
- [Approved model profiles](APPROVED_MODEL_PROFILES.md): artifact manifests, automatic backend/precision selection and verification scope.
- [Examples index](../examples/README.md): file transcription, microphone/VAD, native streaming, windowed inference and local Node.js recipes.
- [Companion applications](https://github.com/asrjs/demos): Showcase, Playground, Realtime and Benchmark.

## Develop the library

- [Project charter](PROJECT_CHARTER.md): durable product vision and priorities.
- [Architecture](architecture.md): source boundaries and dependency rules.
- [Workspace map](WORKSPACE_CONTEXT.md): maintained applications and historical references.
- [Contributor entrypoint](AGENT_ENTRYPOINT.md): scope, safety and engineering navigation.
- [Model tools](../tools/README.md) and [Whisper export workflow](whisper-export-workflow.md): reuse established porting and graph optimization knowledge.

## Read engineering evidence

[Reports](reports) contain dated experiments, acceptance receipts and handoffs. They establish specific tested configurations; they do not redefine the product or automatically apply to later builds. Historical plans and archives preserve earlier decisions, not an always-active task queue.

The library is the product. Model ports, demos, benchmarks and validation tools serve its accuracy, performance and ease of integration. Prefer work that makes a developer's offline or streaming application simpler while preserving established implementations.
