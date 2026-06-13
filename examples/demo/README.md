# asr.js Whisper 4-Graph Browser Demo

Runs the Whisper large-v3-turbo 4-graph splitgraph path in a browser through the
library pipeline. The demo uses the core package for model loading, mel
processing, decode policy, KV cache handling, and transcript mapping.

```bash
cd examples/demo
npm run dev
```

Open the printed local URL in a real Chrome/Edge window with WebGPU enabled for
WebGPU profiles. WASM profiles require cross-origin isolation for threaded WASM;
the Vite config sets the required headers.

Default model repo:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
```

The q8 WASM preset is the safest browser fallback. The fp16io encoder + fp32
decoder preset is the current WebGPU target.
