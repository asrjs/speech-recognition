# WebGPU Agent Test Pattern

Cross-environment browser GPU test: Hermes agent (WSL) prepares a self-contained test page, another Hermes agent (Windows host) runs it in a real browser and saves results.

## Folder layout

```
webgpu-agent-test/
├── index.html          — Button-driven test page (v4: no auto-start, 2 buttons)
├── jfk2.en.wav         — Test audio fixture (~12s JFK speech)
├── INSTRUCTIONS.md     — Windows agent instructions (server, browser, save)
├── AGENT_CHAT.md       — Shared log between cross-environment agents
├── models/
│   ├── fp16/           — Local fp16 model files (~2.3GB, external data)
│   ├── mixed/          — Mixed precision: q8 encoder (616MB) + fp16 decoder (~1.1GB, ~1.7GB total)
│   └── fp32/           — (legacy, not used in v4)
└── _results/           — Agent saves results here
```

## HTML page requirements

### Version history

| Version | Key features | Bug fixes |
|---------|-------------|-----------|
| v1 | Auto-start, 1 button, HF CDN model download | — |
| v2 | Top-5 logit dump, NaN detection, 3 buttons (WASM added) | Decode loop never runs (stopped variable) |
| v3 | Local models, auto-start removed, VRAM warning | Mel spectrogram magnitude bug, WASM fp16 removed |
| v4 | fp32 removed, Mixed added, 2 buttons only | Loop condition cleanup |
| v5 | Full dropdown suite (WebGPU/WASM, fp16/mixed/q8) | — |
| v6 | fp32 restored, ORT 1.21.0→1.26.0 | MODEL_DIRS.fp32 mapping, hasExt for fp32 |

### v6 page architecture (current)

- **Backend dropdown**: WebGPU / WASM
- **Model dropdown**: fp32 (4.5GB) / fp16 (2.3GB) / Mixed q8 enc+fp16 dec (1.7GB) / q8 (1.4GB)
- **Encoder input dropdown**: Auto / float32 / float16
- **Run + Cancel buttons**
- **VRAM/memory warning**: fp32 (4.5GB) may OOM browser; testler arası F5

### Model dir mapping (critical for adding new variants)

When adding a new model variant to the dropdown, update THREE places in the JS:

```javascript
// 1. MODEL_DIRS — map variant key → directory path
const MODEL_DIRS = {
  fp32:  'models/fp32/',   // ← ADD HERE
  fp16:  'models/fp16/',
  mixed: 'models/mixed/',
  q8:    'models/q8/',
};

// 2. MODEL_SIZES — display size
const MODEL_SIZES = { fp32: '4.5 GB', fp16: '2.3 GB', mixed: '1.7 GB', q8: '1.4 GB' };

// 3. hasExt logic — which variants have external data files
const hasExt = variant === 'fp32' || variant === 'fp16' || variant === 'mixed';  // ← ADD
const encExt = hasExt && variant !== 'mixed' && variant !== 'q8' ? 'encoder_model.onnx.data' : null;
const initExt = hasExt ? 'decoder_init.onnx.data' : null;
const stepExt = hasExt ? 'decoder_step.onnx.data' : null;
```

Failure to update all three causes `Model dir: undefined` → `Failed to load model because protobuf parsing failed` (the page fetches from `models/undefined/` and gets HTML, which isn't valid protobuf).

### Known issues (as of v6)

| Test | Result | Root cause |
|------|--------|------------|
| WebGPU fp16 | ❌ NaN logits | decoder_init ops (Erf, Where) fail on WebGPU EP with fp16 |
| WebGPU mixed | ❌ ConvInteger | q8 ops (ConvInteger) unsupported on WebGPU EP |
| WebGPU q8 | ❌ ConvInteger | Same as mixed |
| WebGPU fp32 | ❓ Untested | Need non-headless browser (2.4GB encoder data exceeds headless fetch limit) |
| WASM any | ❌ OOM/crash | Headless browser memory limits; fp16 tensors also unsupported |

### HTTP server for large files

Python's `http.server` (SimpleHTTP/0.6) is HTTP/1.0 and fails on files >2GB. Use `npx serve` instead:

```powershell
npx serve --cors
# HTTP/1.1, Accept-Ranges: bytes, handles large files
```

fp16's largest file = 1.2GB (works with Python). fp32's encoder data = 2.4GB (requires npx serve).

### ONNX Runtime Web version

Current CDN: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.all.min.js`

1.26.0 may fix WebGPU EP fp16 issues present in 1.21.0. To upgrade:
1. Update CDN link in `<script src="...">`
2. Hard-refresh browser (Ctrl+F5) to bypass cache

Available versions: `npm view onnxruntime-web versions --json`

### BUTTON-driven (NOT auto-start)

Do NOT use `window.addEventListener('load', ...)` auto-start. Add explicit buttons so the controlling agent can sequence tests. **Auto-start broke agent control** in v1-v2 — always use buttons.

## Key bug history (v1→v4)

| # | Issue | Found in | Fixed in |
|---|-------|----------|----------|
| 1 | **Decode loop never runs** (`stopped` variable always false in loop condition) | v1 | v2 |
| 2 | **Mel spectrogram magnitude outer-loop bug** — magnitude written to ALL frames using LAST frame's FFT data, producing identical columns | v2 | v3 |
| 3 | **Auto-start prevents agent control** — page starts inference before agent can interact | v1 | v3 |
| 4 | **Model download every time** — no IndexedDB cache, 2.3-4.5GB each run | v1 | v3 (local models) |
| 5 | **WASM fp16 unsupported** — ORT Web WASM silently crashes on float16 | v1 | v3 (removed) |
| 6 | **VRAM accumulation** — fp16 + fp32 models simultaneously exceed 8GB VRAM | v2 | v3 (warning) |
| 7 | **WebGPU fp16 NaN logits** — decoder_step ops (Erf, Where, LessOrEqual, Tile, Range) fail on WebGPU EP with fp16 | v1 | TBD |

## Seeding the mixed model directory

Mixed = q8 encoder + fp16 decoder:

```bash
mkdir -p /path/to/webgpu-agent-test/models/mixed

# q8 encoder (inline weights, no .data file)
cp /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8/encoder_model.onnx models/mixed/

# fp16 decoder (external data files)
cp /path/to/fp16/decoder_init.onnx /path/to/fp16/decoder_init.onnx.data models/mixed/
cp /path/to/fp16/decoder_step.onnx /path/to/fp16/decoder_step.onnx.data models/mixed/
cp /path/to/fp16/decoder_align.onnx /path/to/fp16/decoder_align.onnx.data models/mixed/
cp /path/to/fp16/*.json models/mixed/
```

The q8 encoder model has NO external data file (weights inline, 616MB). The fp16 decoder models have .data files. Mixed session loading works:
- `encoder_session.create(encoder.onnx)` — no external data needed
- `decoder_init_session.create(decoder_init.onnx, { externalData: [init.data] })`
- `decoder_step_session.create(decoder_step.onnx, { externalData: [step.data] })`

## Folder location

Create under `/mnt/n/github/asrjs/` (Windows N: drive) so the Windows host agent can access it directly at `N:\\github\\asrjs\\webgpu-agent-test\\`.

## INSTRUCTIONS.md template

```markdown
# WebGPU Agent Test — Instructions

## How to run

### 1. Start HTTP server
```powershell
cd N:\\github\\asrjs\\webgpu-agent-test
npx http-server -p 8080 --cors
# or: python -m http.server 8080
```

### 2. Open in browser → click button → save result
1. Open http://localhost:8080/
2. Click "WebGPU fp16" button
3. Wait for Agent Output section
4. Copy result between markers, save to _results/result-fp16-{timestamp}.txt
5. Refresh page (F5) to clear VRAM
6. Click "WebGPU Mixed (q8 enc + fp16 dec)" button
7. Save to _results/result-mixed-{timestamp}.txt
```

## When to use

- Testing WebGPU inference in a real browser (Chrome/Edge with GPU acceleration)
- Cross-environment testing: WSL agent prepares, Windows host agent executes
- Validating model weights work in browser context
- Benchmarking browser inference timing
- Debugging WebGPU EP op support issues (compare variants)

## Diagnostic features

### Top-5 logit dump
```javascript
function topK(arr, k) {
  const indexed = Array.from({ length: Math.min(arr.length, 10000) }, (_, i) => ({ idx: i, val: arr[i] ?? -Infinity }));
  indexed.sort((a, b) => b.val - a.val);
  return indexed.slice(0, k);
}
```

Reasonable top-5 should include text token IDs (e.g. 400="And", 370="so") with positive values. All-special-tokens (>50358) or near-zero values = corrupted output.

### NaN/zero logit detection
```javascript
const hasNaN = Array.from(stepLogits).some(v => isNaN(v));
const allZero = Array.from(stepLogits).every(v => v === 0);
```

## Shared chat log (AGENT_CHAT.md)

Maintain a flat chronological log at `webgpu-agent-test/AGENT_CHAT.md`:

```
## Entry NNN — AgentName (machine, OS, GPU)
**Date:** ...
**Backend:** ...
**Result:** PASS/FAIL

### Findings
...analysis...

### Next steps
...for the other agent...
```

Conventions:
- Newest entries at the bottom (flat chronological)
- Each entry: agent name + machine details + raw output + verdict + analysis + next steps
- Comment block at bottom with format instructions for new agents

## Known WebGPU EP issues

1. **decoder_step ops fail silently on WebGPU EP with fp16**: decoder_step has 611 nodes (opset 17). Suspect ops: Erf(4), Where(2), Tile(1), LessOrEqual(1), Equal(1), Range(2). These are NOT in the core WebGPU supported opset and may trigger fallback paths that produce NaN logits under fp16.
2. **fp32 decoder_step works**: If fp16 fails, fp32 is a reliable fallback (but requires separate model files).
3. **Encoder always works** on WebGPU (no suspect ops, simple matmul/transpose pipeline).
4. **Use `onnx` Python library to inspect graph ops**: `onnx.load(step.onnx)` → iterate `m.graph.node` → `op_counts[n.op_type]` to check for suspect ops.
