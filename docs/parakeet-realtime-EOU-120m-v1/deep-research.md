Absolutely — here is the English version of your report, cleaned up for readability while preserving the technical meaning.

---

# Architectural Report for Porting Parakeet Realtime EOU 120M v1 to ONNX and WebGPU WASM

## Executive summary

This report examines the architecture, tokenizer, input/output expectations, and the technical details that matter most when porting `nvidia/parakeet_realtime_eou_120m-v1` to ONNX and then to a WebGPU/WASM runtime for use in `ysdede/parakeet.js`. The model is a low-latency streaming English ASR model based on a **FastConformer + RNN-T** architecture. It emits an **`<EOU>`** token at the end of each utterance to signal end-of-utterance, and it does **not** produce punctuation or capitalization. ([Hugging Face][1])

The Hugging Face repository does **not** expose the usual detailed files such as `config.json` or `tokenizer.json` directly. Instead, the main checkpoint is distributed as a single **`.nemo`** archive. That means some of the most important details for exact reproduction — such as STFT window/hop sizes, mel feature configuration, normalization behavior, and detailed encoder hyperparameters — are **not visible from the public repo UI** and must be extracted from the `.nemo` package itself. ([Hugging Face][1])

This report relies on the official model card for the hard facts that are explicitly documented, including 16 kHz input, 17 encoder layers, attention context `[70, 1]`, and streaming usage constraints. ([Hugging Face][1]) It also uses two practical secondary sources for porting details: a tokenizer from an ONNX conversion package, and a CoreML conversion package whose metadata exposes dimensions, special IDs, and cache sizes. ([Hugging Face][2])

## Repository inspection and file roles

The official Hugging Face repository is minimal. The key visible assets are the model card and the `.nemo` checkpoint. ([Hugging Face][1])

- `README.md` / model card:
  - describes the model as **streaming ASR with end-of-utterance detection**
  - states the latency target is roughly **80–160 ms**
  - states the model emits `"<EOU>"` at utterance boundaries
  - states it supports **English only**
  - states it does **not** output punctuation or capitalization
  - documents the architecture as **FastConformer-RNNT**
  - documents **17 encoder layers** and attention context **[70, 1]**
  - documents **16 kHz mono waveform** input with a minimum duration requirement of **160 ms** ([Hugging Face][1])

- `.nemo` checkpoint:
  - expected to contain the real model weights, configuration, tokenizer assets, and other metadata
  - not directly inspectable from the repo page without unpacking it locally ([Hugging Face][1])

In standard NeMo practice, a `.nemo` file is an archive containing model config, tokenizer assets, weights, and metadata. The NeMo export docs also confirm that model export behavior and dynamic axes are defined through the model’s export interface, but they do not reveal the hidden config values for this specific model. ([NVIDIA Docs][3])

## Preprocessing pipeline and tokenizer details

### Preprocessing pipeline

The official facts visible from the model card are:

- **Sample rate:** 16 kHz mono ([Hugging Face][1])
- **Minimum audio duration:** 160 ms ([Hugging Face][1])
- **Input format:** 1D audio waveform ([Hugging Face][1])
- **Mel feature dimension:** 128, visible in the CoreML metadata ([Hugging Face][2])

What is **not** explicitly visible from the public repo:

- STFT parameters: `n_fft`, window length, hop length, window type, padding, `center`
- mel settings: `fmin`, `fmax`, mel scale type, power/log convention
- normalization details: mean/variance normalization, clamp behavior, dither, pre-emphasis
- whether any silence trimming or internal VAD is involved
- exact chunk-boundary handling for streaming feature extraction

This is the single biggest technical risk in a browser port. If your JS/WASM preprocessing is even slightly different from the NeMo preprocessing, the model may still run but the WER can degrade sharply, especially because RNN-T models are sensitive to encoder time alignment. ([Hugging Face][1])

The practical conclusion is:

**You should extract `m.cfg.preprocessor` from the `.nemo` model before implementing the browser-side DSP.**

A recommended extraction snippet is:

```python
import nemo.collections.asr as nemo_asr

model_id = "nvidia/parakeet_realtime_eou_120m-v1"
m = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
m.eval()

print("=== PREPROCESSOR CFG ===")
print(m.cfg.preprocessor)

print("=== TOKENIZER ===")
print(type(m.tokenizer))
print("vocab_size:", getattr(m.tokenizer, "vocab_size", None))

print("=== DECODER/JOIN INDEXES ===")
print("decoder.blank_idx:", getattr(m.decoder, "blank_idx", None))
print("decoder.pred_hidden:", getattr(m.decoder, "pred_hidden", None))
print("decoder.pred_rnn_layers:", getattr(m.decoder, "pred_rnn_layers", None))
print("joint.num_extra_outputs:", getattr(m.joint, "num_extra_outputs", None))
```

### Tokenizer details

For RNN-T porting, tokenization matters mainly for **token-id to text** decoding and correct handling of the **`<EOU>`** signal.

The tokenizer visible in the ONNX conversion package shows:

- **model type:** `BPE`
- **normalizer:** `NFKC`
- **decoder:** `Metaspace`
- **replacement:** `" "`
- **prepend_scheme:** `"always"`
- **split:** `true`
- **byte_fallback:** `false`
- **`<unk>` = 0** ([Hugging Face][4])

Special token IDs visible in that tokenizer are:

- **`<EOU>` = 1024**
- **`<EOB>` = 1025** ([Hugging Face][4])

The CoreML metadata reports:

- **`vocab_size` = 1026**
- **`blank_id` = 1026**
- **decoder hidden size = 640**
- **decoder layers = 1**
- **cache channel size = 70**
- **cache time size = 8** ([Hugging Face][2])

That means the RNN-T blank is **not** part of the tokenizer vocabulary. It is an extra output class in the model logits.

### Critical token/id summary

| Item                           | Value |
| ------------------------------ | ----: |
| `<unk>`                        |     0 |
| `<EOU>`                        |  1024 |
| `<EOB>`                        |  1025 |
| `vocab_size`                   |  1026 |
| `blank_id`                     |  1026 |
| output classes including blank |  1027 |

Supported by tokenizer and metadata sources. ([Hugging Face][4])

One oddity is that this tokenizer JSON shows `"merges": []` despite declaring `"type": "BPE"`. For your port, that is not necessarily a blocker, because the runtime need is mainly **detokenization**, not training-time tokenization. In practice, the vocab can be treated as an `id -> piece` lookup table. ([Hugging Face][4])

## Model architecture and runtime behavior

### High-level architecture

From the official model card:

- architecture type: **FastConformer-RNNT**
- encoder: **cache-aware streaming FastConformer**
- encoder layers: **17**
- attention context: **[70, 1]**
- decoder: **RNNT decoder**
- model size: **120M parameters** ([Hugging Face][1])

### Encoder

What is known:

- **17 layers** ([Hugging Face][1])
- **hidden dimension = 512** ([Hugging Face][2])
- **mel input dimension = 128** ([Hugging Face][2])
- **cache channel size = 70**
- **cache time size = 8** ([Hugging Face][2])

Practical cache tensor shapes inferred from the conversion materials are:

- `cacheLastChannel`: `[17, 1, 70, 512]`
- `cacheLastTime`: `[17, 1, 512, 8]`

What remains unspecified without unpacking `.nemo` or inspecting an actual ONNX graph:

- number of attention heads
- FFN expansion size
- convolution kernel sizes
- activation function
- normalization placement/type
- subsampling structure

### Decoder and joint

The CoreML conversion metadata shows:

- **decoder_hidden = 640**
- **decoder_layers = 1** ([Hugging Face][2])

So the practical RNN-T structure is:

- encoder: FastConformer
- prediction network: **1-layer LSTM, hidden size 640**
- joint: produces logits over **1027 classes** including blank

This is highly relevant for ONNX/WebGPU because decoder state must be carried across streaming decode steps.

## Model I/O and deployment structure

### Official NeMo usage

The model card shows the standard NeMo loading route:

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet_realtime_eou_120m-v1"
)
```

It also states:

- runtime: **NeMo 2.5.3+**
- input: 16 kHz mono waveform
- output: string, optionally ending with `<EOU>` ([Hugging Face][1])

### Expected ONNX packaging

A practical ONNX deployment for this model usually splits the RNN-T into:

- `encoder.onnx`
- `decoder_joint.onnx`
- `tokenizer.json` or equivalent vocab map

That packaging is consistent with the public third-party ONNX packaging and with how RNN-T inference is commonly deployed. ([Hugging Face][4])

The big caveat is that the **preprocessor is often not part of the ONNX graph**, so your `parakeet.js` runtime will almost certainly need its own waveform → log-mel pipeline.

## ONNX export strategy and likely issues

The NeMo export docs show that export supports settings like `dynamic_axes`, and the model export interface is the proper way to generate ONNX from NeMo modules. ([NVIDIA Docs][3])

For this model, the two main export choices are:

### 1. Stateless encoder export

Pros:

- simpler graph and I/O
- easier initial validation

Cons:

- worse for real streaming
- may require reprocessing history every chunk

### 2. Stateful cache-aware encoder export

Pros:

- true streaming
- fixed-cost chunk processing

Cons:

- more complex I/O
- more state buffers to manage
- more memory pressure in the browser

For a WebGPU target, stateful export is the right long-term direction, but stateless export is still useful as a correctness baseline.

### Preprocessor export risk

If you try to embed STFT/mel extraction directly into ONNX, exporter/runtime issues can appear around:

- STFT semantics
- complex-number representation
- padding and `center` behavior
- opset compatibility

For browser deployment, a **deterministic JS/WASM preprocessing implementation** is usually safer than embedding feature extraction into ONNX — but only if you first recover the exact `.nemo` preprocessing parameters.

## WebGPU + WASM recommendations for `parakeet.js`

### Runtime bottlenecks

The main browser-side bottlenecks are:

1. **Model size**
   - encoder weights can be very large
   - initial load and cache strategy matter

2. **Streaming cache buffers**
   - cache tensors are nontrivial in size
   - you want persistent buffers, not repeated reallocation

3. **RNN-T greedy decode loop**
   - decoder/joint can require many small sequential steps
   - frequent CPU↔GPU synchronization can kill latency

### Streaming chunk strategy

The model card explicitly targets **80–160 ms latency**, and requires at least **160 ms** of input. ([Hugging Face][1])

Practical implication:

- **160 ms chunk** at 16 kHz = **2560 samples**

That is the safest default chunk size to match the model’s expected operating regime.

### State reset policy

Recommended runtime behavior:

When `nextId === EOU_ID`:

- finalize the transcript
- clear text aggregation state
- reset decoder LSTM state
- usually reset encoder cache too, depending on your turn-taking policy

This behavior should ultimately be validated against reference NeMo inference, but it is the most sensible default for agent-style utterance segmentation.

## `parakeet.js` integration checklist

### Constants to hardcode after verification

```ts
const SAMPLE_RATE = 16000;
const MEL_BINS = 128;
const EOU_ID = 1024;
const EOB_ID = 1025;
const BLANK_ID = 1026;

const ENC_LAYERS = 17;
const ENC_HIDDEN = 512;

const CACHE_LAST_CHANNEL_SHAPE = [17, 1, 70, 512];
const CACHE_LAST_TIME_SHAPE = [17, 1, 512, 8];

const DEC_LAYERS = 1;
const DEC_HIDDEN = 640;
```

Supported by the official model card plus the secondary metadata/tokenizer sources. ([Hugging Face][1])

### Preprocessing pseudocode

```ts
function preprocessFloat32Mono(audio: Float32Array, sr: number): Float32Array {
  // 1) Downmix if needed
  // 2) Resample to 16 kHz
  // 3) Apply amplitude normalization only if the NeMo config requires it
  return audio16k;
}

function logMel128(audio16k: Float32Array): { mel: Float32Array; melFrames: number } {
  // Fill these from m.cfg.preprocessor:
  // n_fft, win_length, hop_length, window_fn, center, pad, normalize, dither, etc.
  return { mel, melFrames };
}
```

### RNN-T greedy loop sketch

```ts
let encCache = initEncoderCache();
let decH = zeros([1, 1, 640]);
let decC = zeros([1, 1, 640]);
let prevToken = BLANK_ID;

for (const chunk of audioChunks16k) {
  const { mel, melFrames } = logMel128(chunk);

  const { encOut, encLen, newEncCache } = runEncoder(mel, melFrames, encCache);
  encCache = newEncCache;

  for (let t = 0; t < encLen; t++) {
    let u = 0;
    while (u < MAX_TOKENS_PER_FRAME) {
      const { logits, hOut, cOut } = runDecoderJoint(encOutFrame(t), prevToken, decH, decC);

      decH = hOut;
      decC = cOut;

      const nextId = argmax(logits);

      if (nextId === BLANK_ID) break;

      if (nextId === EOU_ID) {
        emitUtterance();
        resetStates();
        break;
      }

      appendToken(nextId);
      prevToken = nextId;
      u++;
    }
  }

  emitPartial();
}
```

## Detokenization rules

Use the tokenizer vocab as `id -> piece`.

Recommended decoding behavior:

- do **not** emit `<EOU>` into user-visible text
- treat `<EOU>` as a boundary signal
- do **not** emit blank
- map `▁`-prefixed pieces to spaces in a SentencePiece-style way
- trim leading spaces after reconstruction

This matches the tokenizer’s `Metaspace` decoder behavior. ([Hugging Face][4])

## Validation plan

Two reference baselines are essential:

### 1. NeMo reference inference

Use `ASRModel.from_pretrained` and record:

- final transcripts
- partial transcripts if available
- exact `<EOU>` behavior

### 2. ONNX Runtime CPU baseline

Before WebGPU, compare:

- encoder outputs
- decoder/joint logits
- token sequences
- final transcript text

This gives you a stable baseline before browser-side optimization.

### Minimum test set categories

- short utterances around the 160 ms threshold
- speech + silence + speech sequences
- utterances with clear end-of-utterance pauses
- no-punctuation expectations, since the model is uncased/unpunctuated by design ([Hugging Face][1])

## Main porting risks

### 1. Preprocessor mismatch

Highest risk.

Symptoms:

- poor WER
- unstable EOU timing
- odd transcript drift

Fix:

- extract exact NeMo preprocessor config first

### 2. Cache shape or semantics mismatch

Symptoms:

- short tests pass, long streaming tests degrade
- corrupted outputs after several chunks
- OOM from bad buffer handling

Fix:

- lock shapes exactly to reported cache dimensions
- reuse persistent buffers

### 3. Decoder/joint synchronization overhead

Symptoms:

- bad latency despite GPU usage

Fix:

- keep decoder+joint together if possible
- reduce CPU/GPU round-trips
- cap `MAX_TOKENS_PER_FRAME`

### 4. Mis-handling `<EOU>` and blank

Symptoms:

- utterances split too early or too late
- transcript corruption

Fix:

- treat `<EOU>` as boundary only
- never detokenize blank

## Final assessment

The model is well-suited for an ONNX + WebGPU/WASM port because its publicly documented shape is relatively compact:

- 16 kHz mono input
- 128 mel bins
- 17-layer streaming FastConformer encoder
- 1-layer LSTM prediction network
- explicit `<EOU>` signaling
- manageable special-token layout ([Hugging Face][1])

But several **critical implementation details remain unspecified until you inspect the `.nemo` contents**, especially:

- STFT settings
- mel filterbank settings
- normalization behavior
- internal encoder hyperparameters
- exact export-time I/O signatures

So the right conclusion is:

**Do not implement the final browser preprocessor or finalize the ONNX export interface until you extract the real NeMo config from the `.nemo` archive.**

[1]: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 'nvidia/parakeet_realtime_eou_120m-v1 · Hugging Face'
[2]: https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml/blob/main/metadata.json 'metadata.json · FluidInference/parakeet-realtime-eou-120m-coreml at main'
[3]: https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/core/export.html 'Exporting NeMo Models — NVIDIA NeMo Framework User Guide'
[4]: https://huggingface.co/altunenes/parakeet-rs/blob/main/realtime_eou_120m-v1-onnx/tokenizer.json 'realtime_eou_120m-v1-onnx/tokenizer.json · altunenes/parakeet-rs at main'
