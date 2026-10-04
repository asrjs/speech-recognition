#!/usr/bin/env node
/**
 * Greedy Qwen3-ASR decode against our own fp32 graphs.
 *
 * Three measured facts drive this runner.
 *
 * The audio front takes 128-bin mel features plus a 2-D feature mask and emits
 * one embedding per audio frame, so the two inputs have different ranks: features
 * are [batch, bins, frames] and the mask is [batch, frames].
 *
 * The prompt is 9 text tokens, then one placeholder per audio frame, then 6 more
 * text tokens. Only the audio side changes between inputs, so the text
 * embeddings are baked and concatenated around the audio embeddings.
 *
 * The decoder step carries a KV cache as two stacked tensors,
 * [layers, batch, kv_heads, past_length, head_dim]. The graph must be traced
 * against a non-empty past or its attention mask specialises to one position.
 *
 * The raw output carries a language tag and an <asr_text> marker, so the
 * transcription is what follows the marker.
 */
import { readFileSync } from 'node:fs';

import * as ort from 'onnxruntime-node';

function argValue(name, fallback) {
  const entry = process.argv.find((value) => value.startsWith(`--${name}=`));
  return entry ? entry.slice(name.length + 3) : fallback;
}

function argFlag(name) {
  return process.argv.includes(`--${name}`);
}

/** Reads a float32 .npy written by numpy; its header is a Python dict literal. */
function readNpy(path) {
  const buffer = readFileSync(path);
  // .npy v1.0: 6-byte magic, 2-byte version, then a uint16 header length.
  // Reading four bytes here pulls two bytes of header text into the length and
  // shifts every data offset, which is what made this reader return nothing.
  const headerLength = buffer.readUInt16LE(8);
  const header = buffer.toString('utf8', 10, 10 + headerLength);
  if (!header.includes("'descr': '<f4'")) {
    throw new Error(`${path} is not a float32 npy: ${header.trim()}`);
  }
  const shape = /'shape': \(([^)]*)\)/.exec(header);
  const dims = shape[1].split(',').map((value) => Number(value.trim())).filter((value) => value > 0);
  const count = dims.reduce((a, b) => a * b, 1);
  const offset = 10 + headerLength;
  // Copy rather than view: a Node Buffer's memory need not be 4-byte aligned, so
  // constructing a Float32Array over it can fail on the byte offset.
  // The Float32Array(ArrayBuffer) form copies and works; Float32Array.from() on an
  // ArrayBuffer treats it as an iterable, and an ArrayBuffer is not iterable, so it
  // silently yields length zero.
  const data = new Float32Array(
    buffer.buffer.slice(buffer.byteOffset + offset, buffer.byteOffset + offset + count * 4),
  );

  return { data, dims };
}

const MODEL_DIR = argValue('model-dir', 'models/onnx/qwen3-asr-0.6b');
const FEATURES = argValue('features', `${MODEL_DIR}/jfk_features.npy`);
const MASK = argValue('features-mask', `${MODEL_DIR}/jfk_features_mask.npy`);
const LAYERS = 28;
const HEADS = 8;
const HEAD_DIM = 128;

const front = await ort.InferenceSession.create(`${MODEL_DIR}/audio_front_model.onnx`, {
  executionProviders: ['cpu'],
});
const decoder = await ort.InferenceSession.create(`${MODEL_DIR}/decoder_step_model.onnx`, {
  executionProviders: ['cpu'],
});

const features = readNpy(FEATURES);
const mask = readNpy(MASK);
const prefix = readNpy(`${MODEL_DIR}/prompt_prefix_embeds.npy`);
const suffix = readNpy(`${MODEL_DIR}/prompt_suffix_embeds.npy`);

const started = performance.now();
// The window schedule is a graph input on the dynamo-exported front, so it is
// baked per fixture rather than recomputed here.
const cuSeqlensPath = argValue('cu-seqlens', FEATURES.replace('_features.npy', '_cu_seqlens.bin'));
const cuSeqlensRaw = readFileSync(cuSeqlensPath);
const cuSeqlens = new Int32Array(
  cuSeqlensRaw.buffer.slice(cuSeqlensRaw.byteOffset, cuSeqlensRaw.byteOffset + cuSeqlensRaw.byteLength),
);

const audio = await front.run({
  input_features: new ort.Tensor('float32', features.data, features.dims),
  input_features_mask: new ort.Tensor('float32', mask.data, mask.dims),
  cu_seqlens: new ort.Tensor('int32', cuSeqlens, [cuSeqlens.length]),
});
const audioEmbeds = audio.audio_embeds;
const frames = audioEmbeds.dims[0];

const hidden = audioEmbeds.dims[1] ?? audioEmbeds.dims[audioEmbeds.dims.length - 1];
const promptLength = prefix.dims[1] + frames + suffix.dims[1];
const prompt = new Float32Array(promptLength * hidden);
prompt.set(prefix.data.subarray(0, prefix.dims[1] * hidden), 0);
prompt.set(audioEmbeds.data.subarray(0, frames * hidden), prefix.dims[1] * hidden);
prompt.set(suffix.data.subarray(0, suffix.dims[1] * hidden), (prefix.dims[1] + frames) * hidden);

const zeroPast = () =>
  new ort.Tensor('float32', new Float32Array(0), [LAYERS, 1, HEADS, 0, HEAD_DIM]);

let result = await decoder.run({
  inputs_embeds: new ort.Tensor('float32', prompt, [1, promptLength, hidden]),
  past_key: zeroPast(),
  past_value: zeroPast(),
});
let presentKey = result.present_key;
let presentValue = result.present_value;

const embed = await ort.InferenceSession.create(`${MODEL_DIR}/embed_tokens_model.onnx`, {
  executionProviders: ['cpu'],
});
const tokenizer = JSON.parse(readFileSync(`${MODEL_DIR}/tokenizer.json`, 'utf8'));
const vocabSource = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
const vocab = new Map(Object.entries(vocabSource).map(([token, id]) => [Number(id), token]));
// The asr_text marker is an added token, so it is absent from model.vocab and would
// otherwise decode to nothing, leaving the language tag glued to the transcript.
const added = tokenizer.added_tokens ?? Object.values(tokenizer.added_tokens_decoder ?? {});
for (const entry of added) {
  const content = entry?.content ?? entry;
  const id = entry?.id;
  if (typeof content === 'string' && typeof id === 'number') {
    vocab.set(id, content);
  }
}
const eosIds = new Set([151643, 151645]);
// Measured: 151704 is <asr_text>, the marker between the language tag and the words.
const ASR_TEXT_MARKER = 151704;

const tokenIds = [];
const steps = [promptLength];
for (let step = 0; step < 256; step += 1) {
  const logits = result.logits;
  const last = logits.data.length / logits.dims[2] - 1;
  let bestId = 0;
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.dims[2]; i += 1) {
    const value = logits.data[last * logits.dims[2] + i];
    if (value > best) {
      best = value;
      bestId = i;
    }
  }
  if (eosIds.has(bestId)) {
    break;
  }
  tokenIds.push(bestId);
  steps.push(1);
  // The next step needs an embedding for the emitted token; without a graph for
  // embed_tokens the loop stops after the first generated token.
  if (argFlag('one-token')) {
    break;
  }
  // The reference emits a language tag then an <asr_text> marker before the
  // transcription, so stopping there is what a library would do with that token.
  const embedded = await embed.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(bestId)]), [1, 1]),
  });
  result = await decoder.run({
    inputs_embeds: embedded.embeddings,
    past_key: presentKey,
    past_value: presentValue,
  });
  presentKey = result.present_key;
  presentValue = result.present_value;
}

const raw = tokenIds.map((id) => vocab.get(id) ?? '').join('');
// Qwen uses byte-level BPE, so the space marker is U+0120 rather than U+2581.
const markerIndex = tokenIds.indexOf(ASR_TEXT_MARKER);
const body = markerIndex >= 0 ? tokenIds.slice(markerIndex + 1) : tokenIds;
const text = body.map((id) => (vocab.get(id) ?? '').replace(/\u0120/g, ' ')).join('').trim();

console.log(
  JSON.stringify({
    status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
    backend: { id: 'cpu', engine: 'onnxruntime-node' },
    metrics: {
      totalMs: Number((performance.now() - started).toFixed(3)),
      audioFrames: frames,
      promptLength,
      decodeSteps: steps.length,
      emittedTokenCount: tokenIds.length,
    },
    transcript: { canonical: { text, tokens: tokenIds.map((id) => ({ id })) }, warnings: [] },
  }),
);
