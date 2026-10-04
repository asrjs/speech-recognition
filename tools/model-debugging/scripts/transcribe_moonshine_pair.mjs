#!/usr/bin/env node
/**
 * Greedy Moonshine decode against our own fp32 graphs.
 *
 * Status: UNVERIFIED against the oracle. The export pipeline is measured, but
 * this runner's token sequence does not yet match the JFK baseline. Two fixes
 * from an earlier iteration are already applied here:
 *
 * - Cross-attention memory is a fixed projection of the encoder states, so
 *   `cross_past` stays constant across steps instead of being fed back from
 *   `cross_present`.
 * - The projection emits `[batch, frames, heads * headDim]` but the cache wants
 *   `[batch, heads, frames, headDim]`. That is a transpose, not a reinterpret:
 *   reshaping alone keeps the element count and silently scrambles the data,
 *   which shows up as a repeating token loop rather than an error.
 *
 * `onnxruntime-node` returns run results keyed by output name, not positionally.
 */
import { readFileSync } from 'node:fs';

import * as ort from 'onnxruntime-node';

function readPcm(path) {
  const bytes = readFileSync(path);
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = bytes.toString('ascii', offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4);
    if (id === 'data') {
      const pcm = new Float32Array(size / 2);
      for (let index = 0; index < pcm.length; index += 1) {
        pcm[index] = bytes.readInt16LE(offset + 8 + index * 2) / 32768;
      }
      return pcm;
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error(`No data chunk in ${path}`);
}

function argValue(name, fallback) {
  const entry = process.argv.find((value) => value.startsWith(`--${name}=`));

  return entry ? entry.slice(name.length + 3) : fallback;
}

/** Reorders `[batch, frames, heads * headDim]` into cache layout `[batch, heads, frames, headDim]`. */
function toCacheLayout(tensor, heads) {
  const [batch, frames, hidden] = tensor.dims;
  const headDim = hidden / heads;
  const source = tensor.data;
  const out = new Float32Array(batch * heads * frames * headDim);
  let cursor = 0;
  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let f = 0; f < frames; f += 1) {
        const base = b * frames * hidden + f * hidden + h * headDim;
        for (let d = 0; d < headDim; d += 1) {
          out[cursor] = source[base + d];
          cursor += 1;
        }
      }
    }
  }

  return out;
}

/** SentencePiece vocabularies appear either as a list or as an id-to-token map. */
function buildVocab(tokenizer) {
  const source = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
  if (Array.isArray(source)) {
    return source;
  }

  const entries = Object.entries(source).map(([token, id]) => [Number(id), token]);

  return entries.sort((a, b) => a[0] - b[0]).map(([, token]) => token);
}

const artifactDir = argValue('model-dir', 'models/onnx/moonshine-tiny-fp32');
const audioPath = argValue('audio', 'tools/data/fixtures/audio/jfk-short.wav');
const maxTokens = Number(argValue('max-tokens', '256'));
const slots = 12;
const heads = 8;
const headDim = 36;

const generation = JSON.parse(readFileSync(`${artifactDir}/generation_config.json`, 'utf8'));
const vocab = buildVocab(JSON.parse(readFileSync(`${artifactDir}/tokenizer.json`, 'utf8')));

const encoder = await ort.InferenceSession.create(`${artifactDir}/encoder_model.onnx`, {
  executionProviders: ['cpu'],
});
const crossKv = await ort.InferenceSession.create(`${artifactDir}/decoder_cross_kv_model.onnx`, {
  executionProviders: ['cpu'],
});
const decoder = await ort.InferenceSession.create(`${artifactDir}/decoder_with_past_model.onnx`, {
  executionProviders: ['cpu'],
});
if (process.env.MOON_DEBUG) {
  console.error('encoder outputs', encoder.outputNames);
  console.error('crossKv outputs', crossKv.outputNames.slice(0, 3));
  console.error('decoder inputs', decoder.inputNames);
  console.error('vocab size', vocab.length);
}

const started = performance.now();
const pcm = readPcm(audioPath);
const encoderOut = await encoder.run({
  input_values: new ort.Tensor('float32', pcm, [1, pcm.length]),
});
const states = encoderOut.last_hidden_state ?? encoderOut[encoder.outputNames[0]];
const frames = states.dims[1];

const projectedRaw = await crossKv.run({ encoder_states: states });
const projected = crossKv.outputNames.map((name) => projectedRaw[name]);
const crossData = new Float32Array(slots * frames * heads * headDim);
for (let slot = 0; slot < slots; slot += 1) {
  const cache = toCacheLayout(projected[slot], heads);
  crossData.set(cache, slot * frames * heads * headDim);
}

// Cross-attention memory is constant for the whole utterance: never reassigned.
const crossPast = new ort.Tensor('float32', crossData, [slots, 1, heads, frames, headDim]);
// Self-attention history starts empty, exactly like PyTorch's DynamicCache: the
// decoder creates the BOS entry itself. An earlier revision seeded it from the
// checkpoint because cross-attention was dead in the graph; that made the seed
// look load-bearing when it never was.
let selfPast = new ort.Tensor('float32', new Float32Array(0), [slots, 1, heads, 0, headDim]);

const tokenIds = [];
let steps = 0;
let nextToken = generation.bos_token_id;
for (; steps < maxTokens; steps += 1) {
  const result = await decoder.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(nextToken)]), [1, 1]),
    self_past: selfPast,
    cross_past: crossPast,
    encoder_states: states,
  });
  const logits = result.logits ?? result[decoder.outputNames[0]];
  let bestId = 0;
  let best = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.data.length; index += 1) {
    if (logits.data[index] > best) {
      best = logits.data[index];
      bestId = index;
    }
  }
  if (bestId === generation.eos_token_id) {
    steps += 1;
    break;
  }
  tokenIds.push(bestId);
  nextToken = bestId;
  selfPast = result.self_present ?? result[decoder.outputNames[1]];
}

const text = tokenIds
  .map((id) => (vocab[id] ?? '').replace(/▁/g, ' '))
  .join('')
  .trim();

console.log(
  JSON.stringify({
    status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
    backend: { id: 'wasm', engine: 'onnxruntime-node', provider: 'cpu' },
    input: { path: audioPath, samples: pcm.length },
    transcript: {
      canonical: { text, tokens: tokenIds.map((id) => ({ id })) },
      warnings: [],
      meta: {
        metrics: {
          totalMs: Number((performance.now() - started).toFixed(3)),
          encoderFrameCount: frames,
          decodeIterations: steps,
          emittedTokenCount: tokenIds.length,
        },
      },
    },
  }),
);
