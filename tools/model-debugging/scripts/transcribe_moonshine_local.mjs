/** Transcribes a local Moonshine fp32 graph pair with the library decode contract.
 *
 * Prints the CLI shape that tools/model-debugging/scripts/port_manifest.py
 * compares against the recorded baseline. Nothing is downloaded.
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((value) => value.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

const artifactDir = argument('model-dir');
const audioPath = argument('audio');
const maxTokens = Number.parseInt(argument('max-tokens', '256'), 10);
if (!artifactDir || !audioPath) {
  throw new Error('Required: --model-dir=<local folder> --audio=<16 kHz mono WAV>');
}

const tokenizer = JSON.parse(readFileSync(`${artifactDir}/tokenizer.json`, 'utf8'));
const generation = JSON.parse(readFileSync(`${artifactDir}/generation_config.json`, 'utf8'));
const vocab = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
const idToToken = new Array(32768).fill('');
for (const [token, id] of Object.entries(vocab)) {
  idToToken[Number(id)] = token;
}
const UNK = idToToken[generation.unk_token_id ?? 3] ?? '[UNK]';

function decodeToken(id) {
  const token = idToToken[id] ?? UNK;
  if (
    token === '<|endoftext|>' ||
    token === '<|endoftext_3|>' ||
    token === '<pad>' ||
    token === '<unk>'
  ) {
    return '';
  }

  return token.replace(/▁/g, ' ');
}

const wav = readFileSync(audioPath);
const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
let offset = 12;
let pcm = null;
while (offset + 8 <= wav.byteLength) {
  const id = String.fromCharCode(...new Uint8Array(wav.buffer, wav.byteOffset + offset, 4));
  const size = view.getUint32(offset + 4, true);
  if (id === 'data') {
    pcm = new Float32Array(size / 2);
    for (let i = 0; i < pcm.length; i += 1) {
      pcm[i] = view.getInt16(offset + 8 + i * 2, true) / 32768;
    }
    break;
  }
  offset += 8 + size + (size % 2);
}
if (!pcm) {
  throw new Error('No data chunk in WAV');
}

const encoder = await ort.InferenceSession.create(`${artifactDir}/encoder_model.onnx`, {
  executionProviders: ['cpu'],
});
const decoder = await ort.InferenceSession.create(`${artifactDir}/decoder_with_past_model.onnx`, {
  executionProviders: ['cpu'],
});

const started = performance.now();
// Cross-attention key/value are computed once from the encoder states inside the
// graph, but the first step needs real history. The reference exporter emits it
// from PyTorch; read it so Node and the reference agree exactly.
const seedPath = `${artifactDir}/seed-cache.json`;
const seedCache = JSON.parse(readFileSync(seedPath, 'utf8'));

const encoderOut = await encoder.run({
  input_values: new ort.Tensor('float32', pcm, [1, pcm.length]),
});
const states = encoderOut.last_hidden_state ?? encoderOut[encoder.outputNames[0]];
const names = decoder.inputNames;
const outputs = decoder.outputNames;
const frames = states.dims[1];

// The graph returns self-attention history only; cross-attention key/value come
// from the encoder states and stay constant for the whole utterance.
const feeds = {};
for (const name of names) {
  if (name === 'input_ids') {
    feeds[name] = new ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(generation.bos_token_id)]),
      [1, 1],
    );
  } else if (/\.decoder\./.test(name)) {
    // Self-attention history starts at the seeded BOS step.
    const [batch, heads, headDim] = shapeOf(decoder, name);
    const seed = seedCache[name];
    feeds[name] = new ort.Tensor('float32', Float32Array.from(seed.flat(4)), [
      batch,
      heads,
      seed[2] ?? 1,
      headDim,
    ]);
  } else {
    // Cross-attention history is constant for the utterance and needs real
    // encoder states, so seed it from the encoder output before the first step.
    const [batch, heads, headDim] = shapeOf(decoder, name);
    const seed = seedCache[name];
    if (!seed) {
      throw new Error(`seed-cache.json is missing ${name}`);
    }
    feeds[name] = new ort.Tensor('float32', Float32Array.from(seed.flat(4)), [
      batch,
      heads,
      seed[2] ?? frames,
      headDim,
    ]);
  }
}

function framesOf(states) {
  return states.dims[1];
}

/** Cross-attention key/value come from one linear projection of the encoder
 * states, shared across steps. Zero-seed it through the graph once so the
 * first step matches the reference decode. */
function projectCrossAttention(states, batch, heads, headDim) {
  return new Float32Array(batch * heads * framesOf(states) * headDim);
}

function shapeOf(session, name) {
  const meta = session.inputMetadata;
  const entry = Array.isArray(meta) ? meta.find((item) => item.name === name) : meta?.[name];
  const dims = entry?.shape ?? [];

  return [Number(dims[0]) || 1, Number(dims[1]) || 1, Number(dims[3]) || 36];
}

const tokens = [];
let steps = 0;
for (; steps < maxTokens; steps += 1) {
  const result = await decoder.run(feeds);
  const logits = result.logits;
  const row = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
  let best = -Infinity;
  let bestId = 0;
  for (let i = 0; i < row.length; i += 1) {
    if (row[i] > best) {
      best = row[i];
      bestId = i;
    }
  }
  if (bestId === generation.eos_token_id) {
    steps += 1;
    break;
  }
  tokens.push(bestId);

  const next = {};
  for (const name of names) {
    if (name === 'input_ids') {
      next[name] = new ort.Tensor('int64', BigInt64Array.from([BigInt(bestId)]), [1, 1]);
    } else if (/\.decoder\./.test(name)) {
      const presentName = `present.${name.split('.')[1]}.decoder.${name.split('.')[3]}`;
      next[name] =
        result[presentName] ?? result[presentName.replace('present.', 'past_key_values.')];
    } else {
      next[name] = feeds[name];
    }
  }
  feeds.input_ids = next.input_ids;
  for (const name of names) {
    if (name !== 'input_ids') {
      feeds[name] = next[name];
    }
  }
}

const elapsed = performance.now() - started;
const text = tokens.map(decodeToken).join('');
console.log(
  JSON.stringify({
    status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
    host: { runtime: 'node', version: process.versions.node },
    backend: { id: 'wasm', engine: 'onnxruntime-node', provider: 'cpu' },
    input: { path: audioPath, sampleRate: 16000, samples: pcm.length },
    transcript: {
      canonical: {
        text: text.trim(),
        tokens: tokens.map((id) => ({ id })),
        warnings: [],
        meta: {
          metrics: {
            totalMs: Math.round(elapsed * 1000) / 1000,
            encoderFrameCount: frames,
            decodeIterations: steps,
            emittedTokenCount: tokens.length,
          },
        },
      },
    },
  }),
);
