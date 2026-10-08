/**
 * Real-graph integration: MoonshineStreamingSession over the exported fp32
 * streaming graphs in Node onnxruntime, on the pinned JFK effective PCM.
 *
 * Gates:
 * 1. pushAudio in small chunks, then finalize: the streamed framing must
 *    reproduce the whole-utterance hidden-state count, and the decoded text
 *    must equal the official streaming oracle transcript,
 * 2. a preview taken mid-stream decodes to text over the committed frames,
 * 3. a second finalize after reset() reproduces the same result.
 *
 * Requires the graphs at .cache/moonshine-streaming-onnx (run the export
 * scripts first). Skipped when the graphs or tokenizer are absent so the
 * default suite stays hermetic.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';
import { MoonshineStreamingSession, embeddedFrameCount } from '../src/models/moonshine/streaming-session.js';
import { MoonshineTokenizer } from '../src/models/moonshine/session.js';

const GRAPH_DIR = resolve('.cache/moonshine-streaming-onnx');
const SNAPSHOT = resolve('.cache/moonshine-streaming-tiny');
const WAV = resolve('tools/data/fixtures/audio/jfk-short.wav');
const EXPECTED_TEXT = 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.';

const graphsReady = existsSync(resolve(GRAPH_DIR, 'encoder_chunked_model.onnx'))
  && existsSync(resolve(GRAPH_DIR, 'encoder_model.onnx'))
  && existsSync(resolve(GRAPH_DIR, 'decoder_prefill_model.onnx'))
  && existsSync(resolve(GRAPH_DIR, 'decoder_step_model.onnx'))
  && existsSync(resolve(SNAPSHOT, 'tokenizer.json'));

function effectivePcm(): Float32Array {
  const buffer = readFileSync(WAV);
  const view = new DataView(buffer.buffer, buffer.byteOffset);
  // Walk RIFF chunks to the real data chunk (the fixture carries metadata).
  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const id = String.fromCharCode(buffer[offset]!, buffer[offset + 1]!, buffer[offset + 2]!, buffer[offset + 3]!);
    const size = view.getUint32(offset + 4, true);
    if (id === 'data') {
      const samples = Math.floor(size / 2);
      const pcm = new Float32Array(samples);
      for (let index = 0; index < samples; index += 1) {
        pcm[index] = view.getInt16(offset + 8 + index * 2, true) / 32768;
      }
      return pcm;
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error('WAV data chunk not found');
}

function tokenizer(): MoonshineTokenizer {
  const json = JSON.parse(readFileSync(resolve(SNAPSHOT, 'tokenizer.json'), 'utf-8'));
  return MoonshineTokenizer.fromTokenizerJson(json, 0);
}

function session(emit = 256, dtype: 'float32' | 'float16' = 'float32') {
  const dir = dtype === 'float16' ? resolve('.cache/moonshine-streaming-onnx-fp16') : GRAPH_DIR;
  const options = {
    ort,
    chunkedEncoder: { run: async (feeds) => (await sessionFor(dir, 'encoder_chunked_model.onnx')).run(feeds) },
    wholeEncoder: { run: async (feeds) => (await sessionFor(dir, 'encoder_model.onnx')).run(feeds) },
    prefill: { run: async (feeds) => (await sessionFor(dir, 'decoder_prefill_model.onnx')).run(feeds) },
    step: { run: async (feeds) => (await sessionFor(dir, 'decoder_step_model.onnx')).run(feeds) },
    tokenizer: tokenizer(),
    bosTokenId: 1,
    eosTokenId: 2,
    emitFrames: emit,
    dtype,
  };
  return new MoonshineStreamingSession(options);
}

const sessionCache = new Map<string, Promise<any>>();
function sessionFor(dir: string, name: string) {
  const key = resolve(dir, name);
  let created = sessionCache.get(key);
  if (!created) {
    created = ort.InferenceSession.create(key, { executionProviders: ['cpu'] });
    sessionCache.set(key, created);
  }
  return created;
}

const fp16Ready = existsSync(resolve('.cache/moonshine-streaming-onnx-fp16', 'encoder_chunked_model.onnx'));

/** Probes once whether the pinned onnxruntime-node build can feed float16 inputs. */
let fp16Probe: Promise<boolean> | null = null;
function fp16FeedsSupported(): Promise<boolean> {
  fp16Probe = fp16Probe ?? (async () => {
    try {
      const probeSession = await sessionFor('.cache/moonshine-streaming-onnx-fp16', 'encoder_chunked_model.onnx');
      await probeSession.run({
        input_values: new ort.Tensor('float16', new Uint16Array(16 * 80), [1, 16, 80]),
        embedded_context: new ort.Tensor('float16', new Uint16Array(90 * 320), [1, 90, 320]),
        context_len: new ort.Tensor('int64', BigInt64Array.from([0n]), []),
        conv1_tail: new ort.Tensor('float16', new Uint16Array(320 * 4), [1, 320, 4]),
        conv2_tail: new ort.Tensor('float16', new Uint16Array(640 * 4), [1, 640, 4]),
        frames1: new ort.Tensor('int64', BigInt64Array.from([0n]), []),
        frames2: new ort.Tensor('int64', BigInt64Array.from([0n]), []),
      });
      return true;
    } catch {
      return false;
    }
  })();
  return fp16Probe;
}

describe.skipIf(!graphsReady)('MoonshineStreamingSession on real fp32 graphs', () => {
  it('streams chunked audio and finalizes to the official oracle transcript', async () => {
    const pcm = effectivePcm();
    const streaming = session(256);
    const chunk = 16000; // one second of audio per push
    for (let offset = 0; offset + chunk <= pcm.length; offset += chunk) {
      await streaming.pushAudio(pcm.subarray(offset, offset + chunk));
    }
    const result = await streaming.finalize();
    expect(result.text).toBe(EXPECTED_TEXT);
    expect(result.encoderFrames).toBe(embeddedFrameCount(pcm.length / 80));
  }, 600000);

  it('emits a mid-stream preview before finalize', async () => {
    const pcm = effectivePcm();
    const streaming = session(256);
    await streaming.pushAudio(pcm.subarray(0, 48000));
    const preview = await streaming.preview();
    expect(preview.encoderFrames).toBeGreaterThan(0);
    expect(preview.text.length).toBeGreaterThan(0);
    await streaming.pushAudio(pcm.subarray(48000));
    const result = await streaming.finalize();
    expect(result.text).toBe(EXPECTED_TEXT);
  }, 600000);

  it('reset allows a second stream with the same outcome', async () => {
    const pcm = effectivePcm();
    const streaming = session(64);
    await streaming.pushAudio(pcm);
    const first = await streaming.finalize();
    expect(first.text).toBe(EXPECTED_TEXT);
    streaming.reset();
    await streaming.pushAudio(pcm);
    const second = await streaming.finalize();
    expect(second.text).toBe(first.text);
    expect(second.tokenIds).toEqual(first.tokenIds);
  }, 600000);

  it('drives the fp16 graph set to the same oracle transcript when the Node ORT build can feed float16', async () => {
    if (!fp16Ready) {
      return;
    }
    if (!(await fp16FeedsSupported())) {
      // onnxruntime-node cannot read Node's Float16Array in the native bridge
      // (minimal Relu repro: "not enough space: expected 8, got 0" on 1.25-dev,
      // 1.30.0 and 1.31-dev; works when Float16Array is hidden, i.e. the JS
      // wrapper's Uint16 fallback). The fp16 graphs are export-level verified
      // through Python ORT (moonshine-streaming-fp16-chunked-2026-10-09.json,
      // chunked tokens == oracle); onnxruntime-web uses a separate tensor path
      // and is exercised by the browser validation slice.
      console.warn('[skip] pinned onnxruntime-node build cannot feed float16 tensors');
      return;
    }
    const pcm = effectivePcm();
    const streaming = session(64, 'float16');
    await streaming.pushAudio(pcm);
    const result = await streaming.finalize();
    expect(result.text).toBe(EXPECTED_TEXT);
    expect(result.encoderFrames).toBe(embeddedFrameCount(pcm.length / 80));
  }, 600000);
});
