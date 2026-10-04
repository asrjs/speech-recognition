/**
 * Gates the library Moonshine session against the exported graphs.
 *
 * The migration replaced a per-utterance seed cache with in-graph cross
 * attention, so the test asserts the behaviour that change was meant to buy:
 * a decode that matches the recorded oracle, and a decode that runs without any
 * seed artifact at all.
 *
 * It is skipped when the graphs or the fixture are absent, because the artifacts
 * are large and local; the skip is visible rather than silent.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const MODEL_DIR = resolve(process.cwd(), 'models/onnx/moonshine-tiny-fp32');
const FIXTURE = resolve(process.cwd(), 'tools/data/fixtures/audio/jfk-short.wav');
const ORACLE = resolve(
  process.cwd(),
  'tools/data/results/moonshine/moonshine-tiny-baseline-2026-10-04.json',
);
const REQUIRED = [
  'encoder_model.onnx',
  'decoder_cross_kv_model.onnx',
  'decoder_with_past_model.onnx',
];
const available = REQUIRED.every((name) => existsSync(resolve(MODEL_DIR, name)));

function readPcm(path: string): Float32Array {
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
  throw new Error(`no data chunk in ${path}`);
}

describe.skipIf(!available)('MoonshineSession', () => {
  it('decodes the JFK fixture to the recorded oracle with no seed artifact', async () => {
    // Imported lazily so the skip above applies before ORT is touched.
    const ort = await import('onnxruntime-node');
    const { MoonshineSession, MoonshineTokenizer } =
      await import('../src/models/moonshine/session.js');

    expect(existsSync(resolve(MODEL_DIR, 'seed-cache.json'))).toBe(false);

    const receipt = JSON.parse(readFileSync(resolve(MODEL_DIR, 'export-receipt.json'), 'utf8'));
    const geometry = receipt.graphs.decoder.interface;
    const generation = JSON.parse(
      readFileSync(resolve(MODEL_DIR, 'generation_config.json'), 'utf8'),
    );
    const tokenizer = MoonshineTokenizer.fromTokenizerJson(
      JSON.parse(readFileSync(resolve(MODEL_DIR, 'tokenizer.json'), 'utf8')),
      generation.eos_token_id,
    );
    const session = new MoonshineSession({
      ort: ort as never,
      encoder: await ort.InferenceSession.create(`${MODEL_DIR}/encoder_model.onnx`, {
        executionProviders: ['cpu'],
      }),
      crossKv: await ort.InferenceSession.create(`${MODEL_DIR}/decoder_cross_kv_model.onnx`, {
        executionProviders: ['cpu'],
      }),
      decoder: await ort.InferenceSession.create(`${MODEL_DIR}/decoder_with_past_model.onnx`, {
        executionProviders: ['cpu'],
      }),
      tokenizer,
      bosTokenId: generation.bos_token_id,
      eosTokenId: generation.eos_token_id,
      slots: geometry.slots,
      heads: geometry.heads,
      headDim: geometry.headDim,
    });

    const result = await session.transcribe(readPcm(FIXTURE));
    const oracle = JSON.parse(readFileSync(ORACLE, 'utf8')) as {
      clips: Record<string, { text: string; token_ids: number[] }>;
    };

    expect(result.text).toBe(oracle.clips['jfk-short']!.text);
    expect(result.tokenIds).toEqual(oracle.clips['jfk-short']!.token_ids);
    expect(result.encoderFrames).toBeGreaterThan(0);
    expect(result.steps).toBeGreaterThan(0);
  }, 300_000);
});
