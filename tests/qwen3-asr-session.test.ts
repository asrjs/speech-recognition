/**
 * Gates the registered Qwen3-ASR library session against the exported graphs.
 *
 * The oracle match was first proven by a standalone runner. This test exists so
 * the path that actually ships carries the same evidence: the session in
 * src/models/qwen3-asr, loading the real graphs and decoding the JFK fixture.
 *
 * It skips when the artifacts are absent, because they are large and local.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const MODEL_DIR = resolve(process.cwd(), 'models/onnx/qwen3-asr-0.6b');
const ORACLE = resolve(
  process.cwd(),
  'tools/data/results/model-candidates/qwen3-asr-0.6b-baseline-2026-10-04.json',
);
const REQUIRED = [
  'audio_front_model.onnx',
  'decoder_step_model.onnx',
  'embed_tokens_model.onnx',
  'tokenizer.json',
  'jfk_features.npy',
  'jfk_features_mask.npy',
  'prompt_prefix_embeds.npy',
  'prompt_suffix_embeds.npy',
];
const available = REQUIRED.every((name) => existsSync(resolve(MODEL_DIR, name)));

/** Reads a float32 .npy; its header is a Python dict literal, not JSON. */
function readNpy(path: string): { data: Float32Array; dims: number[] } {
  const buffer = readFileSync(path);
  // .npy v1.0 is a 6-byte magic, a 2-byte version, then a uint16 header length.
  const headerLength = buffer.readUInt16LE(8);
  const header = buffer.toString('utf8', 10, 10 + headerLength);
  if (!header.includes("'descr': '<f4'")) {
    throw new Error(`${path} is not a float32 npy`);
  }
  const matched = /'shape': \(([^)]*)\)/.exec(header);
  const dims = (matched?.[1] ?? '')
    .split(',')
    .map((value) => Number(value.trim()))
    .filter((value) => value > 0);
  const count = dims.reduce((a, b) => a * b, 1);
  const offset = 10 + headerLength;
  // The Float32Array(ArrayBuffer) form copies. Float32Array.from() on an
  // ArrayBuffer treats it as an iterable, and an ArrayBuffer is not iterable,
  // so it silently yields length zero.
  const data = new Float32Array(
    buffer.buffer.slice(buffer.byteOffset + offset, buffer.byteOffset + offset + count * 4),
  );
  return { data, dims };
}

describe.skipIf(!available)('Qwen3AsrSession', () => {
  it('decodes the JFK fixture to the recorded oracle', async () => {
    const ort = await import('onnxruntime-node');
    const { Qwen3AsrSession, Qwen3AsrTokenizer } =
      await import('../src/models/qwen3-asr/session.js');

    const create = (name: string) =>
      ort.InferenceSession.create(`${MODEL_DIR}/${name}`, { executionProviders: ['cpu'] });

    const features = readNpy(resolve(MODEL_DIR, 'jfk_features.npy'));
    const mask = readNpy(resolve(MODEL_DIR, 'jfk_features_mask.npy'));
    const prefix = readNpy(resolve(MODEL_DIR, 'prompt_prefix_embeds.npy'));
    const suffix = readNpy(resolve(MODEL_DIR, 'prompt_suffix_embeds.npy'));

    const session = new Qwen3AsrSession({
      ort: ort as never,
      front: await create('audio_front_model.onnx'),
      decoder: await create('decoder_step_model.onnx'),
      embed: await create('embed_tokens_model.onnx'),
      tokenizer: new Qwen3AsrTokenizer(
        JSON.parse(readFileSync(resolve(MODEL_DIR, 'tokenizer.json'), 'utf8')),
      ),
      prefixEmbeds: prefix.data,
      suffixEmbeds: suffix.data,
      layers: 28,
      kvHeads: 8,
      headDim: 128,
    });

    const result = await session.transcribe(features.data, features.dims, mask.data, mask.dims);
    const oracle = JSON.parse(readFileSync(ORACLE, 'utf8')) as {
      clips: Record<string, { text: string }>;
    };

    expect(result.text).toBe(oracle.clips['jfk-short']!.text);
    expect(result.language).toBe('English');
    expect(result.audioFrames).toBe(143);
    expect(result.tokenIds.length).toBeGreaterThan(20);
  }, 600_000);
});
