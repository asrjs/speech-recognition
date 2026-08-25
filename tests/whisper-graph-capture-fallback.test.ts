import { describe, expect, it, vi } from 'vitest';
import { createWhisperOrtSessionWithGraphCaptureFallback } from '../src/models/whisper-seq2seq/executor.js';
import type { OrtModuleLike, OrtSessionLike } from '../src/models/whisper-seq2seq/ort.js';

function createFakeOrt(
  create: (url: string, options?: Record<string, unknown>) => Promise<OrtSessionLike>,
): OrtModuleLike {
  return {
    env: { wasm: {} },
    Tensor: class {
      readonly data = new Float32Array(0);
      readonly dims: readonly number[] = [];
    } as unknown as OrtModuleLike['Tensor'],
    InferenceSession: { create },
  };
}

describe('Whisper graph-capture fallback', () => {
  it('retries a WebGPU session without capture and reports the original error', async () => {
    const calls: Record<string, unknown>[] = [];
    const expectedSession: OrtSessionLike = { run: vi.fn(async () => ({})) };
    const ort = createFakeOrt(async (_url, options) => {
      calls.push(options ?? {});
      if (calls.length === 1) {
        throw new Error(
          'This session cannot use the graph capture feature as all compute graph nodes have not been partitioned',
        );
      }
      return expectedSession;
    });
    const onFallback = vi.fn();

    const session = await createWhisperOrtSessionWithGraphCaptureFallback(
      ort,
      'https://example.test/decoder_step.onnx',
      { backendId: 'webgpu', enableGraphCapture: true },
      onFallback,
    );

    expect(session).toBe(expectedSession);
    expect(calls).toHaveLength(2);
    expect(calls[0]?.enableGraphCapture).toBe(true);
    expect(calls[1]?.enableGraphCapture).toBeUndefined();
    expect(onFallback).toHaveBeenCalledOnce();
  });

  it('does not hide ordinary WASM session errors', async () => {
    const error = new Error('invalid model');
    const ort = createFakeOrt(async () => {
      throw error;
    });

    await expect(
      createWhisperOrtSessionWithGraphCaptureFallback(
        ort,
        'https://example.test/decoder_step.onnx',
        { backendId: 'wasm', enableGraphCapture: true },
      ),
    ).rejects.toBe(error);
  });

  it('does not retry an unrelated WebGPU session error', async () => {
    const error = new Error('external data file not found');
    let calls = 0;
    const ort = createFakeOrt(async () => {
      calls += 1;
      throw error;
    });

    await expect(
      createWhisperOrtSessionWithGraphCaptureFallback(
        ort,
        'https://example.test/decoder_step.onnx',
        { backendId: 'webgpu', enableGraphCapture: true },
      ),
    ).rejects.toBe(error);
    expect(calls).toBe(1);
  });
});
