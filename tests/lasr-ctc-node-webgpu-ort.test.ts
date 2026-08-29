import { describe, expect, it, vi } from 'vitest';
import type { Mock } from 'vitest';

import { createOrtSession, initOrt } from '../src/models/lasr-ctc/ort.js';

const nodeOrtState = vi.hoisted(() => ({ create: undefined as unknown }));

vi.mock('onnxruntime-node', () => ({
  default: {
    env: { wasm: {} },
    Tensor: class {},
    InferenceSession: {
      create: (nodeOrtState.create = vi.fn(async () => ({
        run: async () => ({}),
        release: async () => undefined,
      }))),
    },
  },
}));

import * as nodeOrtModule from 'onnxruntime-node';

type NodeOrtNamespace = typeof import('onnxruntime-node');

const mockedNodeOrt = nodeOrtModule as unknown as NodeOrtNamespace & {
  default: {
    InferenceSession: { create: Mock };
  };
};

describe('lasr-ctc shared ORT selection in Node-like runtimes', () => {
  it('loads onnxruntime-node for WebGPU backends', async () => {
    const ort = await initOrt('webgpu');
    expect(ort.env).toBe(mockedNodeOrt.default.env);
  });

  it('passes a plain webgpu execution provider to the native session', async () => {
    const ort = await initOrt('webgpu');
    await createOrtSession(ort, 'C:/models/model.onnx', { backendId: 'webgpu' });
    expect(mockedNodeOrt.default.InferenceSession.create).toHaveBeenCalledTimes(1);
    const [, options] = mockedNodeOrt.default.InferenceSession.create.mock.calls[0] as [
      string,
      Record<string, unknown>,
    ];
    expect(options.executionProviders).toEqual(['webgpu']);
  });

  it('keeps the wasm backend on onnxruntime-web', async () => {
    const ort = await initOrt('wasm');
    expect(ort.env).not.toBe(mockedNodeOrt.default.env);
  });
});
