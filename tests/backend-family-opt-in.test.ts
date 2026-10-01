import { describe, expect, it, vi } from 'vitest';
import { createSpeechRuntime } from '../src/runtime/session.js';
import type { ExecutionBackend, SpeechModelFactory } from '../src/types/index.js';

function backend(
  id: string,
  requiresModelOptIn = false,
): ExecutionBackend & {
  readonly requiresModelOptIn: boolean;
} {
  return {
    id,
    displayName: id,
    requiresModelOptIn,
    async probeCapabilities() {
      return {
        id,
        displayName: id,
        available: true,
        priority: id === 'new-engine' ? 100 : 60,
        environments: ['node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32'],
        supportsFp16: false,
        supportsInt8: false,
        supportsSharedArrayBuffer: false,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      };
    },
    async createExecutionContext() {
      throw new Error('A capability lease is not needed for this factory test.');
    },
  };
}

function factory(supportedBackendIds?: readonly string[]): SpeechModelFactory & {
  readonly supportedBackendIds?: readonly string[];
} {
  return {
    family: 'test-family',
    supportedBackendIds,
    supports: () => true,
    createModel: vi.fn(async (request, context) => ({
      info: {
        family: 'test-family',
        modelId: request.modelId,
        classification: {} as never,
      },
      backend: context.backend,
      async createSession() {
        throw new Error('No inference is claimed by this factory test.');
      },
      dispose() {},
    })),
  };
}

describe('model-family backend opt-in', () => {
  it('rejects an explicit guarded backend before an unadapted family can create a model', async () => {
    const family = factory();
    const runtime = createSpeechRuntime({
      backends: [backend('new-engine', true)],
      modelFamilies: [family],
    });
    try {
      await expect(
        runtime.loadModel({
          family: 'test-family',
          modelId: 'test',
          backend: 'new-engine',
        }),
      ).rejects.toMatchObject({ code: 'capability-mismatch' });
      expect(family.createModel).not.toHaveBeenCalled();
    } finally {
      await runtime.dispose();
    }
  });

  it('filters guarded backends from automatic family selection without opt-in', async () => {
    const runtime = createSpeechRuntime({
      backends: [backend('new-engine', true), backend('wasm')],
      modelFamilies: [factory()],
    });
    try {
      const model = await runtime.loadModel({
        family: 'test-family',
        modelId: 'test',
        selectionCriteria: { preferredBackendIds: ['new-engine', 'wasm'] },
      });
      expect(model.backend.id).toBe('wasm');
    } finally {
      await runtime.dispose();
    }
  });

  it('honors a family allowlist for ordinary backends too', async () => {
    const family = factory(['wasm']);
    const runtime = createSpeechRuntime({
      backends: [backend('other')],
      modelFamilies: [family],
    });
    try {
      await expect(
        runtime.loadModel({
          family: 'test-family',
          modelId: 'test',
          backend: 'other',
        }),
      ).rejects.toMatchObject({ code: 'capability-mismatch' });
      expect(family.createModel).not.toHaveBeenCalled();
    } finally {
      await runtime.dispose();
    }
  });

  it('allows an explicitly opted-in family and keeps backend identity', async () => {
    const runtime = createSpeechRuntime({
      backends: [backend('new-engine', true)],
      modelFamilies: [factory(['new-engine'])],
    });
    try {
      const model = await runtime.loadModel({
        family: 'test-family',
        modelId: 'test',
        backend: 'new-engine',
      });
      expect(model.backend.id).toBe('new-engine');
    } finally {
      await runtime.dispose();
    }
  });

  it('preserves legacy factories on unguarded backends', async () => {
    const runtime = createSpeechRuntime({
      backends: [backend('wasm')],
      modelFamilies: [factory()],
    });
    try {
      const model = await runtime.loadModel({ family: 'test-family', modelId: 'test' });
      expect(model.backend.id).toBe('wasm');
    } finally {
      await runtime.dispose();
    }
  });
});
