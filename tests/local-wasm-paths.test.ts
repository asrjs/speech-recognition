import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';
import { loadBuiltInSpeechModel } from '../src/runtime/builtins.js';
import {
  ParakeetModel,
  loadParakeetModelFromLocalEntries,
} from '../src/presets/parakeet/compat.js';

vi.mock('../src/runtime/builtins.js', () => ({ loadBuiltInSpeechModel: vi.fn() }));
afterEach(() => vi.restoreAllMocks());

describe('local ORT engine asset prefix forwarding (simulated engines)', () => {
  it.each(['function', 'static'] as const)(
    'legacy %s local loading forwards engine assets',
    async (kind) => {
      vi.spyOn(URL, 'createObjectURL').mockImplementation(() => 'blob:fixture');
      vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      const loadModel = vi.fn(async () => ({
        createSession: async () => ({ dispose: vi.fn(async () => undefined) }),
        dispose: vi.fn(async () => undefined),
      }));
      const entries = createSpeechModelLocalEntries([
        new File(['not-a-graph'], 'encoder-model.onnx'),
        new File(['not-a-graph'], 'decoder_joint-model.onnx'),
        new File(['token'], 'vocab.txt'),
      ]);
      const options = {
        runtime: { loadModel } as never,
        wasmPaths: '/engine/local/',
        encoderQuant: 'fp32' as const,
        decoderQuant: 'fp32' as const,
        preprocessorBackend: 'js' as const,
      };
      const model =
        kind === 'static'
          ? await ParakeetModel.fromLocalEntries(entries, options)
          : (await loadParakeetModelFromLocalEntries(entries, options)).model;
      try {
        expect(loadModel).toHaveBeenCalledWith(
          expect.objectContaining({
            options: { source: expect.objectContaining({ wasmPaths: '/engine/local/' }) },
          }),
        );
      } finally {
        await model.dispose();
      }
    },
  );
  it.each(['parakeet-tdt-0.6b-v2', 'parakeet-tdt-0.6b-v3'])(
    '%s forwards explicit engine assets through real local resolution',
    async (modelId) => {
      vi.spyOn(URL, 'createObjectURL').mockImplementation(() => 'blob:fixture');
      const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      vi.mocked(loadBuiltInSpeechModel).mockResolvedValue({
        model: { info: { modelId } },
        session: {},
        runtime: {},
        transcribe: vi.fn(),
        dispose: vi.fn(async () => undefined),
      } as never);
      const entries = createSpeechModelLocalEntries([
        new File(['not-a-graph'], 'encoder-model.onnx'),
        new File(['not-a-graph'], 'decoder_joint-model.onnx'),
        new File(['token'], 'vocab.txt'),
      ]);
      const loaded = await loadSpeechModelFromLocalEntries({
        modelId,
        entries,
        encoderQuant: 'fp32',
        decoderQuant: 'fp32',
        preprocessorBackend: 'js',
        wasmPaths: '/engine/ort-1.29.0/',
      });
      try {
        expect(loadBuiltInSpeechModel).toHaveBeenLastCalledWith(
          expect.objectContaining({
            options: { source: expect.objectContaining({ wasmPaths: '/engine/ort-1.29.0/' }) },
          }),
        );
      } finally {
        await loaded.dispose();
      }
      expect(revoke).toHaveBeenCalledTimes(3);
    },
  );

  it.each(['/engine/ort-1.29.0/', undefined])(
    'legacy direct loading preserves prefix %s without changing defaults',
    async (wasmPaths) => {
      const loadModel = vi.fn(async () => ({
        createSession: async () => ({ dispose: vi.fn(async () => undefined) }),
        dispose: vi.fn(async () => undefined),
      }));
      const model = await ParakeetModel.fromUrls({
        encoderUrl: 'blob:encoder',
        decoderUrl: 'blob:decoder',
        tokenizerUrl: 'blob:vocab',
        runtime: { loadModel } as never,
        wasmPaths,
      });
      try {
        expect(loadModel).toHaveBeenCalledWith(
          expect.objectContaining({
            options: { source: expect.objectContaining({ wasmPaths }) },
          }),
        );
      } finally {
        await model.dispose();
      }
    },
  );
});
