import { afterEach, describe, expect, it, vi } from 'vitest';
import { initOrt as nemo } from '../src/models/nemo-tdt/ort.js';
import { initOrt as lasr } from '../src/models/lasr-ctc/ort.js';
import { initOrt as wav2vec } from '../src/models/wav2vec2/ort.js';
import { initQwenOrt as qwen } from '../src/models/qwen-asr/ort.js';
import { initWhisperOrt as whisper } from '../src/models/whisper-seq2seq/ort.js';

afterEach(() => vi.unstubAllGlobals());

describe('family WASM thread configuration', () => {
  it.each([
    ['nemo', nemo],
    ['lasr', lasr],
    ['wav2vec', wav2vec],
    ['qwen', qwen],
    ['whisper', whisper],
  ] as const)(
    '%s clamps explicit threads and preserves single-thread fallback',
    async (_name, init) => {
      vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
      expect((await init('wasm', { cpuThreads: 32 })).env.wasm.numThreads).toBe(10);
      expect((await init('wasm', { cpuThreads: 2 })).env.wasm.numThreads).toBe(2);
      vi.stubGlobal('SharedArrayBuffer', undefined);
      expect((await init('wasm', { cpuThreads: 32 })).env.wasm.numThreads).toBe(1);
    },
  );
  it('retains Whisper single-thread default', async () => {
    vi.stubGlobal('navigator', { hardwareConcurrency: 16 });
    expect((await whisper('wasm')).env.wasm.numThreads).toBe(1);
  });
});
