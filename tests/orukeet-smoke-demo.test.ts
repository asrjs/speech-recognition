import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ load: vi.fn(), entries: vi.fn(), decode: vi.fn() }));
vi.mock('../src/runtime/local-browser.ts', () => ({
  createSpeechModelLocalEntries: mocks.entries,
  loadSpeechModelFromLocalEntries: mocks.load,
}));
vi.mock('../src/browser-media.ts', () => ({ decodeWavToMonoPcm: mocks.decode }));
vi.mock('../examples/demo/src/windowed-controller.js', () => ({
  runWindowedController: async (options: {
    transcribe: (request: unknown) => Promise<unknown>;
  }) => {
    await options.transcribe({
      pcm: new Float32Array(16000),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 16000,
      reason: 'final',
      signal: null,
    });
  },
}));

describe('Orukeet demo output contract (no real engine)', () => {
  let run: () => Promise<void>;
  let fields: Record<
    string,
    {
      value: string;
      textContent: string;
      disabled: boolean;
      files: unknown[];
      addEventListener: (event: string, listener: () => Promise<void>) => void;
    }
  >;
  let dispose: ReturnType<typeof vi.fn>;
  let transcribe: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    fields = Object.fromEntries(
      ['backend', 'precision', 'repeat', 'model', 'audio', 'run', 'result', 'status'].map((id) => [
        id,
        {
          value: '',
          textContent: '',
          disabled: false,
          files: [],
          addEventListener: (_event: string, listener: () => Promise<void>) => {
            if (id === 'run') run = listener;
          },
        },
      ]),
    );
    fields.backend.value = 'webgpu-hybrid';
    fields.precision.value = 'fp32';
    fields.repeat.value = '2';
    fields.model.files = [{}];
    fields.audio.files = [{ arrayBuffer: async () => new ArrayBuffer(0) }];
    vi.stubGlobal('document', { getElementById: (id: string) => fields[id] });
    vi.stubGlobal('location', { search: '' });
    vi.stubGlobal('crossOriginIsolated', true);
    vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => ({ info: {} }) } });
    vi.stubGlobal('__ORT_WASM_BASE', '/local-wasm/');
    mocks.entries.mockReturnValue([]);
    mocks.decode.mockReturnValue({
      pcm: new Float32Array(16000),
      numberOfChannels: 1,
      sampleRate: 16000,
    });
    dispose = vi.fn().mockResolvedValue(undefined);
    transcribe = vi.fn();
    mocks.load.mockResolvedValue({ selection: {}, transcribeMonoPcm: transcribe, dispose });
    await import('../examples/demo/src/orukeet-smoke.js');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function envelope(text: unknown = '', tokenIds: unknown = []) {
    return {
      canonical: { text, warnings: [], meta: { metrics: {} } },
      native: { debug: { tokenIds } },
    };
  }

  async function result() {
    await run();
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(fields.run.disabled).toBe(false);
    return JSON.parse(fields.result.textContent);
  }

  it('accepts repeated empty text and explicit empty tokens for silence', async () => {
    transcribe.mockResolvedValue(envelope());
    const receipt = await result();
    expect(receipt.status).toBe('PASS_ORUKEET_SMOKE');
    expect(receipt.disposed).toBe(true);
    expect(receipt.transcripts).toHaveLength(2);
    expect(receipt.transcripts.map((row: { tokenIds: number[] }) => row.tokenIds)).toEqual([
      [],
      [],
    ]);
    expect(transcribe).toHaveBeenCalledTimes(2);
  });

  it('keeps ordinary speech and repeat mismatch behavior', async () => {
    transcribe
      .mockResolvedValueOnce(envelope('hello', [3]))
      .mockResolvedValueOnce(envelope('other', [4]));
    const receipt = await result();
    expect(receipt.status).toBe('FAIL_ORUKEET_SMOKE');
    expect(receipt.error).toContain('Repeated transcripts differ');
  });

  it.each([undefined, null, 'missing', [true], [-1], [1.5]])(
    'rejects malformed native token IDs: %s',
    async (ids) => {
      const output = envelope('hello', [3]);
      output.native.debug.tokenIds = ids;
      transcribe.mockResolvedValue(output);
      expect((await result()).status).toBe('FAIL_ORUKEET_SMOKE');
    },
  );

  it.each([undefined, null, 0])('rejects missing or malformed canonical text: %s', async (text) => {
    const output = envelope();
    output.canonical.text = text;
    transcribe.mockResolvedValue(output);
    expect((await result()).status).toBe('FAIL_ORUKEET_SMOKE');
  });

  it.each([undefined, []])(
    'controller distinguishes missing diagnostics from explicit empty IDs: %s',
    async (ids) => {
      vi.stubGlobal('location', { search: '?controller=1' });
      vi.resetModules();
      await import('../examples/demo/src/orukeet-smoke.js');
      const output = envelope('hello', []);
      output.native.debug.tokenIds = ids;
      transcribe
        .mockResolvedValueOnce(envelope('hello', [3]))
        .mockResolvedValueOnce(envelope('hello', [3]))
        .mockResolvedValueOnce(output);
      const receipt = await result();
      expect(receipt.status).toBe(ids === undefined ? 'FAIL_ORUKEET_SMOKE' : 'PASS_ORUKEET_SMOKE');
      if (ids !== undefined) expect(receipt.controller.requests[0].tokenIds).toEqual([]);
    },
  );
});
