import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import type { createSpeechRuntime } from '@asrjs/speech-recognition';
import {
  parseNativeExampleArgs,
  prepareNativeExampleInput,
  runNativeLocalExample,
  formatNativeExampleProgress,
} from '../examples/nemo-tdt-native-local.mjs';

function fixture(context: { onTestFinished: (callback: () => void) => void }) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-native-example-'));
  context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const name of ['encoder-model.onnx', 'decoder_joint-model.onnx', 'vocab.txt']) {
    fs.writeFileSync(path.join(directory, name), 'simulation only, not an ONNX graph');
  }
  const config = {
    sampleRate: 16000,
    frameShiftSeconds: 0.01,
    subsamplingFactor: 8,
    melBins: 128,
    predictionHiddenSize: 640,
    predictionLayers: 2,
    maxSymbolsPerStep: 10,
    vocabularySize: 8193,
    tokenizer: { kind: 'sentencepiece', blankTokenId: 8192 },
  };
  const configPath = path.join(directory, 'config.json');
  fs.writeFileSync(configPath, JSON.stringify(config));
  const audio = Buffer.alloc(48);
  audio.write('RIFF', 0);
  audio.writeUInt32LE(40, 4);
  audio.write('WAVEfmt ', 8);
  audio.writeUInt32LE(16, 16);
  audio.writeUInt16LE(1, 20);
  audio.writeUInt16LE(1, 22);
  audio.writeUInt32LE(16000, 24);
  audio.writeUInt32LE(32000, 28);
  audio.writeUInt16LE(2, 32);
  audio.writeUInt16LE(16, 34);
  audio.write('data', 36);
  audio.writeUInt32LE(4, 40);
  audio.writeInt16LE(16384, 44);
  audio.writeInt16LE(-16384, 46);
  const audioPath = path.join(directory, 'audio.wav');
  fs.writeFileSync(audioPath, audio);
  return {
    directory,
    configPath,
    audioPath,
    config,
    options: parseNativeExampleArgs([
      '--model-dir',
      directory,
      '--config',
      configPath,
      '--model-id',
      'test/candidate',
      '--audio',
      audioPath,
      '--threads',
      '2',
    ]),
  };
}

function simulation() {
  const transcript = {
    canonical: { text: 'simulation only', warnings: [] },
    native: { debug: { tokenIds: Array.from({ length: 45 }, (_, index) => index) } },
  };
  const session = { transcribe: vi.fn(async (_audio: unknown, _options: unknown) => transcript) };
  const model = {
    backend: { id: 'native-cpu' },
    createSession: vi.fn(async () => session),
  };
  const runtime = {
    loadModel: vi.fn(async (_request: unknown) => model),
    dispose: vi.fn(async () => {}),
  };
  const dependencies = {
    // Deliberately partial runtime fixture: no native bindings or model acceptance.
    createRuntime: vi.fn(
      (_options: unknown) => runtime as unknown as ReturnType<typeof createSpeechRuntime>,
    ),
    getEngineVersions: vi.fn(async () => ({
      common: 'simulated-common',
      node: 'simulated-binding',
    })),
  };
  return { transcript, session, model, runtime, dependencies };
}

describe('native local consumer example CLI (not model acceptance)', () => {
  it('shows help without artifacts or engine initialization', () => {
    expect(parseNativeExampleArgs(['--help'])).toEqual({ help: true });
  });

  it('labels runtime loading and transcription progress using their actual contracts', () => {
    expect(formatNativeExampleProgress({ phase: 'model-load:start', message: 'Loading.' })).toBe(
      '[model-load:start] Loading.',
    );
    expect(
      formatNativeExampleProgress({ stage: 'decode', progress: 0.5, message: 'Decoding.' }),
    ).toBe('[decode] Decoding.');
  });

  it.for([[], ['--unknown'], ['positional'], ['--model-id', '']])(
    'rejects incomplete or invalid arguments %j',
    (args) => {
      expect(() => parseNativeExampleArgs(args)).toThrow();
    },
  );

  const validArgs = [
    '--model-dir',
    'models/test',
    '--config',
    'config.json',
    '--model-id',
    'test/candidate',
    '--audio',
    'audio.wav',
  ];

  it.for(['0', '-1', '1.5', '99999999999999999999'])(
    'rejects invalid --threads %s after required arguments pass',
    (threads) => {
      expect(() => parseNativeExampleArgs([...validArgs, `--threads=${threads}`])).toThrow(
        '--threads must be a positive safe integer.',
      );
    },
  );

  it('uses the documented thread default and accepts an explicit positive integer', () => {
    expect(parseNativeExampleArgs(validArgs).threads).toBe(4);
    expect(parseNativeExampleArgs([...validArgs, '--threads', '2']).threads).toBe(2);
  });

  it.for(['model-dir', 'config', 'model-id', 'audio'])(
    'rejects empty --%s with the other required arguments present',
    (key) => {
      expect(() => parseNativeExampleArgs([...validArgs, `--${key}=`])).toThrow(
        `Required --${key} is missing or empty.`,
      );
    },
  );

  it('loads owned PCM and explicit candidate paths without resampling', async (context) => {
    const { options } = fixture(context);
    const prepared = await prepareNativeExampleInput(options);
    expect(prepared.audio.channels[0]).toEqual(new Float32Array([0.5, -0.5]));
    expect(prepared.audio.sampleRate).toBe(16000);
    expect(prepared.source.encoderBackend).toBe('native-cpu');
    expect(prepared.source.decoderBackend).toBe('native-cpu');
    expect(prepared.source.preprocessorBackend).toBe('js');
    expect(prepared.source.cpuThreads).toBe(2);
    expect(prepared.source.artifacts.encoderUrl).toMatch(/^file:/);
    expect(prepared.source.artifacts.encoderDataUrl).toBeUndefined();
  });

  it('requests native opt-in and preserves the full canonical/native output', async (context) => {
    const { options } = fixture(context);
    const { dependencies, runtime, session, transcript } = simulation();
    const result = await runNativeLocalExample(options, dependencies);
    expect(runtime.loadModel).toHaveBeenCalledWith(
      expect.objectContaining({
        family: 'nemo-tdt',
        modelId: 'test/candidate',
        backend: 'native-cpu',
        selectionCriteria: {
          allowExperimental: true,
          environment: 'node',
          requiredPrecision: 'fp32',
        },
      }),
    );
    expect(session.transcribe).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        responseFlavor: 'canonical+native',
        returnTokenIds: true,
        returnFrameIndices: true,
        returnTdtSteps: true,
        gridBatching: false,
      }),
    );
    expect(result.transcript).toBe(transcript);
    expect(result.transcript).toMatchObject({
      native: { debug: { tokenIds: transcript.native.debug.tokenIds } },
    });
    expect(runtime.dispose).toHaveBeenCalledOnce();
    expect(result.status).toBe('RUN_COMPLETED_NOT_MODEL_ACCEPTANCE');
  });

  it.for(['missing graph', 'empty sidecar', 'invalid config', 'rate mismatch', 'empty audio'])(
    'fails preflight before creating any runtime for %s',
    async (kind, context) => {
      const { options, directory, configPath, audioPath } = fixture(context);
      if (kind === 'missing graph') fs.unlinkSync(path.join(directory, 'encoder-model.onnx'));
      if (kind === 'empty sidecar')
        fs.writeFileSync(path.join(directory, 'encoder-model.onnx.data'), '');
      if (kind === 'invalid config')
        fs.writeFileSync(configPath, JSON.stringify({ sampleRate: 16000 }));
      if (kind === 'rate mismatch') {
        const audio = fs.readFileSync(audioPath);
        audio.writeUInt32LE(22050, 24);
        audio.writeUInt32LE(44100, 28);
        fs.writeFileSync(audioPath, audio);
      }
      if (kind === 'empty audio') {
        const audio = fs.readFileSync(audioPath).subarray(0, 44);
        audio.writeUInt32LE(36, 4);
        audio.writeUInt32LE(0, 40);
        fs.writeFileSync(audioPath, audio);
      }
      const { dependencies } = simulation();
      await expect(runNativeLocalExample(options, dependencies)).rejects.toThrow();
      expect(dependencies.createRuntime).not.toHaveBeenCalled();
    },
  );

  it.for(['load', 'session', 'transcribe', 'version'])(
    'awaits runtime cleanup after a %s failure',
    async (kind, context) => {
      const { options } = fixture(context);
      const { dependencies, runtime, model, session } = simulation();
      const error = new Error(`simulated ${kind} failure`);
      if (kind === 'load') runtime.loadModel.mockRejectedValueOnce(error);
      if (kind === 'session') model.createSession.mockRejectedValueOnce(error);
      if (kind === 'transcribe') session.transcribe.mockRejectedValueOnce(error);
      if (kind === 'version') dependencies.getEngineVersions.mockRejectedValueOnce(error);
      await expect(runNativeLocalExample(options, dependencies)).rejects.toBe(error);
      expect(runtime.dispose).toHaveBeenCalledOnce();
    },
  );

  it('does not resolve successfully before runtime-owned cleanup completes', async (context) => {
    const { options } = fixture(context);
    const { dependencies, runtime } = simulation();
    let release!: () => void;
    runtime.dispose.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    let settled = false;
    const running = runNativeLocalExample(options, dependencies).then(() => {
      settled = true;
    });
    await vi.waitFor(() => expect(runtime.dispose).toHaveBeenCalledOnce());
    expect(settled).toBe(false);
    release();
    await running;
    expect(settled).toBe(true);
  });

  it('never reports success on cleanup failure', async (context) => {
    const { options } = fixture(context);
    const { dependencies, runtime } = simulation();
    const error = new Error('simulated cleanup failure');
    runtime.dispose.mockRejectedValueOnce(error);
    await expect(runNativeLocalExample(options, dependencies)).rejects.toBe(error);
  });

  it('retains both primary and cleanup errors', async (context) => {
    const { options } = fixture(context);
    const { dependencies, runtime, session } = simulation();
    const primary = new Error('simulated transcription failure');
    const cleanup = new Error('simulated cleanup failure');
    session.transcribe.mockRejectedValueOnce(primary);
    runtime.dispose.mockRejectedValueOnce(cleanup);
    await expect(runNativeLocalExample(options, dependencies)).rejects.toMatchObject({
      errors: [primary, cleanup],
    });
  });
});
