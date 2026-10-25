import { describe, expect, it } from 'vitest';
import { OrtNemotronRnntExecutor } from '../src/models/nemotron-rnnt/executor.js';
import {
  getNemotronRnntStableFrameEnd,
  parseNemotronRnntBundleConfig,
  parseNemotronRnntConfig,
  planNemotronRnntEncoderWindows,
} from '../src/models/nemotron-rnnt/config.js';
import { nemotronBuiltInLocalModelAdapter } from '../src/presets/nemotron/local-adapter.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';

const windowingManifest = {
  format: 'asrjs.nemotron-rnnt',
  schemaVersion: 1,
  family: 'nemotron-rnnt',
  encoderWindowing: {
    firstChunkFrames: 49,
    shiftFrames: 56,
    leftContextFrames: 9,
  },
};

describe('Nemotron declared encoder windowing', () => {
  it('keeps legacy behavior when the bundle has no recognized windowing manifest', () => {
    expect(parseNemotronRnntBundleConfig(undefined)).toBeUndefined();
    expect(parseNemotronRnntBundleConfig({ sample_rate: 16000 })).toBeUndefined();
    expect(parseNemotronRnntConfig('legacy').encoderWindowing).toBeUndefined();
    expect(parseNemotronRnntConfig('legacy').chunkFrames).toBe(65);
  });

  it('accepts only a versioned Nemotron contract with valid geometry', () => {
    expect(parseNemotronRnntBundleConfig(windowingManifest)).toEqual({
      encoderWindowing: {
        firstChunkFrames: 49,
        shiftFrames: 56,
        leftContextFrames: 9,
      },
    });
    expect(() => parseNemotronRnntBundleConfig({ ...windowingManifest, schemaVersion: 2 })).toThrow(
      /Unsupported Nemotron bundle schema version/,
    );
    expect(() =>
      parseNemotronRnntBundleConfig({
        ...windowingManifest,
        encoderWindowing: { firstChunkFrames: 49, shiftFrames: 0, leftContextFrames: 9 },
      }),
    ).toThrow(/positive integer/);
    expect(() =>
      parseNemotronRnntBundleConfig({
        ...windowingManifest,
        encoderWindowing: { firstChunkFrames: 49, shiftFrames: 55, leftContextFrames: 9 },
      }),
    ).toThrow(/shiftFrames \+ leftContextFrames must equal it/);
    expect(() =>
      parseNemotronRnntBundleConfig({
        ...windowingManifest,
        encoderWindowing: { firstChunkFrames: 66, shiftFrames: 56, leftContextFrames: 9 },
      }),
    ).toThrow(/firstChunkFrames must not exceed it/);
  });

  it('validates direct config overrides and rejects malformed planner inputs', () => {
    expect(() =>
      parseNemotronRnntConfig('invalid-zero', {
        encoderWindowing: { firstChunkFrames: 49, shiftFrames: 0, leftContextFrames: 9 },
      }),
    ).toThrow(/positive integer/);
    expect(() =>
      parseNemotronRnntConfig('invalid-nan', {
        encoderWindowing: {
          firstChunkFrames: Number.NaN,
          shiftFrames: 56,
          leftContextFrames: 9,
        },
      }),
    ).toThrow(/positive integer/);
    expect(() =>
      parseNemotronRnntConfig('invalid-geometry', {
        encoderWindowing: { firstChunkFrames: 49, shiftFrames: 55, leftContextFrames: 9 },
      }),
    ).toThrow(/shiftFrames \+ leftContextFrames must equal it/);
    expect(() =>
      planNemotronRnntEncoderWindows(0, 10, {
        firstChunkFrames: 49,
        shiftFrames: 0,
        leftContextFrames: 9,
      }),
    ).toThrow(/positive integer/);
    expect(() =>
      getNemotronRnntStableFrameEnd(10, {
        firstChunkFrames: 49,
        shiftFrames: 0,
        leftContextFrames: 9,
      }),
    ).toThrow(/positive integer/);
  });

  it('plans the first, steady and final NeMo windows without forcing seven rows per window', () => {
    const metadata = parseNemotronRnntBundleConfig(windowingManifest)?.encoderWindowing;
    expect(metadata).toBeDefined();
    const plan = planNemotronRnntEncoderWindows(0, 1101, metadata!);
    expect(plan).toHaveLength(20);
    expect(plan[0]).toEqual({ startFrame: 0, inputFrames: 49, advanceFrames: 49, first: true });
    expect(plan[1]).toEqual({ startFrame: 40, inputFrames: 65, advanceFrames: 56, first: false });
    expect(plan.at(-1)).toEqual({
      startFrame: 1048,
      inputFrames: 53,
      advanceFrames: 44,
      first: false,
    });
  });

  it('plans cumulative streaming updates on stable boundaries and adds left context once', () => {
    const metadata = parseNemotronRnntBundleConfig(windowingManifest)?.encoderWindowing;
    expect(metadata).toBeDefined();
    expect(planNemotronRnntEncoderWindows(0, 49, metadata!)).toEqual([
      { startFrame: 0, inputFrames: 49, advanceFrames: 49, first: true },
    ]);
    expect(planNemotronRnntEncoderWindows(49, 105, metadata!)).toEqual([
      { startFrame: 40, inputFrames: 65, advanceFrames: 56, first: false },
    ]);
    expect(planNemotronRnntEncoderWindows(1057, 1101, metadata!)).toEqual([
      { startFrame: 1048, inputFrames: 53, advanceFrames: 44, first: false },
    ]);
    expect(
      [48, 49, 104, 105, 1097].map((frames) => getNemotronRnntStableFrameEnd(frames, metadata!)),
    ).toEqual([0, 49, 49, 105, 1057]);
  });

  it('reads the declared contract from a local bundle config.json', async () => {
    const entries = [
      'encoder.onnx',
      'decoder.onnx',
      'joint.onnx',
      'tokens.txt',
      ['config.json', JSON.stringify(windowingManifest)] as const,
    ].map((entry) => {
      const name = Array.isArray(entry) ? entry[0] : entry;
      const contents = Array.isArray(entry) ? entry[1] : 'model';
      return {
        path: `bundle/${name}`,
        basename: name,
        file: new File([contents], name),
      };
    });
    const resolved = await nemotronBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'nemotron-local',
      entries,
    } as never);
    try {
      expect((resolved.builtInLoadOptions as { config?: unknown }).config).toEqual({
        encoderWindowing: {
          firstChunkFrames: 49,
          shiftFrames: 56,
          leftContextFrames: 9,
        },
      });
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('leaves unrelated local config.json files on the legacy path', async () => {
    const entries = ['encoder.onnx', 'decoder.onnx', 'joint.onnx', 'tokens.txt'].map((name) => ({
      path: `legacy/${name}`,
      basename: name,
      file: new File(['model'], name),
    }));
    entries.push({
      path: 'legacy/config.json',
      basename: 'config.json',
      file: new File([JSON.stringify({ sample_rate: 16000 })], 'config.json'),
    });
    const resolved = await nemotronBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'nemotron-legacy',
      entries,
    } as never);
    try {
      expect(resolved.builtInLoadOptions).not.toHaveProperty('config');
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('uses actual encoded_lengths for a full offline window plan and rejects a mislabeled first graph', async () => {
    class FakeTensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: number[],
      ) {}
      dispose(): void {}
    }

    const run = async (mismatchFirstWindow: boolean) => {
      const inputFrames: number[] = [];
      let call = 0;
      const config = parseNemotronRnntConfig('nemotron-window-test', {
        encoderWindowing: { firstChunkFrames: 49, shiftFrames: 56, leftContextFrames: 9 },
        encoderCache: {
          channelLayers: 1,
          channelFrames: 1,
          channelDim: 1,
          timeLayers: 1,
          timeFrames: 1,
          timeDim: 1,
        },
      });
      const executor = new OrtNemotronRnntExecutor(
        'nemotron-window-test',
        {} as never,
        config,
        'wasm',
        undefined,
      );
      const encoderSession = {
        async run(feeds: Record<string, FakeTensor>) {
          call += 1;
          const frames = feeds.audio_signal!.dims[1]!;
          inputFrames.push(frames);
          const outputFrames = mismatchFirstWindow && call === 1 ? 5 : frames === 53 ? 6 : 7;
          return {
            outputs: new FakeTensor('float32', new Float32Array(outputFrames * 1024).fill(call), [
              1,
              outputFrames,
              1024,
            ]),
            encoded_lengths: new FakeTensor(
              'int64',
              new BigInt64Array([BigInt(outputFrames)]),
              [1],
            ),
            cache_last_channel_next: new FakeTensor('float32', new Float32Array(1), [1, 1, 1, 1]),
            cache_last_time_next: new FakeTensor('float32', new Float32Array(1), [1, 1, 1, 1]),
            cache_last_channel_len_next: new FakeTensor('int64', new BigInt64Array([0n]), [1]),
          };
        },
      };
      const loaded = {
        ort: { Tensor: FakeTensor },
        encoderSession,
        encoderWindowing: config.encoderWindowing,
      };
      const invoke = (
        executor as unknown as {
          runStreamingEncoder(
            loaded: typeof loaded,
            features: Float32Array,
            frameCount: number,
            promptId: number,
            signal: null,
            cacheState?: undefined,
            frameStart?: number,
            frameEnd?: number,
          ): Promise<Float32Array>;
        }
      ).runStreamingEncoder.bind(executor);
      const output = invoke(loaded, new Float32Array(128 * 1101), 1101, 0, null);
      return { executor, inputFrames, output };
    };

    const successful = await run(false);
    const output = await successful.output;
    expect(successful.inputFrames).toEqual([49, ...Array<number>(18).fill(65), 53]);
    expect(output.length / 1024).toBe(139);
    await successful.executor.dispose();

    const mislabeled = await run(true);
    await expect(mislabeled.output).rejects.toThrow(
      /a 49-frame first window must produce 7 encoded frames; graph produced 5/,
    );
    expect(mislabeled.inputFrames).toEqual([49]);
    await mislabeled.executor.dispose();
  });

  it('uses stable cumulative boundaries and repeats only the declared left context in streaming', async () => {
    class FakeTensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: number[],
      ) {}
      dispose(): void {}
    }

    const inputWindows: Array<{ frames: number; data: Float32Array }> = [];
    const config = parseNemotronRnntConfig('nemotron-window-stream-test', {
      encoderWindowing: { firstChunkFrames: 49, shiftFrames: 56, leftContextFrames: 9 },
      encoderCache: {
        channelLayers: 1,
        channelFrames: 1,
        channelDim: 1,
        timeLayers: 1,
        timeFrames: 1,
        timeDim: 1,
      },
    });
    const executor = new OrtNemotronRnntExecutor(
      'nemotron-window-stream-test',
      {} as never,
      config,
      'wasm',
      undefined,
    );
    const encoderSession = {
      async run(feeds: Record<string, FakeTensor>) {
        const audio = feeds.audio_signal!;
        const frames = audio.dims[1]!;
        inputWindows.push({ frames, data: (audio.data as Float32Array).slice() });
        return {
          outputs: new FakeTensor('float32', new Float32Array(7 * 1024), [1, 7, 1024]),
          encoded_lengths: new FakeTensor('int64', new BigInt64Array([7n]), [1]),
          cache_last_channel_next: new FakeTensor('float32', new Float32Array(1), [1, 1, 1, 1]),
          cache_last_time_next: new FakeTensor('float32', new Float32Array(1), [1, 1, 1, 1]),
          cache_last_channel_len_next: new FakeTensor('int64', new BigInt64Array([0n]), [1]),
        };
      },
    };
    const predictorSession = {
      async run() {
        return {
          decoder_output: new FakeTensor('float32', new Float32Array(640), [1, 640, 1]),
          h_out: new FakeTensor('float32', new Float32Array(1280), [2, 1, 640]),
          c_out: new FakeTensor('float32', new Float32Array(1280), [2, 1, 640]),
        };
      },
    };
    const jointSession = {
      async run(feeds: Record<string, FakeTensor>) {
        const frameCount = feeds.encoder_output!.dims[1]!;
        const logits = new Float32Array(frameCount * 13088).fill(-1);
        for (let frame = 0; frame < frameCount; frame += 1) {
          logits[frame * 13088 + 13087] = 1;
        }
        return { joint_output: new FakeTensor('float32', logits, [1, frameCount, 1, 13088]) };
      },
    };
    const tokenizer = new ParakeetTokenizer(
      Array.from({ length: 13088 }, (_, index) => (index === 13087 ? '<blk>' : `t${index}`)),
      { blankId: 13087 },
    );
    const loaded = {
      ort: { Tensor: FakeTensor },
      tokenizer,
      encoderSession,
      predictorSession,
      jointSession,
      warnings: [],
      promptIdDefault: 0,
      encoderWindowing: config.encoderWindowing,
    };
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise =
      Promise.resolve(loaded);

    try {
      const pcm = Float32Array.from({ length: 17_280 }, (_, index) => Math.sin(index / 23) * 0.1);
      await executor.transcribeStreaming!(
        { sampleRate: 16000, durationSeconds: 0.52, channels: [pcm.slice(0, 8320)] },
        {},
        {} as never,
        false,
      );
      await executor.transcribeStreaming!(
        { sampleRate: 16000, durationSeconds: 1.08, channels: [pcm] },
        {},
        {} as never,
        false,
      );
      expect(inputWindows.map(({ frames }) => frames)).toEqual([49, 65]);
      expect(inputWindows[1]!.data.slice(0, 9 * 128)).toEqual(
        inputWindows[0]!.data.slice(40 * 128, 49 * 128),
      );
    } finally {
      await executor.dispose();
    }
  });
});
