import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';
import { loadSpeechModel } from '../src/runtime/load.js';
import {
  aggregateNemotronRnntFrameConfidences,
  buildEmptyNemotronRnntTranscript,
  buildNemotronRnntTranscriptDetails,
  withNemotronRnntControl,
} from '../src/models/nemotron-rnnt/transcript-details.js';
import { resolveNemotronRnntArtifacts } from '../src/models/nemotron-rnnt/ort.js';
import { parseNemotronRnntConfig } from '../src/models/nemotron-rnnt/config.js';
import { OrtNemotronRnntExecutor } from '../src/models/nemotron-rnnt/executor.js';
import {
  createNemotronPresetFactory,
  resolveNemotronArtifactSource,
  resolveNemotronPresetManifest,
} from '../src/presets/nemotron/factory.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';
import type { NemotronRnntArtifactSource } from '../src/models/nemotron-rnnt/types.js';
import { listSpeechModels, getSpeechModelDescriptor } from '../src/runtime/catalog.js';

describe('Nemotron RNNT built-in family', () => {
  it('is discoverable as a built-in model family', () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    const family = runtime
      .listModelFamilies()
      .find((candidate) => candidate.family === 'nemotron-rnnt');
    expect(family).toBeDefined();
    expect(family?.supports('nemotron-3.5-asr-streaming-0.6b')).toBe(true);
    expect(family?.supports('nvidia/nemotron-3.5-asr-streaming-0.6b')).toBe(true);
    expect(family?.supports('nvidia/parakeet-tdt-0.6b-v3')).toBe(false);
  });

  it('is registered as a built-in preset', () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    const presets = runtime.listPresets().map((factory) => factory.preset);
    expect(presets).toContain('nemotron');
  });

  it('is discoverable through the built-in model catalog (listSpeechModels)', () => {
    const nemotron = listSpeechModels().find(
      (model) => model.preset === 'nemotron',
    );
    expect(nemotron).toBeDefined();
    expect(nemotron?.modelId).toBe('nemotron-3.5-asr-streaming-0.6b');
    expect(nemotron?.displayName).toBe('Nemotron 3.5 ASR Streaming 0.6B');
    expect(nemotron?.languages).toEqual(
      expect.arrayContaining(['en', 'tr', 'auto']),
    );
    // repoId is populated only when manifest sources are enabled;
    // the catalog still discovers the model by id/alias regardless.
    // Aliases resolve through the catalog too.
    const aliased = getSpeechModelDescriptor('nvidia/nemotron-3.5-asr-streaming-0.6b');
    expect(aliased?.modelId).toBe('nemotron-3.5-asr-streaming-0.6b');
  });

  it('exposes a stub scaffold (honestly labeled) when no artifact source is given', async () => {
    const model = await loadSpeechModel({
      family: 'nemotron-rnnt',
      modelId: 'nemotron-3.5-asr-streaming-0.6b',
      backend: 'wasm',
    });
    const result = (await model.transcribe(
      {
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 1600,
        durationSeconds: 0.1,
        channels: [new Float32Array(1600)],
      },
      { responseFlavor: 'native' },
    )) as unknown as {
      warnings?: readonly { code: string }[];
      utteranceText: string;
    };
    expect(
      result.warnings?.some((w) => w.code === 'nemotron-rnnt.stubbed-decoder'),
    ).toBe(true);
    expect(typeof result.utteranceText).toBe('string');
    await model.dispose();
  });

});

describe('Nemotron preset manifest', () => {
  it('resolves the streaming model id and its aliases', () => {
    const byId = resolveNemotronPresetManifest('nemotron-3.5-asr-streaming-0.6b');
    expect(byId?.preset).toBe('nemotron');
    expect(
      resolveNemotronPresetManifest('nvidia/nemotron-3.5-asr-streaming-0.6b'),
    ).toBeDefined();
    expect(
      resolveNemotronPresetManifest(
        'ysdede/nemotron-3.5-asr-streaming-0.6b-onnx',
      ),
    ).toBeDefined();
    expect(resolveNemotronPresetManifest('parakeet-tdt-0.6b-v2')).toBeUndefined();
  });

  it('points the default source at our fp16 export repository', () => {
    const source = resolveNemotronArtifactSource('nemotron-3.5-asr-streaming-0.6b');
    expect(source?.kind).toBe('huggingface');
    if (source?.kind === 'huggingface') {
      expect(source.repoId).toBe(
        'ysdede/nemotron-3.5-asr-streaming-0.6b-onnx',
      );
    }
  });

  it('maps requests to the nemotron-rnnt family and honors manifest sources', async () => {
    const factory = createNemotronPresetFactory({ useManifestSource: true });
    const request = await factory.resolveModelRequest(
      { preset: 'nemotron', modelId: 'nemotron-3.5-asr-streaming-0.6b' },
      {} as never,
    );
    expect(request.family).toBe('nemotron-rnnt');
    expect(request.resolvedPreset).toBe('nemotron');
    expect(request.options?.source?.kind).toBe('huggingface');

    const withoutManifest = createNemotronPresetFactory({ useManifestSource: false });
    const bare = await withoutManifest.resolveModelRequest(
      { preset: 'nemotron', modelId: 'nemotron-3.5-asr-streaming-0.6b' },
      {} as never,
    );
    expect(bare.options?.source).toBeUndefined();
  });
});

describe('Nemotron RNNT config and artifact contract', () => {
  it('defaults to the published Nemotron 3.5 contract', () => {
    const config = parseNemotronRnntConfig('nemotron-3.5-asr-streaming-0.6b');
    expect(config.vocabularySize).toBe(13088);
    expect(config.blankTokenId).toBe(13087);
    expect(config.promptIds).toEqual({ auto: 101, en: 0, tr: 18 });
    expect(config.chunkFrames).toBe(65);
    expect(config.encoderOutputFramesPerChunk).toBe(7);
    expect(config.encoderCache).toEqual({
      channelLayers: 24,
      channelFrames: 56,
      channelDim: 1024,
      timeLayers: 24,
      timeFrames: 8,
      timeDim: 1024,
    });
    expect(config.predictionHiddenSize).toBe(640);
    expect(config.tokenizer.kind).toBe('bpe');
  });

  it('deep-merges prompt/cache/tokenizer overrides', () => {
    const config = parseNemotronRnntConfig('test', {
      chunkFrames: 32,
      promptIds: { en: 7 },
      encoderCache: { channelFrames: 40 },
      tokenizer: { unkTokenId: 0 },
    });
    expect(config.chunkFrames).toBe(32);
    expect(config.promptIds).toEqual({ auto: 101, en: 7, tr: 18 });
    expect(config.encoderCache.channelFrames).toBe(40);
    expect(config.encoderCache.channelLayers).toBe(24);
    expect(config.tokenizer.unkTokenId).toBe(0);
    expect(config.tokenizer.blankTokenId).toBe(13087);
  });

  it('resolves the upstream HF filename contract including external data', () => {
    const source: NemotronRnntArtifactSource = {
      kind: 'huggingface',
      repoId: 'onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4',
    };
    const { artifacts } = resolveNemotronRnntArtifacts(source, 'wasm');
    expect(artifacts.encoderUrl).toContain('/resolve/main/encoder.onnx');
    expect(artifacts.decoderUrl).toContain('/resolve/main/decoder.onnx');
    expect(artifacts.jointUrl).toContain('/resolve/main/joint.onnx');
    expect(artifacts.tokenizerUrl).toContain('/resolve/main/tokenizer.json');
    expect(artifacts.encoderDataUrl).toContain('/resolve/main/encoder.onnx.data');
    expect(artifacts.decoderDataUrl).toContain('/resolve/main/decoder.onnx.data');
    expect(artifacts.jointDataUrl).toContain('/resolve/main/joint.onnx.data');
  });

  it('passes direct artifacts through untouched', () => {
    const source: NemotronRnntArtifactSource = {
      kind: 'direct',
      artifacts: {
        encoderUrl: 'file:///m/encoder.onnx',
        decoderUrl: 'file:///m/decoder.onnx',
        jointUrl: 'file:///m/joint.onnx',
        tokenizerUrl: 'file:///m/vocab.txt',
      },
    };
    const { artifacts } = resolveNemotronRnntArtifacts(source, 'wasm');
    expect(artifacts.encoderUrl).toBe('file:///m/encoder.onnx');
    expect(artifacts.tokenizerUrl).toBe('file:///m/vocab.txt');
    expect(artifacts.encoderDataUrl).toBeUndefined();
  });

  it('defaults to a hybrid split under a webgpu backend: webgpu encoder, wasm predictor/joint', () => {
    const source: NemotronRnntArtifactSource = {
      kind: 'huggingface',
      repoId: 'onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4',
    };
    const { options } = resolveNemotronRnntArtifacts(source, 'webgpu');
    expect(options.ortBackend).toBe('webgpu');
    expect(options.encoderBackend).toBe('webgpu');
    expect(options.decoderBackend).toBe('wasm');
  });

  it('honors explicit decoder backend requests and wasm fallbacks', () => {
    const base: NemotronRnntArtifactSource = {
      kind: 'direct',
      artifacts: {
        encoderUrl: 'file:///m/encoder.onnx',
        decoderUrl: 'file:///m/decoder.onnx',
        jointUrl: 'file:///m/joint.onnx',
        tokenizerUrl: 'file:///m/vocab.txt',
      },
    };
    const fullWebgpu = resolveNemotronRnntArtifacts(
      { ...base, encoderBackend: 'webgpu', decoderBackend: 'webgpu' },
      'wasm',
    );
    expect(fullWebgpu.options.ortBackend).toBe('webgpu');
    expect(fullWebgpu.options.decoderBackend).toBe('webgpu');

    const allWasm = resolveNemotronRnntArtifacts(base, 'wasm');
    expect(allWasm.options.ortBackend).toBe('wasm');
    expect(allWasm.options.encoderBackend).toBe('wasm');
    expect(allWasm.options.decoderBackend).toBe('wasm');
  });
});

describe('Nemotron tokenizer.json loading', () => {
  function writeTempTokenizerJson(content: string): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemotron-tok-'));
    const file = path.join(dir, 'tokenizer.json');
    fs.writeFileSync(file, content, 'utf-8');
    return pathToFileURL(file).toString();
  }

  it('treats the array index as the authoritative id for pair-array vocabs', async () => {
    // Mirrors the upstream Nemotron 3.5 INT4 shape: [token, placeholder]
    // pairs where placeholder ids are -1 and the index is the real id.
    const url = writeTempTokenizerJson(
      JSON.stringify({
        model: {
          type: 'BPE',
          unk_id: 0,
          vocab: [['<unk>', 0], ['▁And', -1], ['▁', -1], ['<blank>', -1]],
        },
      }),
    );
    const tokenizer = await ParakeetTokenizer.fromTokenizerJson(url, {
      blankId: 3,
    });
    expect(tokenizer.vocabSize).toBe(4);
    expect(tokenizer.blankId).toBe(3);
    expect(tokenizer.idsToTokens([1, 2])).toEqual(['▁And', '▁']);
    expect(tokenizer.decode([1])).toBe('And');
  });

  it('honors explicit ids for object-map vocabs', async () => {
    const url = writeTempTokenizerJson(
      JSON.stringify({
        model: {
          type: 'BPE',
          vocab: { '<unk>': 0, '▁hello': 1, '▁world': 2, '<blank>': 3 },
        },
      }),
    );
    const tokenizer = await ParakeetTokenizer.fromTokenizerJson(url, {
      blankId: 3,
    });
    expect(tokenizer.vocabSize).toBe(4);
    expect(tokenizer.decode([1, 2])).toBe('hello world');
  });

  it('rejects tokenizer files without a vocab mapping', async () => {
    const url = writeTempTokenizerJson(JSON.stringify({ model: { type: 'BPE' } }));
    await expect(ParakeetTokenizer.fromTokenizerJson(url)).rejects.toThrow(
      /no model\.vocab/,
    );
  });
});

describe('Nemotron transcript helpers', () => {
  const tokenizer = new ParakeetTokenizer(
    ['<unk>', '▁And', '▁so', '<en-US>', '<blank>'],
    { blankId: 4 },
  );

  it('aggregates sparse frame confidence maps without gaps', () => {
    const stats = new Map<number, { sum: number; count: number }>([
      [0, { sum: 1.8, count: 2 }],
      [2, { sum: 0.9, count: 1 }],
    ]);
    expect(aggregateNemotronRnntFrameConfidences(stats)).toEqual([
      0.9, 0, 0.9,
    ]);
    expect(aggregateNemotronRnntFrameConfidences(new Map())).toEqual([]);
  });

  it('marks <en-US>/<tr-TR> tokens as lang segments in the control summary', () => {
    const details = buildNemotronRnntTranscriptDetails(
      tokenizer,
      [1, 2, 3],
      [5, 6, 9],
      [0.9, 0.8, 0.7],
      [-0.1, -0.2, -0.3],
      { frameTimeSeconds: 0.08 },
    );
    const transcript = withNemotronRnntControl({
      utteranceText: details.utteranceText,
      tokens: details.tokens,
      specialTokens: details.specialTokens,
    });
    expect(transcript.control?.containsLangSegment).toBe(true);
    expect(transcript.control?.langSegmentTokenIds).toEqual([3]);
    expect(details.specialTokens[0]?.kind).toBe('lang-segment');
    expect(details.utteranceText).toBe('And so');
  });

  it('builds an honestly empty transcript for silent input', () => {
    const empty = buildEmptyNemotronRnntTranscript([]);
    expect(empty.utteranceText).toBe('');
    expect(empty.tokens).toHaveLength(0);
    expect(empty.control?.containsLangSegment).toBe(false);
  });
});

describe('Nemotron RNNT predictor state contract', () => {
  it('keeps frontend, encoder caches, and RNNT state invariant across cumulative partitions', async () => {
    class FakeTensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: number[],
      ) {}
      dispose(): void {}
    }

    const runPartitioned = async (
      cuts: number[],
      hooks: { failPredictorOnce?: boolean; beforeEncoderRun?: (call: number) => Promise<void> } = {},
      execute = true,
    ) => {
      let encoderCall = 0;
      let predictorFailed = false;
      const cacheInputs: number[] = [];
      const featureChunks: number[][] = [];
      const predictorInputs: Array<{ target: number; h: number; c: number }> = [];
      const encoderSession = {
        async run(feeds: Record<string, FakeTensor>) {
          encoderCall += 1;
          await hooks.beforeEncoderRun?.(encoderCall);
          cacheInputs.push((feeds.cache_last_channel!.data as Float32Array)[0] ?? -1);
          featureChunks.push(Array.from(feeds.audio_signal!.data as Float32Array));
          return {
            outputs: new FakeTensor('float32', new Float32Array(7 * 1024).fill(encoderCall), [1, 7, 1024]),
            cache_last_channel_next: new FakeTensor('float32', new Float32Array(24 * 56 * 1024).fill(encoderCall), [1, 24, 56, 1024]),
            cache_last_time_next: new FakeTensor('float32', new Float32Array(24 * 1024 * 8).fill(encoderCall), [1, 24, 1024, 8]),
            cache_last_channel_len_next: new FakeTensor('int64', new BigInt64Array([BigInt(encoderCall)]), [1]),
          };
        },
      };
      const predictorSession = {
        async run(feeds: Record<string, FakeTensor>) {
          if (hooks.failPredictorOnce && !predictorFailed) {
            predictorFailed = true;
            throw new Error('injected predictor failure');
          }
          const target = Number((feeds.targets!.data as BigInt64Array)[0]);
          const h = (feeds.h_in!.data as Float32Array)[0] ?? 0;
          const c = (feeds.c_in!.data as Float32Array)[0] ?? 0;
          predictorInputs.push({ target, h, c });
          return {
            decoder_output: new FakeTensor('float32', new Float32Array(640).fill(target), [1, 640, 1]),
            h_out: new FakeTensor('float32', new Float32Array(1280).fill(h + 1), [2, 1, 640]),
            c_out: new FakeTensor('float32', new Float32Array(1280).fill(c + 1), [2, 1, 640]),
          };
        },
      };
      const jointSession = {
        async run(feeds: Record<string, FakeTensor>) {
          const enc = feeds.encoder_output!.data as Float32Array;
          const dec = feeds.decoder_output!.data as Float32Array;
          const frameCount = feeds.encoder_output!.dims[1]!;
          const vocabSize = 4;
          const logits = new Float32Array(frameCount * vocabSize).fill(-10);
          const target = dec[0] ?? 0;
          for (let frame = 0; frame < frameCount; frame += 1) {
            const marker = enc[frame * 1024] ?? 0;
            const winner = marker >= 4 && target === 1 ? 2
              : marker >= 3 && target === 3 ? 1
                : 3;
            logits[frame * vocabSize + winner] = 10;
          }
          return { joint_output: new FakeTensor('float32', logits, [1, frameCount, 1, vocabSize]) };
        },
      };
      const config = parseNemotronRnntConfig('nemotron-test', {
        maxDecodeSteps: 20, maxOutputTokens: 20, jointWindowFrames: 2, vocabularySize: 4,
      });
      const executor = new OrtNemotronRnntExecutor('nemotron-test', {} as never, config, 'wasm', undefined);
      const tokenizer = new ParakeetTokenizer(['<unk>', '▁one', '▁two', '<blank>'], { blankId: 3 });
      const loaded = {
        ort: { Tensor: FakeTensor }, tokenizer, encoderSession, predictorSession, jointSession,
        warnings: [], promptIdDefault: 0,
      };
      (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve(loaded);
      const pcm = Float32Array.from({ length: 42000 }, (_, i) => Math.sin(i / 29) * 0.1);
      const transcribe = (input: Float32Array, finalize: boolean) => executor.transcribeStreaming!(
        { sampleRate: 16000, durationSeconds: input.length / 16000, channels: [input] },
        { returnTokenIds: true, returnFrameIndices: true }, {} as never, finalize,
      );
      let offset = 0;
      let result: NemotronRnntNativeTranscript | undefined;
      if (execute) {
        for (const end of [...cuts, pcm.length]) {
          const cumulative = pcm.slice(0, end);
          result = await transcribe(cumulative, end === pcm.length);
          offset = end;
        }
      }
      if (execute) expect(offset).toBe(pcm.length);
      return { result: result!, cacheInputs, featureChunks, predictorInputs, executor, transcribe, pcm };
    };

    const fine = await runPartitioned([20000, 40000]);
    const coarse = await runPartitioned([27000]);
    expect(fine.cacheInputs).toEqual([0, 1, 2, 3, 4]);
    expect(coarse.cacheInputs).toEqual(fine.cacheInputs);
    expect(coarse.featureChunks).toEqual(fine.featureChunks);
    expect(coarse.result.debug?.tokenIds).toEqual(fine.result.debug?.tokenIds);
    expect(coarse.result.debug?.frameIndices).toEqual(fine.result.debug?.frameIndices);
    expect(fine.result.debug?.tokenIds).toEqual([1, 2]);
    expect(fine.predictorInputs.some((input) => input.target === 1 && input.h === 1 && input.c === 1)).toBe(true);
    expect(fine.predictorInputs.some((input) => input.target === 2 && input.h === 2 && input.c === 2)).toBe(true);
    expect(fine.result.warnings?.some((warning) =>
      warning.code === 'nemotron-rnnt.streaming-numerical-parity-unverified')).toBe(true);
    await expect(fine.executor.transcribeStreaming!(
      { sampleRate: 16000, durationSeconds: 42001 / 16000, channels: [new Float32Array(42001)] },
      {}, {} as never, false,
    )).rejects.toThrow('finalized');

    const priorCacheCount = fine.cacheInputs.length;
    fine.executor.resetStreaming();
    await fine.executor.transcribeStreaming!(
      { sampleRate: 16000, durationSeconds: 1, channels: [new Float32Array(16000)] },
      {}, {} as never, true,
    );
    expect(fine.cacheInputs[priorCacheCount]).toBe(0);

    const fixture = await runPartitioned([0], { failPredictorOnce: true }, false);
    await expect(fixture.transcribe(fixture.pcm, true)).rejects.toThrow('injected predictor failure');
    expect(fixture.cacheInputs).toEqual([0, 1, 2, 3, 4]);
    await fixture.transcribe(fixture.pcm, true);
    expect(fixture.cacheInputs.slice(5)).toEqual([0, 6, 7, 8, 9]);

    let markEntered!: () => void;
    let release!: () => void;
    const entered = new Promise<void>((resolve) => { markEntered = resolve; });
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const fixturePromise = runPartitioned([], {
      beforeEncoderRun: async (call) => {
        if (call === 1) {
          markEntered();
          await gate;
        }
      },
    }, false);
    const resetFixture = await fixturePromise;
    const pending = resetFixture.transcribe(resetFixture.pcm, true);
    await entered;
    resetFixture.executor.resetStreaming();
    release();
    await pending;
    const cacheCount = resetFixture.cacheInputs.length;
    await resetFixture.transcribe(new Float32Array(16000), true);
    expect(resetFixture.cacheInputs[cacheCount]).toBe(0);
  });

  it('feeds only the latest label with the carried state on each predictor step', async () => {
    class FakeTensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: number[],
      ) {}

      dispose(): void {}
    }

    const predictorInputs: Array<{
      targets: number[];
      targetDims: number[];
      h0: number;
      c0: number;
    }> = [];
    const jointDecoderDims: number[][] = [];
    let jointCalls = 0;
    let malformedLogits = false;
    const predictorSession = {
      async run(feeds: Record<string, FakeTensor>) {
        const targets = feeds.targets!;
        const h = feeds.h_in!;
        const c = feeds.c_in!;
        predictorInputs.push({
          targets: Array.from(targets.data as BigInt64Array, Number),
          targetDims: targets.dims,
          h0: (h.data as Float32Array)[0] ?? -1,
          c0: (c.data as Float32Array)[0] ?? -1,
        });
        const step = predictorInputs.length;
        return {
          decoder_output: new FakeTensor(
            'float32',
            new Float32Array(640).fill(step),
            [1, 640, 1],
          ),
          h_out: new FakeTensor('float32', new Float32Array(1280).fill(step), [2, 1, 640]),
          c_out: new FakeTensor('float32', new Float32Array(1280).fill(step * 2), [2, 1, 640]),
        };
      },
    };
    const jointSession = {
      async run(feeds: Record<string, FakeTensor>) {
        jointCalls += 1;
        const enc = feeds.encoder_output!;
        const dec = feeds.decoder_output!;
        jointDecoderDims.push(dec.dims);
        const frameCount = enc.dims[1]!;
        const targetCount = dec.dims[1]!;
        const vocabSize = 4;
        const logits = new Float32Array(
          frameCount * targetCount * vocabSize - (malformedLogits ? 1 : 0),
        ).fill(-10);
        for (let frame = 0; frame < frameCount; frame += 1) {
          const winner = jointCalls === 1 && frame === 0 ? 1 : 3;
          logits[
            frame * targetCount * vocabSize + (targetCount - 1) * vocabSize + winner
          ] = 10;
        }
        return {
          joint_output: new FakeTensor(
            'float32',
            logits,
            [1, frameCount, targetCount, vocabSize],
          ),
        };
      },
    };
    const executor = new OrtNemotronRnntExecutor(
      'nemotron-test',
      {} as never,
      parseNemotronRnntConfig('nemotron-test', {
        predictionHiddenSize: 640,
        jointWindowFrames: 2,
        maxDecodeSteps: 10,
        maxOutputTokens: 10,
      }),
      'wasm',
      undefined,
    );
    const runGreedyDecoder = (
      executor as unknown as {
        runGreedyDecoder(
          loaded: unknown,
          encFlat: Float32Array,
          blankId: number,
          distributionSize: number,
          signal: null,
        ): Promise<{ tokenIds: number[] }>;
      }
    ).runGreedyDecoder.bind(executor);

    const result = await runGreedyDecoder(
      {
        ort: { Tensor: FakeTensor },
        predictorSession,
        jointSession,
      },
      new Float32Array(2 * 1024),
      3,
      4,
      null,
    );

    expect(result.tokenIds).toEqual([1]);
    expect(predictorInputs.map((input) => input.targets)).toEqual([[3], [1]]);
    expect(predictorInputs.map((input) => input.targetDims)).toEqual([[1, 1], [1, 1]]);
    expect(predictorInputs.map((input) => [input.h0, input.c0])).toEqual([[0, 0], [1, 2]]);
    expect(jointDecoderDims).toEqual([[1, 1, 640], [1, 1, 640]]);
    malformedLogits = true;
    await expect(runGreedyDecoder(
      { ort: { Tensor: FakeTensor }, predictorSession, jointSession },
      new Float32Array(2 * 1024), 3, 4, null,
    )).rejects.toThrow('joint logits shape mismatch');
  });
});

function readWavPcm16Mono(filePath: string): { pcm: Float32Array; sampleRate: number } {
  const buf = fs.readFileSync(filePath);
  let offset = 12;
  let dataOffset = 0;
  let dataLength = 0;
  let sampleRate = 16000;
  while (offset < buf.length - 8) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      sampleRate = buf.readUInt32LE(offset + 12);
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  const samples = dataLength / 2;
  const pcm = new Float32Array(samples);
  for (let i = 0; i < samples; i += 1) {
    pcm[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  }
  return { pcm, sampleRate };
}

async function loadNemotronDirect(modelDir: string) {
  const fileUrl = (p: string) => pathToFileURL(p).toString();
  return loadSpeechModel({
    family: 'nemotron-rnnt',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
    backend: 'wasm',
    options: {
      source: {
        kind: 'direct',
        artifacts: {
          encoderUrl: fileUrl(path.join(modelDir, 'encoder.onnx')),
          decoderUrl: fileUrl(path.join(modelDir, 'decoder.onnx')),
          jointUrl: fileUrl(path.join(modelDir, 'joint.onnx')),
          tokenizerUrl: fileUrl(path.join(modelDir, 'vocab.txt')),
        },
      },
    },
  } as never);
}

describe('Nemotron ONNX fixture smoke', () => {
  const modelDir =
    process.env.NEMOTRON_INT4_DIR ??
    'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles';

  function weightsAvailable(): boolean {
    return ['encoder.onnx', 'decoder.onnx', 'joint.onnx', 'vocab.txt']
      .map((file) => path.join(modelDir, file))
      .every((file) => fs.existsSync(file));
  }

  const jfkFixture = path.resolve('tools/data/fixtures/audio/jfk-short.wav');
  const librivoxFixture = path.resolve(
    'tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav',
  );

  it.skipIf(
    !process.env.ASRJS_FIXTURE_SMOKE || !weightsAvailable() || !fs.existsSync(jfkFixture),
  )(
    'reproduces the parity transcript through the public API',
    async () => {
      const { pcm } = readWavPcm16Mono(jfkFixture);
      const model = await loadNemotronDirect(modelDir);
      const result = (await model.transcribe(pcm, { responseFlavor: 'native' })) as {
        utteranceText: string;
        tokens?: readonly unknown[];
      };
      await model.dispose();

      expect(result.utteranceText).toBe(
        'And so my fellow Americans ask not what your country can do for you at what you can do for your country',
      );
      expect(result.tokens?.length).toBe(41);
    },
    600_000,
  );

  it.skipIf(
    !process.env.ASRJS_FIXTURE_SMOKE_LONG || !weightsAvailable() || !fs.existsSync(librivoxFixture),
  )(
    'decodes the 40.6 s speech/silence fixture with windowed joint scanning',
    async () => {
      const { pcm } = readWavPcm16Mono(librivoxFixture);
      const model = await loadNemotronDirect(modelDir);
      const result = (await model.transcribe(pcm, { responseFlavor: 'native' })) as {
        utteranceText: string;
        tokens?: readonly unknown[];
      };
      await model.dispose();

    // Deterministic on fixed weights. Note honestly: the community INT4
    // streaming encoder degrades on this fixture relative to the NeMo
    // full-context oracle (some words drop/garble); that is a model
    // parity limitation documented in the goal file, not a decode bug.
    expect(result.tokens?.length).toBe(88);
    expect(result.utteranceText).toContain('Librivox recordings are in the public domain');
  }, 900_000);
});
