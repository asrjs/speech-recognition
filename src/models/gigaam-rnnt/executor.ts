import { normalizePcmInput } from '../../audio/index.js';
import { confidenceFromLogits, argmax } from '../../inference/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import type { AbortSignalLike, AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks, TranscriptWarning } from '../../types/index.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import { createOrtSession, disposeOrtOutputs, initOrt, releaseOrtSession, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import { GigaAmJsPreprocessor } from './frontend.js';
import { GigaAmRnntTokenizer } from './tokenizer.js';
import type { LasrCtcNativeToken } from '../lasr-ctc/types.js';
import { resolveGigaAmRnntBackends, type GigaAmRnntArtifactSource, type GigaAmRnntModelConfig, type GigaAmRnntModelOptions, type GigaAmRnntNativeTranscript, type GigaAmRnntTranscriptionOptions, type ResolvedGigaAmRnntBackends } from './types.js';

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly encoder: OrtSessionLike;
  readonly decoder: OrtSessionLike;
  readonly joint: OrtSessionLike;
  readonly tokenizer: GigaAmRnntTokenizer;
  readonly warnings: readonly TranscriptWarning[];
  readonly backends: ResolvedGigaAmRnntBackends;
}

function tensor(ort: OrtModuleLike, type: 'float32' | 'int64' | 'int32', data: ArrayBufferView, dims: readonly number[]): OrtTensorLike {
  return new ort.Tensor(type, data, dims);
}

function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
}

function firstOutput(outputs: Record<string, OrtTensorLike>, ...names: string[]): OrtTensorLike {
  for (const name of names) if (outputs[name]) return outputs[name]!;
  const output = Object.values(outputs)[0];
  if (!output) throw new Error('GigaAM RNN-T graph returned no output.');
  return output;
}

function tensorData(value: OrtTensorLike): Float32Array {
  if (value.type !== 'float16') {
    return value.data instanceof Float32Array
      ? new Float32Array(value.data)
      : Float32Array.from(value.data as unknown as ArrayLike<number>);
  }
  const source = value.data as unknown as ArrayLike<number>;
  const result = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const bits = Number(source[index] ?? 0);
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;
    result[index] = exponent === 0
      ? sign * (mantissa / 1024) * 2 ** -14
      : exponent === 31
        ? (mantissa ? NaN : sign * Infinity)
        : sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return result;
}

function hfUrl(repoId: string, revision: string, filename: string): string {
  return `https://huggingface.co/${repoId.split('/').map(encodeURIComponent).join('/')}/resolve/${encodeURIComponent(revision)}/${filename.split('/').map(encodeURIComponent).join('/')}`;
}

export class OrtGigaAmRnntExecutor {
  private readonly preprocessor = new GigaAmJsPreprocessor();
  private readonly handles: ResolvedAssetHandle[] = [];
  private readonly source?: GigaAmRnntArtifactSource;
  private readonly provider?: AssetProvider;
  private readonly hooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly state?: Promise<LoadedState>;
  private disposed = false;
  private disposePromise?: Promise<void>;
  // Speculative batch widths grow geometrically from the minimum when a
  // batch window finds no emission (the predictor state is unchanged, so the
  // scored blank rows are exact progress) and reset on every emission, so
  // wasted pre-emission rows stay proportional to the blank-run length. Full
  // suffix batching is free for a dispatch-bound GPU joint but compute-bound
  // waste for a WASM joint, hence the bounded growth.
  private static readonly joinerBatchMinRows = 2;
  private static readonly joinerBatchMaxRows = 64;
  // Speculative batched-joiner state. The joint graph is row-parallel (the
  // leading dimension is the frame batch); a graph that rejects batched
  // shapes latches batching off permanently for this executor.
  private joinerRowSize?: number;
  private joinerBatchAllowed = true;
  private joinerBatchWidth = OrtGigaAmRnntExecutor.joinerBatchMinRows;

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly config: GigaAmRnntModelConfig,
    options?: GigaAmRnntModelOptions,
    dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks; readonly signal?: import('../../types/index.js').AbortSignalLike | null } = {},
  ) {
    this.source = options?.source;
    this.provider = dependencies.assetProvider;
    this.hooks = dependencies.runtimeHooks;
    this.signal = dependencies.signal;
    if (this.source) this.state = this.initialize();
  }

  private async resolve(source: Extract<GigaAmRnntArtifactSource, { kind: 'huggingface' }>, filename: string): Promise<string> {
    if (!this.provider) return hfUrl(source.repoId, source.revision ?? 'main', filename);
    const handle = await this.provider.resolve({
      id: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`,
      provider: 'huggingface',
      repoId: source.repoId,
      revision: source.revision ?? 'main',
      filename,
      cacheKey: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`,
      onProgress: (event) => this.hooks?.onProgress?.({
        phase: 'asset:download',
        modelId: this.modelId,
        file: filename,
        loaded: event.loaded,
        total: event.total,
        percent: event.total ? Math.round(event.loaded / event.total * 100) : event.done && !event.aborted ? 100 : undefined,
        isComplete: Boolean(event.done) && !event.aborted,
        aborted: event.aborted,
        message: event.aborted ? `Cancelled ${filename}.` : event.done ? `Prepared ${filename}.` : `Downloading ${filename}.`,
      }),
    });
    this.handles.push(handle);
    const url = await handle.getLocator('url');
    if (!url) throw new Error(`Could not create a URL locator for "${filename}".`);
    return url;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw createExperimentalArtifactMissingError('gigaam-rnnt', this.modelId);
    let encoderUrl: string;
    let decoderUrl: string;
    let jointUrl: string;
    let tokenizerUrl: string;
    let encoderDataUrl: string | undefined;
    let decoderDataUrl: string | undefined;
    let jointDataUrl: string | undefined;
    let encoderDataFilename: string | undefined;
    let decoderDataFilename: string | undefined;
    let jointDataFilename: string | undefined;
    if (this.source.kind === 'direct') {
      ({
        encoderUrl, decoderUrl, jointUrl, tokenizerUrl,
        encoderDataUrl, decoderDataUrl, jointDataUrl,
        encoderDataFilename, decoderDataFilename, jointDataFilename,
      } = this.source.artifacts);
    } else {
      const encoderFilename = this.source.encoderFilename ?? 'v3_e2e_rnnt_encoder.onnx';
      const decoderFilename = this.source.decoderFilename ?? 'v3_e2e_rnnt_decoder.onnx';
      const jointFilename = this.source.jointFilename ?? 'v3_e2e_rnnt_joint.onnx';
      const tokenizerFilename = this.source.tokenizerFilename ?? 'v3_e2e_rnnt_vocab.txt';
      encoderUrl = await this.resolve(this.source, encoderFilename);
      decoderUrl = await this.resolve(this.source, decoderFilename);
      jointUrl = await this.resolve(this.source, jointFilename);
      tokenizerUrl = await this.resolve(this.source, tokenizerFilename);
      encoderDataFilename = this.source.encoderDataFilename;
      decoderDataFilename = this.source.decoderDataFilename;
      jointDataFilename = this.source.jointDataFilename;
      if (encoderDataFilename) encoderDataUrl = await this.resolve(this.source, encoderDataFilename);
      if (decoderDataFilename) decoderDataUrl = await this.resolve(this.source, decoderDataFilename);
      if (jointDataFilename) jointDataUrl = await this.resolve(this.source, jointDataFilename);
    }
    const backends = resolveGigaAmRnntBackends(this.source, this.backendId);
    const ort = await initOrt(backends.ortBackend, {
      wasmPaths: this.source.wasmPaths,
      cpuThreads: this.source.cpuThreads,
      signal: this.signal,
    });
    const make = (url: string, backendId: 'webgpu' | 'wasm', dataUrl?: string, dataPath?: string) => createOrtSession(ort, url, {
      backendId,
      externalDataUrl: dataUrl,
      externalDataPath: dataPath,
      signal: this.signal,
    });
    // ORT's WASM provider rejects concurrent initWasm() calls, and a WebGPU
    // session can still initialize WASM for fallback kernels. Keep mixed and
    // all-WASM loads serial; only the all-WebGPU composition is safe to probe
    // concurrently. This preserves the historical hybrid path while making
    // the fully GPU composition's startup overlap an evidence-backed option.
    type SessionName = 'encoder' | 'decoder' | 'joint';
    type SessionSpec = {
      readonly name: SessionName;
      readonly backend: 'webgpu' | 'wasm';
      readonly url: string;
      readonly dataUrl?: string;
      readonly dataPath?: string;
    };
    const specs: readonly SessionSpec[] = [
      { name: 'encoder', backend: backends.encoderBackend, url: encoderUrl, dataUrl: encoderDataUrl, dataPath: encoderDataFilename },
      { name: 'decoder', backend: backends.decoderBackend, url: decoderUrl, dataUrl: decoderDataUrl, dataPath: decoderDataFilename },
      { name: 'joint', backend: backends.jointBackend, url: jointUrl, dataUrl: jointDataUrl, dataPath: jointDataFilename },
    ];
    const created = new Map<SessionName, OrtSessionLike>();
    if (this.source.parallelSessionInitialization === true && specs.every((spec) => spec.backend === 'webgpu')) {
      const results = await Promise.allSettled(specs.map((spec) => make(spec.url, spec.backend, spec.dataUrl, spec.dataPath)));
      const failed = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failed) {
        results.forEach((result) => {
          if (result.status === 'fulfilled') releaseOrtSession(result.value);
        });
        throw failed.reason;
      }
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') created.set(specs[index]!.name, result.value);
      });
    } else {
      try {
        for (const spec of specs) created.set(spec.name, await make(spec.url, spec.backend, spec.dataUrl, spec.dataPath));
      } catch (error) {
        created.forEach((session) => releaseOrtSession(session));
        throw error;
      }
    }
    const encoder = created.get('encoder')!;
    const decoder = created.get('decoder')!;
    const joint = created.get('joint')!;
    if (this.disposed) {
      releaseOrtSession(encoder);
      releaseOrtSession(decoder);
      releaseOrtSession(joint);
      throw new Error(`GigaAM RNN-T executor was disposed during load for "${this.modelId}".`);
    }
    let tokenizer: GigaAmRnntTokenizer;
    try {
      tokenizer = await GigaAmRnntTokenizer.fromUrl(tokenizerUrl, this.signal);
    } catch (error) {
      releaseOrtSession(encoder);
      releaseOrtSession(decoder);
      releaseOrtSession(joint);
      throw error;
    }
    if (this.disposed) {
      releaseOrtSession(encoder);
      releaseOrtSession(decoder);
      releaseOrtSession(joint);
      throw new Error(`GigaAM RNN-T executor was disposed during load for "${this.modelId}".`);
    }
    return { ort, encoder, decoder, joint, tokenizer, warnings: [], backends };
  }

  async ready(): Promise<void> {
    if (!this.state) throw createExperimentalArtifactMissingError('gigaam-rnnt', this.modelId);
    await this.state;
  }

  async transcribe(input: AudioBufferLike, options: GigaAmRnntTranscriptionOptions = {}): Promise<GigaAmRnntNativeTranscript> {
    if (this.disposed) throw new Error(`GigaAM RNN-T executor is disposed for "${this.modelId}".`);
    const state = await this.state;
    if (!state) throw createExperimentalArtifactMissingError('gigaam-rnnt', this.modelId);
    const audio = normalizePcmInput(input).toMono();
    const started = nowMs();
    const preprocessStarted = nowMs();
    const prepared = this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    const preprocessMs = nowMs() - preprocessStarted;
    if (prepared.frameCount <= 0) return { utteranceText: '', isFinal: true, warnings: [...state.warnings] };

    const featureTensor = tensor(state.ort, 'float32', prepared.features, [1, this.config.nMels, prepared.frameCount]);
    const lengthTensor = tensor(state.ort, 'int64', BigInt64Array.from([BigInt(prepared.frameCount)]), [1]);
    let encoderOutputs: Record<string, OrtTensorLike>;
    const encodeStarted = nowMs();
    try {
      encoderOutputs = await state.encoder.run({ audio_signal: featureTensor, length: lengthTensor });
    } finally {
      featureTensor.dispose?.();
      lengthTensor.dispose?.();
    }

    let hidden = 0;
    let timeSteps = 0;
    let frames = 0;
    let encData!: Float32Array;
    try {
      const encoded = firstOutput(encoderOutputs, 'encoded');
      const encodedLength = firstOutput(encoderOutputs, 'encoded_len');
      const encDims = [...encoded.dims];
      if (encDims.length !== 3 || encDims[0] !== 1) {
        throw new Error(`Unexpected GigaAM RNN-T encoder shape: [${encDims.join(', ')}].`);
      }
      hidden = encDims[1] ?? 0;
      timeSteps = encDims[2] ?? 0;
      frames = Math.min(timeSteps, Number((encodedLength.data as unknown as ArrayLike<number | bigint>)[0] ?? timeSteps));
      encData = new Float32Array(tensorData(encoded));
    } finally {
      disposeOrtOutputs(encoderOutputs);
    }
    const encodeMs = nowMs() - encodeStarted;
    const blankId = state.tokenizer.blankId;
    const layers = Math.max(1, this.config.predictionRnnLayers ?? 1);
    const predHidden = this.config.predictionHiddenSize;
    const ids: number[] = [];
    const tokens: LasrCtcNativeToken[] = [];
    let lastLabel = blankId;
    let hasState = false;
    const h = new Float32Array(layers * predHidden);
    const c = new Float32Array(layers * predHidden);
    let decodeIterations = 0;
    const decodeStarted = nowMs();

    // Greedy RNN-T decode. The joint graph is row-parallel: its leading
    // dimension is the frame batch, so all remaining frames can be scored
    // speculatively against the current predictor state in one run. Blank
    // rows never change the predictor state, so the first non-blank row is
    // exactly what the per-frame sequential loop would emit. After an
    // emission the predictor advances and the suffix of frames INCLUDING the
    // emitting frame is re-batched, so a second token on the same frame is
    // still found. Frames that hit maxTokensPerFrame advance without further
    // scoring, matching the sequential loop. Predictor outputs for the
    // current (label, h, c) are cached, so a run of blank frames costs one
    // joint dispatch per frame and zero decoder dispatches. Graphs that
    // reject batched shapes latch batching off permanently and take the
    // identical sequential path.
    let frame = 0;
    let frameEmitted = 0;
    let joinerBatchRuns = 0;
    let predictor: {
      readonly label: number;
      readonly dec: Float32Array;
      readonly nextH: Float32Array;
      readonly nextC: Float32Array;
    } | undefined;

    const runPredictor = async (label: number): Promise<{ dec: Float32Array; nextH: Float32Array; nextC: Float32Array }> => {
      const target = tensor(state.ort, 'int64', BigInt64Array.from([BigInt(label)]), [1, 1]);
      const hTensor = tensor(state.ort, 'float32', hasState ? h.slice() : new Float32Array(layers * predHidden), [layers, 1, predHidden]);
      const cTensor = tensor(state.ort, 'float32', hasState ? c.slice() : new Float32Array(layers * predHidden), [layers, 1, predHidden]);
      let decoderOutputs: Record<string, OrtTensorLike>;
      try {
        decoderOutputs = await state.decoder.run({ x: target, hi: hTensor, ci: cTensor });
      } catch {
        decoderOutputs = await state.decoder.run({ x: target, 'h.1': hTensor, 'c.1': cTensor });
      } finally {
        target.dispose?.();
        hTensor.dispose?.();
        cTensor.dispose?.();
      }
      try {
        const decoderOut = firstOutput(decoderOutputs, 'dec');
        const nextHOutput = firstOutput(decoderOutputs, 'ho', 'h');
        const nextCOutput = firstOutput(decoderOutputs, 'co', 'c');
        return { dec: tensorData(decoderOut), nextH: tensorData(nextHOutput), nextC: tensorData(nextCOutput) };
      } finally {
        disposeOrtOutputs(decoderOutputs);
      }
    };

    const ensurePredictor = async (): Promise<NonNullable<typeof predictor>> => {
      const label = hasState ? lastLabel : blankId;
      if (!predictor || predictor.label !== label) {
        const run = await runPredictor(label);
        predictor = { label, ...run };
      }
      return predictor;
    };

    const gatherEncoderRow = (atFrame: number, target: Float32Array, offset: number): void => {
      for (let index = 0; index < hidden; index += 1) {
        target[offset + index] = encData[index * timeSteps + atFrame] ?? 0;
      }
    };

    const advancePredictor = (nextH: Float32Array, nextC: Float32Array, tokenId: number): void => {
      lastLabel = tokenId;
      hasState = true;
      h.set(nextH.subarray(0, h.length));
      c.set(nextC.subarray(0, c.length));
      predictor = undefined;
    };

    const pushToken = (atFrame: number, tokenId: number, logits: Float32Array, logitsOffset: number, vocab: number): void => {
      const confidence = confidenceFromLogits(logits.subarray(logitsOffset, logitsOffset + vocab), tokenId, vocab);
      ids.push(tokenId);
      const startTime = roundTimestampSeconds(atFrame * this.config.featureHopSeconds * this.config.rawStride);
      tokens.push({
        index: tokens.length,
        id: options.returnTokenIds ? tokenId : undefined,
        text: state.tokenizer.decodeTokenPiece(tokenId),
        startTime,
        endTime: roundTimestampSeconds((atFrame + 1) * this.config.featureHopSeconds * this.config.rawStride),
        confidence: roundMetric(confidence.confidence, 4),
        logitIndex: options.returnLogitIndices ? atFrame : undefined,
      });
    };

    while (frame < frames) {
      throwIfDecodeAborted(options.signal);
      if (frameEmitted >= this.config.maxTokensPerFrame) {
        frame += 1;
        frameEmitted = 0;
        continue;
      }
      const remaining = frames - frame;
      if (this.joinerBatchAllowed && this.joinerRowSize !== undefined && remaining > 1) {
        const rowSize = this.joinerRowSize;
        const width = Math.min(remaining, this.joinerBatchWidth, OrtGigaAmRnntExecutor.joinerBatchMaxRows);
        const current = await ensurePredictor();
        const encBatch = new Float32Array(width * hidden);
        for (let row = 0; row < width; row += 1) gatherEncoderRow(frame + row, encBatch, row * hidden);
        const decBatch = new Float32Array(width * current.dec.length);
        for (let row = 0; row < width; row += 1) decBatch.set(current.dec, row * current.dec.length);
        const encTensor = tensor(state.ort, 'float32', encBatch, [width, hidden, 1]);
        const decTensor = tensor(state.ort, 'float32', decBatch, [width, current.dec.length, 1]);
        let batchValues: Float32Array | undefined;
        let batchRowParallel = false;
        try {
          const batchOutputs = await state.joint.run({ enc: encTensor, dec: decTensor });
          const batchLogits = firstOutput(batchOutputs, 'joint');
          batchValues = tensorData(batchLogits);
          const batchDims = [...batchLogits.dims];
          batchRowParallel = batchDims.length >= 2
            && batchDims[0] === width
            && batchDims[batchDims.length - 1] === rowSize
            && batchValues.length >= width * rowSize;
          if (!batchRowParallel) this.joinerBatchAllowed = false;
          disposeOrtOutputs(batchOutputs);
        } catch (error) {
          if (error instanceof PipelineAbortedError) throw error;
          this.joinerBatchAllowed = false;
          batchValues = undefined;
        } finally {
          encTensor.dispose?.();
          decTensor.dispose?.();
        }
        if (batchValues && batchRowParallel) {
          joinerBatchRuns += 1;
          let emissionRow = -1;
          let emissionToken = blankId;
          for (let row = 0; row < width; row += 1) {
            // The shared argmax returns an absolute index; the row-local
            // token id is the offset within the flattened logits block.
            const tokenId = argmax(batchValues, row * rowSize, rowSize) - row * rowSize;
            if (tokenId !== blankId) {
              emissionRow = row;
              emissionToken = tokenId;
              break;
            }
          }
          if (emissionRow === -1) {
            // The whole window was blank: the predictor state is unchanged,
            // so these frames need no further scoring. Widen the window for
            // the next run (the blank run may continue) and keep going.
            decodeIterations += width;
            frame += width;
            frameEmitted = 0;
            this.joinerBatchWidth = Math.min(width * 2, OrtGigaAmRnntExecutor.joinerBatchMaxRows);
            continue;
          }
          this.joinerBatchWidth = OrtGigaAmRnntExecutor.joinerBatchMinRows;
          decodeIterations += emissionRow + 1;
          const emissionFrame = frame + emissionRow;
          pushToken(emissionFrame, emissionToken, batchValues, emissionRow * rowSize, rowSize);
          advancePredictor(current.nextH, current.nextC, emissionToken);
          frame = emissionFrame;
          frameEmitted = emissionRow === 0 ? frameEmitted + 1 : 1;
          continue;
        }
      }
      for (let emitted = frameEmitted; emitted < this.config.maxTokensPerFrame; emitted += 1) {
        throwIfDecodeAborted(options.signal);
        const current = await ensurePredictor();
        const encFrame = new Float32Array(hidden);
        gatherEncoderRow(frame, encFrame, 0);
        const encTensor = tensor(state.ort, 'float32', encFrame, [1, hidden, 1]);
        const decTensor = tensor(state.ort, 'float32', current.dec, [1, current.dec.length, 1]);
        let jointOutputs: Record<string, OrtTensorLike>;
        try {
          jointOutputs = await state.joint.run({ enc: encTensor, dec: decTensor });
        } finally {
          encTensor.dispose?.();
          decTensor.dispose?.();
        }
        let logits: Float32Array;
        let logitsDims: readonly number[];
        try {
          const logitsTensor = firstOutput(jointOutputs, 'joint');
          logits = tensorData(logitsTensor);
          logitsDims = [...logitsTensor.dims];
        } finally {
          disposeOrtOutputs(jointOutputs);
        }
        if (this.joinerRowSize === undefined && logitsDims.length >= 2 && logitsDims[0] === 1) {
          this.joinerRowSize = logitsDims[logitsDims.length - 1];
        }
        const vocab = logits.length;
        const tokenId = argmax(logits, 0, vocab);
        decodeIterations += 1;
        if (tokenId === blankId) break;
        pushToken(frame, tokenId, logits, 0, vocab);
        advancePredictor(current.nextH, current.nextC, tokenId);
      }
      frame += 1;
      frameEmitted = 0;
    }

    const finished = nowMs();
    const totalMs = finished - started;
    const decodeMs = finished - decodeStarted;
    const componentBackends = state.backends ?? resolveGigaAmRnntBackends(undefined, this.backendId);
    return {
      utteranceText: state.tokenizer.decode(ids),
      isFinal: true,
      tokens,
      confidence: { tokenAverage: tokens.length ? tokens.reduce((sum, item) => sum + (item.confidence ?? 0), 0) / tokens.length : 0 },
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        totalMs: roundMetric(totalMs),
        wallMs: roundMetric(totalMs),
        audioDurationSec: roundMetric(audio.durationSeconds, 4),
        rtf: audio.durationSeconds ? roundMetric(totalMs / (audio.durationSeconds * 1000), 4) : 0,
        rtfx: audio.durationSeconds ? roundMetric(audio.durationSeconds / (totalMs / 1000), 4) : undefined,
        preprocessorBackend: 'js',
        encoderBackend: componentBackends.encoderBackend,
        decoderBackend: componentBackends.decoderBackend,
        jointBackend: componentBackends.jointBackend,
        encoderFrameCount: frames,
        decodeIterations,
        joinerBatchRuns,
        emittedTokenCount: tokens.length,
      },
      warnings: [...state.warnings],
    };
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    if (this.state) {
      try {
        const loaded = await this.state;
        releaseOrtSession(loaded.encoder);
        releaseOrtSession(loaded.decoder);
        releaseOrtSession(loaded.joint);
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.handles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
