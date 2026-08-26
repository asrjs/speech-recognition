import { argmax } from '../../inference/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import type { AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks } from '../../types/index.js';
import { createOrtSession, initOrt, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import type { XAsrArtifactSource, XAsrModelConfig, XAsrModelOptions, XAsrNativeTranscript, XAsrStateTensorSpec, XAsrTranscriptionOptions } from './types.js';
import { XAsrJsFrontend } from './frontend.js';
import { XAsrTokenizer } from './tokenizer.js';

export interface XAsrStreamState {
  readonly audio: Float32Array;
  readonly features: Float32Array;
  readonly encodedFrames: number;
  readonly inputFrames: number;
  readonly tokenIds: readonly number[];
  readonly encoderStates: readonly OrtTensorLike[];
}

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly encoder: OrtSessionLike;
  readonly decoder: OrtSessionLike;
  readonly joiner: OrtSessionLike;
  readonly tokenizer: XAsrTokenizer;
}

function urlFor(repo: string, revision: string, file: string, subfolder?: string): string {
  const parts = [repo, revision, subfolder, file].filter(Boolean).join('/');
  return `https://huggingface.co/${parts.split('/').map(encodeURIComponent).join('/')}`
    .replace(`/${encodeURIComponent(revision)}/`, `/resolve/${encodeURIComponent(revision)}/`);
}

function first(outputs: Record<string, OrtTensorLike>, preferred?: string): OrtTensorLike {
  if (preferred && outputs[preferred]) return outputs[preferred]!;
  const value = Object.values(outputs)[0];
  if (!value) throw new Error('X-ASR graph returned no outputs.');
  return value;
}

function readFloat(value: OrtTensorLike): Float32Array {
  if (value.type !== 'float16' && value.data instanceof Float32Array) return value.data;
  const source = value.data as unknown as ArrayLike<number>;
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const bits = Number(source[i] ?? 0);
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;
    out[i] = exponent === 0 ? sign * (mantissa / 1024) * 2 ** -14 : exponent === 31 ? (mantissa ? NaN : sign * Infinity) : sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return out;
}

function cloneState(ort: OrtModuleLike, spec: XAsrStateTensorSpec): OrtTensorLike {
  const data = spec.type === 'float16' ? new Uint16Array(spec.dims.reduce((a, b) => a * b, 1)) : spec.type === 'int64' ? new BigInt64Array(spec.dims.reduce((a, b) => a * b, 1)) : spec.type === 'int32' ? new Int32Array(spec.dims.reduce((a, b) => a * b, 1)) : new Float32Array(spec.dims.reduce((a, b) => a * b, 1));
  return new ort.Tensor(spec.type, data, spec.dims);
}

export interface XAsrExecutor {
  ready(): Promise<void>;
  transcribe(audio: AudioBufferLike, options?: XAsrTranscriptionOptions): Promise<XAsrNativeTranscript>;
  createStream(): XAsrStreamState;
  pushStream(state: XAsrStreamState, audio: Float32Array, final?: boolean, options?: XAsrTranscriptionOptions): Promise<{ state: XAsrStreamState; transcript: XAsrNativeTranscript }>;
  dispose(): void;
}

export class OrtXAsrExecutor implements XAsrExecutor {
  private readonly source?: XAsrArtifactSource;
  private readonly provider?: AssetProvider;
  private readonly hooks?: SpeechRuntimeHooks;
  private readonly handles: ResolvedAssetHandle[] = [];
  private readonly frontend = new XAsrJsFrontend();
  private readonly state?: Promise<LoadedState>;

  constructor(private readonly modelId: string, private readonly backendId: string, private readonly config: XAsrModelConfig, options?: XAsrModelOptions, dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks } = {}) {
    this.source = options?.source; this.provider = dependencies.assetProvider; this.hooks = dependencies.runtimeHooks;
    if (this.source) this.state = this.initialize();
  }

  private async resolve(source: Extract<XAsrArtifactSource, { kind: 'huggingface' }>, file: string): Promise<string> {
    const revision = source.revision ?? 'main';
    if (!this.provider) return urlFor(source.repoId, revision, file, source.subfolder);
    const handle = await this.provider.resolve({ id: `huggingface:${source.repoId}:${revision}:${file}`, provider: 'huggingface', repoId: source.repoId, revision, filename: source.subfolder ? `${source.subfolder}/${file}` : file, cacheKey: `huggingface:${source.repoId}:${revision}:${file}`, onProgress: (event) => this.hooks?.onProgress?.({ phase: 'asset:download', modelId: this.modelId, file, loaded: event.loaded, total: event.total, percent: event.total ? Math.round(event.loaded / event.total * 100) : event.done ? 100 : undefined, isComplete: event.done, message: event.done ? `Prepared ${file}.` : `Downloading ${file}.` }) });
    this.handles.push(handle); const locator = await handle.getLocator('url');
    if (!locator) throw new Error(`Could not create a URL locator for "${file}".`);
    return locator;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw new Error(`No X-ASR artifact source is configured for "${this.modelId}".`);
    let encoderUrl: string; let decoderUrl: string; let joinerUrl: string; let tokenizerUrl: string;
    let encoderDataUrl: string | undefined; let decoderDataUrl: string | undefined; let joinerDataUrl: string | undefined;
    let encoderDataFilename: string | undefined; let decoderDataFilename: string | undefined; let joinerDataFilename: string | undefined;
    if (this.source.kind === 'direct') {
      ({ encoderUrl, decoderUrl, joinerUrl, tokenizerUrl, encoderDataUrl, decoderDataUrl, joinerDataUrl, encoderDataFilename, decoderDataFilename, joinerDataFilename } = this.source.artifacts);
    } else {
      encoderUrl = await this.resolve(this.source, this.source.encoderFilename ?? 'encoder-160ms.onnx');
      decoderUrl = await this.resolve(this.source, this.source.decoderFilename ?? 'decoder-160ms.onnx');
      joinerUrl = await this.resolve(this.source, this.source.joinerFilename ?? 'joiner-160ms.onnx');
      tokenizerUrl = await this.resolve(this.source, this.source.tokenizerFilename ?? 'tokens.txt');
      encoderDataFilename = this.source.encoderDataFilename; decoderDataFilename = this.source.decoderDataFilename; joinerDataFilename = this.source.joinerDataFilename;
      if (encoderDataFilename) encoderDataUrl = await this.resolve(this.source, encoderDataFilename);
      if (decoderDataFilename) decoderDataUrl = await this.resolve(this.source, decoderDataFilename);
      if (joinerDataFilename) joinerDataUrl = await this.resolve(this.source, joinerDataFilename);
    }
    const ort = await initOrt(this.backendId, { cpuThreads: this.source.cpuThreads, wasmPaths: this.source.wasmPaths });
    const backend = this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm';
    const make = (url: string, dataUrl?: string, dataPath?: string) => createOrtSession(ort, url, { backendId: backend, enableProfiling: this.source?.enableProfiling, externalDataUrl: dataUrl, externalDataPath: dataPath });
    const [encoder, decoder, joiner, tokenizer] = await Promise.all([make(encoderUrl, encoderDataUrl, encoderDataFilename), make(decoderUrl, decoderDataUrl, decoderDataFilename), make(joinerUrl, joinerDataUrl, joinerDataFilename), XAsrTokenizer.fromUrl(tokenizerUrl)]);
    return { ort, encoder, decoder, joiner, tokenizer };
  }

  async ready(): Promise<void> { if (!this.state) throw new Error(`No X-ASR artifact source is configured for "${this.modelId}".`); await this.state; }

  createStream(): XAsrStreamState {
    if (!this.source) throw new Error(`No X-ASR artifact source is configured for "${this.modelId}".`);
    return { audio: new Float32Array(0), features: new Float32Array(0), encodedFrames: 0, inputFrames: 0, tokenIds: [], encoderStates: this.config.graph.encoderStateInputs.map(() => undefined as unknown as OrtTensorLike) as OrtTensorLike[] };
  }

  private async decodeFeatures(state: XAsrStreamState, features: Float32Array, final: boolean): Promise<XAsrStreamState> {
    const loaded = await this.state!; const graph = this.config.graph;
    let pending = features; let nextState = [...state.encoderStates]; let encodedFrames = state.encodedFrames; const tokens = [...state.tokenIds];
    while (pending.length >= graph.encoderFrameSize * this.config.featureDim || (final && pending.length > 0)) {
      const needed = graph.encoderFrameSize * this.config.featureDim; const chunk = new Float32Array(needed); chunk.set(pending.subarray(0, Math.min(needed, pending.length))); pending = pending.subarray(Math.min(needed, pending.length));
      const featureTensor = new loaded.ort.Tensor('float32', chunk, [1, graph.encoderFrameSize, this.config.featureDim]);
      const feeds: Record<string, unknown> = { [graph.featureInputName ?? 'features']: featureTensor };
      graph.encoderStateInputs.forEach((spec, index) => { feeds[spec.name] = nextState[index] ?? cloneState(loaded.ort, spec); });
      let output: Record<string, OrtTensorLike>;
      try { output = await loaded.encoder.run(feeds); } finally { featureTensor.dispose?.(); graph.encoderStateInputs.forEach((spec, index) => { if (!nextState[index] && feeds[spec.name]) (feeds[spec.name] as OrtTensorLike).dispose?.(); }); }
      const encoded = first(output, graph.encoderOutputName); const encodedData = readFloat(encoded); const dims = [...encoded.dims]; const frames = dims[1] ?? 0; const hidden = dims[2] ?? (frames ? encodedData.length / frames : 0);
      const outputStates = graph.encoderStateOutputs ?? Object.keys(output).filter((name) => name !== (graph.encoderOutputName ?? Object.keys(output)[0]));
      nextState = outputStates
        .map((name) => output[name])
        .filter((value): value is OrtTensorLike => Boolean(value));
      const decoderInput = graph.decoderInputName ?? Object.keys(loaded.decoder.inputMetadata ?? {})[0] ?? 'y';
      const decoderOutputName = graph.decoderOutputName;
      const context = new Int32Array(graph.decoderContextSize); context.fill(0); context.set(tokens.slice(-graph.decoderContextSize));
      const decoderTensor = new loaded.ort.Tensor('int32', context, [1, graph.decoderContextSize]); const decoderOutput = await loaded.decoder.run({ [decoderInput]: decoderTensor }); decoderTensor.dispose?.(); let decoderValue = first(decoderOutput, decoderOutputName); let decoderData = readFloat(decoderValue);
      for (let frame = 0; frame < frames; frame += 1) {
        const enc = new Float32Array(hidden); enc.set(encodedData.subarray(frame * hidden, frame * hidden + hidden));
        const encTensor = new loaded.ort.Tensor('float32', enc, [1, 1, hidden]); const decTensor = new loaded.ort.Tensor('float32', decoderData, decoderValue.dims); const joinerOutput = await loaded.joiner.run({ [graph.joinerEncoderInputName ?? Object.keys(loaded.joiner.inputMetadata ?? {})[0] ?? 'encoder_out']: encTensor, [graph.joinerDecoderInputName ?? Object.keys(loaded.joiner.inputMetadata ?? {})[1] ?? 'decoder_out']: decTensor }); encTensor.dispose?.(); decTensor.dispose?.(); const logits = first(joinerOutput, graph.joinerOutputName); const values = readFloat(logits); const id = argmax(values, 0, values.length); logits.dispose?.(); if (id !== 0) { tokens.push(id); decoderValue.dispose?.(); const nextContext = new Int32Array(graph.decoderContextSize); nextContext.fill(0); nextContext.set(tokens.slice(-graph.decoderContextSize)); const nextTensor = new loaded.ort.Tensor('int32', nextContext, [1, graph.decoderContextSize]); const nextOutput = await loaded.decoder.run({ [decoderInput]: nextTensor }); nextTensor.dispose?.(); decoderValue = first(nextOutput, decoderOutputName); decoderData = readFloat(decoderValue); Object.values(nextOutput).forEach((value) => { if (value !== decoderValue) value.dispose?.(); }); }
      }
      decoderValue.dispose?.();
      encoded.dispose?.(); Object.values(output).forEach((value) => { if (!nextState.includes(value) && value !== encoded) value.dispose?.(); }); encodedFrames += frames;
    }
    return { audio: state.audio, features: pending, encodedFrames, inputFrames: state.inputFrames, tokenIds: tokens, encoderStates: nextState };
  }

  async pushStream(state: XAsrStreamState, audio: Float32Array, final = false, options: XAsrTranscriptionOptions = {}): Promise<{ state: XAsrStreamState; transcript: XAsrNativeTranscript }> {
    const started = nowMs(); const allAudio = new Float32Array(state.audio.length + audio.length); allAudio.set(state.audio); allAudio.set(audio, state.audio.length); const allFeatures = this.frontend.process(allAudio); const newFeatures = allFeatures.subarray(state.inputFrames * this.config.featureDim); const combined = new Float32Array(state.features.length + newFeatures.length); combined.set(state.features); combined.set(newFeatures, state.features.length); const next = await this.decodeFeatures({ ...state, audio: allAudio }, combined, final); const ids = [...next.tokenIds]; const tokenizer = (await this.state!).tokenizer; const transcript: XAsrNativeTranscript = { utteranceText: tokenizer.decode(ids), isFinal: final, tokens: options.returnTokenIds ? ids.map((id, index) => ({ index, id, text: tokenizer.decodeTokenPiece(id), startTime: roundTimestampSeconds(index * this.config.featureHopSeconds), endTime: roundTimestampSeconds((index + 1) * this.config.featureHopSeconds) })) : undefined, metrics: { preprocessMs: 0, encodeMs: roundMetric(nowMs() - started), decodeMs: 0, totalMs: roundMetric(nowMs() - started), wallMs: roundMetric(nowMs() - started), audioDurationSec: roundMetric(allAudio.length / this.config.sampleRate, 4), encoderFrameCount: next.encodedFrames, emittedTokenCount: ids.length, preprocessorBackend: 'js' }, warnings: [] }; return { state: { ...next, features: next.features, inputFrames: allFeatures.length / this.config.featureDim }, transcript };
  }

  async transcribe(audio: AudioBufferLike, options: XAsrTranscriptionOptions = {}): Promise<XAsrNativeTranscript> { const state = this.createStream(); const pcm = audio.channels?.[0] ?? (audio.data instanceof Float32Array ? audio.data : Float32Array.from(audio.data ?? [])); return (await this.pushStream(state, pcm, true, options)).transcript; }
  dispose(): void { this.handles.forEach((handle) => void handle.dispose()); }
}
