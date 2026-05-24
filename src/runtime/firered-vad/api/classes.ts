import type {
  CmvnStats,
  FireRedAedConfig,
  FireRedAedDetectResult,
  FireRedFrameResult,
  FireRedRuntimeOptions,
  FireRedStreamVadConfig,
  FireRedVadConfig,
  FireRedVadDetectResult,
  FireredVadStreamPackedCreateOptions,
  NormalizedFireRedAedConfig,
  NormalizedFireRedStreamVadConfig,
  NormalizedFireRedVadConfig,
  StreamVadFrameResult,
} from '../types.js';
import { AudioFeat, StreamingPackedAudioFeat } from '../core/audio-feat.js';
import { normalizeAudioInput, type AudioInput } from '../core/audio-input.js';
import {
  createOrtFireRedBackend,
  createZeroStreamCache,
  flattenFeatFrames,
  type FireRedBackend,
} from '../core/backend.js';
import { loadDefaultCmvn, cmvnFromArrays } from '../core/cmvn.js';
import { normalizeAedConfig, normalizeStreamVadConfig, normalizeVadConfig } from '../core/config.js';
import {
  DEFAULT_MODEL_URLS,
  FIRERED_GITHUB_RAW_ONNX_BASE,
  FRAME_LENGTH_SAMPLE,
  FRAME_PER_SECONDS,
} from '../core/constants.js';
import { loadBinaryResource, resolveModelUrl } from '../core/loader.js';
import { StreamVadPostprocessor } from '../core/stream-postprocessor.js';
import { roundTo } from '../core/util.js';
import { VadPostprocessor } from '../core/vad-postprocessor.js';

interface RuntimeInit {
  readonly backend: FireRedBackend;
  readonly cmvn: CmvnStats;
}

const DEFAULT_PRETRAINED_DIR = FIRERED_GITHUB_RAW_ONNX_BASE;

async function loadCmvnFromRuntimeOptions(
  runtimeOptions: FireRedRuntimeOptions,
  modelDir?: string,
): Promise<CmvnStats> {
  if (runtimeOptions.cmvn) {
    return cmvnFromArrays(runtimeOptions.cmvn.means, runtimeOptions.cmvn.istd);
  }
  const explicitUrl = runtimeOptions.modelUrls?.cmvnJsonUrl;
  if (explicitUrl) {
    try {
      const bytes = await loadBinaryResource(explicitUrl);
      const text = new TextDecoder().decode(bytes);
      const parsed = JSON.parse(text) as { means: number[]; istd: number[] };
      return cmvnFromArrays(parsed.means, parsed.istd);
    } catch {
      // fallback to defaults below
    }
  }
  if (modelDir) {
    const maybeCmvnUrl = resolveModelUrl(modelDir, 'cmvn.json');
    try {
      const bytes = await loadBinaryResource(maybeCmvnUrl);
      const text = new TextDecoder().decode(bytes);
      const parsed = JSON.parse(text) as { means: number[]; istd: number[] };
      return cmvnFromArrays(parsed.means, parsed.istd);
    } catch {
      // use default cmvn below
    }
  }
  return loadDefaultCmvn();
}

function withModelDirOverrides(
  runtimeOptions: FireRedRuntimeOptions,
  modelDir?: string,
): FireRedRuntimeOptions {
  if (!modelDir) {
    return runtimeOptions;
  }
  return {
    ...runtimeOptions,
    modelUrls: {
      ...runtimeOptions.modelUrls,
      vadUrl: runtimeOptions.modelUrls?.vadUrl ?? resolveModelUrl(modelDir, 'fireredvad_vad.onnx'),
      streamVadWithCacheUrl:
        runtimeOptions.modelUrls?.streamVadWithCacheUrl ??
        resolveModelUrl(modelDir, 'fireredvad_stream_vad_with_cache.onnx'),
      aedUrl: runtimeOptions.modelUrls?.aedUrl ?? resolveModelUrl(modelDir, 'fireredvad_aed.onnx'),
      cmvnJsonUrl: runtimeOptions.modelUrls?.cmvnJsonUrl ?? resolveModelUrl(modelDir, 'cmvn.json'),
    },
  };
}

async function initRuntime(
  runtimeOptions: FireRedRuntimeOptions,
  modelDir?: string,
): Promise<RuntimeInit> {
  const merged = withModelDirOverrides(runtimeOptions, modelDir);
  const backend = await createOrtFireRedBackend(merged);
  const cmvn = await loadCmvnFromRuntimeOptions(merged, modelDir);
  return { backend, cmvn };
}

function makeStreamResult(
  frameIdx: number,
  confidence: number,
  isSpeech: boolean,
): FireRedFrameResult {
  return {
    confidence,
    is_speech: isSpeech,
    isSpeech,
    frame_offset: frameIdx,
    frameOffset: frameIdx,
  };
}

function probsToNumberList(probs: Float32Array): number[] {
  return Array.from(probs, (value) => value);
}

function withOptionalWavPath<T extends object>(value: T, wavPath?: string): T {
  if (!wavPath) {
    return value;
  }
  return {
    ...value,
    wav_path: wavPath,
    wavPath,
  } as T;
}

function mergeChunkProbabilities(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

class FireRedBase {
  protected readonly backend: FireRedBackend;
  protected readonly cmvn: CmvnStats;

  protected constructor(runtime: RuntimeInit) {
    this.backend = runtime.backend;
    this.cmvn = runtime.cmvn;
  }

  async dispose(): Promise<void> {
    await this.backend.dispose();
  }
}

export class FireRedStreamVad extends FireRedBase {
  readonly config: NormalizedFireRedStreamVadConfig;
  private readonly audioFeat: AudioFeat;
  private readonly postprocessor: StreamVadPostprocessor;
  private modelCaches = createZeroStreamCache();

  private constructor(runtime: RuntimeInit, config: NormalizedFireRedStreamVadConfig) {
    super(runtime);
    this.config = config;
    this.audioFeat = new AudioFeat(runtime.cmvn);
    this.postprocessor = new StreamVadPostprocessor({
      smoothWindowSize: config.smooth_window_size,
      speechThreshold: config.speech_threshold,
      padStartFrame: config.pad_start_frame,
      minSpeechFrame: config.min_speech_frame,
      maxSpeechFrame: config.max_speech_frame,
      minSilenceFrame: config.min_silence_frame,
    });
  }

  static async from_pretrained(
    model_dir = DEFAULT_PRETRAINED_DIR,
    config: FireRedStreamVadConfig = {},
  ): Promise<FireRedStreamVad> {
    const normalized = normalizeStreamVadConfig(config);
    const runtime = await initRuntime(normalized, model_dir);
    return new FireRedStreamVad(runtime, normalized);
  }

  static async fromPretrained(
    modelDir = DEFAULT_PRETRAINED_DIR,
    config: FireRedStreamVadConfig = {},
  ): Promise<FireRedStreamVad> {
    return FireRedStreamVad.from_pretrained(modelDir, config);
  }

  reset(): void {
    this.modelCaches.fill(0);
    this.audioFeat.reset();
    this.postprocessor.reset();
  }

  set_mode(mode = 0): void {
    if (mode === 0) {
      this.config.speech_threshold = 0.3;
      this.config.min_speech_frame = 8;
      this.config.min_silence_frame = 20;
    } else if (mode === 1) {
      this.config.speech_threshold = 0.5;
      this.config.min_speech_frame = 10;
      this.config.min_silence_frame = 15;
    } else if (mode === 2) {
      this.config.speech_threshold = 0.7;
      this.config.min_speech_frame = 15;
      this.config.min_silence_frame = 10;
    } else if (mode === 3) {
      this.config.speech_threshold = 0.9;
      this.config.min_speech_frame = 20;
      this.config.min_silence_frame = 5;
    }
    this.postprocessor.speechThreshold = this.config.speech_threshold;
    this.postprocessor.minSpeechFrame = this.config.min_speech_frame;
    this.postprocessor.minSilenceFrame = this.config.min_silence_frame;
  }

  setMode(mode = 0): void {
    this.set_mode(mode);
  }

  async detect_frame(audio_frame: ArrayLike<number>): Promise<StreamVadFrameResult> {
    if (audio_frame.length !== FRAME_LENGTH_SAMPLE) {
      throw new RangeError(`Expected ${FRAME_LENGTH_SAMPLE} samples, got ${audio_frame.length}.`);
    }
    const frames = this.audioFeat.extract(audio_frame);
    if (frames.length === 0) {
      return this.postprocessor.processOneFrame(0);
    }
    const output = await this.backend.runStream(flattenFeatFrames(frames), this.modelCaches);
    this.modelCaches = output.caches.slice();
    return this.postprocessor.processOneFrame(output.probs[0] ?? 0);
  }

  async detectFrame(audioFrame: ArrayLike<number>): Promise<StreamVadFrameResult> {
    return this.detect_frame(audioFrame);
  }

  async detect_chunk(audio_chunk: ArrayLike<number>): Promise<StreamVadFrameResult[]> {
    const frames = this.audioFeat.extract(audio_chunk);
    if (frames.length === 0) {
      return [];
    }
    const output = await this.backend.runStream(flattenFeatFrames(frames), this.modelCaches);
    this.modelCaches = output.caches.slice();
    const probs = probsToNumberList(output.probs);
    return probs.map((prob) => this.postprocessor.processOneFrame(prob));
  }

  async detectChunk(audioChunk: ArrayLike<number>): Promise<StreamVadFrameResult[]> {
    return this.detect_chunk(audioChunk);
  }

  static results_to_timestamps(results: StreamVadFrameResult[]): Array<[number, number]> {
    const sorted = [...results].sort((a, b) => a.frame_idx - b.frame_idx);
    const frameTimestamps: Array<[number, number]> = [];
    let start = -1;
    let end = -1;

    for (const result of sorted) {
      if (result.is_speech_start) {
        start = Math.max(0, result.speech_start_frame - 1);
        end = -1;
      } else if (result.is_speech_end) {
        end = Math.max(0, result.speech_end_frame - 1);
        frameTimestamps.push([start, end]);
        start = -1;
        end = -1;
      }
    }

    if (start !== -1 && sorted.length > 0) {
      end = sorted[sorted.length - 1]!.frame_idx - 1;
      frameTimestamps.push([start, end]);
    }

    return frameTimestamps.map(([s, e]) => [roundTo(s / FRAME_PER_SECONDS, 3), roundTo(e / FRAME_PER_SECONDS, 3)]);
  }

  static resultsToTimestamps(results: StreamVadFrameResult[]): Array<[number, number]> {
    return FireRedStreamVad.results_to_timestamps(results);
  }

  async detect_full(
    audio: AudioInput,
  ): Promise<[StreamVadFrameResult[], FireRedVadDetectResult]> {
    this.reset();
    const normalized = await normalizeAudioInput(audio);
    const frames = this.audioFeat.extract(normalized.pcm16);
    const frameResults: StreamVadFrameResult[] = [];
    if (frames.length > 0) {
      const output = await this.backend.runStream(flattenFeatFrames(frames), this.modelCaches);
      const probs = probsToNumberList(output.probs);
      for (const prob of probs) {
        frameResults.push(this.postprocessor.processOneFrame(prob));
      }
    }
    const duration = normalized.pcm16.length / normalized.sampleRate;
    const timestamps = FireRedStreamVad.results_to_timestamps(frameResults);
    this.reset();
    return [
      frameResults,
      withOptionalWavPath(
        {
        dur: roundTo(duration, 3),
        timestamps,
        },
        normalized.wavPath,
      ),
    ];
  }

  async detectFull(audio: AudioInput): Promise<[StreamVadFrameResult[], FireRedVadDetectResult]> {
    return this.detect_full(audio);
  }
}

export class FireRedVad extends FireRedBase {
  readonly config: NormalizedFireRedVadConfig;
  private readonly audioFeat: AudioFeat;
  private readonly postprocessor: VadPostprocessor;

  private constructor(runtime: RuntimeInit, config: NormalizedFireRedVadConfig) {
    super(runtime);
    this.config = config;
    this.audioFeat = new AudioFeat(runtime.cmvn);
    this.postprocessor = new VadPostprocessor({
      smoothWindowSize: config.smooth_window_size,
      probThreshold: config.speech_threshold,
      minSpeechFrame: config.min_speech_frame,
      maxSpeechFrame: config.max_speech_frame,
      minSilenceFrame: config.min_silence_frame,
      mergeSilenceFrame: config.merge_silence_frame,
      extendSpeechFrame: config.extend_speech_frame,
    });
  }

  static async from_pretrained(
    model_dir = DEFAULT_PRETRAINED_DIR,
    config: FireRedVadConfig = {},
  ): Promise<FireRedVad> {
    const normalized = normalizeVadConfig(config);
    const runtime = await initRuntime(normalized, model_dir);
    return new FireRedVad(runtime, normalized);
  }

  static async fromPretrained(
    modelDir = DEFAULT_PRETRAINED_DIR,
    config: FireRedVadConfig = {},
  ): Promise<FireRedVad> {
    return FireRedVad.from_pretrained(modelDir, config);
  }

  async detect(
    audio: AudioInput,
    do_postprocess = true,
  ): Promise<[FireRedVadDetectResult | null, Float32Array]> {
    const normalized = await normalizeAudioInput(audio);
    const frames = this.audioFeat.extract(normalized.pcm16);
    if (frames.length === 0) {
      return [
        withOptionalWavPath(
          {
          dur: 0,
          timestamps: [],
          },
          normalized.wavPath,
        ),
        new Float32Array(0),
      ];
    }

    const chunkMax = this.config.chunk_max_frame;
    const chunkProbs: Float32Array[] = [];
    for (let start = 0; start < frames.length; start += chunkMax) {
      const chunk = frames.slice(start, Math.min(frames.length, start + chunkMax));
      const output = await this.backend.runVad(flattenFeatFrames(chunk));
      chunkProbs.push(output.probs);
    }
    const probs = mergeChunkProbabilities(chunkProbs);
    if (!do_postprocess) {
      return [null, probs];
    }

    const duration = normalized.pcm16.length / normalized.sampleRate;
    const decisions = this.postprocessor.process(probsToNumberList(probs));
    return [
      withOptionalWavPath(
        {
        dur: roundTo(duration, 3),
        timestamps: this.postprocessor.decisionToSegment(decisions, duration),
        },
        normalized.wavPath,
      ),
      probs,
    ];
  }
}

export class FireRedAed extends FireRedBase {
  static readonly IDX2EVENT: Record<number, 'speech' | 'singing' | 'music'> = {
    0: 'speech',
    1: 'singing',
    2: 'music',
  };

  readonly config: NormalizedFireRedAedConfig;
  private readonly audioFeat: AudioFeat;
  private readonly eventPostprocessors: Record<string, VadPostprocessor>;

  private constructor(runtime: RuntimeInit, config: NormalizedFireRedAedConfig) {
    super(runtime);
    this.config = config;
    this.audioFeat = new AudioFeat(runtime.cmvn);
    this.eventPostprocessors = {
      speech: new VadPostprocessor({
        smoothWindowSize: config.smooth_window_size,
        probThreshold: config.speech_threshold,
        minSpeechFrame: config.min_event_frame,
        maxSpeechFrame: config.max_event_frame,
        minSilenceFrame: config.min_silence_frame,
        mergeSilenceFrame: config.merge_silence_frame,
        extendSpeechFrame: config.extend_speech_frame,
      }),
      singing: new VadPostprocessor({
        smoothWindowSize: config.smooth_window_size,
        probThreshold: config.singing_threshold,
        minSpeechFrame: config.min_event_frame,
        maxSpeechFrame: config.max_event_frame,
        minSilenceFrame: config.min_silence_frame,
        mergeSilenceFrame: config.merge_silence_frame,
        extendSpeechFrame: config.extend_speech_frame,
      }),
      music: new VadPostprocessor({
        smoothWindowSize: config.smooth_window_size,
        probThreshold: config.music_threshold,
        minSpeechFrame: config.min_event_frame,
        maxSpeechFrame: config.max_event_frame,
        minSilenceFrame: config.min_silence_frame,
        mergeSilenceFrame: config.merge_silence_frame,
        extendSpeechFrame: config.extend_speech_frame,
      }),
    };
  }

  static async from_pretrained(
    model_dir = DEFAULT_PRETRAINED_DIR,
    config: FireRedAedConfig = {},
  ): Promise<FireRedAed> {
    const normalized = normalizeAedConfig(config);
    const runtime = await initRuntime(normalized, model_dir);
    return new FireRedAed(runtime, normalized);
  }

  static async fromPretrained(
    modelDir = DEFAULT_PRETRAINED_DIR,
    config: FireRedAedConfig = {},
  ): Promise<FireRedAed> {
    return FireRedAed.from_pretrained(modelDir, config);
  }

  async detect(audio: AudioInput): Promise<[FireRedAedDetectResult, Float32Array]> {
    const normalized = await normalizeAudioInput(audio);
    const frames = this.audioFeat.extract(normalized.pcm16);
    if (frames.length === 0) {
      return [
        withOptionalWavPath(
          {
          dur: 0,
          event2timestamps: { speech: [], singing: [], music: [] },
          event2ratio: { speech: 0, singing: 0, music: 0 },
          },
          normalized.wavPath,
        ),
        new Float32Array(0),
      ];
    }

    const chunkMax = this.config.chunk_max_frame;
    const chunkProbs: Float32Array[] = [];
    for (let start = 0; start < frames.length; start += chunkMax) {
      const chunk = frames.slice(start, Math.min(frames.length, start + chunkMax));
      const output = await this.backend.runAed(flattenFeatFrames(chunk));
      chunkProbs.push(output.probs);
    }
    const probs = mergeChunkProbabilities(chunkProbs);
    const duration = normalized.pcm16.length / normalized.sampleRate;
    const event2timestamps: Record<string, Array<[number, number]>> = {};
    const event2ratio: Record<string, number> = {};

    for (const [idxRaw, event] of Object.entries(FireRedAed.IDX2EVENT)) {
      const idx = Number(idxRaw);
      const eventProbs: number[] = [];
      for (let t = 0; t < frames.length; t += 1) {
        eventProbs.push(probs[t * 3 + idx] ?? 0);
      }
      const post = this.eventPostprocessors[event];
      if (!post) {
        continue;
      }
      const decisions = post.process(eventProbs);
      event2timestamps[event] = post.decisionToSegment(decisions, duration);
      const threshold = this.config[`${event}_threshold` as keyof NormalizedFireRedAedConfig] as number;
      const ratio =
        eventProbs.length === 0
          ? 0
          : eventProbs.filter((prob) => prob >= threshold).length / eventProbs.length;
      event2ratio[event] = roundTo(ratio, 3);
    }

    return [
      withOptionalWavPath(
        {
        dur: roundTo(duration, 3),
        event2timestamps,
        event2ratio,
        },
        normalized.wavPath,
      ),
      probs,
    ];
  }
}

export class FireredVadStreamPacked extends FireRedBase {
  private readonly audioFeat: StreamingPackedAudioFeat;
  private modelCaches = createZeroStreamCache();
  private audioBuffer: number[] = [];
  private frameOffset = 0;
  private threshold = 0.5;

  private constructor(runtime: RuntimeInit, threshold: number) {
    super(runtime);
    this.audioFeat = new StreamingPackedAudioFeat(runtime.cmvn);
    this.threshold = threshold;
  }

  static async create(options: FireredVadStreamPackedCreateOptions = {}): Promise<FireredVadStreamPacked> {
    const modelUrl = options.modelUrl ?? DEFAULT_MODEL_URLS.streamVadWithCacheUrl;
    const base = modelUrl.replace(/[\\/][^\\/]+$/, '/');
    const runtimeOptions: FireRedRuntimeOptions = {
      ...options,
      modelUrls: {
        ...(options.modelUrls ?? {}),
        streamVadWithCacheUrl: modelUrl,
        vadUrl: options.modelUrls?.vadUrl ?? `${base}fireredvad_vad.onnx`,
        aedUrl: options.modelUrls?.aedUrl ?? `${base}fireredvad_aed.onnx`,
        cmvnJsonUrl: options.modelUrls?.cmvnJsonUrl ?? options.cmvnJsonUrl,
      },
    };
    const runtime = await initRuntime(runtimeOptions);
    return new FireredVadStreamPacked(runtime, options.threshold ?? 0.5);
  }

  static async firered_vad_create(
    options: FireredVadStreamPackedCreateOptions = {},
  ): Promise<FireredVadStreamPacked> {
    return FireredVadStreamPacked.create(options);
  }

  async process_stream(audio_data: ArrayLike<number>): Promise<FireRedFrameResult> {
    for (let i = 0; i < audio_data.length; i += 1) {
      this.audioBuffer.push(audio_data[i] ?? 0);
    }
    while (this.audioBuffer.length > FRAME_LENGTH_SAMPLE) {
      this.audioBuffer.shift();
    }
    if (this.audioBuffer.length < FRAME_LENGTH_SAMPLE) {
      return makeStreamResult(this.frameOffset, 0, false);
    }
    const frame = this.audioFeat.extractSingleFrame(this.audioBuffer);
    if (!frame) {
      return makeStreamResult(this.frameOffset, 0, false);
    }
    this.frameOffset += 1;
    const output = await this.backend.runStream(frame, this.modelCaches);
    this.modelCaches = output.caches.slice();
    const confidence = output.probs[0] ?? 0;
    const isSpeech = confidence > this.threshold;
    return makeStreamResult(this.frameOffset, confidence, isSpeech);
  }

  async processStream(audioData: ArrayLike<number>): Promise<FireRedFrameResult> {
    return this.process_stream(audioData);
  }

  reset(): void {
    this.audioBuffer = [];
    this.frameOffset = 0;
    this.modelCaches.fill(0);
    this.audioFeat.reset();
  }

  get_frame_offset(): number {
    return this.frameOffset;
  }

  getFrameOffset(): number {
    return this.get_frame_offset();
  }

  async destroy(): Promise<void> {
    await this.dispose();
  }
}

export async function non_stream_vad(
  audio: AudioInput,
  model_dir = DEFAULT_PRETRAINED_DIR,
  config: FireRedVadConfig = {},
): Promise<FireRedVadDetectResult | null> {
  const vad = await FireRedVad.from_pretrained(model_dir, config);
  try {
    const [result] = await vad.detect(audio, true);
    return result;
  } finally {
    await vad.dispose();
  }
}

export async function nonStreamVad(
  audio: AudioInput,
  modelDir = DEFAULT_PRETRAINED_DIR,
  config: FireRedVadConfig = {},
): Promise<FireRedVadDetectResult | null> {
  return non_stream_vad(audio, modelDir, config);
}

export async function stream_vad_full(
  audio: AudioInput,
  model_dir = DEFAULT_PRETRAINED_DIR,
  config: FireRedStreamVadConfig = {},
): Promise<[StreamVadFrameResult[], FireRedVadDetectResult]> {
  const vad = await FireRedStreamVad.from_pretrained(model_dir, config);
  try {
    return await vad.detect_full(audio);
  } finally {
    await vad.dispose();
  }
}

export async function streamVadFull(
  audio: AudioInput,
  modelDir = DEFAULT_PRETRAINED_DIR,
  config: FireRedStreamVadConfig = {},
): Promise<[StreamVadFrameResult[], FireRedVadDetectResult]> {
  return stream_vad_full(audio, modelDir, config);
}

export async function non_stream_aed(
  audio: AudioInput,
  model_dir = DEFAULT_PRETRAINED_DIR,
  config: FireRedAedConfig = {},
): Promise<FireRedAedDetectResult> {
  const aed = await FireRedAed.from_pretrained(model_dir, config);
  try {
    const [result] = await aed.detect(audio);
    return result;
  } finally {
    await aed.dispose();
  }
}

export async function nonStreamAed(
  audio: AudioInput,
  modelDir = DEFAULT_PRETRAINED_DIR,
  config: FireRedAedConfig = {},
): Promise<FireRedAedDetectResult> {
  return non_stream_aed(audio, modelDir, config);
}
