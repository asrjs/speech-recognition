import { mixAudioBufferChannelsToMono } from './media.js';
import { SincResampler } from '../audio/resample.js';
import { AudioRingBuffer } from './realtime.js';
import {
  STREAMING_DEVICE_SAMPLE_RATE_FALLBACK,
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_TIMELINE_CHUNK_FRAMES,
  framesToMilliseconds,
  resolveStreamingTimelineChunkFrames,
} from './audio-timeline.js';

export interface MicrophoneAudioChunk {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly deviceSampleRate: number | null;
  readonly contextSampleRate: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startTimeSeconds: number;
  readonly endTimeSeconds: number;
}

export interface ScriptProcessorAudioNodeLike {
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null;
  connect(destination: unknown): void;
  disconnect(): void;
}

export interface AudioWorkletPortLike {
  onmessage: ((event: { data: unknown }) => void) | null;
  postMessage?(message: unknown): void;
}

export interface AudioWorkletNodeLike {
  readonly port: AudioWorkletPortLike;
  onprocessorerror: ((event: unknown) => void) | null;
  connect(destination: unknown): void;
  disconnect(): void;
}

export interface AudioWorkletLike {
  addModule(moduleUrl: string): Promise<void>;
}

export interface MediaStreamSourceNodeLike {
  connect(destination: unknown): void;
  disconnect(): void;
}

export interface BrowserAudioContextLike {
  readonly sampleRate: number;
  readonly destination: unknown;
  readonly audioWorklet?: AudioWorkletLike;
  createMediaStreamSource(stream: MediaStream): MediaStreamSourceNodeLike;
  createScriptProcessor(
    bufferSize: number,
    numberOfInputChannels: number,
    numberOfOutputChannels: number,
  ): ScriptProcessorAudioNodeLike;
  close(): Promise<void> | void;
}

export interface BrowserMicrophoneCaptureOptions {
  readonly bufferSize?: number;
  readonly chunkFrames?: number;
  readonly chunkDurationMs?: number;
  readonly targetSampleRate?: number;
  readonly stream?: MediaStream;
  readonly constraints?: MediaStreamConstraints;
  readonly stopTracksOnStop?: boolean;
  readonly getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  readonly createAudioContext?: (sampleRate: number) => BrowserAudioContextLike;
  readonly onChunk: (chunk: MicrophoneAudioChunk) => void;
  readonly onError?: (error: unknown) => void;
}

export interface BrowserMicrophoneCaptureHandle {
  readonly sampleRate: number;
  readonly deviceSampleRate: number | null;
  readonly contextSampleRate: number;
  readonly chunkFrames: number;
  readonly chunkDurationMs: number;
  readonly stream: MediaStream;
  stop(): Promise<void>;
}

export interface BrowserMicrophoneRingCaptureOptions extends Omit<
  BrowserMicrophoneCaptureOptions,
  'onChunk'
> {
  readonly ringBufferDurationSeconds?: number;
  readonly onChunk?: (chunk: MicrophoneAudioChunk) => void;
}

export interface BrowserMicrophoneRingCaptureHandle extends BrowserMicrophoneCaptureHandle {
  readonly ringBuffer: AudioRingBuffer;
  subscribe(listener: (chunk: MicrophoneAudioChunk) => void): () => void;
  getWaveform(
    pointCount?: number,
    frameSpan?: number,
  ): ReturnType<AudioRingBuffer['getMinMaxPairs']>;
  getSamplePoints(
    pointCount?: number,
    frameSpan?: number,
  ): ReturnType<AudioRingBuffer['getSamplePoints']>;
}

function defaultConstraints(): MediaStreamConstraints {
  return {
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  };
}

function resolveGetUserMedia(
  override?: BrowserMicrophoneCaptureOptions['getUserMedia'],
): (constraints: MediaStreamConstraints) => Promise<MediaStream> {
  if (override) {
    return override;
  }
  const getUserMedia = globalThis.navigator?.mediaDevices?.getUserMedia?.bind(
    globalThis.navigator.mediaDevices,
  );
  if (!getUserMedia) {
    throw new Error(
      'startMicrophoneCapture requires navigator.mediaDevices.getUserMedia in this environment.',
    );
  }
  return getUserMedia;
}

function resolveCreateAudioContext(
  override?: BrowserMicrophoneCaptureOptions['createAudioContext'],
): (sampleRate: number) => BrowserAudioContextLike {
  if (override) {
    return override;
  }
  return (sampleRate) => {
    if (typeof globalThis.AudioContext !== 'function') {
      throw new Error('startMicrophoneCapture requires AudioContext in this environment.');
    }
    return new globalThis.AudioContext({ sampleRate });
  };
}

function buildCaptureProcessorModule(): string {
  // The worklet runs in its own global scope, so the shared band-limited
  // resampler is embedded by source instead of imported.
  return `
const AsrjsSincResampler = ${SincResampler.toString()};

class AsrjsCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    this.targetSampleRate = Math.max(1, Number(opts.targetSampleRate) || sampleRate);
    this.targetChunkFrames = Math.max(1, Math.floor(Number(opts.targetChunkFrames) || ${STREAMING_TIMELINE_CHUNK_FRAMES}));
    this.resampler = new AsrjsSincResampler(sampleRate, this.targetSampleRate);
    this.pending = new Float32Array(this.targetChunkFrames);
    this.pendingLength = 0;
  }

  mixToMono(inputChannels) {
    const frameCount = inputChannels[0].length;
    const channelCount = inputChannels.length;
    const mono = new Float32Array(frameCount);
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      let sampleValue = 0;
      for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
        sampleValue += inputChannels[channelIndex][frameIndex] || 0;
      }
      mono[frameIndex] = sampleValue / channelCount;
    }
    return mono;
  }

  emitChunks(samples) {
    let offset = 0;
    while (offset < samples.length) {
      const count = Math.min(this.targetChunkFrames - this.pendingLength, samples.length - offset);
      this.pending.set(samples.subarray(offset, offset + count), this.pendingLength);
      this.pendingLength += count;
      offset += count;
      if (this.pendingLength === this.targetChunkFrames) {
        const out = this.pending;
        this.pending = new Float32Array(this.targetChunkFrames);
        this.pendingLength = 0;
        this.port.postMessage({
          type: 'chunk',
          pcm: out,
          sampleRate: this.targetSampleRate,
        }, [out.buffer]);
      }
    }
  }

  process(inputs) {
    const inputChannels = inputs[0];
    if (!inputChannels?.length || !inputChannels[0]?.length) {
      return true;
    }

    this.emitChunks(this.resampler.process(this.mixToMono(inputChannels)));
    return true;
  }
}

registerProcessor('asrjs-capture-processor', AsrjsCaptureProcessor);
`;
}

function createFixedChunkResampler(options: {
  readonly sourceSampleRate: number;
  readonly targetSampleRate: number;
  readonly chunkFrames: number;
  readonly onChunk: (pcm: Float32Array) => void;
}) {
  const chunkFrames = Math.max(1, options.chunkFrames);
  const resampler = new SincResampler(
    Math.max(1, options.sourceSampleRate),
    Math.max(1, options.targetSampleRate),
  );
  let pending = new Float32Array(chunkFrames);
  let pendingLength = 0;

  return {
    push(chunk: Float32Array) {
      if (!chunk.length) return;
      const samples = resampler.process(chunk);
      let offset = 0;
      while (offset < samples.length) {
        const count = Math.min(chunkFrames - pendingLength, samples.length - offset);
        pending.set(samples.subarray(offset, offset + count), pendingLength);
        pendingLength += count;
        offset += count;
        if (pendingLength === chunkFrames) {
          const out = pending;
          pending = new Float32Array(chunkFrames);
          pendingLength = 0;
          options.onChunk(out);
        }
      }
    },
  };
}

function resolveScriptProcessorBufferSize(chunkFrames: number): number {
  const safeChunkFrames = Math.max(1, Math.floor(chunkFrames));
  let bufferSize = 256;
  while (bufferSize < safeChunkFrames && bufferSize < 16_384) {
    bufferSize *= 2;
  }
  return bufferSize;
}

async function createAudioWorkletCaptureNode(
  audioContext: BrowserAudioContextLike,
  options: BrowserMicrophoneCaptureOptions,
): Promise<AudioWorkletNodeLike | null> {
  if (!audioContext.audioWorklet || typeof globalThis.AudioWorkletNode !== 'function') {
    return null;
  }

  const processorCode = buildCaptureProcessorModule();
  const moduleUrl = URL.createObjectURL(
    new Blob([processorCode], { type: 'application/javascript' }),
  );

  try {
    await audioContext.audioWorklet.addModule(moduleUrl);
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }

  return new globalThis.AudioWorkletNode(audioContext as AudioContext, 'asrjs-capture-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      targetSampleRate: options.targetSampleRate ?? audioContext.sampleRate,
      targetChunkFrames: options.chunkFrames ?? STREAMING_TIMELINE_CHUNK_FRAMES,
    },
  }) as AudioWorkletNodeLike;
}

export async function startMicrophoneCapture(
  options: BrowserMicrophoneCaptureOptions,
): Promise<BrowserMicrophoneCaptureHandle> {
  const processingSampleRate = options.targetSampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  const chunkFrames =
    options.chunkFrames ??
    resolveStreamingTimelineChunkFrames(processingSampleRate, options.chunkDurationMs);
  const bufferSize = options.bufferSize ?? resolveScriptProcessorBufferSize(chunkFrames);
  const createAudioContext = resolveCreateAudioContext(options.createAudioContext);
  const ownsStream = !options.stream;
  const stream =
    options.stream ??
    (await resolveGetUserMedia(options.getUserMedia)(options.constraints ?? defaultConstraints()));
  const audioTrack = stream.getAudioTracks?.()[0] ?? null;
  const trackSettings = audioTrack?.getSettings?.() ?? null;
  const deviceSampleRate =
    typeof trackSettings?.sampleRate === 'number' ? trackSettings.sampleRate : null;
  const preferredContextSampleRate =
    deviceSampleRate ?? processingSampleRate ?? STREAMING_DEVICE_SAMPLE_RATE_FALLBACK;
  let audioContext: BrowserAudioContextLike | null = null;
  let source: MediaStreamSourceNodeLike | null = null;
  let workletNode: AudioWorkletNodeLike | null = null;
  let scriptProcessor: ScriptProcessorAudioNodeLike | null = null;
  let processor: AudioWorkletNodeLike | ScriptProcessorAudioNodeLike | null = null;
  let stopped = false;
  let stopPromise: Promise<void> | null = null;

  const stopResources = (): Promise<void> => {
    if (stopPromise) {
      return stopPromise;
    }
    stopped = true;
    // Assign the shared promise before invoking potentially reentrant releases.
    stopPromise = Promise.resolve().then(async () => {
      const errors: unknown[] = [];
      const attempt = (release: () => void) => {
        try {
          release();
        } catch (error) {
          errors.push(error);
        }
      };
      if (scriptProcessor) {
        attempt(() => {
          scriptProcessor!.onaudioprocess = null;
        });
      }
      if (workletNode) {
        attempt(() => {
          workletNode!.port.onmessage = null;
        });
        attempt(() => {
          workletNode!.onprocessorerror = null;
        });
      }
      attempt(() => processor?.disconnect());
      attempt(() => source?.disconnect());
      try {
        await audioContext?.close();
      } catch (error) {
        errors.push(error);
      }
      if (options.stopTracksOnStop ?? ownsStream) {
        attempt(() => {
          for (const track of stream.getTracks()) {
            attempt(() => track.stop());
          }
        });
      }
      if (errors.length === 1) throw errors[0];
      if (errors.length > 1) throw new AggregateError(errors, 'Microphone capture cleanup failed.');
    });
    return stopPromise;
  };

  let currentFrame = 0;

  const emitChunk = (pcm: Float32Array, sampleRate: number) => {
    if (stopped || !audioContext) {
      return;
    }
    const startFrame = currentFrame;
    currentFrame += pcm.length;
    options.onChunk({
      pcm,
      sampleRate,
      deviceSampleRate,
      contextSampleRate: audioContext.sampleRate,
      startFrame,
      endFrame: currentFrame,
      startTimeSeconds: startFrame / sampleRate,
      endTimeSeconds: currentFrame / sampleRate,
    });
  };

  try {
    audioContext = createAudioContext(preferredContextSampleRate);
    source = audioContext.createMediaStreamSource(stream);
    workletNode = await createAudioWorkletCaptureNode(audioContext, {
      ...options,
      chunkFrames,
      targetSampleRate: processingSampleRate,
    });
    scriptProcessor = workletNode ? null : audioContext.createScriptProcessor(bufferSize, 1, 1);
    processor = workletNode ?? scriptProcessor;
    if (!processor) {
      throw new Error('Unable to create a microphone capture processor.');
    }

    if (workletNode) {
      workletNode.port.onmessage = (event) => {
        try {
          const payload = event.data as
            | {
                readonly type?: string;
                readonly pcm?: Float32Array;
                readonly sampleRate?: number;
              }
            | undefined;

          if (payload?.type !== 'chunk' || !(payload.pcm instanceof Float32Array)) {
            return;
          }

          emitChunk(payload.pcm, payload.sampleRate ?? processingSampleRate);
        } catch (error) {
          options.onError?.(error);
        }
      };
      workletNode.onprocessorerror = (error) => {
        options.onError?.(error);
      };
    } else {
      const fallbackResampler = createFixedChunkResampler({
        sourceSampleRate: audioContext.sampleRate,
        targetSampleRate: processingSampleRate,
        chunkFrames,
        onChunk: (pcm) => emitChunk(pcm, processingSampleRate),
      });
      scriptProcessor!.onaudioprocess = (event: AudioProcessingEvent) => {
        try {
          fallbackResampler.push(mixAudioBufferChannelsToMono(event.inputBuffer));
        } catch (error) {
          options.onError?.(error);
        }
      };
    }

    source.connect(processor);
    processor.connect(audioContext.destination);
  } catch (error) {
    try {
      await stopResources();
    } catch (cleanupError) {
      throw new AggregateError(
        [error, cleanupError],
        'Microphone capture initialization and cleanup failed.',
      );
    }
    throw error;
  }

  return {
    sampleRate: processingSampleRate,
    deviceSampleRate,
    contextSampleRate: audioContext.sampleRate,
    chunkFrames,
    chunkDurationMs: framesToMilliseconds(chunkFrames, processingSampleRate),
    stream,
    stop: stopResources,
  };
}

export async function startMicrophoneRingCapture(
  options: BrowserMicrophoneRingCaptureOptions,
): Promise<BrowserMicrophoneRingCaptureHandle> {
  const processingSampleRate = options.targetSampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  const ringBuffer = new AudioRingBuffer({
    sampleRate: processingSampleRate,
    durationSeconds: options.ringBufferDurationSeconds ?? 12,
  });
  const listeners = new Set<(chunk: MicrophoneAudioChunk) => void>();

  const handle = await startMicrophoneCapture({
    ...options,
    onChunk(chunk) {
      ringBuffer.write(chunk.pcm);
      options.onChunk?.(chunk);
      for (const listener of listeners) {
        listener(chunk);
      }
    },
  });

  return {
    ...handle,
    ringBuffer,
    subscribe(listener: (chunk: MicrophoneAudioChunk) => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getWaveform(pointCount?: number, frameSpan?: number) {
      return ringBuffer.getMinMaxPairs(pointCount, frameSpan);
    },
    getSamplePoints(pointCount?: number, frameSpan?: number) {
      return ringBuffer.getSamplePoints(pointCount, frameSpan);
    },
  };
}
