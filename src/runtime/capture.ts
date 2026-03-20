import { mixAudioBufferChannelsToMono } from './media.js';
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
  onaudioprocess: ((event: any) => void) | null;
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
  return `
class AsrjsCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    this.targetSampleRate = Math.max(1, Number(opts.targetSampleRate) || sampleRate);
    this.targetChunkFrames = Math.max(1, Math.floor(Number(opts.targetChunkFrames) || ${STREAMING_TIMELINE_CHUNK_FRAMES}));
    this.sourceRate = sampleRate;
    this.rateRatio = this.sourceRate / this.targetSampleRate;
    this.sourceBuffer = new Float32Array(Math.max(2048, Math.ceil(this.rateRatio * this.targetChunkFrames * 4) + 2));
    this.sourceLength = 0;
    this.sourceReadIndex = 0;
  }

  ensureCapacity(required) {
    if (required <= this.sourceBuffer.length) return;
    let nextLength = this.sourceBuffer.length;
    while (nextLength < required) {
      nextLength *= 2;
    }
    const next = new Float32Array(nextLength);
    next.set(this.sourceBuffer.subarray(0, this.sourceLength), 0);
    this.sourceBuffer = next;
  }

  compactSourceBuffer() {
    const consumed = Math.floor(this.sourceReadIndex);
    if (consumed <= 0) return;
    if (consumed < this.sourceLength) {
      this.sourceBuffer.copyWithin(0, consumed, this.sourceLength);
    }
    this.sourceLength -= consumed;
    this.sourceReadIndex -= consumed;
  }

  appendInput(inputChannels) {
    if (!inputChannels?.length || !inputChannels[0]?.length) return;
    const frameCount = inputChannels[0].length;
    const channelCount = inputChannels.length;
    this.ensureCapacity(this.sourceLength + frameCount + 2);

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      let sampleValue = 0;
      for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
        sampleValue += inputChannels[channelIndex][frameIndex] || 0;
      }
      this.sourceBuffer[this.sourceLength + frameIndex] = sampleValue / channelCount;
    }

    this.sourceLength += frameCount;
  }

  emitAvailableChunks() {
    const requiredSourceFrames = () =>
      this.sourceReadIndex + Math.max(1, (this.targetChunkFrames - 1) * this.rateRatio + 1);

    while (this.sourceLength >= requiredSourceFrames()) {
      const out = new Float32Array(this.targetChunkFrames);
      for (let index = 0; index < this.targetChunkFrames; index += 1) {
        const sourcePosition = this.sourceReadIndex + index * this.rateRatio;
        const sourceIndex = Math.floor(sourcePosition);
        const fraction = sourcePosition - sourceIndex;
        const left = this.sourceBuffer[sourceIndex] || 0;
        const right = this.sourceBuffer[Math.min(sourceIndex + 1, this.sourceLength - 1)] || left;
        out[index] = left + (right - left) * fraction;
      }

      this.sourceReadIndex += this.targetChunkFrames * this.rateRatio;
      this.port.postMessage({
        type: 'chunk',
        pcm: out,
        sampleRate: this.targetSampleRate,
      }, [out.buffer]);

      if (this.sourceReadIndex >= 2048) {
        this.compactSourceBuffer();
      }
    }
  }

  process(inputs) {
    const inputChannels = inputs[0];
    if (!inputChannels?.length || !inputChannels[0]?.length) {
      return true;
    }

    this.appendInput(inputChannels);
    this.emitAvailableChunks();
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
  const sourceSampleRate = Math.max(1, options.sourceSampleRate);
  const targetSampleRate = Math.max(1, options.targetSampleRate);
  const chunkFrames = Math.max(1, options.chunkFrames);
  const rateRatio = sourceSampleRate / targetSampleRate;
  let sourceBuffer = new Float32Array(
    Math.max(2048, Math.ceil(rateRatio * chunkFrames * 4) + 2),
  );
  let sourceLength = 0;
  let sourceReadIndex = 0;

  const ensureCapacity = (required: number) => {
    if (required <= sourceBuffer.length) return;
    let nextLength = sourceBuffer.length;
    while (nextLength < required) {
      nextLength *= 2;
    }
    const next = new Float32Array(nextLength);
    next.set(sourceBuffer.subarray(0, sourceLength), 0);
    sourceBuffer = next;
  };

  const compactSourceBuffer = () => {
    const consumed = Math.floor(sourceReadIndex);
    if (consumed <= 0) return;
    if (consumed < sourceLength) {
      sourceBuffer.copyWithin(0, consumed, sourceLength);
    }
    sourceLength -= consumed;
    sourceReadIndex -= consumed;
  };

  const append = (chunk: Float32Array) => {
    if (!chunk.length) return;
    ensureCapacity(sourceLength + chunk.length + 2);
    sourceBuffer.set(chunk, sourceLength);
    sourceLength += chunk.length;
  };

  const emitAvailableChunks = () => {
    const requiredSourceFrames = () =>
      sourceReadIndex + Math.max(1, (chunkFrames - 1) * rateRatio + 1);

    while (sourceLength >= requiredSourceFrames()) {
      const out = new Float32Array(chunkFrames);
      for (let index = 0; index < chunkFrames; index += 1) {
        const sourcePosition = sourceReadIndex + index * rateRatio;
        const sourceIndex = Math.floor(sourcePosition);
        const fraction = sourcePosition - sourceIndex;
        const left = sourceBuffer[sourceIndex] ?? 0;
        const right =
          sourceBuffer[Math.min(sourceIndex + 1, sourceLength - 1)] ?? left;
        out[index] = left + (right - left) * fraction;
      }

      sourceReadIndex += chunkFrames * rateRatio;
      options.onChunk(out);

      if (sourceReadIndex >= 2048) {
        compactSourceBuffer();
      }
    }
  };

  return {
    push(chunk: Float32Array) {
      append(chunk);
      emitAvailableChunks();
    },
  };
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

  return new globalThis.AudioWorkletNode(
    audioContext as AudioContext,
    'asrjs-capture-processor',
    {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: {
        targetSampleRate: options.targetSampleRate ?? audioContext.sampleRate,
        targetChunkFrames: options.chunkFrames ?? STREAMING_TIMELINE_CHUNK_FRAMES,
      },
    },
  ) as AudioWorkletNodeLike;
}

export async function startMicrophoneCapture(
  options: BrowserMicrophoneCaptureOptions,
): Promise<BrowserMicrophoneCaptureHandle> {
  const processingSampleRate = options.targetSampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  const chunkFrames =
    options.chunkFrames ?? resolveStreamingTimelineChunkFrames(processingSampleRate);
  const bufferSize = options.bufferSize ?? chunkFrames;
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
  const audioContext = createAudioContext(preferredContextSampleRate);
  const source = audioContext.createMediaStreamSource(stream);
  let currentFrame = 0;

  const emitChunk = (pcm: Float32Array, sampleRate: number) => {
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

  const workletNode = await createAudioWorkletCaptureNode(audioContext, {
    ...options,
    chunkFrames,
    targetSampleRate: processingSampleRate,
  });
  const scriptProcessor =
    workletNode ? null : audioContext.createScriptProcessor(bufferSize, 1, 1);
  const processor = workletNode ?? scriptProcessor;
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
    scriptProcessor!.onaudioprocess = (event: any) => {
      try {
        fallbackResampler.push(mixAudioBufferChannelsToMono(event.inputBuffer));
      } catch (error) {
        options.onError?.(error);
      }
    };
  }

  source.connect(processor);
  processor.connect(audioContext.destination);

  return {
    sampleRate: processingSampleRate,
    deviceSampleRate,
    contextSampleRate: audioContext.sampleRate,
    chunkFrames,
    chunkDurationMs: framesToMilliseconds(chunkFrames, processingSampleRate),
    stream,
    async stop() {
      if (scriptProcessor) {
        scriptProcessor.onaudioprocess = null;
      }
      if (workletNode) {
        workletNode.port.onmessage = null;
      }
      processor.disconnect();
      source.disconnect();
      await audioContext.close();

      if (options.stopTracksOnStop ?? ownsStream) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
      }
    },
  };
}
