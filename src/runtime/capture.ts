import { mixAudioBufferChannelsToMono, type AudioBufferLikeForDecode } from './media.js';

export interface MicrophoneAudioChunk {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
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

export interface MediaStreamSourceNodeLike {
  connect(destination: unknown): void;
  disconnect(): void;
}

export interface BrowserAudioContextLike {
  readonly sampleRate: number;
  readonly destination: unknown;
  createMediaStreamSource(stream: MediaStream): MediaStreamSourceNodeLike;
  createScriptProcessor(bufferSize: number, numberOfInputChannels: number, numberOfOutputChannels: number): ScriptProcessorAudioNodeLike;
  close(): Promise<void> | void;
}

export interface BrowserMicrophoneCaptureOptions {
  readonly bufferSize?: number;
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
  readonly stream: MediaStream;
  stop(): Promise<void>;
}

function defaultConstraints(): MediaStreamConstraints {
  return {
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  };
}

function resolveGetUserMedia(
  override?: BrowserMicrophoneCaptureOptions['getUserMedia']
): (constraints: MediaStreamConstraints) => Promise<MediaStream> {
  if (override) {
    return override;
  }
  const getUserMedia = globalThis.navigator?.mediaDevices?.getUserMedia?.bind(globalThis.navigator.mediaDevices);
  if (!getUserMedia) {
    throw new Error('startMicrophoneCapture requires navigator.mediaDevices.getUserMedia in this environment.');
  }
  return getUserMedia;
}

function resolveCreateAudioContext(
  override?: BrowserMicrophoneCaptureOptions['createAudioContext']
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

export async function startMicrophoneCapture(
  options: BrowserMicrophoneCaptureOptions
): Promise<BrowserMicrophoneCaptureHandle> {
  const bufferSize = options.bufferSize ?? 4096;
  const createAudioContext = resolveCreateAudioContext(options.createAudioContext);
  const ownsStream = !options.stream;
  const stream = options.stream ?? await resolveGetUserMedia(options.getUserMedia)(options.constraints ?? defaultConstraints());
  const audioContext = createAudioContext(options.targetSampleRate ?? 16000);
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
  let currentFrame = 0;

  processor.onaudioprocess = (event) => {
    try {
      const pcm = mixAudioBufferChannelsToMono(event.inputBuffer);
      const startFrame = currentFrame;
      currentFrame += pcm.length;
      options.onChunk({
        pcm,
        sampleRate: audioContext.sampleRate,
        startFrame,
        endFrame: currentFrame,
        startTimeSeconds: startFrame / audioContext.sampleRate,
        endTimeSeconds: currentFrame / audioContext.sampleRate
      });
    } catch (error) {
      options.onError?.(error);
    }
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  return {
    sampleRate: audioContext.sampleRate,
    stream,
    async stop() {
      processor.onaudioprocess = null;
      processor.disconnect();
      source.disconnect();
      await audioContext.close();

      if (options.stopTracksOnStop ?? ownsStream) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
      }
    }
  };
}
