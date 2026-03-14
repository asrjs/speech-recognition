export interface AudioBufferLikeForDecode {
  readonly numberOfChannels: number;
  readonly length: number;
  readonly sampleRate: number;
  getChannelData(channel: number): Float32Array;
}

export interface AudioContextLikeForDecode {
  decodeAudioData(buffer: ArrayBuffer): Promise<AudioBufferLikeForDecode>;
  close?: () => Promise<void> | void;
}

export interface DecodedMonoAudio {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly durationSec: number;
  readonly numberOfChannels: number;
}

export interface DecodeAudioSourceOptions {
  readonly targetSampleRate?: number;
  readonly createAudioContext?: (sampleRate: number) => AudioContextLikeForDecode;
  readonly fetchImpl?: typeof fetch;
}

export function mixAudioBufferChannelsToMono(audioBuffer: AudioBufferLikeForDecode): Float32Array {
  const channelCount = Math.max(1, audioBuffer.numberOfChannels);
  if (channelCount === 1) {
    return new Float32Array(audioBuffer.getChannelData(0));
  }

  const mono = new Float32Array(audioBuffer.length);
  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channel = audioBuffer.getChannelData(channelIndex);
    for (let frameIndex = 0; frameIndex < audioBuffer.length; frameIndex += 1) {
      mono[frameIndex] = (mono[frameIndex] ?? 0) + ((channel[frameIndex] ?? 0) / channelCount);
    }
  }
  return mono;
}

export async function resolveAudioSourceToBlob(
  source: string | Blob,
  options: Pick<DecodeAudioSourceOptions, 'fetchImpl'> = {}
): Promise<Blob> {
  if (typeof source !== 'string') {
    return source;
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error('resolveAudioSourceToBlob requires fetch when source is a URL.');
  }

  const response = await fetchImpl(source);
  if (!response.ok) {
    throw new Error(`Audio fetch failed: ${response.status}`);
  }

  return await response.blob();
}

export async function decodeAudioSourceToMonoPcm(
  source: string | Blob,
  options: DecodeAudioSourceOptions = {}
): Promise<DecodedMonoAudio> {
  const targetSampleRate = options.targetSampleRate ?? 16000;
  const blob = await resolveAudioSourceToBlob(source, options);
  const arrayBuffer = await blob.arrayBuffer();

  const createAudioContext = options.createAudioContext ?? ((sampleRate: number) => {
    if (typeof globalThis.AudioContext !== 'function') {
      throw new Error('decodeAudioSourceToMonoPcm requires AudioContext in this environment.');
    }
    return new globalThis.AudioContext({ sampleRate });
  });

  const audioContext = createAudioContext(targetSampleRate);
  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const pcm = mixAudioBufferChannelsToMono(decoded);
    return {
      pcm,
      sampleRate: decoded.sampleRate,
      durationSec: decoded.length / decoded.sampleRate,
      numberOfChannels: decoded.numberOfChannels
    };
  } finally {
    await audioContext.close?.();
  }
}
