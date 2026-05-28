export type WhisperStrideSeconds = number | readonly [leftSeconds: number, rightSeconds: number];

export interface WhisperChunkPlan {
  readonly index: number;
  readonly startSample: number;
  readonly endSample: number;
  readonly inputLengthSamples: number;
  readonly isFirst: boolean;
  readonly isLast: boolean;
  /** Transformers.js/HF-compatible `[inputLength, leftStride, rightStride]` in samples. */
  readonly stride: readonly [inputLengthSamples: number, leftStrideSamples: number, rightStrideSamples: number];
  readonly startTime: number;
  readonly endTime: number;
}

export function planWhisperChunks(
  audioLengthSamples: number,
  sampleRate: number,
  chunkLengthSeconds: number,
  strideLengthSeconds?: WhisperStrideSeconds,
): WhisperChunkPlan[] {
  validatePositiveInteger(audioLengthSamples, 'audioLengthSamples');
  validatePositiveNumber(sampleRate, 'sampleRate');

  const chunkSamples = Math.round(chunkLengthSeconds * sampleRate);
  if (chunkLengthSeconds <= 0 || chunkSamples <= 0 || audioLengthSamples <= chunkSamples) {
    return [createChunk(0, 0, audioLengthSamples, audioLengthSamples, sampleRate, 0, 0)];
  }

  const [leftStrideSamples, rightStrideSamples] = resolveStrideSamples(
    strideLengthSeconds ?? chunkLengthSeconds / 6,
    chunkLengthSeconds,
    sampleRate,
  );
  const jumpSamples = chunkSamples - leftStrideSamples - rightStrideSamples;
  if (jumpSamples <= 0) {
    throw new Error('Whisper chunk stride settings must leave positive forward progress.');
  }

  const chunks: WhisperChunkPlan[] = [];
  let startSample = 0;
  while (startSample < audioLengthSamples) {
    const endSample = Math.min(audioLengthSamples, startSample + chunkSamples);
    const isFirst = startSample === 0;
    const isLast = endSample >= audioLengthSamples;
    chunks.push(
      createChunk(
        chunks.length,
        startSample,
        endSample,
        audioLengthSamples,
        sampleRate,
        isFirst ? 0 : leftStrideSamples,
        isLast ? 0 : rightStrideSamples,
      ),
    );

    if (isLast) {
      break;
    }
    startSample += jumpSamples;
  }

  return chunks;
}

function createChunk(
  index: number,
  startSample: number,
  endSample: number,
  audioLengthSamples: number,
  sampleRate: number,
  leftStrideSamples: number,
  rightStrideSamples: number,
): WhisperChunkPlan {
  const inputLengthSamples = endSample - startSample;
  return {
    index,
    startSample,
    endSample,
    inputLengthSamples,
    isFirst: startSample === 0,
    isLast: endSample >= audioLengthSamples,
    stride: [inputLengthSamples, leftStrideSamples, rightStrideSamples],
    startTime: startSample / sampleRate,
    endTime: endSample / sampleRate,
  };
}

function resolveStrideSamples(
  strideLengthSeconds: WhisperStrideSeconds,
  chunkLengthSeconds: number,
  sampleRate: number,
): readonly [number, number] {
  if (typeof strideLengthSeconds === 'number') {
    validateNonNegativeNumber(strideLengthSeconds, 'stride');
    if (strideLengthSeconds >= chunkLengthSeconds / 2) {
      throw new Error('Whisper symmetric stride must be less than half of chunk length.');
    }
    const strideSamples = Math.round(strideLengthSeconds * sampleRate);
    return [strideSamples, strideSamples];
  }

  const [leftSeconds, rightSeconds] = strideLengthSeconds;
  validateNonNegativeNumber(leftSeconds, 'left stride');
  validateNonNegativeNumber(rightSeconds, 'right stride');
  if (leftSeconds + rightSeconds >= chunkLengthSeconds) {
    throw new Error('Whisper left and right stride lengths must sum to less than chunk length.');
  }
  return [Math.round(leftSeconds * sampleRate), Math.round(rightSeconds * sampleRate)];
}

function validatePositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validatePositiveNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive number.`);
  }
}

function validateNonNegativeNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a non-negative number.`);
  }
}
