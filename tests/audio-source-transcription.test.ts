import { beforeEach, describe, expect, it, vi } from 'vitest';

const { loadBuiltInSpeechModel, decodeAudioSourceToMonoPcm } = vi.hoisted(() => ({
  loadBuiltInSpeechModel: vi.fn(),
  decodeAudioSourceToMonoPcm: vi.fn(),
}));

vi.mock('../src/runtime/builtins.js', () => ({ loadBuiltInSpeechModel }));
vi.mock('../src/runtime/media.js', () => ({ decodeAudioSourceToMonoPcm }));

import { transcribeSpeechFromAudioSource } from '../src/runtime/load.js';

const inference = {
  sampleRate: 22050,
  maxInputDurationSec: 3600,
  recommendedWindowDurationSec: 3600,
  minWindowDurationSec: 1,
  maxWindowDurationSec: 3600,
  autoWindowThresholdSec: 3600,
  defaultOverlapSec: 0,
  supportsWordTimestamps: false,
  supportsSegmentTimestamps: false,
  defaultSegmentationStrategy: 'none' as const,
  defaultMergeStrategy: 'concat' as const,
};

function createLoadedHandle(dispose = vi.fn(async () => undefined)) {
  const transcribe = vi.fn(async () => ({ text: 'recognized' }) as never);
  const handle = {
    runtime: {},
    model: {
      info: { family: 'test-family', modelId: 'test-model', classification: {}, inference },
    },
    session: {},
    transcribe,
    dispose,
  };
  return { handle, transcribe, dispose };
}

function decodedAudio(sampleRate = 22050) {
  return {
    pcm: new Float32Array([0.1, 0.2]),
    sampleRate,
    durationSec: 2 / sampleRate,
    numberOfChannels: 1,
    metrics: {},
  };
}

describe('transcribeSpeechFromAudioSource', () => {
  beforeEach(() => {
    vi.mocked(loadBuiltInSpeechModel).mockReset();
    vi.mocked(decodeAudioSourceToMonoPcm).mockReset();
    vi.mocked(decodeAudioSourceToMonoPcm).mockResolvedValue(decodedAudio() as never);
  });

  it('uses the loaded model sample rate, forwards transcription options, and disposes', async () => {
    const { handle, transcribe, dispose } = createLoadedHandle();
    vi.mocked(loadBuiltInSpeechModel).mockResolvedValue(handle as never);
    const file = new Blob(['audio'], { type: 'audio/wav' });
    const signal = new AbortController().signal;

    await transcribeSpeechFromAudioSource(file, {
      modelId: 'test-model',
      signal,
      audioOptions: { strategy: 'native-rate', resamplerQuality: 'linear' },
      transcribeOptions: { detail: 'words', signal },
    });

    expect(decodeAudioSourceToMonoPcm).toHaveBeenCalledWith(file, {
      strategy: 'native-rate',
      resamplerQuality: 'linear',
      targetSampleRate: 22050,
      signal,
    });
    expect(transcribe).toHaveBeenCalledOnce();
    const [audio, options] = transcribe.mock.calls[0]!;
    expect(audio).toMatchObject({ sampleRate: 22050, numberOfChannels: 1 });
    expect(options).toMatchObject({ detail: 'words', signal });
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('disposes after decode failure and preserves both decode and cleanup errors', async () => {
    const decodeError = new Error('audio decode failed');
    const cleanupError = new Error('model cleanup failed');
    const { handle, dispose } = createLoadedHandle(
      vi.fn(async () => {
        throw cleanupError;
      }),
    );
    vi.mocked(loadBuiltInSpeechModel).mockResolvedValue(handle as never);
    vi.mocked(decodeAudioSourceToMonoPcm).mockRejectedValue(decodeError);

    const error = await transcribeSpeechFromAudioSource(new Blob(['bad']), {
      modelId: 'test-model',
    }).catch((value: unknown) => value);

    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual([decodeError, cleanupError]);
    expect((error as AggregateError).cause).toBe(decodeError);
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('does not transcribe if cancellation arrives during audio decoding', async () => {
    const { handle, transcribe, dispose } = createLoadedHandle();
    vi.mocked(loadBuiltInSpeechModel).mockResolvedValue(handle as never);
    let finishDecode!: (value: ReturnType<typeof decodedAudio>) => void;
    vi.mocked(decodeAudioSourceToMonoPcm).mockReturnValue(
      new Promise((resolve) => {
        finishDecode = resolve;
      }) as never,
    );
    const abort = new AbortController();
    const result = transcribeSpeechFromAudioSource(new Blob(['audio']), {
      modelId: 'test-model',
      signal: abort.signal,
    });
    await vi.waitFor(() => expect(decodeAudioSourceToMonoPcm).toHaveBeenCalledOnce());
    abort.abort();
    finishDecode(decodedAudio());

    await expect(result).rejects.toMatchObject({ name: 'PipelineAbortedError' });
    expect(transcribe).not.toHaveBeenCalled();
    expect(dispose).toHaveBeenCalledOnce();
  });

  it.each(['load signal', 'transcription signal'] as const)(
    'does not load the model when the %s is already aborted',
    async (signalOwner) => {
      const abort = new AbortController();
      abort.abort();
      vi.mocked(loadBuiltInSpeechModel).mockResolvedValue(createLoadedHandle().handle as never);
      const options =
        signalOwner === 'load signal'
          ? { modelId: 'test-model', signal: abort.signal }
          : { modelId: 'test-model', transcribeOptions: { signal: abort.signal } };

      await expect(
        transcribeSpeechFromAudioSource(new Blob(['audio']), options),
      ).rejects.toMatchObject({ name: 'PipelineAbortedError' });
      expect(loadBuiltInSpeechModel).not.toHaveBeenCalled();
      expect(decodeAudioSourceToMonoPcm).not.toHaveBeenCalled();
    },
  );
});
