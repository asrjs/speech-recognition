import type { LoadSpeechModelOptions, PartialTranscript } from '@asrjs/speech-recognition';
import { loadSpeechModel, PcmAudioBuffer } from '@asrjs/speech-recognition';

/**
 * Consume ordered mono Float32 PCM chunks at one sample rate with a model that
 * provides native streaming. Chunks are wrapped as PCM; this function does not
 * resample them. The caller owns capture and chooses a compatible model.
 */
export async function transcribeNativeStream(
  loadOptions: LoadSpeechModelOptions,
  chunks: AsyncIterable<Float32Array>,
  sampleRate: number,
  onRevision: (result: PartialTranscript) => void,
): Promise<PartialTranscript> {
  const loaded = await loadSpeechModel(loadOptions);
  let transcriber: Awaited<ReturnType<typeof loaded.createStreamingTranscriber>> | undefined;
  let failure: { error: unknown } | undefined;

  const throwIfAborted = () => {
    if (loadOptions.signal?.aborted) {
      const error = new Error('Streaming transcription was aborted.');
      error.name = 'AbortError';
      throw error;
    }
  };

  try {
    throwIfAborted();
    if (!loaded.supportsStreaming) {
      throw new Error(`Model "${loaded.model.info.modelId}" does not support native streaming.`);
    }

    transcriber = await loaded.createStreamingTranscriber({ emitPartials: true });
    throwIfAborted();

    for await (const chunk of chunks) {
      throwIfAborted();
      // Each chunk must already use sampleRate; wrapping does not resample it.
      const audio = PcmAudioBuffer.fromMono(chunk, sampleRate);
      // Await each push to preserve chunk order and model-owned streaming state.
      const revision = await transcriber.pushAudio(audio);
      throwIfAborted();
      onRevision(revision);
    }

    throwIfAborted();
    const final = await transcriber.finalize();
    throwIfAborted();
    onRevision(final);
    return final;
  } catch (error) {
    // Keep even a thrown null/undefined value available if cleanup also fails.
    failure = { error };
    throw error;
  } finally {
    // The loaded handle tracks created stream resources and owns their cleanup.
    try {
      await loaded.dispose();
    } catch (cleanupError) {
      if (failure) {
        throw new AggregateError(
          [failure.error, cleanupError],
          'Streaming transcription and resource cleanup failed.',
          { cause: failure.error },
        );
      }
      throw cleanupError;
    }
  }
}
