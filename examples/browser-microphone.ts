import type { LoadedSpeechModel, TranscriptResult } from '@asrjs/speech-recognition';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeMicrophoneControllerOptions,
} from '@asrjs/speech-recognition/browser';
import type { RealtimeTranscriptionUpdate } from '@asrjs/speech-recognition/realtime';

/**
 * The caller owns the loaded model. Start capture from a user gesture and
 * await microphone.dispose() before loaded.dispose() when leaving the screen.
 */
export function createMicrophoneTranscriptionExample(
  loaded: LoadedSpeechModel,
  onResult: (result: TranscriptResult, update: RealtimeTranscriptionUpdate) => void,
  onError: (error: unknown) => void,
  vad: Pick<
    BrowserRealtimeMicrophoneControllerOptions,
    'config' | 'tenVadConfig' | 'tenVadOptions' | 'fireRedVadConfig' | 'fireRedVadOptions'
  > = {},
) {
  const microphone = createBrowserRealtimeMicrophoneController({
    ...vad,
    model: loaded,
    micMode: 'speech-detect',
    transcriptionContext: {},
    onTranscript(update) {
      // The update already carries capture-relative times and context revisions.
      onResult(update.canonical, update);
    },
  });
  let previousError = '';
  const unsubscribe = microphone.subscribe((state) => {
    if (state.micError && state.micError !== previousError) onError(new Error(state.micError));
    previousError = state.micError;
  });
  return {
    ...microphone,
    async dispose() {
      unsubscribe();
      await microphone.dispose();
    },
  };
}
