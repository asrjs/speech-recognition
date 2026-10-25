import {
  transcribeSpeechFromAudioSource,
  type LoadSpeechModelOptions,
} from '@asrjs/speech-recognition';

/** Pass an input element's File, a Blob, or an audio URL in a browser. */
export async function transcribeBrowserFile(
  source: string | Blob,
  loadOptions: LoadSpeechModelOptions,
) {
  return transcribeSpeechFromAudioSource(source, {
    ...loadOptions,
    transcribeOptions: { responseFlavor: 'canonical' },
  });
}
