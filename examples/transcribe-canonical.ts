import {
  transcribeSpeechFromMonoPcm,
  type LoadSpeechModelOptions,
} from '@asrjs/speech-recognition';

/** One recording, using the caller's model source and real audio sample rate. */
export async function transcribeCanonicalExample(
  pcm: Float32Array,
  sampleRate: number,
  loadOptions: LoadSpeechModelOptions,
) {
  // The one-shot helper owns loading, inference and awaited cleanup.
  return transcribeSpeechFromMonoPcm(pcm, sampleRate, {
    ...loadOptions,
    transcribeOptions: {
      detail: 'segments',
      responseFlavor: 'canonical',
    },
  });
}
