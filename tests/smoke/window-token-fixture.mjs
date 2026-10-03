/** Bounded stock-v3 smoke fixture, not a library audio-duration policy. */
export function createWindowTokenFixture(source) {
  if (!(source instanceof Float32Array) || source.length === 0 || source.length > 19 * 16000) {
    throw new Error('Window-token smoke requires nonempty mono 16kHz PCM at most 19 seconds');
  }
  const windowSamples = 20 * 16000;
  const pcm = new Float32Array(windowSamples * 2);
  pcm.set(source);
  pcm.set(source, windowSamples);
  return {
    pcm,
    options: { returnTokens: true, windowing: 'force', windowDurationSeconds: 20, overlapSeconds: 0 },
    windowing: { windowSeconds: 20, durationSeconds: 40 },
  };
}
