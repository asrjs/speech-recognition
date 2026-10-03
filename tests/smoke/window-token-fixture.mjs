/** Bounded stock-v3 smoke fixture, not a library audio-duration policy. */
export function createWindowTokenFixture(source, overlap = false) {
  const maxSeconds = overlap ? 5 : 19;
  if (!(source instanceof Float32Array) || source.length === 0 || source.length > maxSeconds * 16000) {
    throw new Error(`Window-token smoke requires nonempty mono 16kHz PCM at most ${maxSeconds} seconds`);
  }
  const windowSamples = 20 * 16000;
  const pcm = new Float32Array((overlap ? 34 : 40) * 16000);
  pcm.set(source, overlap ? 15 * 16000 : 0);
  if (!overlap) pcm.set(source, windowSamples);
  return {
    pcm,
    options: { returnTokens: true, windowing: 'force', windowDurationSeconds: 20, overlapSeconds: overlap ? 6 : 0 },
    windowing: { windowSeconds: 20, durationSeconds: overlap ? 34 : 40, ...(overlap ? { overlapSeconds: 6 } : {}) },
  };
}
