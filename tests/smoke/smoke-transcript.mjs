/** Smoke equality only; not WER normalization or numerical model parity. */
export function matchesSmokeTranscript(actual, expected) {
  const normalize = (text) => text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
  const target = normalize(expected);
  return target.length > 0 && normalize(actual) === target;
}
