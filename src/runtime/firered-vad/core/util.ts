export function roundTo(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function isNodeRuntime(): boolean {
  return (
    typeof process !== 'undefined' &&
    typeof process.versions === 'object' &&
    typeof process.versions?.node === 'string'
  );
}

export function isLikelyHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

export function looksLikeFileUrl(value: string): boolean {
  return /^file:\/\//i.test(value);
}
