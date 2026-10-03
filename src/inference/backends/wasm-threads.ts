/** Browser-safe WASM worker cap; native ORT thread policy is separate. */
export function resolveWasmThreads(requested?: number, defaultThreads?: number): number {
  const reported = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : undefined;
  const hardwareThreads =
    typeof reported === 'number' && Number.isFinite(reported) && reported >= 1
      ? Math.floor(reported)
      : 4;
  const limit = Math.floor(hardwareThreads / 2) + 2;
  const selected = requested ?? defaultThreads ?? limit;
  return Number.isFinite(selected) && selected >= 1 ? Math.min(Math.floor(selected), limit) : 1;
}
