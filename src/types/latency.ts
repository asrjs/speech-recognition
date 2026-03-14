export interface FrameOffset {
  readonly frameIndex: number;
  readonly timeSeconds: number;
}

export interface LatencyMetrics {
  readonly processLatencyMs: number;
  readonly rtf: number;
  readonly firstTokenLatencyMs?: number;
}
