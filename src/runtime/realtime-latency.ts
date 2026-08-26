/**
 * Latency and stability tracking for the realtime transcription controller.
 *
 * The controller consumes microphone audio roughly in real time, so a useful
 * latency definition is anchored to the audio timeline: for any audio frame,
 * we know the wall-clock moment it was ingested. This tracker records those
 * ingest marks and, for each published update, measures how long after its
 * audio boundary the update was emitted. That yields the operator-relevant
 * numbers: first-partial latency (speech start to first text), end-of-utterance
 * latency (speech end to final commit), per-update pipeline lag, transcribe
 * processing latency, and committed-text stability (shrink/duplicate counts).
 *
 * All time reads go through an injectable `now` clock so behavior is unit
 * testable with a deterministic fake clock.
 */

export interface RealtimeLatencyTrackerOptions {
  /** Audio sample rate used to interpret frame numbers. */
  readonly sampleRate: number;
  /** Injectable monotonic wall clock in milliseconds. Defaults to performance.now. */
  readonly now?: () => number;
  /** Maximum completed utterances retained in the summary. Defaults to 50. */
  readonly maxHistory?: number;
}

export interface RealtimeLatencyUpdateRecord {
  readonly kind: 'partial' | 'final';
  readonly trigger: string;
  readonly windowStartFrame: number;
  readonly windowEndFrame: number;
  /** Audio frame marking utterance speech end, when known (final updates). */
  readonly speechEndFrame?: number | null;
  readonly revision: number;
  readonly committedText: string;
  readonly previewText: string;
  /**
   * Wall-clock duration of the transcribe callback that produced this update.
   * When omitted, the tracker derives it from the most recent
   * noteTranscribeStart mark instead.
   */
  readonly processLatencyMs?: number;
}

export interface RealtimeLatencyUpdateSummary {
  readonly kind: 'partial' | 'final';
  readonly trigger: string;
  readonly revision: number;
  /** Wall-clock delay between ingesting windowEndFrame and emitting the update. */
  readonly emitLagMs: number | null;
  readonly processLatencyMs: number;
}

export interface RealtimeUtteranceLatency {
  /** Wall time from ingesting the utterance's first audio to the first partial. */
  readonly firstPartialLatencyMs: number | null;
  /** Wall time from ingesting the utterance's speech end to the final update. */
  readonly endOfUtteranceLatencyMs: number | null;
  readonly partialCount: number;
  readonly finalCount: number;
  /** Times the committed prefix became shorter between updates (rework). */
  readonly commitShrinkCount: number;
  /** Times an update added no new committed or preview text (stall/duplicate). */
  readonly stagnantUpdateCount: number;
  readonly updates: readonly RealtimeLatencyUpdateSummary[];
}

export interface RealtimeLatencySummary {
  readonly sampleRate: number;
  readonly inProgressUtterance: RealtimeUtteranceLatency | null;
  readonly completedUtterances: readonly RealtimeUtteranceLatency[];
  readonly lastFirstPartialLatencyMs: number | null;
  readonly lastEndOfUtteranceLatencyMs: number | null;
  readonly meanProcessLatencyMs: number | null;
  readonly p50ProcessLatencyMs: number | null;
  readonly p95ProcessLatencyMs: number | null;
  readonly meanEmitLagMs: number | null;
  readonly p50EmitLagMs: number | null;
  readonly p95EmitLagMs: number | null;
  readonly totalUpdates: number;
  readonly totalPartials: number;
  readonly totalFinals: number;
  readonly totalCommitShrinkCount: number;
  readonly totalStagnantUpdateCount: number;
}

interface IngestMark {
  readonly frame: number;
  readonly wallMs: number;
}

const DEFAULT_HISTORY = 50;
const MAX_INGEST_MARKS = 4096;

function percentile(values: readonly number[], quantile: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil(quantile * sorted.length) - 1);
  return sorted[Math.max(0, index)] ?? null;
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

interface UtteranceState {
  firstPartialLatencyMs: number | null;
  endOfUtteranceLatencyMs: number | null;
  partialCount: number;
  finalCount: number;
  commitShrinkCount: number;
  stagnantUpdateCount: number;
  updates: RealtimeLatencyUpdateSummary[];
  lastCommittedLength: number;
  lastTotalTextLength: number;
}

function createUtteranceState(): UtteranceState {
  return {
    firstPartialLatencyMs: null,
    endOfUtteranceLatencyMs: null,
    partialCount: 0,
    finalCount: 0,
    commitShrinkCount: 0,
    stagnantUpdateCount: 0,
    updates: [],
    lastCommittedLength: 0,
    lastTotalTextLength: 0,
  };
}

export class RealtimeLatencyTracker {
  readonly sampleRate: number;
  private readonly now: () => number;
  private readonly maxHistory: number;
  private ingestMarks: IngestMark[] = [];
  private utterance = createUtteranceState();
  private completed: RealtimeUtteranceLatency[] = [];
  private processSamples: number[] = [];
  private emitLagSamples: number[] = [];
  private totalUpdates = 0;
  private totalPartials = 0;
  private totalFinals = 0;
  private totalCommitShrink = 0;
  private totalStagnant = 0;
  private lastTranscribeStartWall: number | null = null;

  constructor(options: RealtimeLatencyTrackerOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('RealtimeLatencyTracker requires a positive sampleRate.');
    }
    this.sampleRate = options.sampleRate;
    this.now =
      options.now ??
      (() =>
        typeof performance !== 'undefined'
          ? performance.now()
          : (Date as unknown as { now(): number }).now());
    this.maxHistory = Math.max(1, Math.floor(options.maxHistory ?? DEFAULT_HISTORY));
  }

  /** Record that audio ingestion has reached `endFrame` at the current wall time. */
  noteIngest(endFrame: number): void {
    const last = this.ingestMarks[this.ingestMarks.length - 1];
    if (last && endFrame <= last.frame) {
      return;
    }
    this.ingestMarks.push({ frame: endFrame, wallMs: this.now() });
    if (this.ingestMarks.length > MAX_INGEST_MARKS) {
      this.ingestMarks = this.ingestMarks.slice(-MAX_INGEST_MARKS / 2);
    }
  }

  /** Mark the wall time immediately before a transcribe round-trip starts. */
  noteTranscribeStart(): void {
    this.lastTranscribeStartWall = this.now();
  }

  /** Publish one controller update and fold it into utterance/session metrics. */
  noteUpdate(record: RealtimeLatencyUpdateRecord): void {
    const emitWall = this.now();
    const ingestWall = this.ingestWallTimeForFrame(record.windowEndFrame);
    const emitLagMs = ingestWall === null ? null : emitWall - ingestWall;
    const processLatencyMs =
      record.processLatencyMs ??
      (this.lastTranscribeStartWall === null ? 0 : emitWall - this.lastTranscribeStartWall);

    this.totalUpdates += 1;
    this.processSamples.push(processLatencyMs);
    if (emitLagMs !== null) {
      this.emitLagSamples.push(emitLagMs);
    }
    if (record.kind === 'final') {
      this.totalFinals += 1;
    } else {
      this.totalPartials += 1;
    }

    const state = this.utterance;
    const committedLength = record.committedText.length;
    if (committedLength < state.lastCommittedLength) {
      state.commitShrinkCount += 1;
      this.totalCommitShrink += 1;
    }
    state.lastCommittedLength = Math.max(committedLength, 0);

    const totalTextLength = committedLength + record.previewText.length;
    if (totalTextLength <= state.lastTotalTextLength && totalTextLength > 0) {
      state.stagnantUpdateCount += 1;
      this.totalStagnant += 1;
    }
    state.lastTotalTextLength = Math.max(totalTextLength, state.lastTotalTextLength);

    if (record.kind === 'partial') {
      state.partialCount += 1;
      if (state.firstPartialLatencyMs === null) {
        const startWall = this.ingestWallTimeForFrame(record.windowStartFrame);
        state.firstPartialLatencyMs = startWall === null ? null : emitWall - startWall;
      }
    }

    state.updates.push({
      kind: record.kind,
      trigger: record.trigger,
      revision: record.revision,
      emitLagMs,
      processLatencyMs,
    });

    if (record.kind === 'final') {
      state.finalCount += 1;
      const speechEndFrame = record.speechEndFrame ?? record.windowEndFrame;
      const speechEndWall = this.ingestWallTimeForFrame(speechEndFrame);
      if (speechEndWall !== null) {
        state.endOfUtteranceLatencyMs = emitWall - speechEndWall;
      }
      this.closeUtterance();
    }
  }

  /** Discard the in-progress utterance (new session) while keeping history. */
  reset(): void {
    this.utterance = createUtteranceState();
    this.ingestMarks = [];
    this.lastTranscribeStartWall = null;
  }

  getSummary(): RealtimeLatencySummary {
    const completed = [...this.completed];
    const inProgress =
      this.utterance.updates.length > 0 ||
      this.utterance.partialCount > 0 ||
      this.utterance.finalCount > 0
        ? this.snapshotUtterance(this.utterance)
        : null;
    const lastCompleted = completed.length > 0 ? completed[completed.length - 1] : null;
    return {
      sampleRate: this.sampleRate,
      inProgressUtterance: inProgress,
      completedUtterances: completed,
      lastFirstPartialLatencyMs: lastCompleted?.firstPartialLatencyMs ?? null,
      lastEndOfUtteranceLatencyMs: lastCompleted?.endOfUtteranceLatencyMs ?? null,
      meanProcessLatencyMs: mean(this.processSamples),
      p50ProcessLatencyMs: percentile(this.processSamples, 0.5),
      p95ProcessLatencyMs: percentile(this.processSamples, 0.95),
      meanEmitLagMs: mean(this.emitLagSamples),
      p50EmitLagMs: percentile(this.emitLagSamples, 0.5),
      p95EmitLagMs: percentile(this.emitLagSamples, 0.95),
      totalUpdates: this.totalUpdates,
      totalPartials: this.totalPartials,
      totalFinals: this.totalFinals,
      totalCommitShrinkCount: this.totalCommitShrink,
      totalStagnantUpdateCount: this.totalStagnant,
    };
  }

  /** Wall time when ingestion first reached (or passed) the given audio frame. */
  private ingestWallTimeForFrame(frame: number): number | null {
    for (const mark of this.ingestMarks) {
      if (mark.frame >= frame) {
        return mark.wallMs;
      }
    }
    return null;
  }

  private closeUtterance(): void {
    if (this.utterance.updates.length === 0) {
      return;
    }
    this.completed.push(this.snapshotUtterance(this.utterance));
    if (this.completed.length > this.maxHistory) {
      this.completed = this.completed.slice(-this.maxHistory);
    }
    this.utterance = createUtteranceState();
  }

  private snapshotUtterance(
    state: ReturnType<typeof createUtteranceState>,
  ): RealtimeUtteranceLatency {
    return {
      firstPartialLatencyMs: state.firstPartialLatencyMs,
      endOfUtteranceLatencyMs: state.endOfUtteranceLatencyMs,
      partialCount: state.partialCount,
      finalCount: state.finalCount,
      commitShrinkCount: state.commitShrinkCount,
      stagnantUpdateCount: state.stagnantUpdateCount,
      updates: [...state.updates],
    };
  }
}
