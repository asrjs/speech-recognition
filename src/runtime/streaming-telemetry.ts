/**
 * Streaming telemetry counters and lifecycle observability for ASR.js.
 *
 * Implements real-time tracking of:
 * - Conservation of audio mass (input, accepted, emitted, dropped, duplicated)
 * - Lifecycle events (capture start/stop, vad speech start/end)
 * - Resource handles (active Web Workers, active AudioContexts)
 * - Memory high water mark and disposal timing
 */

export interface StreamingTelemetrySnapshot {
  readonly inputSampleCount: number;
  readonly acceptedSampleCount: number;
  readonly emittedSampleCount: number;
  readonly droppedSamples: number;
  readonly duplicatedSamples: number;
  readonly captureStartCount: number;
  readonly captureStopCount: number;
  readonly vadStartCount: number;
  readonly vadEndCount: number;
  readonly activeWorkerCount: number;
  readonly activeAudioContextCount: number;
  readonly memoryHighWaterBytes: number;
  readonly disposalDurationMs: number | null;
}

export interface AudioConservationVerification {
  readonly ok: boolean;
  readonly unaccountedDroppedSamples: number;
  readonly detail: string;
}

export class StreamingTelemetryCollector {
  private _inputSampleCount = 0;
  private _acceptedSampleCount = 0;
  private _emittedSampleCount = 0;
  private _droppedSamples = 0;
  private _duplicatedSamples = 0;
  private _captureStartCount = 0;
  private _captureStopCount = 0;
  private _vadStartCount = 0;
  private _vadEndCount = 0;
  private _activeWorkerCount = 0;
  private _activeAudioContextCount = 0;
  private _memoryHighWaterBytes = 0;
  private _disposalDurationMs: number | null = null;

  recordInputSamples(samples: number): void {
    if (samples > 0) {
      this._inputSampleCount += samples;
      this.sampleMemory();
    }
  }

  recordAcceptedSamples(samples: number): void {
    if (samples > 0) {
      this._acceptedSampleCount += samples;
    }
  }

  recordEmittedSamples(samples: number): void {
    if (samples > 0) {
      this._emittedSampleCount += samples;
    }
  }

  recordDroppedSamples(samples: number): void {
    if (samples > 0) {
      this._droppedSamples += samples;
    }
  }

  recordDuplicatedSamples(samples: number): void {
    if (samples > 0) {
      this._duplicatedSamples += samples;
    }
  }

  recordCaptureStart(): void {
    this._captureStartCount += 1;
    this._activeAudioContextCount += 1;
  }

  recordCaptureStop(): void {
    this._captureStopCount += 1;
    if (this._activeAudioContextCount > 0) {
      this._activeAudioContextCount -= 1;
    }
  }

  recordVadStart(): void {
    this._vadStartCount += 1;
  }

  recordVadEnd(): void {
    this._vadEndCount += 1;
  }

  incrementActiveWorkers(): void {
    this._activeWorkerCount += 1;
  }

  decrementActiveWorkers(): void {
    if (this._activeWorkerCount > 0) {
      this._activeWorkerCount -= 1;
    }
  }

  incrementActiveAudioContexts(): void {
    this._activeAudioContextCount += 1;
  }

  decrementActiveAudioContexts(): void {
    if (this._activeAudioContextCount > 0) {
      this._activeAudioContextCount -= 1;
    }
  }

  recordDisposal(durationMs: number): void {
    this._disposalDurationMs = Math.max(0, durationMs);
  }

  recordMemory(bytes: number): void {
    if (bytes > this._memoryHighWaterBytes) {
      this._memoryHighWaterBytes = bytes;
    }
  }

  sampleMemory(): void {
    if (typeof performance !== 'undefined' && (performance as any).memory?.usedJSHeapSize) {
      this.recordMemory((performance as any).memory.usedJSHeapSize);
    } else if (typeof process !== 'undefined' && typeof process.memoryUsage === 'function') {
      try {
        this.recordMemory(process.memoryUsage().heapUsed);
      } catch {
        // ignore
      }
    }
  }

  getSnapshot(): StreamingTelemetrySnapshot {
    return {
      inputSampleCount: this._inputSampleCount,
      acceptedSampleCount: this._acceptedSampleCount,
      emittedSampleCount: this._emittedSampleCount,
      droppedSamples: this._droppedSamples,
      duplicatedSamples: this._duplicatedSamples,
      captureStartCount: this._captureStartCount,
      captureStopCount: this._captureStopCount,
      vadStartCount: this._vadStartCount,
      vadEndCount: this._vadEndCount,
      activeWorkerCount: this._activeWorkerCount,
      activeAudioContextCount: this._activeAudioContextCount,
      memoryHighWaterBytes: this._memoryHighWaterBytes,
      disposalDurationMs: this._disposalDurationMs,
    };
  }

  /**
   * Asserts conservation of audio mass.
   * In a lossless or perfectly accounted pipeline:
   * unaccounted = acceptedSampleCount - (emittedSampleCount + bufferedSamples + droppedSamples)
   */
  verifyConservation(bufferedSamples = 0): AudioConservationVerification {
    const unaccounted = Math.max(
      0,
      this._acceptedSampleCount - (this._emittedSampleCount + bufferedSamples + this._droppedSamples),
    );
    const ok = unaccounted === 0 && this._droppedSamples === 0 && this._duplicatedSamples === 0;
    return {
      ok,
      unaccountedDroppedSamples: unaccounted,
      detail: ok
        ? 'Conservation satisfied: 0 dropped, 0 duplicated, 0 unaccounted samples.'
        : `Conservation violation: ${unaccounted} unaccounted dropped, ${this._droppedSamples} explicit dropped, ${this._duplicatedSamples} duplicated.`,
    };
  }

  reset(): void {
    this._inputSampleCount = 0;
    this._acceptedSampleCount = 0;
    this._emittedSampleCount = 0;
    this._droppedSamples = 0;
    this._duplicatedSamples = 0;
    this._captureStartCount = 0;
    this._captureStopCount = 0;
    this._vadStartCount = 0;
    this._vadEndCount = 0;
    this._activeWorkerCount = 0;
    this._activeAudioContextCount = 0;
    this._memoryHighWaterBytes = 0;
    this._disposalDurationMs = null;
  }
}

let defaultTelemetryCollector: StreamingTelemetryCollector | null = null;

export function getGlobalStreamingTelemetry(): StreamingTelemetryCollector {
  if (!defaultTelemetryCollector) {
    defaultTelemetryCollector = new StreamingTelemetryCollector();
  }
  return defaultTelemetryCollector;
}

export function setGlobalStreamingTelemetry(collector: StreamingTelemetryCollector | null): void {
  defaultTelemetryCollector = collector;
}

export function resetGlobalStreamingTelemetry(): void {
  defaultTelemetryCollector?.reset();
}

export function createStreamingTelemetryCollector(): StreamingTelemetryCollector {
  return new StreamingTelemetryCollector();
}
