export interface NoiseFloorTrackerConfig {
  readonly initialNoiseFloor: number;
  readonly fastAdaptationRate: number;
  readonly slowAdaptationRate: number;
  readonly minBackgroundDurationSec: number;
}

export type NoiseFloorObservationSource =
  | 'confirmed-silence-window'
  | 'rejected-candidate-window';

export interface NoiseFloorTrackerState {
  readonly noiseFloor: number;
  readonly noiseFloorDbfs: number;
  readonly backgroundAverage: number;
  readonly backgroundAverageDbfs: number;
  readonly confirmedSilenceDurationSec: number;
  readonly recentBackgroundObservationCount: number;
}

const MIN_NOISE_FLOOR = 0.00001;
const MAX_BACKGROUND_OBSERVATIONS = 64;
const ROBUST_BACKGROUND_LOWER_FRACTION = 0.6;
const REJECTED_CANDIDATE_ADAPTATION_SCALE = 0.5;

export function amplitudeToDbfs(value: number, floorDbfs = -100): number {
  if (!Number.isFinite(value) || value <= 0) {
    return floorDbfs;
  }
  return Math.max(floorDbfs, 20 * Math.log10(value));
}

function clampFinitePositive(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  return value;
}

function pushObservation(target: number[], value: number): void {
  target.push(value);
  if (target.length > MAX_BACKGROUND_OBSERVATIONS) {
    target.shift();
  }
}

function computeRobustBackgroundAverage(observations: readonly number[]): number | null {
  if (observations.length === 0) {
    return null;
  }
  const sorted = [...observations].sort((left, right) => left - right);
  const sampleCount = Math.max(
    1,
    Math.floor(sorted.length * ROBUST_BACKGROUND_LOWER_FRACTION),
  );
  const slice = sorted.slice(0, sampleCount);
  const sum = slice.reduce((total, value) => total + value, 0);
  return sum / Math.max(1, slice.length);
}

export class NoiseFloorTracker {
  private config: NoiseFloorTrackerConfig;
  private noiseFloor: number;
  private confirmedSilenceDurationSec = 0;
  private recentBackgroundObservations: number[] = [];

  constructor(config: NoiseFloorTrackerConfig) {
    this.config = {
      ...config,
    };
    this.noiseFloor = clampFinitePositive(config.initialNoiseFloor, MIN_NOISE_FLOOR);
  }

  updateConfig(config: Partial<NoiseFloorTrackerConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
    if (
      config.initialNoiseFloor !== undefined &&
      Number.isFinite(config.initialNoiseFloor) &&
      config.initialNoiseFloor > 0
    ) {
      this.noiseFloor = clampFinitePositive(config.initialNoiseFloor, MIN_NOISE_FLOOR);
    }
  }

  observeWindow(
    source: NoiseFloorObservationSource,
    energy: number,
    durationSec: number,
  ): NoiseFloorTrackerState {
    const safeEnergy = clampFinitePositive(energy, MIN_NOISE_FLOOR);
    const safeDurationSec = Math.max(0, Number.isFinite(durationSec) ? durationSec : 0);

    if (source === 'confirmed-silence-window') {
      this.confirmedSilenceDurationSec += safeDurationSec;
    } else {
      this.confirmedSilenceDurationSec = 0;
    }

    pushObservation(this.recentBackgroundObservations, safeEnergy);
    const backgroundAverage =
      computeRobustBackgroundAverage(this.recentBackgroundObservations) ?? safeEnergy;
    const adaptationRate =
      source === 'confirmed-silence-window'
        ? this.resolveConfirmedSilenceAdaptationRate()
        : Math.max(
            0.0001,
            Math.min(this.config.fastAdaptationRate, this.config.slowAdaptationRate) *
              REJECTED_CANDIDATE_ADAPTATION_SCALE,
          );

    this.noiseFloor =
      this.noiseFloor * (1 - adaptationRate) + backgroundAverage * adaptationRate;
    this.noiseFloor = Math.max(MIN_NOISE_FLOOR, this.noiseFloor);

    return this.getState();
  }

  getState(): NoiseFloorTrackerState {
    const backgroundAverage =
      computeRobustBackgroundAverage(this.recentBackgroundObservations) ?? this.noiseFloor;
    return {
      noiseFloor: this.noiseFloor,
      noiseFloorDbfs: amplitudeToDbfs(this.noiseFloor),
      backgroundAverage,
      backgroundAverageDbfs: amplitudeToDbfs(backgroundAverage),
      confirmedSilenceDurationSec: this.confirmedSilenceDurationSec,
      recentBackgroundObservationCount: this.recentBackgroundObservations.length,
    };
  }

  reset(): void {
    this.noiseFloor = clampFinitePositive(this.config.initialNoiseFloor, MIN_NOISE_FLOOR);
    this.confirmedSilenceDurationSec = 0;
    this.recentBackgroundObservations = [];
  }

  private resolveConfirmedSilenceAdaptationRate(): number {
    if (this.confirmedSilenceDurationSec >= this.config.minBackgroundDurationSec) {
      return this.config.slowAdaptationRate;
    }
    const blend = Math.max(
      0,
      Math.min(1, this.confirmedSilenceDurationSec / this.config.minBackgroundDurationSec),
    );
    return (
      this.config.fastAdaptationRate * (1 - blend) + this.config.slowAdaptationRate * blend
    );
  }
}
