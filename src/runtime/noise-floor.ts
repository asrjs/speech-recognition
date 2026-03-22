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
  readonly confirmedSilenceAverage: number;
  readonly confirmedSilenceAverageDbfs: number;
  readonly rejectedCandidateAverage: number;
  readonly rejectedCandidateAverageDbfs: number;
  readonly confirmedBackgroundObservationCount: number;
  readonly rejectedBackgroundObservationCount: number;
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

function clampUnitInterval(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(1, Math.max(0, value));
}

function sanitizeNoiseFloorConfig(
  config: NoiseFloorTrackerConfig,
  fallback: NoiseFloorTrackerConfig,
): NoiseFloorTrackerConfig {
  return {
    initialNoiseFloor: Math.max(
      MIN_NOISE_FLOOR,
      clampFinitePositive(
        config.initialNoiseFloor,
        fallback.initialNoiseFloor,
      ),
    ),
    fastAdaptationRate: clampUnitInterval(
      config.fastAdaptationRate,
      fallback.fastAdaptationRate,
    ),
    slowAdaptationRate: clampUnitInterval(
      config.slowAdaptationRate,
      fallback.slowAdaptationRate,
    ),
    minBackgroundDurationSec: Math.max(
      0,
      Number.isFinite(config.minBackgroundDurationSec)
        ? config.minBackgroundDurationSec
        : fallback.minBackgroundDurationSec,
    ),
  };
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
  private recentConfirmedSilenceObservations: number[] = [];
  private recentRejectedCandidateObservations: number[] = [];

  constructor(config: NoiseFloorTrackerConfig) {
    this.config = sanitizeNoiseFloorConfig(config, {
      initialNoiseFloor: MIN_NOISE_FLOOR,
      fastAdaptationRate: 0,
      slowAdaptationRate: 0,
      minBackgroundDurationSec: 0,
    });
    this.noiseFloor = this.config.initialNoiseFloor;
  }

  updateConfig(config: Partial<NoiseFloorTrackerConfig>): void {
    const previousInitialNoiseFloor = this.config.initialNoiseFloor;
    const nextConfig = sanitizeNoiseFloorConfig(
      {
        ...this.config,
        ...config,
      },
      this.config,
    );
    this.config = nextConfig;
    if (this.config.initialNoiseFloor !== previousInitialNoiseFloor) {
      this.reset();
    }
  }

  observeWindow(
    source: NoiseFloorObservationSource,
    energy: number,
    durationSec: number,
  ): NoiseFloorTrackerState {
    if (
      !Number.isFinite(energy) ||
      energy < 0 ||
      !Number.isFinite(durationSec) ||
      durationSec <= 0
    ) {
      return this.getState();
    }

    const safeEnergy = Math.max(MIN_NOISE_FLOOR, energy);
    const safeDurationSec = durationSec;

    if (source === 'confirmed-silence-window') {
      this.confirmedSilenceDurationSec += safeDurationSec;
      pushObservation(this.recentConfirmedSilenceObservations, safeEnergy);
    } else {
      this.confirmedSilenceDurationSec = 0;
      pushObservation(this.recentRejectedCandidateObservations, safeEnergy);
    }

    const backgroundAverage = this.computeBackgroundAverage(safeEnergy);
    const adaptationRate =
      source === 'confirmed-silence-window'
        ? this.resolveConfirmedSilenceAdaptationRate()
        : Math.max(
            0,
            Math.min(this.config.fastAdaptationRate, this.config.slowAdaptationRate) *
              REJECTED_CANDIDATE_ADAPTATION_SCALE,
          );

    this.noiseFloor =
      this.noiseFloor * (1 - adaptationRate) + backgroundAverage * adaptationRate;
    this.noiseFloor = Math.max(MIN_NOISE_FLOOR, this.noiseFloor);

    return this.getState();
  }

  getState(): NoiseFloorTrackerState {
    const confirmedSilenceAverage =
      computeRobustBackgroundAverage(this.recentConfirmedSilenceObservations) ?? this.noiseFloor;
    const rejectedCandidateAverage =
      computeRobustBackgroundAverage(this.recentRejectedCandidateObservations) ?? this.noiseFloor;
    const backgroundAverage = this.computeBackgroundAverage(this.noiseFloor);
    return {
      noiseFloor: this.noiseFloor,
      noiseFloorDbfs: amplitudeToDbfs(this.noiseFloor),
      backgroundAverage,
      backgroundAverageDbfs: amplitudeToDbfs(backgroundAverage),
      confirmedSilenceDurationSec: this.confirmedSilenceDurationSec,
      confirmedSilenceAverage,
      confirmedSilenceAverageDbfs: amplitudeToDbfs(confirmedSilenceAverage),
      rejectedCandidateAverage,
      rejectedCandidateAverageDbfs: amplitudeToDbfs(rejectedCandidateAverage),
      confirmedBackgroundObservationCount: this.recentConfirmedSilenceObservations.length,
      rejectedBackgroundObservationCount: this.recentRejectedCandidateObservations.length,
    };
  }

  reset(): void {
    this.noiseFloor = clampFinitePositive(this.config.initialNoiseFloor, MIN_NOISE_FLOOR);
    this.confirmedSilenceDurationSec = 0;
    this.recentConfirmedSilenceObservations = [];
    this.recentRejectedCandidateObservations = [];
  }

  private resolveConfirmedSilenceAdaptationRate(): number {
    if (this.config.minBackgroundDurationSec <= 0) {
      return this.config.slowAdaptationRate;
    }
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

  private computeBackgroundAverage(fallback: number): number {
    const confirmedSilenceAverage = computeRobustBackgroundAverage(
      this.recentConfirmedSilenceObservations,
    );
    const rejectedCandidateAverage = computeRobustBackgroundAverage(
      this.recentRejectedCandidateObservations,
    );

    if (confirmedSilenceAverage !== null && rejectedCandidateAverage !== null) {
      return confirmedSilenceAverage * 0.8 + rejectedCandidateAverage * 0.2;
    }
    if (confirmedSilenceAverage !== null) {
      return confirmedSilenceAverage;
    }
    if (rejectedCandidateAverage !== null) {
      return rejectedCandidateAverage;
    }
    return clampFinitePositive(fallback, this.noiseFloor);
  }
}
