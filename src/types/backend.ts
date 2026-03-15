export type BackendEnvironment = 'browser' | 'node' | 'worker';
export type PrecisionMode = 'fp32' | 'fp16' | 'int8';
export type AccelerationClass = 'cpu' | 'gpu' | 'npu' | 'hybrid';

export interface BackendCapabilities {
  readonly id: string;
  readonly displayName: string;
  readonly available: boolean;
  readonly priority: number;
  readonly environments: readonly BackendEnvironment[];
  readonly acceleration: readonly AccelerationClass[];
  readonly supportedPrecisions: readonly PrecisionMode[];
  readonly supportsFp16: boolean;
  readonly supportsInt8: boolean;
  readonly supportsSharedArrayBuffer: boolean;
  readonly requiresSharedArrayBuffer: boolean;
  readonly fallbackSuitable: boolean;
  readonly experimental?: boolean;
  readonly provider?: string;
  readonly adapter?: string;
  readonly notes: readonly string[];
}

export interface BackendProbeContext {
  readonly environment?: BackendEnvironment;
  readonly preferredDevice?: AccelerationClass;
}

export interface BackendSelectionCriteria {
  readonly preferredBackendIds?: readonly string[];
  readonly requiredPrecision?: PrecisionMode;
  readonly preferAcceleration?: boolean;
  readonly requireSharedArrayBuffer?: boolean;
  readonly environment?: BackendEnvironment;
  readonly allowExperimental?: boolean;
}

export interface BackendExecutionRequest {
  readonly modelFamily: string;
  readonly modelId: string;
  readonly precision?: PrecisionMode;
  readonly artifactHints?: readonly string[];
}

export interface BackendExecutionContext {
  readonly backendId: string;
  readonly capabilities: BackendCapabilities;
  readonly provider?: string;
  dispose(): Promise<void> | void;
}

export interface ExecutionBackend {
  readonly id: string;
  readonly displayName: string;
  probeCapabilities(context?: BackendProbeContext): Promise<BackendCapabilities>;
  createExecutionContext(request: BackendExecutionRequest): Promise<BackendExecutionContext>;
}
