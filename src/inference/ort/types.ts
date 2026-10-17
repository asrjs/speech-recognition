/**
 * Unified ORT Runtime Layer - Types & Interfaces
 *
 * Model-agnostic interfaces for ONNX Runtime (Node native and Web/WASM/WebGPU).
 */

export interface OrtWasmEnvLike {
  wasmPaths?: string;
  numThreads?: number;
  simd?: boolean;
  proxy?: boolean;
}

export interface OrtWebGpuEnvLike {
  profiling?: unknown;
  device?: unknown;
  [key: string]: unknown;
}

export interface OrtEnvLike {
  wasm: OrtWasmEnvLike;
  webgpu?: OrtWebGpuEnvLike;
  versions?: {
    common?: string;
    [key: string]: string | undefined;
  };
  logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal';
  logSeverityLevel?: number;
  [key: string]: unknown;
}

export interface OrtTensorLike<TData extends ArrayBufferView = ArrayBufferView> {
  readonly data: TData;
  readonly dims: readonly number[];
  readonly type?: string;
  readonly location?: string;
  readonly gpuBuffer?: unknown;
  getData?(releaseData?: boolean): Promise<TData>;
  dispose?(): void;
}

export interface OrtSessionLike {
  readonly inputNames?: readonly string[];
  readonly outputNames?: readonly string[];
  readonly inputMetadata?:
    | readonly {
        readonly name?: string;
        readonly type?: string;
        readonly shape?: readonly (number | string)[];
      }[]
    | Record<string, { readonly type?: string; readonly shape?: readonly (number | string)[] }>;
  run(
    feeds: Record<string, unknown>,
    fetchesOrOptions?: unknown,
    options?: Record<string, unknown>,
  ): Promise<Record<string, OrtTensorLike>>;
  release?(): void | Promise<void>;
}

export interface OrtTensorConstructor {
  new <TData extends ArrayBufferView>(
    type: string,
    data: TData,
    dims: readonly number[],
  ): OrtTensorLike<TData>;
  fromGpuBuffer?(
    buffer: unknown,
    options: {
      dataType?: string;
      dims: readonly number[];
      download?: () => Promise<ArrayBufferView>;
      dispose?: () => void;
    },
  ): OrtTensorLike;
}

export interface OrtModuleLike {
  readonly env: OrtEnvLike;
  readonly Tensor?: OrtTensorConstructor | unknown;
  readonly InferenceSession: {
    create(
      model: string | Uint8Array,
      options?: Record<string, unknown>,
    ): Promise<OrtSessionLike>;
  };
}

export type OrtExecutionProviderOption =
  | string
  | {
      readonly name: string;
      readonly deviceType?: 'gpu' | 'cpu';
      readonly powerPreference?: 'high-performance' | 'low-power' | 'default';
      readonly [key: string]: unknown;
    };

export interface UnifiedOrtInitOptions {
  readonly customWasmPaths?: string;
  readonly wasmPaths?: string;
  readonly ortVersion?: string;
  readonly cpuThreads?: number;
  readonly simd?: boolean;
  readonly proxy?: boolean;
  readonly logSeverityLevel?: number;
  readonly enableProfiling?: boolean;
  readonly signal?: { readonly aborted: boolean } | null;
}

export type OrtPreferredOutputLocation =
  | 'cpu'
  | 'gpu-buffer'
  | Readonly<Record<string, 'cpu' | 'gpu-buffer'>>;

export interface UnifiedExternalDataFile {
  readonly dataUrl: string;
  readonly path: string;
}

export interface UnifiedOrtSessionOptions {
  readonly backendId?: string;
  readonly deviceType?: 'gpu' | 'cpu';
  readonly powerPreference?: 'high-performance' | 'low-power';
  readonly allowWasmFallback?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly graphOptimizationLevel?: 'all' | 'basic' | 'disabled';
  readonly executionMode?: 'sequential' | 'parallel';
  readonly enableCpuMemArena?: boolean;
  readonly enableMemPattern?: boolean;
  readonly preferredOutputLocation?: OrtPreferredOutputLocation;
  readonly enableGraphCapture?: boolean;
  readonly freeDimensionOverrides?: Record<string, number>;
  readonly externalData?: readonly UnifiedExternalDataFile[];
  readonly externalDataUrl?: string;
  readonly externalDataPath?: string;
  readonly webgpuOptions?: Record<string, unknown>;
  readonly signal?: { readonly aborted: boolean } | null;
  readonly sessionOptions?: Record<string, unknown>;
}

export interface UnifiedSessionCreateResult {
  readonly session: OrtSessionLike;
  readonly resolvedBackendId: 'webgpu' | 'wasm' | 'native-cpu';
  readonly fellBackToWasm: boolean;
}
