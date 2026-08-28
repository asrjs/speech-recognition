export interface TenVadInitPayload {
  readonly hopSize: number;
  readonly threshold: number;
  readonly scriptUrl: string;
  readonly wasmUrl: string;
  readonly fallbackScriptUrl: string | null;
  readonly fallbackWasmUrl: string | null;
}

export interface TenVadProcessPayload {
  readonly samples: Float32Array;
  readonly globalSampleOffset: number;
}

export interface TenVadUpdateConfigPayload {
  readonly hopSize: number;
  readonly threshold: number;
}

export type TenVadControlMessage =
  | { readonly type: 'INIT'; readonly payload: TenVadInitPayload }
  | { readonly type: 'RESET'; readonly payload: null }
  | { readonly type: 'UPDATE_CONFIG'; readonly payload: TenVadUpdateConfigPayload }
  | { readonly type: 'DISPOSE'; readonly payload: null };

export type TenVadProcessMessage = {
  readonly type: 'PROCESS';
  readonly payload: TenVadProcessPayload;
  readonly id?: number;
};

export type TenVadRequestMessage =
  | (TenVadControlMessage & { readonly id: number })
  | TenVadProcessMessage;

export interface TenVadResultPayload {
  readonly probabilities: Float32Array;
  readonly flags: Uint8Array;
  readonly globalSampleOffset: number;
  readonly hopCount: number;
}

interface TenVadSuccessPayload {
  readonly success: true;
}

export type TenVadResponseMessage =
  | {
      readonly type: 'INIT';
      readonly id: number;
      readonly payload: TenVadSuccessPayload & { readonly version: string };
    }
  | { readonly type: 'RESULT'; readonly payload: TenVadResultPayload }
  | { readonly type: 'RESET'; readonly id: number; readonly payload: TenVadSuccessPayload }
  | { readonly type: 'UPDATE_CONFIG'; readonly id: number; readonly payload: TenVadSuccessPayload }
  | { readonly type: 'DISPOSE'; readonly id: number; readonly payload: TenVadSuccessPayload }
  | { readonly type: 'ERROR'; readonly id: number; readonly payload: string };

export type TenVadResponsePayload = TenVadResponseMessage['payload'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isRequestId(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isPositiveFiniteNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value > 0;
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string';
}

function isFloat32Array(value: unknown): value is Float32Array {
  return Object.prototype.toString.call(value) === '[object Float32Array]';
}

function isUint8Array(value: unknown): value is Uint8Array {
  return Object.prototype.toString.call(value) === '[object Uint8Array]';
}

function isTenVadInitPayload(value: unknown): value is TenVadInitPayload {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isPositiveFiniteNumber(value.hopSize) &&
    isFiniteNumber(value.threshold) &&
    value.threshold >= 0 &&
    typeof value.scriptUrl === 'string' &&
    value.scriptUrl.length > 0 &&
    typeof value.wasmUrl === 'string' &&
    value.wasmUrl.length > 0 &&
    isNullableString(value.fallbackScriptUrl) &&
    isNullableString(value.fallbackWasmUrl)
  );
}

function isTenVadProcessPayload(value: unknown): value is TenVadProcessPayload {
  if (!isRecord(value)) {
    return false;
  }
  return isFloat32Array(value.samples) && isFiniteNumber(value.globalSampleOffset);
}

function isTenVadUpdateConfigPayload(value: unknown): value is TenVadUpdateConfigPayload {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isPositiveFiniteNumber(value.hopSize) && isFiniteNumber(value.threshold) && value.threshold >= 0
  );
}

export function isTenVadWorkerRequest(value: unknown): value is TenVadRequestMessage {
  if (!isRecord(value) || typeof value.type !== 'string') {
    return false;
  }

  switch (value.type) {
    case 'INIT':
      return isRequestId(value.id) && isTenVadInitPayload(value.payload);
    case 'PROCESS':
      return (
        isTenVadProcessPayload(value.payload) && (value.id === undefined || isRequestId(value.id))
      );
    case 'RESET':
    case 'DISPOSE':
      return isRequestId(value.id) && value.payload === null;
    case 'UPDATE_CONFIG':
      return isRequestId(value.id) && isTenVadUpdateConfigPayload(value.payload);
    default:
      return false;
  }
}

function isSuccessPayload(value: unknown): value is TenVadSuccessPayload {
  return isRecord(value) && value.success === true;
}

function isTenVadResultPayload(value: unknown): value is TenVadResultPayload {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isFloat32Array(value.probabilities) &&
    isUint8Array(value.flags) &&
    isFiniteNumber(value.globalSampleOffset) &&
    typeof value.hopCount === 'number' &&
    Number.isInteger(value.hopCount) &&
    value.hopCount > 0 &&
    value.probabilities.length >= value.hopCount &&
    value.flags.length >= value.hopCount
  );
}

export function isTenVadWorkerResponse(value: unknown): value is TenVadResponseMessage {
  if (!isRecord(value) || typeof value.type !== 'string') {
    return false;
  }

  switch (value.type) {
    case 'INIT':
      return (
        isRequestId(value.id) &&
        isSuccessPayload(value.payload) &&
        isRecord(value.payload) &&
        typeof value.payload.version === 'string'
      );
    case 'RESULT':
      return isTenVadResultPayload(value.payload);
    case 'RESET':
    case 'UPDATE_CONFIG':
    case 'DISPOSE':
      return isRequestId(value.id) && isSuccessPayload(value.payload);
    case 'ERROR':
      return isRequestId(value.id) && typeof value.payload === 'string';
    default:
      return false;
  }
}
