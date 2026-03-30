import {
  TenVadAdapter,
  resolveDefaultTenVadAssetUrls,
  resolveSupportedTenVadHopSize,
  resolveTenVadAssetUrls,
  type TenVadAdapterConfig,
  type TenVadAdapterOptions,
  type TenVadRecentResult,
  type TenVadAssetUrls,
  type ResolvedTenVadAssetUrls,
} from './ten-vad-browser.js';

export type FireRedVadAdapterConfig = TenVadAdapterConfig;
export type FireRedVadAdapterOptions = TenVadAdapterOptions;
export type FireRedVadRecentResult = TenVadRecentResult;
export type FireRedVadModelUrls = TenVadAssetUrls;
export type ResolvedFireRedVadModelUrls = ResolvedTenVadAssetUrls;

export class FireRedVadAdapter extends TenVadAdapter {}

export function resolveSupportedFireRedVadHopSize(
  sampleRate?: number,
  preferredHopSize?: number,
): number {
  return resolveSupportedTenVadHopSize(sampleRate, preferredHopSize);
}

export function resolveDefaultFireRedVadModelUrls(): FireRedVadModelUrls {
  return resolveDefaultTenVadAssetUrls();
}

export function resolveFireRedVadModelUrls(config: FireRedVadAdapterConfig = {}): ResolvedFireRedVadModelUrls {
  return resolveTenVadAssetUrls(config);
}
