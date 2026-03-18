import {
  getDefaultNemoTdtWeightSetup,
  normalizeNemoTdtWeightBackend,
  pickPreferredNemoTdtQuantization,
  type NemoTdtWeightSetup,
} from '../nemo-tdt/weights.js';
import type { NemoRnntQuantization } from './types.js';

export type NemoRnntWeightSetup = NemoTdtWeightSetup;

export function normalizeNemoRnntWeightBackend(backendId: string): 'webgpu' | 'wasm' {
  return normalizeNemoTdtWeightBackend(backendId);
}

export function getDefaultNemoRnntWeightSetup(backendId: string): NemoRnntWeightSetup {
  return getDefaultNemoTdtWeightSetup(backendId);
}

export function pickPreferredNemoRnntQuantization(
  available: readonly NemoRnntQuantization[],
  backendId: string,
  role: 'encoder' | 'decoder',
): NemoRnntQuantization {
  return pickPreferredNemoTdtQuantization(available, backendId, role);
}
