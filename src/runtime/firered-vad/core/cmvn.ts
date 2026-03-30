import defaultCmvnJson from '../assets/cmvn.json' with { type: 'json' };
import type { CmvnStats } from '../types.js';

interface CmvnJson {
  readonly means: number[];
  readonly istd: number[];
}

export function loadDefaultCmvn(): CmvnStats {
  const data = defaultCmvnJson as CmvnJson;
  return {
    means: new Float32Array(data.means),
    istd: new Float32Array(data.istd),
  };
}

export function cmvnFromArrays(means: ArrayLike<number>, istd: ArrayLike<number>): CmvnStats {
  return {
    means: Float32Array.from(means),
    istd: Float32Array.from(istd),
  };
}
