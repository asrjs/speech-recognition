/**
 * Whisper Quality Gates — re-exports from standalone quality/ module.
 *
 * These functions have been moved to src/quality/ for model-agnostic use.
 * Re-exported here for backward compatibility.
 *
 * @deprecated Import from src/quality/ directly for model-agnostic usage.
 */

export {
  compressionRatioGate,
  logProbGate,
  noSpeechGate,
  entropyGate,
  evaluateGates,
} from '../../quality/index.js';
