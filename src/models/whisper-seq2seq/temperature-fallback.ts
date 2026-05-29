/**
 * Temperature Fallback — re-exports from standalone quality/ module.
 * @deprecated Import from src/quality/ directly.
 */

export {
  DEFAULT_TEMPERATURES,
  withTemperatureFallback,
  type FallbackResult,
  type TranscribeAttempt,
} from '../../quality/index.js';
