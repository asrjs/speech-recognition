import type { RuntimeLogger } from '../types/index.js';

/**
 * Library-wide sink for recoverable warnings raised outside a runtime, such as
 * cache fallbacks, Hugging Face listing failures and best-effort session
 * release. Runtime-scoped events still go to `SpeechRuntimeHooks.logger`.
 */
export type SpeechDiagnosticsLogger = Pick<RuntimeLogger, 'warn'>;

const consoleDiagnosticsLogger: SpeechDiagnosticsLogger = {
  warn(message, meta) {
    const detail = meta && 'error' in meta && Object.keys(meta).length === 1 ? meta.error : meta;
    if (detail === undefined) {
      console.warn(message);
    } else {
      console.warn(message, detail);
    }
  },
};

let diagnosticsLogger: SpeechDiagnosticsLogger | null = consoleDiagnosticsLogger;

/**
 * Redirects library warnings to an application logger. Pass `null` to silence
 * them; omit the argument to restore the default `console.warn` output.
 */
export function setSpeechDiagnosticsLogger(
  logger: SpeechDiagnosticsLogger | null = consoleDiagnosticsLogger,
): void {
  diagnosticsLogger = logger;
}

/** Reports a recoverable library warning. A failing logger never throws here. */
export function reportSpeechWarning(message: string, meta?: Record<string, unknown>): void {
  try {
    diagnosticsLogger?.warn?.(message, meta);
  } catch {
    // Diagnostics must not change the outcome of the operation being reported.
  }
}
