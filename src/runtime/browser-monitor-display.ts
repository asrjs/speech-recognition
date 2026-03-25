export const BROWSER_WAVEFORM_SCALE_OPTIONS = [
  { value: 'physical', label: 'Physical', note: 'raw amplitude' },
  { value: 'focus', label: 'Focus', note: 'speech-weighted gain' },
  { value: 'adaptive', label: 'Balanced', note: 'outlier-resistant gain' }
] as const;

export type BrowserWaveformScaleMode =
  (typeof BROWSER_WAVEFORM_SCALE_OPTIONS)[number]['value'];

export const BROWSER_MONITOR_SURFACE_OPTIONS = [
  { value: 'full', label: 'Full', note: 'compact stats + waveform + footer metrics' },
  { value: 'stats-only', label: 'Stats Only', note: 'compact stats only' },
  { value: 'waveform-only', label: 'Waveform Only', note: 'waveform only' },
] as const;

export type BrowserMonitorSurfaceMode =
  (typeof BROWSER_MONITOR_SURFACE_OPTIONS)[number]['value'];

export const DEFAULT_BROWSER_MONITOR_SURFACE_MODE: BrowserMonitorSurfaceMode = 'full';
export const DEFAULT_BROWSER_WAVEFORM_SCALE_MODE: BrowserWaveformScaleMode = 'physical';

export const BROWSER_BINARY_TOGGLE_OPTIONS = [
  { value: 'on', label: 'show' },
  { value: 'off', label: 'hide' },
] as const;

export const BROWSER_REVERSED_BINARY_TOGGLE_OPTIONS = [
  { value: 'off', label: 'hide' },
  { value: 'on', label: 'show' },
] as const;

export type BrowserMonitorDisplayControlDefinition = {
  id: 'surface' | 'scale' | 'speech-threshold' | 'ten-vad-threshold' | 'pre-vad-overlay';
  label: string;
  note: string;
};

export const BROWSER_MONITOR_DISPLAY_CONTROL_DEFINITIONS = {
  surface: {
    id: 'surface',
    label: 'Surface',
    note: 'compact stats, waveform, and footer layout',
  },
  scale: {
    id: 'scale',
    label: 'View',
    note: 'waveform gain model',
  },
  speechThreshold: {
    id: 'speech-threshold',
    label: 'Gate lines',
    note: 'waveform legend',
  },
  tenVadThreshold: {
    id: 'ten-vad-threshold',
    label: 'TEN line',
    note: 'diagnostic only',
  },
  preVadOverlay: {
    id: 'pre-vad-overlay',
    label: 'RMS',
    note: 'diagnostic',
  },
} as const satisfies Record<
  'surface' | 'scale' | 'speechThreshold' | 'tenVadThreshold' | 'preVadOverlay',
  BrowserMonitorDisplayControlDefinition
>;

export function shouldRenderBrowserCompactStats(
  mode: BrowserMonitorSurfaceMode,
): boolean {
  return mode !== 'waveform-only';
}

export function shouldRenderBrowserWaveform(
  mode: BrowserMonitorSurfaceMode,
): boolean {
  return mode !== 'stats-only';
}

export function shouldRenderBrowserWaveformFooterMetrics(
  mode: BrowserMonitorSurfaceMode,
): boolean {
  return mode === 'full';
}

export function shouldDisableBrowserWaveformVisualControls(
  mode: BrowserMonitorSurfaceMode,
): boolean {
  return mode === 'stats-only';
}
