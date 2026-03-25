export interface BrowserCompactStatsSourceSnapshot {
  readonly rough?: {
    readonly snr?: number | null;
    readonly snrThreshold?: number | null;
    readonly minSnrThreshold?: number | null;
    readonly levelDbfs?: number | null;
    readonly levelWindowDbfs?: number | null;
    readonly isSpeech?: boolean | null;
  } | null;
  readonly activeSegment?: unknown;
  readonly recentSegments?: readonly unknown[] | null;
  readonly recentDecisions?: ReadonlyArray<{ readonly message?: string | null }> | null;
  readonly capture?: {
    readonly processingSampleRate?: number | null;
    readonly inputSampleRate?: number | null;
  } | null;
  readonly plot?: {
    readonly startFrame: number;
    readonly endFrame: number;
  } | null;
}

export interface BrowserCompactStatsSnapshot {
  readonly speaking: boolean;
  readonly currentSnr: number | null;
  readonly snrThreshold: number | null;
  readonly minSnrThreshold: number | null;
  readonly signalDbfs: number | null;
  readonly averageDbfs: number | null;
  readonly gateDbfs: number | null;
  readonly noiseDbfs: number | null;
  readonly targetRate: number | null;
  readonly visibleSeconds: number | null;
  readonly recentSegments: number;
  readonly recentRejected: number;
}

export interface BrowserCompactStatsRenderer {
  update(snapshot: BrowserCompactStatsSnapshot | null | undefined): void;
  dispose(): void;
}

function formatThreshold(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '--';
}

function formatCompactDbfs(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '--';
}

function formatCompactHz(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)}Hz` : 'N/A';
}

function formatCompactSeconds(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)}s` : '--';
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function metricHtml(label: string, value: string): string {
  return `<span style="display:inline-flex;align-items:baseline;gap:4px;white-space:nowrap;"><span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;">${escapeHtml(label)}</span><span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:700;color:#0f172a;">${escapeHtml(value)}</span></span>`;
}

function miniHtml(label: string, value: string, active = false): string {
  return `<span style="display:inline-flex;align-items:baseline;gap:4px;white-space:nowrap;"><span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;">${escapeHtml(label)}</span><span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;font-weight:700;color:${active ? '#047857' : '#0f172a'};">${escapeHtml(value)}</span></span>`;
}

export function resolveBrowserCompactStatsSnapshot(
  snapshot: BrowserCompactStatsSourceSnapshot | null | undefined,
  liveNoiseDbfs: number | null | undefined,
  liveSpeechDbfs: number | null | undefined,
): BrowserCompactStatsSnapshot {
  const visibleSeconds =
    snapshot?.plot && typeof snapshot?.capture?.processingSampleRate === 'number'
      ? (snapshot.plot.endFrame - snapshot.plot.startFrame) /
        Math.max(1, snapshot.capture.processingSampleRate)
      : null;
  const recentRejected = Math.max(
    0,
    (snapshot?.recentDecisions ?? []).filter((entry) =>
      String(entry?.message ?? '').includes('Segment rejected'),
    ).length,
  );
  return {
    speaking: Boolean(snapshot?.activeSegment || snapshot?.rough?.isSpeech),
    currentSnr:
      typeof snapshot?.rough?.snr === 'number' && Number.isFinite(snapshot.rough.snr)
        ? snapshot.rough.snr
        : null,
    snrThreshold:
      typeof snapshot?.rough?.snrThreshold === 'number' &&
      Number.isFinite(snapshot.rough.snrThreshold)
        ? snapshot.rough.snrThreshold
        : null,
    minSnrThreshold:
      typeof snapshot?.rough?.minSnrThreshold === 'number' &&
      Number.isFinite(snapshot.rough.minSnrThreshold)
        ? snapshot.rough.minSnrThreshold
        : null,
    signalDbfs:
      typeof snapshot?.rough?.levelDbfs === 'number' &&
      Number.isFinite(snapshot.rough.levelDbfs)
        ? snapshot.rough.levelDbfs
        : null,
    averageDbfs:
      typeof snapshot?.rough?.levelWindowDbfs === 'number' &&
      Number.isFinite(snapshot.rough.levelWindowDbfs)
        ? snapshot.rough.levelWindowDbfs
        : null,
    gateDbfs:
      typeof liveSpeechDbfs === 'number' && Number.isFinite(liveSpeechDbfs)
        ? liveSpeechDbfs
        : null,
    noiseDbfs:
      typeof liveNoiseDbfs === 'number' && Number.isFinite(liveNoiseDbfs)
        ? liveNoiseDbfs
        : null,
    targetRate:
      typeof snapshot?.capture?.processingSampleRate === 'number' &&
      Number.isFinite(snapshot.capture.processingSampleRate)
        ? snapshot.capture.processingSampleRate
        : typeof snapshot?.capture?.inputSampleRate === 'number' &&
            Number.isFinite(snapshot.capture.inputSampleRate)
          ? snapshot.capture.inputSampleRate
          : null,
    visibleSeconds,
    recentSegments: snapshot?.recentSegments?.length ?? 0,
    recentRejected,
  };
}

export function createBrowserCompactStatsRenderer(
  container: HTMLElement,
): BrowserCompactStatsRenderer {
  container.setAttribute('role', 'status');
  container.setAttribute('aria-live', 'polite');
  container.setAttribute('aria-label', 'Live detector summary');

  return {
    update(snapshot: BrowserCompactStatsSnapshot | null | undefined): void {
      const snrActive =
        typeof snapshot?.currentSnr === 'number' &&
        typeof snapshot?.snrThreshold === 'number' &&
        snapshot.currentSnr >= snapshot.snrThreshold;
      const snrPercent =
        typeof snapshot?.currentSnr === 'number' && Number.isFinite(snapshot.currentSnr)
          ? clamp((snapshot.currentSnr / 20) * 100, 0, 100)
          : 0;
      const snrThresholdPercent =
        typeof snapshot?.snrThreshold === 'number' && Number.isFinite(snapshot.snrThreshold)
          ? clamp((snapshot.snrThreshold / 20) * 100, 0, 100)
          : 0;
      const minThresholdPercent =
        typeof snapshot?.minSnrThreshold === 'number' &&
        Number.isFinite(snapshot.minSnrThreshold)
          ? clamp((snapshot.minSnrThreshold / 20) * 100, 0, 100)
          : 0;

      container.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:4px;margin-bottom:8px;padding:6px 8px;border-radius:8px;background:linear-gradient(180deg, rgba(59, 130, 246, 0.06), rgba(15, 23, 42, 0.02));border:1px solid rgba(148, 163, 184, 0.28);">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;min-width:0;">
            <div style="display:flex;align-items:center;gap:6px;min-width:126px;flex-shrink:0;">
              <span aria-hidden="true" style="width:8px;height:8px;border-radius:999px;flex-shrink:0;background:${snapshot?.speaking ? '#10b981' : 'rgba(148, 163, 184, 0.8)'};box-shadow:${snapshot?.speaking ? '0 0 0 2px rgba(16, 185, 129, 0.16)' : 'none'};"></span>
              <span style="position:relative;width:72px;height:8px;overflow:hidden;border-radius:999px;background:rgba(51, 65, 85, 0.16);border:1px solid rgba(148, 163, 184, 0.22);">
                <span style="display:block;height:100%;width:${snrPercent}%;background:linear-gradient(90deg, #ef4444 0%, #f59e0b 42%, #10b981 100%);box-shadow:${snrActive ? '0 0 6px rgba(16, 185, 129, 0.28)' : 'none'};"></span>
                <span aria-hidden="true" style="position:absolute;top:-1px;bottom:-1px;width:1px;background:#38bdf8;left:${snrThresholdPercent}%;transform:translateX(-50%);"></span>
                <span aria-hidden="true" style="position:absolute;top:-1px;bottom:-1px;width:1px;background:#10b981;opacity:0.85;left:${minThresholdPercent}%;transform:translateX(-50%);"></span>
              </span>
              <span style="min-width:24px;text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;font-weight:700;color:${snrActive ? '#047857' : '#0f172a'};">${escapeHtml(formatThreshold(snapshot?.currentSnr))}</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px 12px;min-width:0;">
              ${metricHtml('Sig', formatCompactDbfs(snapshot?.signalDbfs))}
              ${metricHtml('Avg', formatCompactDbfs(snapshot?.averageDbfs))}
              ${metricHtml('Gate', formatCompactDbfs(snapshot?.gateDbfs))}
              ${metricHtml('Noise', formatCompactDbfs(snapshot?.noiseDbfs))}
              ${metricHtml('Segs', `${snapshot?.recentSegments ?? 0}/${snapshot?.recentRejected ?? 0}`)}
            </div>
          </div>
          <div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;padding-top:4px;border-top:1px solid rgba(148, 163, 184, 0.22);">
            ${miniHtml('Audio', formatCompactHz(snapshot?.targetRate))}
            ${miniHtml('Buf', formatCompactSeconds(snapshot?.visibleSeconds))}
            ${miniHtml('SNR', formatThreshold(snapshot?.currentSnr), snrActive)}
            ${miniHtml('Thr', formatThreshold(snapshot?.snrThreshold))}
          </div>
        </div>
      `;
    },
    dispose(): void {
      container.innerHTML = '';
    },
  };
}
