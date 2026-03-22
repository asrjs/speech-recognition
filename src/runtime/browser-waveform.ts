import type { BrowserRealtimePlot } from './browser-realtime.js';

export const BROWSER_WAVEFORM_CANVAS_MAX_DPR = 2;
export const BROWSER_WAVEFORM_CANVAS_HEIGHT = 156;

const LANE_HEIGHT = 22;
const LANE_INSET = 8;
const LANE_GAP = 8;

export interface BrowserWaveformRenderSegment {
  readonly startFrame: number;
  readonly endFrame: number;
}

export interface BrowserWaveformRenderFrame {
  readonly waveform?: unknown;
  readonly plot?: BrowserRealtimePlot | null;
  readonly recentSegments?: readonly BrowserWaveformRenderSegment[];
  readonly activeSegment?: BrowserWaveformRenderSegment | null;
}

export interface BrowserWaveformRenderOptions {
  readonly backgroundColor?: string;
  readonly waveformScaleMode?: 'physical' | 'adaptive';
  readonly waveformTargetPeak?: number;
  readonly waveformMaxDisplayGain?: number;
  readonly showSpeechThreshold?: boolean;
  readonly speechThresholdDbfs?: number | null;
}

const DEFAULT_RENDER_OPTIONS: Required<BrowserWaveformRenderOptions> = {
  backgroundColor: '#fbfcfd',
  waveformScaleMode: 'adaptive',
  waveformTargetPeak: 0.82,
  waveformMaxDisplayGain: 12,
  showSpeechThreshold: false,
  speechThresholdDbfs: null,
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function amplitudeToDbfs(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return -100;
  }
  return 20 * Math.log10(value);
}

function getLaneLayout(height: number) {
  const topLaneTop = LANE_INSET;
  const topLaneBottom = topLaneTop + LANE_HEIGHT;
  const bottomLaneTop = height - LANE_INSET - LANE_HEIGHT;
  const bottomLaneBottom = bottomLaneTop + LANE_HEIGHT;
  const waveformTop = topLaneBottom + LANE_GAP;
  const waveformBottom = bottomLaneTop - LANE_GAP;

  return {
    topLaneTop,
    topLaneBottom,
    bottomLaneTop,
    bottomLaneBottom,
    waveformTop,
    waveformBottom,
  };
}

function drawGrid(context: CanvasRenderingContext2D, width: number, height: number): void {
  context.strokeStyle = 'rgba(148, 163, 184, 0.12)';
  context.lineWidth = 1;

  for (let row = 1; row < 4; row += 1) {
    const y = (height / 4) * row;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function resolveWaveformDisplayGain(
  columns: readonly BrowserRealtimePlot['columns'][number][],
  options: Required<BrowserWaveformRenderOptions>,
): number {
  if (options.waveformScaleMode === 'physical') {
    return 1;
  }
  let peak = 0;
  for (const column of columns) {
    if (!column?.hasData) {
      continue;
    }
    peak = Math.max(
      peak,
      Math.abs(Number(column.waveformMin ?? 0)),
      Math.abs(Number(column.waveformMax ?? 0)),
    );
  }
  if (!Number.isFinite(peak) || peak <= 0.00001) {
    return 1;
  }
  return clamp(options.waveformTargetPeak / peak, 1, options.waveformMaxDisplayGain);
}

function drawWaveformAxis(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  displayGain: number,
  options: Required<BrowserWaveformRenderOptions>,
): void {
  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);
  const toY = (amplitude: number) => waveformTop + (1 - clamp(amplitude, -1, 1)) * 0.5 * laneHeight;

  context.save();
  context.strokeStyle = 'rgba(93, 115, 135, 0.18)';
  context.beginPath();
  context.moveTo(0, toY(1));
  context.lineTo(width, toY(1));
  context.moveTo(0, toY(0));
  context.lineTo(width, toY(0));
  context.moveTo(0, toY(-1));
  context.lineTo(width, toY(-1));
  context.stroke();

  context.fillStyle = 'rgba(93, 115, 135, 0.72)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText('+1.0', 8, toY(1) + 10);
  context.fillText('0', 8, toY(0) - 4);
  context.fillText('-1.0', 8, toY(-1) - 4);
  if (options.waveformScaleMode === 'adaptive' && displayGain > 1.05) {
    context.fillText(`display x${displayGain.toFixed(1)}`, width - 92, toY(1) + 10);
  }
  context.restore();
}

function drawSpeechThreshold(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  columns: readonly BrowserRealtimePlot['columns'][number][],
  displayGain: number,
  options: Required<BrowserWaveformRenderOptions>,
): void {
  if (!options.showSpeechThreshold || !Number.isFinite(options.speechThresholdDbfs)) {
    return;
  }
  const thresholdAmplitude = 10 ** ((options.speechThresholdDbfs as number) / 20);
  const scaledThreshold = clamp(thresholdAmplitude * displayGain, 0, 1);
  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);
  const toY = (amplitude: number) => waveformTop + (1 - clamp(amplitude, -1, 1)) * 0.5 * laneHeight;
  let thresholdHit = false;
  for (const column of columns) {
    if (!column?.hasData) {
      continue;
    }
    const peak = Math.max(
      Math.abs(Number(column.waveformMin ?? 0)),
      Math.abs(Number(column.waveformMax ?? 0)),
    );
    if (peak >= thresholdAmplitude) {
      thresholdHit = true;
      break;
    }
  }

  context.save();
  context.strokeStyle = thresholdHit ? 'rgba(239, 68, 68, 0.9)' : 'rgba(245, 158, 11, 0.92)';
  context.lineWidth = 1;
  context.setLineDash([5, 3]);
  context.beginPath();
  context.moveTo(0, toY(scaledThreshold));
  context.lineTo(width, toY(scaledThreshold));
  context.moveTo(0, toY(-scaledThreshold));
  context.lineTo(width, toY(-scaledThreshold));
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = thresholdHit ? 'rgba(239, 68, 68, 0.95)' : 'rgba(180, 83, 9, 0.92)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText(
    `speech thr ${options.speechThresholdDbfs?.toFixed(0)} dBFS ${thresholdHit ? 'hit' : 'clear'}`,
    8,
    toY(scaledThreshold) - 4,
  );
  context.restore();
}

function drawSpans(
  context: CanvasRenderingContext2D,
  displayStartFrame: number,
  totalFrames: number,
  segments: readonly BrowserWaveformRenderSegment[] | undefined,
  activeSegment: BrowserWaveformRenderSegment | null | undefined,
  width: number,
  height: number,
): void {
  const toX = (frame: number) => ((frame - displayStartFrame) / totalFrames) * width;

  if (Array.isArray(segments)) {
    context.fillStyle = 'rgba(148, 163, 184, 0.08)';
    for (const segment of segments) {
      const x0 = clamp(toX(segment.startFrame), 0, width);
      const x1 = clamp(toX(segment.endFrame), 0, width);
      if (x1 > x0) {
        context.fillRect(x0, 0, x1 - x0, height);
      }
    }
  }

  if (activeSegment) {
    const x0 = clamp(toX(activeSegment.startFrame), 0, width);
    const x1 = clamp(toX(activeSegment.endFrame), 0, width);
    context.fillStyle = 'rgba(93, 115, 135, 0.12)';
    context.fillRect(x0, 0, Math.max(2, x1 - x0), height);
  }
}

function drawWaveformGateOverlay(
  context: CanvasRenderingContext2D,
  columns: readonly (BrowserRealtimePlot['columns'][number] & { readonly gateMode: string })[],
  width: number,
  height: number,
): void {
  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);

  context.fillStyle = '#f4f8fc';
  context.fillRect(0, waveformTop, width, laneHeight);

  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    if (!column?.hasData) {
      continue;
    }
    if (column.activeSegment) {
      context.fillStyle = 'rgba(16, 185, 129, 0.14)';
    } else if (column.recentSegment) {
      context.fillStyle = 'rgba(93, 115, 135, 0.08)';
    } else if (column.detectorPass) {
      context.fillStyle = 'rgba(59, 130, 246, 0.1)';
    } else {
      continue;
    }
    context.fillRect(index, waveformTop, 1, laneHeight);
  }
}

function drawWaveformAmplitude(
  context: CanvasRenderingContext2D,
  columns: readonly (BrowserRealtimePlot['columns'][number] & { readonly gateMode: string })[],
  height: number,
  displayGain: number,
): void {
  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);
  const toY = (amplitude: number) =>
    waveformTop + (1 - clamp(amplitude * displayGain, -1, 1)) * 0.5 * laneHeight;

  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    if (!column?.hasData) {
      continue;
    }
    const min = clamp(Number(column.waveformMin ?? 0), -1, 1);
    const max = clamp(Number(column.waveformMax ?? 0), -1, 1);
    const y0 = toY(max);
    const y1 = toY(min);
    if (column.activeSegment) {
      context.fillStyle = 'rgba(5, 150, 105, 0.95)';
    } else if (column.detectorPass) {
      context.fillStyle =
        column.gateMode === 'ten-vad-only' ? 'rgba(37, 99, 235, 0.92)' : 'rgba(59, 130, 246, 0.9)';
    } else if (column.tenVadPass || column.roughPass) {
      context.fillStyle = 'rgba(148, 163, 184, 0.92)';
    } else {
      context.fillStyle = 'rgba(148, 163, 184, 0.42)';
    }
    context.fillRect(index, y0, 1, Math.max(1, y1 - y0));
  }
}

function drawLiveEdge(
  context: CanvasRenderingContext2D,
  plot: BrowserRealtimePlot | null | undefined,
  height: number,
): void {
  if (!plot || plot.livePointIndex < 0) {
    return;
  }
  const x = plot.livePointIndex + 0.5;
  context.save();
  context.strokeStyle = 'rgba(239, 68, 68, 0.95)';
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
  context.restore();
}

function drawTenVadOverlay(
  context: CanvasRenderingContext2D,
  columns: readonly BrowserRealtimePlot['columns'][number][],
  width: number,
  height: number,
): void {
  if (!Array.isArray(columns) || columns.length === 0) {
    return;
  }
  const { topLaneTop } = getLaneLayout(height);
  const laneTop = topLaneTop;
  const laneHeight = LANE_HEIGHT;

  context.save();
  context.fillStyle = 'rgba(37, 99, 235, 0.08)';
  context.fillRect(0, laneTop, width, laneHeight);

  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    const probability = clamp(Number(column?.vadProbability ?? 0), 0, 1);
    const barHeight = Math.max(1, probability * laneHeight);
    context.fillStyle = column?.tenVadPass ? 'rgba(37, 99, 235, 0.82)' : 'rgba(37, 99, 235, 0.3)';
    context.fillRect(index, laneTop + laneHeight - barHeight, 1, barHeight);
  }

  context.fillStyle = 'rgba(37, 99, 235, 0.8)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText('ten-vad', 8, laneTop + 10);
  context.restore();
}

function drawEnergyOverlay(
  context: CanvasRenderingContext2D,
  columns: readonly BrowserRealtimePlot['columns'][number][],
  width: number,
  height: number,
): void {
  if (!Array.isArray(columns) || columns.length === 0) {
    return;
  }
  const { bottomLaneTop, bottomLaneBottom } = getLaneLayout(height);
  const laneHeight = LANE_HEIGHT;
  const laneTop = bottomLaneTop;
  const laneBottom = bottomLaneBottom;

  context.save();
  context.fillStyle = 'rgba(217, 119, 6, 0.08)';
  context.fillRect(0, laneTop, width, laneHeight);
  context.beginPath();
  context.moveTo(0, laneBottom);

  for (let index = 0; index < columns.length; index += 1) {
    const centerX = index + 0.5;
    const energyDbfs = amplitudeToDbfs(Number(columns[index]?.roughEnergy ?? 0));
    const normalizedEnergy = clamp((energyDbfs + 72) / 42, 0, 1);
    const y = laneBottom - normalizedEnergy * (laneHeight - 2);
    context.lineTo(centerX, y);
  }

  context.lineTo(width, laneBottom);
  context.closePath();
  context.fillStyle = 'rgba(217, 119, 6, 0.18)';
  context.fill();

  context.beginPath();
  for (let index = 0; index < columns.length; index += 1) {
    const centerX = index + 0.5;
    const energyDbfs = amplitudeToDbfs(Number(columns[index]?.roughEnergy ?? 0));
    const normalizedEnergy = clamp((energyDbfs + 72) / 42, 0, 1);
    const y = laneBottom - normalizedEnergy * (laneHeight - 2);
    if (index === 0) {
      context.moveTo(centerX, y);
    } else {
      context.lineTo(centerX, y);
    }
  }
  context.strokeStyle = 'rgba(217, 119, 6, 0.95)';
  context.lineWidth = 1.5;
  context.stroke();

  for (let index = 0; index < columns.length; index += 1) {
    if (!columns[index]?.roughPass) {
      continue;
    }
    context.fillStyle = 'rgba(234, 179, 8, 0.16)';
    context.fillRect(index, laneTop, 1, laneHeight);
  }

  context.fillStyle = 'rgba(180, 83, 9, 0.85)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText('pre-vad energy', 8, laneTop + 10);
  context.restore();
}

export function renderBrowserRealtimeWaveformFrame(
  context: CanvasRenderingContext2D,
  frame: BrowserWaveformRenderFrame | null | undefined,
  width: number,
  height = BROWSER_WAVEFORM_CANVAS_HEIGHT,
  options: BrowserWaveformRenderOptions = {},
): void {
  const resolvedOptions = {
    ...DEFAULT_RENDER_OPTIONS,
    ...options,
  };

  context.clearRect(0, 0, width, height);
  context.fillStyle = resolvedOptions.backgroundColor;
  context.fillRect(0, 0, width, height);

  drawGrid(context, width, height);

  if (!frame?.waveform || !frame?.plot || frame.plot.columns.length < 1) {
    context.fillStyle = '#94a3b8';
    context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillText('Waiting for waveform data…', 12, 20);
    return;
  }

  const columns = frame.plot.columns.map((column) => ({
    ...column,
    gateMode: frame.plot?.gateMode ?? 'rough-only',
  }));
  const displayGain = resolveWaveformDisplayGain(columns, resolvedOptions);

  drawSpans(
    context,
    frame.plot.startFrame,
    Math.max(1, frame.plot.endFrame - frame.plot.startFrame),
    frame.recentSegments,
    frame.activeSegment,
    width,
    height,
  );
  drawWaveformAxis(context, width, height, displayGain, resolvedOptions);
  drawTenVadOverlay(context, columns, width, height);
  drawEnergyOverlay(context, columns, width, height);
  drawWaveformGateOverlay(context, columns, width, height);
  drawSpeechThreshold(context, width, height, columns, displayGain, resolvedOptions);
  drawWaveformAmplitude(context, columns, height, displayGain);
  drawLiveEdge(context, frame.plot, height);
}
