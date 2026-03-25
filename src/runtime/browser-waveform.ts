import type { BrowserRealtimePlot } from './browser-realtime.js';

export const BROWSER_WAVEFORM_CANVAS_MAX_DPR = 2;
export const BROWSER_WAVEFORM_CANVAS_HEIGHT = 252;

const LANE_HEIGHT = 22;
const LANE_INSET = 8;
const LANE_GAP = 8;
const SEGMENT_BAND_HEIGHT = 24;
const SEGMENT_BAND_ROWS = 4;

export interface BrowserWaveformRenderSegment {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly logicalStartFrame?: number;
  readonly label?: string;
  readonly reason?: string;
  readonly accepted?: boolean;
}

export interface BrowserWaveformRenderFrame {
  readonly waveform?: unknown;
  readonly plot?: BrowserRealtimePlot | null;
  readonly recentSegments?: readonly BrowserWaveformRenderSegment[];
  readonly activeSegment?: BrowserWaveformRenderSegment | null;
  readonly rough?: {
    readonly snr?: number;
    readonly snrThreshold?: number;
    readonly minSnrThreshold?: number;
    readonly useSnrGate?: boolean;
    readonly isSpeech?: boolean;
  } | null;
}

export interface BrowserWaveformRenderOptions {
  readonly backgroundColor?: string;
  readonly waveformScaleMode?: 'physical' | 'adaptive' | 'focus';
  readonly waveformTargetPeak?: number;
  readonly waveformMaxDisplayGain?: number;
  readonly showSpeechThreshold?: boolean;
  readonly noiseFloorDbfs?: number | null;
  readonly absoluteSpeechFloorDbfs?: number | null;
  readonly speechThresholdDbfs?: number | null;
  readonly onsetThresholdDbfs?: number | null;
  readonly showTenVadThreshold?: boolean;
  readonly tenVadThreshold?: number | null;
  readonly showPreVadOverlay?: boolean;
}

const DEFAULT_RENDER_OPTIONS: Required<BrowserWaveformRenderOptions> = {
  backgroundColor: '#fbfcfd',
  waveformScaleMode: 'focus',
  waveformTargetPeak: 0.82,
  waveformMaxDisplayGain: 12,
  showSpeechThreshold: false,
  noiseFloorDbfs: null,
  absoluteSpeechFloorDbfs: null,
  speechThresholdDbfs: null,
  onsetThresholdDbfs: null,
  showTenVadThreshold: false,
  tenVadThreshold: null,
  showPreVadOverlay: false,
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function dbfsToAmplitude(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return 10 ** (value / 20);
}

function getLaneLayout(height: number) {
  const topLaneTop = LANE_INSET;
  const topLaneBottom = topLaneTop + LANE_HEIGHT;
  const bottomLaneTop = height - LANE_INSET - LANE_HEIGHT;
  const bottomLaneBottom = bottomLaneTop + LANE_HEIGHT;
  const segmentBandTop = topLaneBottom + 2;
  const segmentBandBottom = segmentBandTop + SEGMENT_BAND_HEIGHT;
  const waveformTop = segmentBandBottom + LANE_GAP;
  const waveformBottom = bottomLaneTop - LANE_GAP;

  return {
    topLaneTop,
    topLaneBottom,
    segmentBandTop,
    segmentBandBottom,
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

function drawLaneAxis(
  context: CanvasRenderingContext2D,
  width: number,
  laneTop: number,
  laneHeight: number,
  labels: readonly { readonly value: number; readonly label: string }[],
  color = 'rgba(93, 115, 135, 0.18)',
): void {
  const toY = (value: number) => laneTop + (1 - clamp(value, 0, 1)) * laneHeight;

  context.save();
  context.strokeStyle = color;
  context.lineWidth = 1;
  context.fillStyle = 'rgba(93, 115, 135, 0.72)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';

  for (const entry of labels) {
    const y = toY(entry.value);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
    context.fillText(entry.label, 8, Math.min(laneTop + laneHeight - 2, y - 2));
  }
  context.restore();
}

function resolveWaveformDisplayGain(
  columns: readonly BrowserRealtimePlot['columns'][number][],
  options: Required<BrowserWaveformRenderOptions>,
): number {
  if (options.waveformScaleMode === 'physical') {
    return 1;
  }
  const amplitudes: number[] = [];
  for (const column of columns) {
    if (!column?.hasData) {
      continue;
    }
    const columnPeak = Math.max(
      Math.abs(Number(column.waveformMin ?? 0)),
      Math.abs(Number(column.waveformMax ?? 0)),
    );
    if (Number.isFinite(columnPeak) && columnPeak > 0.00001) {
      amplitudes.push(columnPeak);
    }
  }
  if (amplitudes.length === 0) {
    return 1;
  }
  amplitudes.sort((a, b) => a - b);
  const percentile = options.waveformScaleMode === 'focus' ? 0.9 : 0.97;
  const percentileIndex = clamp(
    Math.round((amplitudes.length - 1) * percentile),
    0,
    amplitudes.length - 1,
  );
  const peak = amplitudes[percentileIndex] ?? amplitudes[amplitudes.length - 1] ?? 0;
  if (!Number.isFinite(peak) || peak <= 0.00001) {
    return 1;
  }
  const targetPeak = options.waveformScaleMode === 'focus' ? 0.92 : 0.84;
  const maxGain = options.waveformScaleMode === 'focus' ? 18 : options.waveformMaxDisplayGain;
  return clamp(targetPeak / peak, 1, maxGain);
}

function mapDisplayAmplitude(
  amplitude: number,
  displayGain: number,
): number {
  return clamp(amplitude * displayGain, -1, 1);
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
  const toY = (amplitude: number) =>
    waveformTop + ((1 - clamp(amplitude, -1, 1)) * 0.5) * laneHeight;
  const axisTop = options.waveformScaleMode === 'physical' ? 1 : mapDisplayAmplitude(1, displayGain);
  const axisMid = 0;
  const axisBottom = -axisTop;

  context.save();
  context.strokeStyle = 'rgba(93, 115, 135, 0.18)';
  context.beginPath();
  context.moveTo(0, toY(axisTop));
  context.lineTo(width, toY(axisTop));
  context.moveTo(0, toY(axisMid));
  context.lineTo(width, toY(axisMid));
  context.moveTo(0, toY(axisBottom));
  context.lineTo(width, toY(axisBottom));
  context.stroke();

  context.fillStyle = 'rgba(93, 115, 135, 0.72)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText('+1.0', 8, toY(axisTop) + 10);
  context.fillText('0', 8, toY(axisMid) - 4);
  context.fillText('-1.0', 8, toY(axisBottom) - 4);
  if (options.waveformScaleMode !== 'physical' && displayGain > 1.05) {
    const detailLabel = options.waveformScaleMode === 'focus' ? 'detail' : 'balanced';
    context.fillText(`display x${displayGain.toFixed(1)} · ${detailLabel}`, width - 132, toY(axisTop) + 10);
  }
  context.restore();
}

function drawSpeechThreshold(
  context: CanvasRenderingContext2D,
  frame: BrowserWaveformRenderFrame | null | undefined,
  columns: readonly (BrowserRealtimePlot['columns'][number] & { readonly gateMode: string })[],
  width: number,
  height: number,
  displayGain: number,
  options: Required<BrowserWaveformRenderOptions>,
): void {
  if (!options.showSpeechThreshold) {
    return;
  }

  const currentSnr = Number(frame?.rough?.snr);
  const snrThreshold = Number(frame?.rough?.snrThreshold);
  const minSnrThreshold = Number(frame?.rough?.minSnrThreshold);
  const useSnrGate = frame?.rough?.useSnrGate === true;
  const adaptiveSnrGateDbfs =
    Number.isFinite(options.noiseFloorDbfs) && Number.isFinite(snrThreshold)
      ? (options.noiseFloorDbfs as number) + snrThreshold
      : null;
  const adaptiveMinSnrGateDbfs =
    Number.isFinite(options.noiseFloorDbfs) && Number.isFinite(minSnrThreshold)
      ? (options.noiseFloorDbfs as number) + minSnrThreshold
      : null;
  const thresholds = [
    {
      key: 'noise',
      label: 'noise',
      dbfs: options.noiseFloorDbfs,
      color: 'rgba(71, 85, 105, 0.82)',
      dash: [3, 3] as number[],
    },
    {
      key: 'floor',
      label: 'floor',
      dbfs: options.absoluteSpeechFloorDbfs,
      color: 'rgba(124, 58, 237, 0.88)',
      dash: [2, 4] as number[],
    },
    {
      key: 'speech',
      label: 'gate',
      dbfs: options.speechThresholdDbfs,
      color: 'rgba(239, 68, 68, 0.92)',
      dash: [5, 3] as number[],
    },
    {
      key: 'onset',
      label: 'onset',
      dbfs: options.onsetThresholdDbfs,
      color: 'rgba(245, 158, 11, 0.92)',
      dash: [4, 3] as number[],
    },
    {
      key: 'snr',
      label: 'snr',
      dbfs: useSnrGate ? adaptiveSnrGateDbfs : null,
      color: 'rgba(14, 165, 233, 0.9)',
      dash: [6, 2] as number[],
    },
    {
      key: 'min-snr',
      label: 'min snr',
      dbfs: useSnrGate ? adaptiveMinSnrGateDbfs : null,
      color: 'rgba(34, 197, 94, 0.9)',
      dash: [2, 2] as number[],
    },
  ].filter((threshold, index, entries) => {
    if (!Number.isFinite(threshold.dbfs)) {
      return false;
    }
    const thresholdDbfs = threshold.dbfs as number;
    if (threshold.key === 'onset' && Number.isFinite(options.speechThresholdDbfs)) {
      return Math.abs(thresholdDbfs - (options.speechThresholdDbfs as number)) >= 0.5;
    }
    return !entries.slice(0, index).some((previous) => (
      Number.isFinite(previous.dbfs) && Math.abs((previous.dbfs as number) - thresholdDbfs) < 0.5
    ));
  });

  if (thresholds.length === 0) {
    return;
  }

  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);
  const toY = (amplitude: number) =>
    waveformTop + ((1 - clamp(amplitude, -1, 1)) * 0.5) * laneHeight;
  const maxColumnAmplitude = columns.reduce((peak, column) => (
    Math.max(
      peak,
      Math.abs(Number(column?.waveformMin ?? 0)),
      Math.abs(Number(column?.waveformMax ?? 0)),
    )
  ), 0);

  context.save();
  context.lineWidth = 1;
  for (const threshold of thresholds) {
    const amplitude = clamp(dbfsToAmplitude(threshold.dbfs as number), 0, 1);
    const scaledThreshold = clamp(mapDisplayAmplitude(amplitude, displayGain), 0, 1);
    const activeAlpha = maxColumnAmplitude >= amplitude ? 1 : 0.55;
    context.strokeStyle = threshold.color.replace(/0\.\d+\)$/, `${activeAlpha})`);
    context.setLineDash(threshold.dash);
    context.beginPath();
    context.moveTo(0, toY(scaledThreshold));
    context.lineTo(width, toY(scaledThreshold));
    context.moveTo(0, toY(-scaledThreshold));
    context.lineTo(width, toY(-scaledThreshold));
    context.stroke();
  }
  context.setLineDash([]);
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const legendRows = thresholds.map((threshold) => ({
    ...threshold,
    text:
      threshold.key === 'snr'
        ? `snr ${snrThreshold.toFixed(1)} dB`
        : threshold.key === 'min-snr'
          ? `min ${minSnrThreshold.toFixed(1)} dB`
          : `${threshold.label} ${(threshold.dbfs as number).toFixed(0)} dBFS`,
  }));
  const legendWidth = legendRows.reduce(
    (maxWidth, row) => Math.max(maxWidth, context.measureText(row.text).width),
    126,
  ) + 28;
  const rowHeight = 14;
  const snrBarHeight = Number.isFinite(currentSnr) ? 22 : 0;
  const legendHeight = legendRows.length * rowHeight + 8 + snrBarHeight;
  const legendX = Math.max(40, width - legendWidth - 10);
  const legendY = waveformTop + 8;

  context.fillStyle = 'rgba(255, 255, 255, 0.82)';
  context.strokeStyle = 'rgba(148, 163, 184, 0.45)';
  context.lineWidth = 1;
  context.beginPath();
  context.roundRect(legendX, legendY, legendWidth, legendHeight, 6);
  context.fill();
  context.stroke();

  legendRows.forEach((row, index) => {
    const rowY = legendY + 12 + index * rowHeight;
    context.strokeStyle = row.color;
    context.lineWidth = 1.5;
    context.setLineDash(row.dash);
    context.beginPath();
    context.moveTo(legendX + 8, rowY - 4);
    context.lineTo(legendX + 18, rowY - 4);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = 'rgba(15, 23, 42, 0.9)';
    context.fillText(row.text, legendX + 22, rowY);
  });

  if (Number.isFinite(currentSnr)) {
    const meterTop = legendY + 8 + legendRows.length * rowHeight + 4;
    const meterLeft = legendX + 8;
    const meterWidth = legendWidth - 16;
    const meterHeight = 10;
    const snrPercent = clamp((currentSnr / 20) * 100, 0, 100);
    const snrThresholdPercent = Number.isFinite(snrThreshold)
      ? clamp((snrThreshold / 20) * 100, 0, 100)
      : 0;
    const minSnrThresholdPercent = Number.isFinite(minSnrThreshold)
      ? clamp((minSnrThreshold / 20) * 100, 0, 100)
      : 0;
    const fillWidth = (meterWidth * snrPercent) / 100;

    const meterGradient = context.createLinearGradient(meterLeft, 0, meterLeft + meterWidth, 0);
    meterGradient.addColorStop(0, 'rgba(239, 68, 68, 0.78)');
    meterGradient.addColorStop(0.45, 'rgba(245, 158, 11, 0.82)');
    meterGradient.addColorStop(1, 'rgba(34, 197, 94, 0.86)');

    context.fillStyle = 'rgba(226, 232, 240, 0.95)';
    context.fillRect(meterLeft, meterTop, meterWidth, meterHeight);
    context.fillStyle = meterGradient;
    context.fillRect(meterLeft, meterTop, fillWidth, meterHeight);

    if (Number.isFinite(snrThreshold)) {
      const markerX = meterLeft + (meterWidth * snrThresholdPercent) / 100;
      context.strokeStyle = 'rgba(14, 165, 233, 0.95)';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(markerX, meterTop - 1);
      context.lineTo(markerX, meterTop + meterHeight + 1);
      context.stroke();
    }
    if (Number.isFinite(minSnrThreshold)) {
      const markerX = meterLeft + (meterWidth * minSnrThresholdPercent) / 100;
      context.strokeStyle = 'rgba(34, 197, 94, 0.95)';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(markerX, meterTop - 1);
      context.lineTo(markerX, meterTop + meterHeight + 1);
      context.stroke();
    }

    context.fillStyle = Number.isFinite(snrThreshold) && currentSnr >= snrThreshold
      ? 'rgba(21, 128, 61, 0.95)'
      : 'rgba(15, 23, 42, 0.88)';
    context.fillText(`snr ${currentSnr.toFixed(1)} dB`, meterLeft, meterTop + meterHeight + 10);
  }
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
  const { segmentBandTop, waveformBottom } = getLaneLayout(height);
  const segmentTop = Math.max(0, segmentBandTop);
  const segmentHeight = Math.max(24, waveformBottom - segmentBandTop + 8);

  if (Array.isArray(segments)) {
    segments.forEach((segment, index) => {
      const x0 = clamp(toX(segment.startFrame), 0, width);
      const x1 = clamp(toX(segment.endFrame), 0, width);
      if (x1 > x0) {
        const isEven = index % 2 === 0;
        const fillStyle = isEven ? 'rgba(148, 163, 184, 0.09)' : 'rgba(203, 213, 225, 0.14)';
        const strokeStyle = isEven ? 'rgba(100, 116, 139, 0.34)' : 'rgba(148, 163, 184, 0.4)';

        context.save();
        context.fillStyle = fillStyle;
        context.strokeStyle = strokeStyle;
        context.lineWidth = 1;
        context.fillRect(x0, segmentTop, x1 - x0, segmentHeight);
        context.strokeRect(x0 + 0.5, segmentTop + 0.5, Math.max(1, x1 - x0 - 1), segmentHeight - 1);
        context.restore();
      }
    });
  }

  if (activeSegment) {
    const x0 = clamp(toX(activeSegment.startFrame), 0, width);
    const x1 = clamp(toX(activeSegment.endFrame), 0, width);
    context.save();
    context.fillStyle = 'rgba(16, 185, 129, 0.12)';
    context.strokeStyle = 'rgba(5, 150, 105, 0.8)';
    context.lineWidth = 1;
    context.fillRect(x0, segmentTop, Math.max(2, x1 - x0), segmentHeight);
    context.strokeRect(x0 + 0.5, segmentTop + 0.5, Math.max(1, x1 - x0 - 1), segmentHeight - 1);
    context.restore();
  }
}

function drawSegmentAnnotations(
  context: CanvasRenderingContext2D,
  displayStartFrame: number,
  totalFrames: number,
  segments: readonly BrowserWaveformRenderSegment[] | undefined,
  activeSegment: BrowserWaveformRenderSegment | null | undefined,
  width: number,
  height: number,
): void {
  const toX = (frame: number) => ((frame - displayStartFrame) / totalFrames) * width;
  const {
    segmentBandTop,
    segmentBandBottom,
    waveformBottom,
  } = getLaneLayout(height);
  const segmentTop = Math.max(0, segmentBandTop);
  const segmentHeight = Math.max(24, waveformBottom - segmentBandTop + 8);
  const rowHeight = Math.max(6, Math.floor(SEGMENT_BAND_HEIGHT / SEGMENT_BAND_ROWS));
  const rowBottoms = Array.from({ length: SEGMENT_BAND_ROWS }, () => -Infinity);

  const drawLabelPill = (
    x0: number,
    x1: number,
    text: string,
    fill: string,
    textColor: string,
  ) => {
    if (x1 - x0 < 24) {
      return;
    }
    context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    const textWidth = context.measureText(text).width;
    const maxWidth = Math.max(0, x1 - x0 - 6);
    const pillWidth = Math.min(Math.max(20, textWidth + 8), maxWidth);
    if (pillWidth < 16) {
      return;
    }
    let rowIndex = 0;
    for (let index = 0; index < rowBottoms.length; index += 1) {
      if (x0 >= rowBottoms[index]!) {
        rowIndex = index;
        break;
      }
      rowIndex = index;
    }
    rowBottoms[rowIndex] = Math.max(rowBottoms[rowIndex]!, x0 + pillWidth + 3);
    const pillX = x0 + 2;
    const pillY = segmentBandTop + rowIndex * rowHeight + 1;
    context.save();
    context.fillStyle = fill;
    context.beginPath();
    context.roundRect(pillX, pillY, pillWidth, rowHeight - 1, 3);
    context.fill();
    context.fillStyle = textColor;
    context.fillText(text, pillX + 4, pillY + rowHeight - 2);
    context.restore();
  };

  context.save();
  context.fillStyle = 'rgba(248, 250, 252, 0.92)';
  context.fillRect(0, segmentBandTop, width, Math.max(1, segmentBandBottom - segmentBandTop));
  context.strokeStyle = 'rgba(203, 213, 225, 0.7)';
  context.beginPath();
  context.moveTo(0, segmentBandBottom + 0.5);
  context.lineTo(width, segmentBandBottom + 0.5);
  context.stroke();

  if (Array.isArray(segments)) {
    segments.forEach((segment, index) => {
      const x0 = clamp(toX(segment.startFrame), 0, width);
      const x1 = clamp(toX(segment.endFrame), 0, width);
      if (x1 <= x0) {
        return;
      }
      const label = segment.label ?? segment.reason ?? `seg ${index + 1}`;
      drawLabelPill(x0, x1, label, 'rgba(255, 255, 255, 0.92)', 'rgba(51, 65, 85, 0.92)');
      if (
        Number.isFinite(segment.logicalStartFrame)
        && (segment.logicalStartFrame as number) > segment.startFrame
        && (segment.logicalStartFrame as number) < segment.endFrame
      ) {
        const onsetX = clamp(toX(segment.logicalStartFrame as number), 0, width);
        context.strokeStyle = 'rgba(14, 165, 233, 0.95)';
        context.lineWidth = 1;
        context.setLineDash([2, 2]);
        context.beginPath();
        context.moveTo(onsetX + 0.5, segmentBandTop);
        context.lineTo(onsetX + 0.5, segmentTop + segmentHeight - 1);
        context.stroke();
        context.setLineDash([]);
      }
    });
  }

  if (activeSegment) {
    const x0 = clamp(toX(activeSegment.startFrame), 0, width);
    const x1 = clamp(toX(activeSegment.endFrame), 0, width);
    const label = activeSegment.label ?? 'live';
    drawLabelPill(x0, x1, label, 'rgba(209, 250, 229, 0.96)', 'rgba(4, 120, 87, 0.96)');
  }
  context.restore();
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
    waveformTop + ((1 - mapDisplayAmplitude(amplitude, displayGain)) * 0.5) * laneHeight;

  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    if (!column?.hasData) {
      continue;
    }
    const min = clamp(Number(column.waveformMin ?? 0), -1, 1);
    const max = clamp(Number(column.waveformMax ?? 0), -1, 1);
    const y0 = toY(max);
    const y1 = toY(min);
    const insideSegment = column.activeSegment || column.recentSegment;
    if (column.activeSegment) {
      context.fillStyle = 'rgba(5, 150, 105, 0.95)';
    } else if (column.detectorPass) {
      context.fillStyle =
        column.gateMode === 'ten-vad-only'
          ? 'rgba(37, 99, 235, 0.92)'
          : 'rgba(59, 130, 246, 0.9)';
    } else if (column.tenVadPass || column.roughPass) {
      context.fillStyle = insideSegment
        ? 'rgba(56, 189, 248, 0.72)'
        : 'rgba(148, 163, 184, 0.92)';
    } else if (insideSegment) {
      context.fillStyle = column.recentSegment
        ? 'rgba(96, 165, 250, 0.62)'
        : 'rgba(45, 212, 191, 0.72)';
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
  options: Required<BrowserWaveformRenderOptions>,
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
  drawLaneAxis(
    context,
    width,
    laneTop,
    laneHeight,
    [
      { value: 1, label: '1.0' },
      { value: 0.5, label: '0.5' },
      { value: 0, label: '0.0' },
    ],
    'rgba(37, 99, 235, 0.12)',
  );

  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    const probability = clamp(Number(column?.vadProbability ?? 0), 0, 1);
    const barHeight = Math.max(1, probability * laneHeight);
    context.fillStyle = column?.tenVadPass ? 'rgba(37, 99, 235, 0.82)' : 'rgba(37, 99, 235, 0.3)';
    context.fillRect(index, laneTop + laneHeight - barHeight, 1, barHeight);
  }

  if (options.showTenVadThreshold && Number.isFinite(options.tenVadThreshold)) {
    const threshold = clamp(options.tenVadThreshold as number, 0, 1);
    const thresholdY = laneTop + (1 - threshold) * laneHeight;
    let thresholdHit = false;
    for (const column of columns) {
      if (Number(column?.vadProbability ?? 0) >= threshold) {
        thresholdHit = true;
        break;
      }
    }
    context.strokeStyle = thresholdHit ? 'rgba(239, 68, 68, 0.92)' : 'rgba(37, 99, 235, 0.92)';
    context.lineWidth = 1;
    context.setLineDash([5, 3]);
    context.beginPath();
    context.moveTo(0, thresholdY);
    context.lineTo(width, thresholdY);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = thresholdHit ? 'rgba(239, 68, 68, 0.95)' : 'rgba(37, 99, 235, 0.95)';
    context.fillText(`thr ${threshold.toFixed(2)}`, 66, Math.max(laneTop + 10, thresholdY - 4));
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
  options: Required<BrowserWaveformRenderOptions>,
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
  drawLaneAxis(
    context,
    width,
    laneTop,
    laneHeight,
    [
      { value: 1, label: '1.0' },
      { value: 0.5, label: '0.5' },
      { value: 0, label: '0.0' },
    ],
    'rgba(217, 119, 6, 0.12)',
  );
  context.beginPath();
  context.moveTo(0, laneBottom);

  for (let index = 0; index < columns.length; index += 1) {
    const centerX = index + 0.5;
    const normalizedEnergy = clamp(Number(columns[index]?.preVadRms ?? 0), 0, 1);
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
    const normalizedEnergy = clamp(Number(columns[index]?.preVadRms ?? 0), 0, 1);
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

  if (options.showSpeechThreshold) {
    const maxEnergy = columns.reduce(
      (peak, column) => Math.max(peak, clamp(Number(column?.preVadRms ?? 0), 0, 1)),
      0,
    );
    const thresholds = [
      {
        dbfs: options.noiseFloorDbfs,
        color: 'rgba(71, 85, 105, 0.82)',
        dash: [3, 3] as number[],
      },
      {
        dbfs: options.absoluteSpeechFloorDbfs,
        color: 'rgba(124, 58, 237, 0.88)',
        dash: [2, 4] as number[],
      },
      {
        dbfs: options.speechThresholdDbfs,
        color: 'rgba(239, 68, 68, 0.92)',
        dash: [5, 3] as number[],
      },
      {
        dbfs: options.onsetThresholdDbfs,
        color: 'rgba(245, 158, 11, 0.92)',
        dash: [4, 3] as number[],
      },
    ].filter((threshold) => Number.isFinite(threshold.dbfs));

    for (const threshold of thresholds) {
      const thresholdAmplitude = clamp(dbfsToAmplitude(threshold.dbfs as number), 0, 1);
      const thresholdY = laneBottom - thresholdAmplitude * (laneHeight - 2);
      const activeAlpha = maxEnergy >= thresholdAmplitude ? 1 : 0.55;
      context.strokeStyle = threshold.color.replace(/0\.\d+\)$/, `${activeAlpha})`);
      context.lineWidth = 1;
      context.setLineDash(threshold.dash);
      context.beginPath();
      context.moveTo(0, thresholdY);
      context.lineTo(width, thresholdY);
      context.stroke();
    }
    context.setLineDash([]);
  }

  context.fillStyle = 'rgba(180, 83, 9, 0.85)';
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText('pre-vad rms', 8, laneTop + 10);
  context.restore();
}

function drawPreVadWaveformOverlay(
  context: CanvasRenderingContext2D,
  columns: readonly BrowserRealtimePlot['columns'][number][],
  height: number,
  displayGain: number,
  enabled: boolean,
): void {
  if (!enabled || !Array.isArray(columns) || columns.length === 0) {
    return;
  }
  const { waveformTop, waveformBottom } = getLaneLayout(height);
  const laneHeight = Math.max(8, waveformBottom - waveformTop);
  const toY = (amplitude: number) =>
    waveformTop + ((1 - mapDisplayAmplitude(amplitude, displayGain)) * 0.5) * laneHeight;

  context.save();
  context.beginPath();
  for (let index = 0; index < columns.length; index += 1) {
    const x = index + 0.5;
    const y = toY(Number(columns[index]?.preVadRms ?? 0));
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  for (let index = columns.length - 1; index >= 0; index -= 1) {
    const x = index + 0.5;
    const y = toY(-Number(columns[index]?.preVadRms ?? 0));
    context.lineTo(x, y);
  }
  context.closePath();
  context.fillStyle = 'rgba(217, 119, 6, 0.12)';
  context.fill();

  context.beginPath();
  for (let index = 0; index < columns.length; index += 1) {
    const x = index + 0.5;
    const y = toY(Number(columns[index]?.preVadRms ?? 0));
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  context.strokeStyle = 'rgba(217, 119, 6, 0.95)';
  context.lineWidth = 1;
  context.stroke();

  context.beginPath();
  for (let index = 0; index < columns.length; index += 1) {
    const x = index + 0.5;
    const y = toY(-Number(columns[index]?.preVadRms ?? 0));
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  context.stroke();
  context.restore();
}

function drawSegmentDebugLegend(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  const { waveformBottom } = getLaneLayout(height);
  const legendX = 12;
  const legendY = waveformBottom - 40;
  const legendWidth = 286;
  const legendHeight = 32;
  const boxWidth = Math.min(legendWidth, width - 24);

  context.save();
  context.fillStyle = 'rgba(255, 255, 255, 0.84)';
  context.strokeStyle = 'rgba(148, 163, 184, 0.4)';
  context.lineWidth = 1;
  context.beginPath();
  context.roundRect(legendX, legendY, boxWidth, legendHeight, 5);
  context.fill();
  context.stroke();

  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillStyle = 'rgba(51, 65, 85, 0.92)';

  context.strokeStyle = 'rgba(100, 116, 139, 0.7)';
  context.strokeRect(legendX + 7.5, legendY + 5.5, 10, 9);
  context.fillText('box start', legendX + 22, legendY + 13);

  const onsetX = legendX + 74.5;
  context.strokeStyle = 'rgba(14, 165, 233, 0.95)';
  context.setLineDash([2, 2]);
  context.beginPath();
  context.moveTo(onsetX, legendY + 4);
  context.lineTo(onsetX, legendY + 16);
  context.stroke();
  context.setLineDash([]);
  context.fillText('onset', legendX + 79, legendY + 13);

  context.fillStyle = 'rgba(96, 165, 250, 0.62)';
  context.fillRect(legendX + 118, legendY + 5, 12, 10);
  context.fillStyle = 'rgba(51, 65, 85, 0.92)';
  context.fillText('included', legendX + 135, legendY + 13);

  context.fillStyle = 'rgba(59, 130, 246, 0.9)';
  context.fillRect(legendX + 206, legendY + 5, 12, 10);
  context.fillStyle = 'rgba(51, 65, 85, 0.92)';
  context.fillText('gate', legendX + 223, legendY + 13);

  context.fillStyle = 'rgba(16, 185, 129, 0.22)';
  context.fillRect(legendX + 7, legendY + 19, 12, 10);
  context.strokeStyle = 'rgba(5, 150, 105, 0.84)';
  context.strokeRect(legendX + 7.5, legendY + 19.5, 11, 9);
  context.fillStyle = 'rgba(51, 65, 85, 0.92)';
  context.fillText('live', legendX + 24, legendY + 27);
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
  drawTenVadOverlay(context, columns, width, height, resolvedOptions);
  drawEnergyOverlay(context, columns, width, height, resolvedOptions);
  drawWaveformGateOverlay(context, columns, width, height);
  drawSpeechThreshold(context, frame, columns, width, height, displayGain, resolvedOptions);
  drawWaveformAmplitude(context, columns, height, displayGain);
  drawSegmentAnnotations(
    context,
    frame.plot.startFrame,
    Math.max(1, frame.plot.endFrame - frame.plot.startFrame),
    frame.recentSegments,
    frame.activeSegment,
    width,
    height,
  );
  drawPreVadWaveformOverlay(
    context,
    columns,
    height,
    displayGain,
    resolvedOptions.showPreVadOverlay,
  );
  drawSegmentDebugLegend(context, width, height);
  drawLiveEdge(context, frame.plot, height);
}
