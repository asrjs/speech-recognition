import type { AbortSignalLike, BaseTranscriptionOptions, TranscriptResult } from '../types/index.js';
import { createTranscriptOutputStage } from './output-sidecars.js';
import type { SubtitleFormat } from './subtitles.js';

export type PipelineSidecars = Readonly<Record<string, unknown>>;

export interface PipelineContext<TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions> {
  readonly input?: unknown;
  readonly options?: TOptions;
  readonly signal?: AbortSignalLike | null;
  readonly transcript?: TranscriptResult;
  readonly sidecars: PipelineSidecars;
  readonly completedStageIds: readonly string[];
}

export interface PipelineStageResult<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
> {
  readonly transcript?: TranscriptResult;
  readonly sidecars?: PipelineSidecars;
  readonly options?: TOptions;
}

export interface PipelineStage<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
> {
  readonly id: string;
  run(
    context: PipelineContext<TOptions>,
  ): PipelineStageResult<TOptions> | void | Promise<PipelineStageResult<TOptions> | void>;
}

export interface CreatePipelineContextOptions<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
> {
  readonly input?: unknown;
  readonly options?: TOptions;
  readonly signal?: AbortSignalLike | null;
  readonly transcript?: TranscriptResult;
  readonly sidecars?: PipelineSidecars;
  readonly completedStageIds?: readonly string[];
}

export class PipelineStageError extends Error {
  readonly stageId: string;
  override readonly cause: unknown;

  constructor(stageId: string, cause: unknown) {
    super(`Pipeline stage "${stageId}" failed.`);
    this.name = 'PipelineStageError';
    this.stageId = stageId;
    this.cause = cause;
  }
}

export class PipelineAbortedError extends Error {
  readonly stageId?: string;

  constructor(stageId?: string) {
    super(stageId ? `Pipeline aborted before stage "${stageId}".` : 'Pipeline aborted.');
    this.name = 'PipelineAbortedError';
    this.stageId = stageId;
  }
}

export function createPipelineContext<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
>(options: CreatePipelineContextOptions<TOptions> = {}): PipelineContext<TOptions> {
  return {
    input: options.input,
    options: options.options,
    signal: options.signal,
    transcript: options.transcript,
    sidecars: { ...(options.sidecars ?? {}) },
    completedStageIds: [...(options.completedStageIds ?? [])],
  };
}

export async function runPipelineStages<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
>(
  initialContext: PipelineContext<TOptions>,
  stages: readonly PipelineStage<TOptions>[],
): Promise<PipelineContext<TOptions>> {
  let context = createPipelineContext(initialContext);

  for (const stage of stages) {
    throwIfAborted(context.signal, stage.id);
    let stageResult: PipelineStageResult<TOptions> | void;
    try {
      stageResult = await stage.run(context);
    } catch (error) {
      if (error instanceof PipelineAbortedError) {
        throw error;
      }
      throw new PipelineStageError(stage.id, error);
    }

    throwIfAborted(context.signal, stage.id);
    context = mergePipelineStageResult(context, stage.id, stageResult);
  }

  return context;
}

export interface SubtitleSidecarStageOptions {
  readonly id?: string;
  readonly formats?: readonly SubtitleFormat[];
}

export function createSubtitleSidecarStage(
  options: SubtitleSidecarStageOptions = {},
): PipelineStage {
  return createTranscriptOutputStage({
    id: options.id ?? 'subtitle-sidecars',
    formats: options.formats ?? ['srt', 'vtt'],
  });
}

function mergePipelineStageResult<TOptions extends BaseTranscriptionOptions>(
  context: PipelineContext<TOptions>,
  stageId: string,
  stageResult: PipelineStageResult<TOptions> | void,
): PipelineContext<TOptions> {
  return {
    input: context.input,
    signal: context.signal,
    options: stageResult?.options ?? context.options,
    transcript: stageResult?.transcript ?? context.transcript,
    sidecars: {
      ...context.sidecars,
      ...(stageResult?.sidecars ?? {}),
    },
    completedStageIds: [...context.completedStageIds, stageId],
  };
}

function throwIfAborted(signal: AbortSignalLike | null | undefined, stageId?: string): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError(stageId);
  }
}
