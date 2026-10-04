import { normalizePcmInput } from '../../audio/index.js';
import { mapLasrCtcNativeToCanonical } from '../lasr-ctc/mapping.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';
import type {
  AudioInputLike,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { loadOmnilingualCtc, type OmnilingualCtcLoadedModel } from './model.js';

export const DEFAULT_OMNILINGUAL_CTC_CLASSIFICATION: ModelClassification = {
  family: 'omnilingual-ctc',
  ecosystem: 'wav2vec2',
  processor: 'raw-waveform-conv',
  encoder: 'wav2vec2-conv',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr',
};

export interface OmnilingualCtcLoadOptions {
  readonly modelUrl: string;
  readonly tokensUrl: string;
  readonly externalDataUrl?: string;
  readonly externalDataFilename?: string;
  readonly sampleRate?: number;
  readonly vocabularySize?: number;
  readonly blankId?: number;
  readonly backendId?: 'wasm' | 'webgpu';
  readonly cpuThreads?: number;
  readonly wasmPaths?: string;
  readonly signal?: AbortSignal | null;
}

/**
 * Registers the omnilingual CTC family.
 *
 * The graph consumes raw samples, so this family has no feature frontend to
 * reconcile; the wav2vec2-line normalization and the shared CTC decoder are the
 * whole runtime path.
 */
export function createOmnilingualCtcModelFamily(): SpeechModelFactory<
  OmnilingualCtcLoadOptions,
  LasrCtcTranscriptionOptions,
  LasrCtcNativeTranscript
> {
  return {
    family: 'omnilingual-ctc',
    classification: DEFAULT_OMNILINGUAL_CTC_CLASSIFICATION,
    supportedBackendIds: ['wasm', 'webgpu', 'native-cpu'],
    supports(modelId: string): boolean {
      const value = modelId.toLowerCase();
      return value.includes('omnilingual') || value.includes('omni-asr');
    },
    async createModel(
      request,
      context,
    ): Promise<
      SpeechModel<OmnilingualCtcLoadOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>
    > {
      const sampleRate = request.options?.sampleRate ?? 16000;
      // Each session owns its own graph. Sharing one across sessions means a
      // second createSession after the first is disposed transcribes through a
      // disposed model, so loading is per-session and disposal is tracked.
      const live = new Set<OmnilingualCtcLoadedModel>();
      let disposed = false;

      return {
        info: {
          modelId: request.modelId,
          family: 'omnilingual-ctc',
          displayName: request.modelId,
          languages: ['mul'],
          backendId: context.backend.id,
        } as never,
        backend: context.backend,
        loadOptions: request.options,
        async createSession(): Promise<
          import('../../types/index.js').SpeechSession<
            LasrCtcTranscriptionOptions,
            LasrCtcNativeTranscript
          >
        > {
          if (disposed) {
            throw new Error('Omnilingual CTC model was disposed.');
          }

          const model = await loadOmnilingualCtc({
            ...request.options,
            backendId:
              request.options?.backendId ?? (context.backend.id === 'webgpu' ? 'webgpu' : 'wasm'),
          } as OmnilingualCtcLoadOptions);
          live.add(model);
          let sessionDisposed = false;

          return {
            async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
              input: AudioInputLike,
              options?: LasrCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor },
            ): Promise<TranscriptResponse<LasrCtcNativeTranscript, TFlavor>> {
              const pcm = normalizePcmInput(input, { sampleRate });
              const mono = pcm.channels[0];
              if (!mono) {
                throw new Error('Omnilingual CTC transcription requires mono audio.');
              }
              const native = await model.transcribe(mono);
              const canonical = mapLasrCtcNativeToCanonical(
                native,
                DEFAULT_OMNILINGUAL_CTC_CLASSIFICATION,
                {
                  detail: options?.detail ?? 'segments',
                  detailLevel: options?.detail ?? 'segments',
                  isFinal: native.isFinal,
                } as never,
              );

              if (options?.responseFlavor === 'canonical+native') {
                return { canonical, native } as TranscriptResponse<
                  LasrCtcNativeTranscript,
                  TFlavor
                >;
              }

              return canonical as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
            },
            dispose(): void {
              if (!sessionDisposed) {
                sessionDisposed = true;
                live.delete(model);
                model.dispose();
              }
            },
          };
        },
        dispose(): void {
          if (disposed) {
            return;
          }
          disposed = true;
          for (const model of live) {
            model.dispose();
          }
          live.clear();
        },
      };
    },
  };
}
