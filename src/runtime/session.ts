import type {
  BackendCapabilities,
  BackendProbeContext,
  BackendSelectionCriteria,
  ExecutionBackend,
  ModelClassification,
  ModelLoadRequest,
  SpeechModel,
  SpeechModelFactory,
  SpeechRuntime,
  SpeechRuntimeHooks
} from '../types/index.js';
import {
  ensureAvailableBackendCandidates,
  sortBackendCandidates,
  type BackendCandidate
} from './backend.js';
import { BackendUnavailableError, ModelLoadError } from './errors.js';
import { createRuntimeHooks } from './logging.js';

export interface DefaultSpeechRuntimeOptions {
  readonly hooks?: SpeechRuntimeHooks;
  readonly backends?: readonly ExecutionBackend[];
  readonly modelFamilies?: readonly SpeechModelFactory<any, any, any>[];
}

export class DefaultSpeechRuntime implements SpeechRuntime {
  private readonly hooks: Required<SpeechRuntimeHooks>;
  private readonly backends = new Map<string, ExecutionBackend>();
  private readonly modelFamilies = new Map<string, SpeechModelFactory<any, any, any>>();

  constructor(options: DefaultSpeechRuntimeOptions = {}) {
    this.hooks = createRuntimeHooks(options.hooks);

    for (const backend of options.backends ?? []) {
      this.registerBackend(backend);
    }

    for (const factory of options.modelFamilies ?? []) {
      this.registerModelFamily(factory);
    }
  }

  registerBackend(backend: ExecutionBackend): this {
    this.backends.set(backend.id, backend);
    this.hooks.logger.debug?.('Registered backend', { backendId: backend.id });
    return this;
  }

  registerModelFamily(factory: SpeechModelFactory<any, any, any>): this {
    this.modelFamilies.set(factory.family, factory);
    this.hooks.logger.debug?.('Registered model family', { family: factory.family });
    return this;
  }

  listBackends(): readonly ExecutionBackend[] {
    return [...this.backends.values()];
  }

  listModelFamilies(): readonly SpeechModelFactory<any, any, any>[] {
    return [...this.modelFamilies.values()];
  }

  async probeBackends(context: BackendProbeContext = {}): Promise<readonly BackendCapabilities[]> {
    return Promise.all(
      this.listBackends().map(async (backend) => {
        try {
          return await backend.probeCapabilities(context);
        } catch (error) {
          this.hooks.logger.warn?.('Backend capability probe failed', {
            backendId: backend.id,
            error
          });

          return {
            id: backend.id,
            displayName: backend.displayName,
            available: false,
            priority: 0,
            environments: context.environment ? [context.environment] : [],
            acceleration: [],
            supportedPrecisions: [],
            supportsFp16: false,
            supportsInt8: false,
            supportsSharedArrayBuffer: false,
            requiresSharedArrayBuffer: false,
            fallbackSuitable: false,
            experimental: false,
            notes: ['Capability probe failed.']
          } satisfies BackendCapabilities;
        }
      })
    );
  }

  async selectBackend(criteria: BackendSelectionCriteria = {}): Promise<ExecutionBackend> {
    const candidates = await this.collectBackendCandidates({
      environment: criteria.environment
    });
    const supported = ensureAvailableBackendCandidates(candidates, criteria);
    return sortBackendCandidates(supported, criteria)[0]!.backend;
  }

  async loadModel<
    TLoadOptions = unknown,
    TTranscriptionOptions = never,
    TNative = unknown
  >(request: ModelLoadRequest<TLoadOptions>): Promise<SpeechModel<TLoadOptions, any, TNative>> {
    const family = this.resolveModelFamily(request);
    if (!family) {
      throw new ModelLoadError(`No model family is registered for model "${request.modelId}".`, {
        family: request.family,
        modelId: request.modelId
      });
    }

    const backend = request.backend
      ? await this.requireBackend(request.backend, request.selectionCriteria)
      : await this.selectBackend(request.selectionCriteria);

    this.hooks.onProgress({
      phase: 'model-load:start',
      backendId: backend.id,
      modelId: request.modelId,
      message: `Loading ${request.modelId} with ${backend.id}.`
    });

    try {
      const model = await family.createModel(request, {
        runtime: this,
        backend,
        hooks: this.hooks
      });

      this.hooks.onProgress({
        phase: 'model-load:complete',
        backendId: backend.id,
        modelId: request.modelId,
        message: `Loaded ${request.modelId} with ${backend.id}.`
      });

      return model as SpeechModel<TLoadOptions, any, TNative>;
    } catch (error) {
      throw new ModelLoadError(`Failed to load model "${request.modelId}".`, {
        backendId: backend.id,
        family: family.family,
        cause: error
      });
    }
  }

  private async collectBackendCandidates(context: BackendProbeContext): Promise<BackendCandidate[]> {
    const candidates = await Promise.all(
      this.listBackends().map(async (backend) => ({
        backend,
        capabilities: await backend.probeCapabilities(context)
      }))
    );

    return candidates;
  }

  private resolveModelFamily(
    request: ModelLoadRequest<unknown>
  ): SpeechModelFactory<any, any, any> | undefined {
    if (request.family) {
      return this.modelFamilies.get(request.family);
    }

    if (request.classification) {
      return this.listModelFamilies().find((family) => this.matchesRequestedClassification(family, request.classification!));
    }

    return this.listModelFamilies().find((family) => family.supports(request.modelId));
  }

  private matchesRequestedClassification(
    factory: SpeechModelFactory<any, any, any>,
    classification: Partial<ModelClassification>
  ): boolean {
    if (factory.matchesClassification) {
      return factory.matchesClassification(classification);
    }

    const candidate = factory.classification;
    if (!candidate) {
      return false;
    }

    return this.classificationContains(candidate, classification);
  }

  private classificationContains(
    candidate: Partial<ModelClassification>,
    requested: Partial<ModelClassification>
  ): boolean {
    return Object.entries(requested).every(([key, value]) => {
      if (value === undefined) {
        return true;
      }

      return candidate[key as keyof ModelClassification] === value;
    });
  }

  private async requireBackend(
    backendId: string,
    criteria: BackendSelectionCriteria | undefined
  ): Promise<ExecutionBackend> {
    const backend = this.backends.get(backendId);
    if (!backend) {
      throw new BackendUnavailableError(`Backend "${backendId}" is not registered.`);
    }

    const candidates = await this.collectBackendCandidates({
      environment: criteria?.environment
    });
    const match = candidates.find((candidate) => candidate.backend.id === backendId);
    if (!match) {
      throw new BackendUnavailableError(`Backend "${backendId}" could not be probed.`);
    }

    ensureAvailableBackendCandidates([match], {
      ...criteria,
      preferredBackendIds: [backendId]
    });

    return backend;
  }
}

export function createSpeechRuntime(options: DefaultSpeechRuntimeOptions = {}): DefaultSpeechRuntime {
  return new DefaultSpeechRuntime(options);
}
