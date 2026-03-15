import { createDefaultAssetProvider } from '../io/index.js';
import type {
  AssetCache,
  AssetProvider,
  BackendCapabilities,
  BackendProbeContext,
  BackendSelectionCriteria,
  ExecutionBackend,
  FamilyModelLoadRequest,
  ModelClassification,
  ModelLoadRequest,
  PresetModelLoadRequest,
  SpeechModel,
  SpeechModelFactory,
  SpeechPresetFactory,
  SpeechRuntime,
  SpeechRuntimeHooks,
} from '../types/index.js';
import {
  ensureAvailableBackendCandidates,
  sortBackendCandidates,
  type BackendCandidate,
} from './backend.js';
import { BackendUnavailableError, ModelLoadError } from './errors.js';
import { createRuntimeHooks } from './logging.js';

export interface DefaultSpeechRuntimeOptions {
  readonly hooks?: SpeechRuntimeHooks;
  readonly backends?: readonly ExecutionBackend[];
  readonly modelFamilies?: readonly SpeechModelFactory<any, any, any>[];
  readonly presets?: readonly SpeechPresetFactory<any, any>[];
  readonly assetProvider?: AssetProvider;
  readonly assetCache?: AssetCache;
}

/**
 * Default runtime implementation used by `createSpeechRuntime()`.
 *
 * It keeps model families and presets in separate registries, resolves preset
 * requests into technical family requests, tracks loaded models, and owns
 * runtime-level disposal.
 */
export class DefaultSpeechRuntime implements SpeechRuntime {
  readonly assetProvider?: AssetProvider;
  readonly assetCache?: AssetCache;

  private readonly hooks: Required<SpeechRuntimeHooks>;
  private readonly backends = new Map<string, ExecutionBackend>();
  private readonly modelFamilies = new Map<string, SpeechModelFactory<any, any, any>>();
  private readonly presets = new Map<string, SpeechPresetFactory<any, any>>();
  private readonly loadedModels = new Set<SpeechModel<any, any, any>>();
  private disposed = false;

  constructor(options: DefaultSpeechRuntimeOptions = {}) {
    this.hooks = createRuntimeHooks(options.hooks);
    this.assetCache = options.assetCache;
    this.assetProvider =
      options.assetProvider ??
      createDefaultAssetProvider({
        cache: options.assetCache,
      });

    for (const backend of options.backends ?? []) {
      this.registerBackend(backend);
    }

    for (const factory of options.modelFamilies ?? []) {
      this.registerModelFamily(factory);
    }

    for (const preset of options.presets ?? []) {
      this.registerPreset(preset);
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

  registerPreset(factory: SpeechPresetFactory<any, any>): this {
    this.presets.set(factory.preset, factory);
    this.hooks.logger.debug?.('Registered preset', { preset: factory.preset });
    return this;
  }

  listBackends(): readonly ExecutionBackend[] {
    return [...this.backends.values()];
  }

  listModelFamilies(): readonly SpeechModelFactory<any, any, any>[] {
    return [...this.modelFamilies.values()];
  }

  listPresets(): readonly SpeechPresetFactory<any, any>[] {
    return [...this.presets.values()];
  }

  async probeBackends(context: BackendProbeContext = {}): Promise<readonly BackendCapabilities[]> {
    return Promise.all(
      this.listBackends().map(async (backend) => {
        try {
          return await backend.probeCapabilities(context);
        } catch (error) {
          this.hooks.logger.warn?.('Backend capability probe failed', {
            backendId: backend.id,
            error,
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
            notes: ['Capability probe failed.'],
          } satisfies BackendCapabilities;
        }
      }),
    );
  }

  async selectBackend(criteria: BackendSelectionCriteria = {}): Promise<ExecutionBackend> {
    const candidates = await this.collectBackendCandidates({
      environment: criteria.environment,
    });
    const supported = ensureAvailableBackendCandidates(candidates, criteria);
    return sortBackendCandidates(supported, criteria)[0]!.backend;
  }

  async loadModel<TLoadOptions = unknown, TNative = unknown>(
    request: ModelLoadRequest<TLoadOptions>,
  ): Promise<SpeechModel<TLoadOptions, any, TNative>> {
    if (this.disposed) {
      throw new ModelLoadError('Cannot load a model from a disposed runtime.');
    }

    const familyRequest = await this.resolveLoadRequest(request);
    const family = this.resolveModelFamily(familyRequest);
    if (!family) {
      throw new ModelLoadError(
        `No model family is registered for model "${familyRequest.modelId}".`,
        {
          family: familyRequest.family,
          modelId: familyRequest.modelId,
          preset: familyRequest.resolvedPreset,
        },
      );
    }

    const backend = familyRequest.backend
      ? await this.requireBackend(familyRequest.backend, familyRequest.selectionCriteria)
      : await this.selectBackend(familyRequest.selectionCriteria);

    this.hooks.onProgress({
      phase: 'model-load:start',
      backendId: backend.id,
      modelId: familyRequest.modelId,
      message: `Loading ${familyRequest.modelId} with ${backend.id}.`,
    });

    try {
      const model = await family.createModel(familyRequest, {
        runtime: this,
        backend,
        hooks: this.hooks,
        assetProvider: this.assetProvider,
        assetCache: this.assetCache,
      });

      this.loadedModels.add(model);

      this.hooks.onProgress({
        phase: 'model-load:complete',
        backendId: backend.id,
        modelId: familyRequest.modelId,
        message: `Loaded ${familyRequest.modelId} with ${backend.id}.`,
      });

      return model as SpeechModel<TLoadOptions, any, TNative>;
    } catch (error) {
      throw new ModelLoadError(`Failed to load model "${familyRequest.modelId}".`, {
        backendId: backend.id,
        family: family.family,
        preset: familyRequest.resolvedPreset,
        cause: error,
      });
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }

    this.disposed = true;

    const models = [...this.loadedModels];
    this.loadedModels.clear();

    await Promise.all(
      models.map(async (model) => {
        await model.dispose();
      }),
    );
  }

  private async resolveLoadRequest(
    request: ModelLoadRequest<unknown>,
  ): Promise<FamilyModelLoadRequest<any>> {
    if (request.family === undefined) {
      return this.resolvePresetRequest(request);
    }

    return request;
  }

  private async resolvePresetRequest(
    request: PresetModelLoadRequest<unknown>,
  ): Promise<FamilyModelLoadRequest<any>> {
    const preset = this.presets.get(request.preset);
    if (!preset) {
      throw new ModelLoadError(`Preset "${request.preset}" is not registered.`, {
        preset: request.preset,
        modelId: request.modelId,
      });
    }

    const resolved = await preset.resolveModelRequest(request, {
      runtime: this,
      hooks: this.hooks,
      assetProvider: this.assetProvider,
      assetCache: this.assetCache,
    });

    return {
      ...resolved,
      backend: resolved.backend ?? request.backend,
      selectionCriteria: resolved.selectionCriteria ?? request.selectionCriteria,
      classification: {
        ...resolved.classification,
        ...request.classification,
      },
    };
  }

  private async collectBackendCandidates(
    context: BackendProbeContext,
  ): Promise<BackendCandidate[]> {
    const candidates = await Promise.all(
      this.listBackends().map(async (backend) => ({
        backend,
        capabilities: await backend.probeCapabilities(context),
      })),
    );

    return candidates;
  }

  private resolveModelFamily(
    request: FamilyModelLoadRequest<unknown>,
  ): SpeechModelFactory<any, any, any> | undefined {
    const byFamily = this.modelFamilies.get(request.family);
    if (byFamily) {
      return byFamily;
    }

    if (request.classification) {
      return this.listModelFamilies().find((family) =>
        this.matchesRequestedClassification(family, request.classification!),
      );
    }

    return this.listModelFamilies().find((family) => family.supports(request.modelId));
  }

  private matchesRequestedClassification(
    factory: SpeechModelFactory<any, any, any>,
    classification: Partial<ModelClassification>,
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
    requested: Partial<ModelClassification>,
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
    criteria: BackendSelectionCriteria | undefined,
  ): Promise<ExecutionBackend> {
    const backend = this.backends.get(backendId);
    if (!backend) {
      throw new BackendUnavailableError(`Backend "${backendId}" is not registered.`);
    }

    const candidates = await this.collectBackendCandidates({
      environment: criteria?.environment,
    });
    const match = candidates.find((candidate) => candidate.backend.id === backendId);
    if (!match) {
      throw new BackendUnavailableError(`Backend "${backendId}" could not be probed.`);
    }

    ensureAvailableBackendCandidates([match], {
      ...criteria,
      preferredBackendIds: [backendId],
    });

    return backend;
  }
}

export function createSpeechRuntime(
  options: DefaultSpeechRuntimeOptions = {},
): DefaultSpeechRuntime {
  return new DefaultSpeechRuntime(options);
}
