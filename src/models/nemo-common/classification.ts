import type { ModelClassification } from '../../types/index.js';

export function createModelClassification(
  base: ModelClassification,
  override: Partial<ModelClassification> = {},
): ModelClassification {
  return {
    ...base,
    ...override,
  };
}

export function describeModelClassification(classification: ModelClassification): string {
  const parts = [
    classification.ecosystem,
    classification.processor,
    classification.encoder,
    classification.topology,
    classification.decoder,
    classification.task,
    classification.family,
  ].filter((part): part is string => typeof part === 'string' && part.length > 0);

  return [...new Set(parts)].join(' / ');
}
