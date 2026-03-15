export type SpeechComponentLayer = 'processor' | 'encoder' | 'decoder' | 'decoding' | 'tokenizer';

export interface ModelComponentDescriptor {
  readonly layer: SpeechComponentLayer;
  readonly module: string;
  readonly implementation: string;
  readonly shared?: boolean;
  readonly notes?: readonly string[];
}

export interface ModelArchitectureDescriptor {
  readonly processor: ModelComponentDescriptor;
  readonly encoder: ModelComponentDescriptor;
  readonly decoder: ModelComponentDescriptor;
  readonly decoding: ModelComponentDescriptor;
  readonly tokenizer: ModelComponentDescriptor;
}

export function createModelArchitecture(
  descriptor: ModelArchitectureDescriptor,
): ModelArchitectureDescriptor {
  return descriptor;
}
