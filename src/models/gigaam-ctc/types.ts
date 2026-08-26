import type {
  LasrCtcArtifactSource,
  LasrCtcModelConfig,
  LasrCtcModelOptions,
} from '../lasr-ctc/types.js';

export interface GigaAmModelConfig extends Omit<
  LasrCtcModelConfig,
  'ecosystem' | 'architecture' | 'processorArchitecture' | 'encoderArchitecture'
> {
  readonly ecosystem: 'gigaam';
  readonly architecture: 'gigaam-ctc';
  readonly processorArchitecture: 'gigaam-fbank';
  readonly encoderArchitecture: 'gigaam-conformer';
  readonly nFft: 320;
  readonly winLength: 320;
  readonly hopLength: 160;
  readonly featureLayout: 'mel-major';
}

export type GigaAmArtifactSource = LasrCtcArtifactSource;
export type GigaAmModelOptions = Omit<LasrCtcModelOptions, 'config' | 'source'> & {
  readonly source?: GigaAmArtifactSource;
  readonly config?: Partial<GigaAmModelConfig>;
};
