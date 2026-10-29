/** A pinned ONNX bundle published for direct loading from the Hugging Face Hub. */
export interface PublishedBundle {
  readonly repoId: string;
  readonly revision: string;
  readonly files: Readonly<Record<string, string>>;
}

/**
 * Build a stable Hub URL for a known artifact. Revisions are immutable commit
 * ids so a catalog entry cannot silently change underneath a cached model.
 */
export function publishedBundleFileUrl(bundle: PublishedBundle, fileKey: string): string {
  const path = bundle.files[fileKey];
  if (!path)
    throw new Error(`Published bundle ${bundle.repoId} has no artifact named "${fileKey}".`);
  const repo = bundle.repoId.split('/').map(encodeURIComponent).join('/');
  const revision = encodeURIComponent(bundle.revision);
  const encodedPath = path.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repo}/resolve/${revision}/${encodedPath}`;
}

export const QWEN3_ASR_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/qwen3-asr-0.6b-onnx',
  revision: 'ac0bedcc0f1614c0b3982940c14bdb169ff2365d',
  files: {
    encoder: 'audio-encoder-dynamic.onnx',
    decoderPrefill: 'decoder-prefill-fp16.onnx',
    decoderStep: 'decoder-step-fp16.onnx',
    decoderData: 'decoder-fp16.onnx.data',
    tokenizer: 'tokenizer/tokenizer.json',
  },
};

export const MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/moonshine-streaming-tiny-onnx',
  revision: 'cb347a0760c2d7d6cf9083eb770376ba108e5b31',
  files: {
    encoder: 'encoder_model.onnx',
    encoderData: 'encoder_model.onnx.data',
    chunkedEncoder: 'encoder_chunked_model.onnx',
    chunkedEncoderData: 'encoder_chunked_model.onnx.data',
    decoderPrefill: 'decoder_prefill_model.onnx',
    decoderPrefillData: 'decoder_prefill_model.onnx.data',
    decoderStep: 'decoder_step_model.onnx',
    decoderStepData: 'decoder_step_model.onnx.data',
    tokenizer: 'tokenizer.json',
    generation: 'generation_config.json',
    config: 'config.json',
    preprocessor: 'preprocessor_config.json',
    processor: 'processor_config.json',
    tokenizerConfig: 'tokenizer_config.json',
  },
};
